import hashlib
import os
import time
from pathlib import Path
from typing import Optional

import faiss
import numpy as np
import torch
from jaxtyping import Bool, Float, Int
from sentence_transformers import SentenceTransformer
from torch import Tensor
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

from delphi import logger

from ..config import ConstructorConfig
from .latents import (
    ActivatingExample,
    ActivationData,
    LatentRecord,
    NonActivatingExample,
)

model_cache: dict[tuple[str, str], SentenceTransformer] = {}


def get_model(name: str, device: str = "cuda") -> SentenceTransformer:
    global model_cache
    if (name, device) not in model_cache:
        logger.info(f"Loading model {name} on device {device}")
        model_cache[(name, device)] = SentenceTransformer(name, device=device)
    return model_cache[(name, device)]


def prepare_non_activating_examples(
    tokens: Int[Tensor, "examples ctx_len"],
    activations: Float[Tensor, "examples ctx_len"],
    distance: float,
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
) -> list[NonActivatingExample]:
    """
    Prepare a list of non-activating examples from input tokens and distance.

    Args:
        tokens: Tokenized input sequences.
        distance: The distance from the neighbouring latent.
    """
    return [
        NonActivatingExample(
            tokens=toks,
            activations=acts,
            distance=distance,
            str_tokens=tokenizer.batch_decode(toks),
        )
        for toks, acts in zip(tokens, activations)
    ]


def _top_k_pools(
    max_buffer: Float[Tensor, "batch"],
    split_activations: Float[Tensor, "activations ctx_len"],
    buffer_tokens: Int[Tensor, "batch ctx_len"],
) -> tuple[Int[Tensor, "examples ctx_len"], Float[Tensor, "examples ctx_len"]]:
    """
    Get the top k activation pools.

    Args:
        max_buffer: The maxima of each context window's activations.
        split_activations: The split activations.
        buffer_tokens: The buffer tokens.
        max_examples: The maximum number of examples.

    Returns:
        The token windows and activation windows.
    """
    sorted_indices = torch.argsort(max_buffer, descending=True)
    activation_windows = split_activations[sorted_indices]
    token_windows = buffer_tokens[sorted_indices]

    return token_windows, activation_windows


def pool_max_activation_windows(
    activations: Float[Tensor, "examples"],
    tokens: Int[Tensor, "windows seq"],
    ctx_indices: Int[Tensor, "examples"],
    index_within_ctx: Int[Tensor, "examples"],
    ctx_len: int,
) -> tuple[Int[Tensor, "examples ctx_len"], Float[Tensor, "examples ctx_len"]]:
    """
    Pool max activation windows from the buffer output and update the latent record.

    Args:
        activations : The activations.
        tokens : The input tokens.
        ctx_indices : The context indices.
        index_within_ctx : The index within the context.
        ctx_len : The context length.
        max_examples : The maximum number of examples.
    Returns:
        The token windows and activation windows.
    """
    # unique_ctx_indices: array of distinct context window indices in order of first
    # appearance. sequential integers from 0 to batch_size * cache_token_length//ctx_len
    # inverses: maps each activation back to its index in unique_ctx_indices
    # (can be used to dereference the context window idx of each activation)
    # lengths: the number of activations per unique context window index
    unique_ctx_indices, inverses, lengths = torch.unique_consecutive(
        ctx_indices, return_counts=True, return_inverse=True
    )

    # Get the max activation magnitude within each context window
    max_buffer = torch.segment_reduce(activations, "max", lengths=lengths)
    # Deduplicate the context windows
    new_tensor = torch.zeros(len(unique_ctx_indices), ctx_len, dtype=activations.dtype)
    new_tensor[inverses, index_within_ctx] = activations
    tokens = tokens[unique_ctx_indices]

    token_windows, activation_windows = _top_k_pools(max_buffer, new_tensor, tokens)

    return token_windows, activation_windows


def pool_centered_activation_windows_old(
    activations: Float[Tensor, "examples"],
    tokens: Float[Tensor, "windows seq"],
    n_windows_per_batch: int,
    ctx_indices: Float[Tensor, "examples"],
    index_within_ctx: Float[Tensor, "examples"],
    ctx_len: int,
) -> tuple[Float[Tensor, "examples ctx_len"], Float[Tensor, "examples ctx_len"]]:
    """
    Similar to pool_max_activation_windows. Doesn't use the ctx_indices that were
    at the start of the batch or the end of the batch, because it always tries
    to have a buffer of ctx_len*5//6 on the left and ctx_len*1//6 on the right.
    To do this, for each window, it joins the contexts from the other windows
    of the same batch, to form a new context, which is then cut to the correct shape,
    centered on the max activation.

    Args:
        activations : The activations.
        tokens : The input tokens.
        ctx_indices : The context indices.
        index_within_ctx : The index within the context.
        ctx_len : The context length.
        max_examples : The maximum number of examples.
    """

    # === Input Assertions ===
    assert activations.dim() == 1, f"activations must be 1D, got {activations.dim()}D"
    assert tokens.dim() == 2, f"tokens must be 2D, got {tokens.dim()}D"
    assert ctx_indices.dim() == 1, f"ctx_indices must be 1D, got {ctx_indices.dim()}D"
    assert (
        index_within_ctx.dim() == 1
    ), f"index_within_ctx must be 1D, got {index_within_ctx.dim()}D"
    assert len(activations) == len(ctx_indices), (
        f"activations and ctx_indices length mismatch: {len(activations)} vs "
        f"{len(ctx_indices)}"
    )
    assert len(activations) == len(index_within_ctx), (
        f"activations and index_within_ctx length mismatch: {len(activations)} vs "
        f"{len(index_within_ctx)}"
    )
    assert (
        tokens.shape[1] == ctx_len
    ), f"tokens seq dim must equal ctx_len: {tokens.shape[1]} vs {ctx_len}"
    assert n_windows_per_batch > 0, "n_windows_per_batch must be positive"
    assert ctx_len > 0, "ctx_len must be positive"
    assert (
        index_within_ctx.min() >= 0
    ), f"index_within_ctx has negative values: min={index_within_ctx.min().item()}"
    assert (
        index_within_ctx.max() < ctx_len
    ), f"index_within_ctx out of bounds: max={index_within_ctx.max().item()}, "
    f"ctx_len={ctx_len}"

    # Get unique context indices and their counts like in pool_max_activation_windows
    unique_ctx_indices, inverses, lengths = torch.unique_consecutive(
        ctx_indices, return_counts=True, return_inverse=True
    )

    # === Assertions after unique_consecutive ===
    assert len(unique_ctx_indices) == len(
        lengths
    ), "unique_ctx_indices and lengths must have same length"
    assert lengths.sum() == len(activations), (
        "lengths sum ({lengths.sum()}) must equal activations count "
        f"({len(activations)})"
    )

    assert inverses.max() < len(
        unique_ctx_indices
    ), "inverses must index into unique_ctx_indices"

    # Get the max activation magnitude within each context window
    max_buffer = torch.segment_reduce(activations, "max", lengths=lengths)

    # === Assertions after segment_reduce ===
    assert len(max_buffer) == len(unique_ctx_indices), (
        f"max_buffer length ({len(max_buffer)}) must match unique windows "
        f"({len(unique_ctx_indices)})"
    )
    assert not torch.isnan(
        max_buffer
    ).any(), "max_buffer contains NaN values after segment_reduce"

    # Get the top max_examples windows
    sorted_values, sorted_indices = torch.sort(max_buffer, descending=True)

    # this tensor has the correct activations for each context window
    temp_tensor = torch.zeros(len(unique_ctx_indices), ctx_len, dtype=activations.dtype)
    temp_tensor[inverses, index_within_ctx] = activations

    # === Assertions after temp_tensor creation ===
    assert (
        temp_tensor.shape[0] == len(unique_ctx_indices)
        and temp_tensor.shape[1] == ctx_len
    ), f"temp_tensor shape mismatch: {temp_tensor.shape}"

    unique_ctx_indices = unique_ctx_indices[sorted_indices]
    temp_tensor = temp_tensor[sorted_indices]

    # if a element in unique_ctx_indices is divisible by n_windows_per_batch it
    # the start of a new batch, so we discard it
    modulo = unique_ctx_indices % n_windows_per_batch
    not_first_position = modulo != 0
    # remove also the elements that are at the end of the batch
    not_last_position = modulo != n_windows_per_batch - 1
    mask = not_first_position & not_last_position

    # === Assertions after mask creation ===
    assert mask.dtype == torch.bool, f"mask must be boolean, got {mask.dtype}"

    unique_ctx_indices = unique_ctx_indices[mask]
    temp_tensor = temp_tensor[mask]
    if len(unique_ctx_indices) == 0:
        return torch.zeros(0, ctx_len), torch.zeros(0, ctx_len)

    # Vectorized operations for all windows at once
    n_windows = len(unique_ctx_indices)

    # === Assertions after filtering ===
    assert n_windows > 0, "n_windows must be positive after filtering"
    assert (
        temp_tensor.shape[0] == n_windows
    ), f"temp_tensor rows ({temp_tensor.shape[0]}) must equal n_windows ({n_windows})"

    # Create indices for previous, current, and next windows
    prev_indices = unique_ctx_indices - 1
    next_indices = unique_ctx_indices + 1

    # === Assertions for prev/next indices ===
    assert (
        prev_indices >= 0
    ).all(), f"prev_indices has negative values (min={prev_indices.min().item()})"
    assert (next_indices < tokens.shape[0]).all(), (
        f"next_indices out of bounds (max={next_indices.max().item()}, "
        f"tokens.shape[0]={tokens.shape[0]})"
    )

    # Create a tensor to hold all concatenated tokens
    all_tokens = torch.cat(
        [tokens[prev_indices], tokens[unique_ctx_indices], tokens[next_indices]], dim=1
    )  # Shape: [n_windows, ctx_len*3]

    # === Assertions after all_tokens creation ===
    assert all_tokens.shape == (
        n_windows,
        ctx_len * 3,
    ), (
        f"all_tokens shape mismatch: {all_tokens.shape} vs expected "
        f"({n_windows}, {ctx_len * 3})"
    )

    # Create tensor for all activations
    final_tensor = torch.zeros((n_windows, ctx_len * 3), dtype=activations.dtype)
    final_tensor[:, ctx_len : ctx_len * 2] = (
        temp_tensor  # Set current window activations
    )

    # Set previous window activations where available
    prev_mask = torch.isin(prev_indices, unique_ctx_indices)
    if prev_mask.any():
        prev_locations = torch.where(
            unique_ctx_indices.unsqueeze(1) == prev_indices.unsqueeze(0)
        )[1]
        # === Assertions for prev_locations ===
        assert len(prev_locations) == prev_mask.sum(), (
            f"prev_locations length ({len(prev_locations)}) must match "
            f"prev_mask count ({prev_mask.sum().item()})"
        )
        assert (
            prev_locations < n_windows
        ).all(), f"prev_locations out of bounds: max={prev_locations.max().item()}"
        final_tensor[prev_mask, :ctx_len] = temp_tensor[prev_locations]

    # Set next window activations where available
    next_mask = torch.isin(next_indices, unique_ctx_indices)
    if next_mask.any():
        next_locations = torch.where(
            unique_ctx_indices.unsqueeze(1) == next_indices.unsqueeze(0)
        )[1]
        # === Assertions for next_locations ===
        assert len(next_locations) == next_mask.sum(), (
            f"next_locations length ({len(next_locations)}) must match "
            f"next_mask count ({next_mask.sum().item()})"
        )
        assert (
            next_locations < n_windows
        ).all(), f"next_locations out of bounds: max={next_locations.max().item()}"
        final_tensor[next_mask, ctx_len * 2 :] = temp_tensor[next_locations]

    # Find max activation indices
    max_activation_indices = torch.argmax(temp_tensor, dim=1) + ctx_len

    # === Assertions for max_activation_indices ===
    assert (
        len(max_activation_indices) == n_windows
    ), "max_activation_indices length mismatch"
    assert (
        max_activation_indices >= ctx_len
    ).all(), "max_activation_indices must be >= ctx_len"
    assert (
        max_activation_indices < ctx_len * 2
    ).all(), "max_activation_indices must be < ctx_len*2"

    # Calculate left for all windows
    left_positions = max_activation_indices - (ctx_len - ctx_len // 4)

    # === Assertions for left_positions ===
    assert (
        left_positions >= 0
    ).all(), f"left_positions has negative values: min={left_positions.min().item()}"
    assert (left_positions + ctx_len <= ctx_len * 3).all(), (
        f"left_positions would exceed bounds: "
        f"max_right={left_positions.max().item() + ctx_len}, limit={ctx_len * 3}"
    )

    # Create index tensors for gathering
    batch_indices = torch.arange(n_windows).unsqueeze(1)
    pos_indices = torch.arange(ctx_len).unsqueeze(0)
    gather_indices = left_positions.unsqueeze(1) + pos_indices

    # === Assertions for gather_indices ===
    assert gather_indices.shape == (
        n_windows,
        ctx_len,
    ), f"gather_indices shape mismatch: {gather_indices.shape}"
    assert (
        gather_indices >= 0
    ).all(), f"gather_indices has negative values: min={gather_indices.min().item()}"
    assert (
        gather_indices < ctx_len * 3
    ).all(), f"gather_indices out of bounds: max={gather_indices.max().item()}"

    # Gather the final windows
    token_windows = all_tokens[batch_indices, gather_indices]
    activation_windows = final_tensor[batch_indices, gather_indices]

    # === Output Assertions ===
    assert (
        token_windows.shape[0] == activation_windows.shape[0]
    ), "token and activation window counts must match"
    assert token_windows.shape[1] == ctx_len, "token_windows must have ctx_len columns"
    assert (
        activation_windows.shape[1] == ctx_len
    ), "activation_windows must have ctx_len columns"
    assert not torch.isnan(
        activation_windows
    ).any(), "activation_windows contains NaN values"

    return token_windows, activation_windows


def pool_centered_activation_windows_new(
    activations: Float[Tensor, "examples"],
    tokens: Float[Tensor, "windows seq"],
    n_windows_per_batch: int,
    ctx_indices: Float[Tensor, "examples"],
    index_within_ctx: Float[Tensor, "examples"],
    ctx_len: int,
) -> tuple[Float[Tensor, "examples ctx_len"], Float[Tensor, "examples ctx_len"]]:
    """
    OPTIMIZED version of pool_centered_activation_windows.

    Key optimizations:
    1. Uses torch.searchsorted instead of O(n^2) torch.where comparisons
    2. Builds index mapping tensor once for O(1) lookups
    3. Avoids creating large intermediate tensors from unsqueeze operations

    Similar to pool_max_activation_windows. Doesn't use the ctx_indices that were
    at the start of the batch or the end of the batch, because it always tries
    to have a buffer of ctx_len*5//6 on the left and ctx_len*1//6 on the right.
    To do this, for each window, it joins the contexts from the other windows
    of the same batch, to form a new context, which is then cut to the correct shape,
    centered on the max activation.

    Args:
        activations : The activations.
        tokens : The input tokens.
        ctx_indices : The context indices.
        index_within_ctx : The index within the context.
        ctx_len : The context length.
        max_examples : The maximum number of examples.
    """

    # === Input Assertions ===
    assert activations.dim() == 1, f"activations must be 1D, got {activations.dim()}D"
    assert tokens.dim() == 2, f"tokens must be 2D, got {tokens.dim()}D"
    assert ctx_indices.dim() == 1, f"ctx_indices must be 1D, got {ctx_indices.dim()}D"
    assert (
        index_within_ctx.dim() == 1
    ), f"index_within_ctx must be 1D, got {index_within_ctx.dim()}D"
    assert len(activations) == len(ctx_indices), (
        f"activations and ctx_indices length mismatch: {len(activations)} vs "
        f"{len(ctx_indices)}"
    )
    assert len(activations) == len(index_within_ctx), (
        f"activations and index_within_ctx length mismatch: {len(activations)} vs "
        f"{len(index_within_ctx)}"
    )
    assert (
        tokens.shape[1] == ctx_len
    ), f"tokens seq dim must equal ctx_len: {tokens.shape[1]} vs {ctx_len}"
    assert n_windows_per_batch > 0, "n_windows_per_batch must be positive"
    assert ctx_len > 0, "ctx_len must be positive"
    assert (
        index_within_ctx.min() >= 0
    ), f"index_within_ctx has negative values: min={index_within_ctx.min().item()}"
    assert index_within_ctx.max() < ctx_len, (
        f"index_within_ctx out of bounds: max={index_within_ctx.max().item()}, "
        f"ctx_len={ctx_len}"
    )

    # Get unique context indices and their counts like in pool_max_activation_windows
    unique_ctx_indices, inverses, lengths = torch.unique_consecutive(
        ctx_indices, return_counts=True, return_inverse=True
    )

    # === Assertions after unique_consecutive ===
    assert len(unique_ctx_indices) == len(
        lengths
    ), "unique_ctx_indices and lengths must have same length"
    assert lengths.sum() == len(activations), (
        f"lengths sum ({lengths.sum()}) must equal activations count "
        f"({len(activations)})"
    )
    assert inverses.max() < len(
        unique_ctx_indices
    ), "inverses must index into unique_ctx_indices"

    # Get the max activation magnitude within each context window
    max_buffer = torch.segment_reduce(activations, "max", lengths=lengths)

    # === Assertions after segment_reduce ===
    assert len(max_buffer) == len(unique_ctx_indices), (
        f"max_buffer length ({len(max_buffer)}) must match unique windows "
        f"({len(unique_ctx_indices)})"
    )
    assert not torch.isnan(
        max_buffer
    ).any(), "max_buffer contains NaN values after segment_reduce"

    # Get the top max_examples windows
    sorted_values, sorted_indices = torch.sort(max_buffer, descending=True)

    # this tensor has the correct activations for each context window
    temp_tensor = torch.zeros(len(unique_ctx_indices), ctx_len, dtype=activations.dtype)
    temp_tensor[inverses, index_within_ctx] = activations

    # === Assertions after temp_tensor creation ===
    assert (
        temp_tensor.shape[0] == len(unique_ctx_indices)
        and temp_tensor.shape[1] == ctx_len
    ), f"temp_tensor shape mismatch: {temp_tensor.shape}"

    unique_ctx_indices = unique_ctx_indices[sorted_indices]
    temp_tensor = temp_tensor[sorted_indices]

    # if a element in unique_ctx_indices is divisible by n_windows_per_batch it
    # the start of a new batch, so we discard it
    modulo = unique_ctx_indices % n_windows_per_batch
    not_first_position = modulo != 0
    # remove also the elements that are at the end of the batch
    not_last_position = modulo != n_windows_per_batch - 1
    mask = not_first_position & not_last_position

    # === Assertions after mask creation ===
    assert mask.dtype == torch.bool, f"mask must be boolean, got {mask.dtype}"

    unique_ctx_indices = unique_ctx_indices[mask]
    temp_tensor = temp_tensor[mask]
    if len(unique_ctx_indices) == 0:
        return torch.zeros(0, ctx_len), torch.zeros(0, ctx_len)

    # Vectorized operations for all windows at once
    n_windows = len(unique_ctx_indices)

    # === Assertions after filtering ===
    assert n_windows > 0, "n_windows must be positive after filtering"
    assert (
        temp_tensor.shape[0] == n_windows
    ), f"temp_tensor rows ({temp_tensor.shape[0]}) must equal n_windows ({n_windows})"

    # Create indices for previous, current, and next windows
    prev_indices = unique_ctx_indices - 1
    next_indices = unique_ctx_indices + 1

    # === Assertions for prev/next indices ===
    assert (
        prev_indices >= 0
    ).all(), f"prev_indices has negative values (min={prev_indices.min().item()})"
    assert (next_indices < tokens.shape[0]).all(), (
        f"next_indices out of bounds (max={next_indices.max().item()}, "
        f"tokens.shape[0]={tokens.shape[0]})"
    )

    # Create a tensor to hold all concatenated tokens
    all_tokens = torch.cat(
        [tokens[prev_indices], tokens[unique_ctx_indices], tokens[next_indices]], dim=1
    )  # Shape: [n_windows, ctx_len*3]

    # === Assertions after all_tokens creation ===
    assert all_tokens.shape == (
        n_windows,
        ctx_len * 3,
    ), (
        f"all_tokens shape mismatch: {all_tokens.shape} vs expected ({n_windows}, "
        f"{ctx_len * 3})"
    )

    # Create tensor for all activations
    final_tensor = torch.zeros((n_windows, ctx_len * 3), dtype=activations.dtype)
    final_tensor[:, ctx_len : ctx_len * 2] = (
        temp_tensor  # Set current window activations
    )

    # === OPTIMIZED SECTION ===
    # Build a mapping from ctx_index value -> position in unique_ctx_indices
    # This allows O(1) lookups instead of O(n^2) comparisons

    # Sort unique_ctx_indices to use searchsorted (need to track original positions)
    sorted_unique, sort_perm = torch.sort(unique_ctx_indices)

    # === Assertions after sorting for searchsorted ===
    assert len(sorted_unique) == n_windows, "sorted_unique length must match n_windows"
    assert len(sort_perm) == n_windows, "sort_perm length must match n_windows"
    assert (
        sorted_unique[1:] >= sorted_unique[:-1]
    ).all(), "sorted_unique must be sorted"

    # For prev_indices: find which ones exist in unique_ctx_indices
    # searchsorted gives us where each prev_index would be inserted
    prev_insert_pos = torch.searchsorted(sorted_unique, prev_indices)
    # Clamp to valid range for comparison
    prev_insert_pos_clamped = prev_insert_pos.clamp(0, len(sorted_unique) - 1)
    # Check if the value at that position actually matches
    prev_mask = sorted_unique[prev_insert_pos_clamped] == prev_indices

    if prev_mask.any():
        # Get the original positions in unique_ctx_indices (before sorting)
        # sort_perm[prev_insert_pos_clamped[prev_mask]] gives the original indices
        prev_locations = sort_perm[prev_insert_pos_clamped[prev_mask]]
        # === Assertions for prev_locations ===
        assert len(prev_locations) == prev_mask.sum(), (
            f"prev_locations length ({len(prev_locations)}) must match "
            f"prev_mask count ({prev_mask.sum().item()})"
        )
        assert (
            prev_locations < n_windows
        ).all(), f"prev_locations out of bounds: max={prev_locations.max().item()}"
        final_tensor[prev_mask, :ctx_len] = temp_tensor[prev_locations]

    # For next_indices: same approach
    next_insert_pos = torch.searchsorted(sorted_unique, next_indices)
    next_insert_pos_clamped = next_insert_pos.clamp(0, len(sorted_unique) - 1)
    next_mask = sorted_unique[next_insert_pos_clamped] == next_indices

    if next_mask.any():
        next_locations = sort_perm[next_insert_pos_clamped[next_mask]]
        # === Assertions for next_locations ===
        assert len(next_locations) == next_mask.sum(), (
            f"next_locations length ({len(next_locations)}) must match "
            f"next_mask count ({next_mask.sum().item()})"
        )
        assert (
            next_locations < n_windows
        ).all(), f"next_locations out of bounds: max={next_locations.max().item()}"
        final_tensor[next_mask, ctx_len * 2 :] = temp_tensor[next_locations]

    # Find max activation indices
    max_activation_indices = torch.argmax(temp_tensor, dim=1) + ctx_len

    # === Assertions for max_activation_indices ===
    assert (
        len(max_activation_indices) == n_windows
    ), "max_activation_indices length mismatch"
    assert (
        max_activation_indices >= ctx_len
    ).all(), "max_activation_indices must be >= ctx_len"
    assert (
        max_activation_indices < ctx_len * 2
    ).all(), "max_activation_indices must be < ctx_len*2"

    # Calculate left for all windows
    left_positions = max_activation_indices - (ctx_len - ctx_len // 4)

    # === Assertions for left_positions ===
    assert (
        left_positions >= 0
    ).all(), f"left_positions has negative values: min={left_positions.min().item()}"
    assert (left_positions + ctx_len <= ctx_len * 3).all(), (
        f"left_positions would exceed bounds: "
        f"max_right={left_positions.max().item() + ctx_len}, limit={ctx_len * 3}"
    )

    # Create index tensors for gathering
    batch_indices = torch.arange(n_windows, device=activations.device).unsqueeze(1)
    pos_indices = torch.arange(ctx_len, device=activations.device).unsqueeze(0)
    gather_indices = left_positions.unsqueeze(1) + pos_indices

    # === Assertions for gather_indices ===
    assert gather_indices.shape == (
        n_windows,
        ctx_len,
    ), f"gather_indices shape mismatch: {gather_indices.shape}"
    assert (
        gather_indices >= 0
    ).all(), f"gather_indices has negative values: min={gather_indices.min().item()}"
    assert (
        gather_indices < ctx_len * 3
    ).all(), f"gather_indices out of bounds: max={gather_indices.max().item()}"

    # Gather the final windows
    token_windows = all_tokens[batch_indices, gather_indices]
    activation_windows = final_tensor[batch_indices, gather_indices]

    # === Output Assertions ===
    assert (
        token_windows.shape[0] == activation_windows.shape[0]
    ), "token and activation window counts must match"
    assert token_windows.shape[1] == ctx_len, "token_windows must have ctx_len columns"
    assert (
        activation_windows.shape[1] == ctx_len
    ), "activation_windows must have ctx_len columns"
    assert not torch.isnan(
        activation_windows
    ).any(), "activation_windows contains NaN values"

    return token_windows, activation_windows


def constructor(
    record: LatentRecord,
    activation_data: ActivationData,
    constructor_cfg: ConstructorConfig,
    tokens: Int[Tensor, "batch seq"],
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    all_data: Optional[dict[int, ActivationData]] = None,
    seed: int = 42,
) -> LatentRecord | None:
    total_start = time.perf_counter()

    cache_ctx_len = tokens.shape[1]
    example_ctx_len = constructor_cfg.example_ctx_len
    source_non_activating = constructor_cfg.non_activating_source
    n_not_active = constructor_cfg.n_non_activating
    min_examples = constructor_cfg.min_examples

    # Get all positions where the latent is active
    prep_start = time.perf_counter()
    flat_indices = (
        activation_data.locations[:, 0] * cache_ctx_len
        + activation_data.locations[:, 1]
    )
    ctx_indices = flat_indices // example_ctx_len
    index_within_ctx = flat_indices % example_ctx_len
    n_windows_per_batch = tokens.shape[1] // example_ctx_len
    reshaped_tokens = tokens.reshape(-1, example_ctx_len)
    n_windows = reshaped_tokens.shape[0]
    unique_batch_pos = ctx_indices.unique()
    mask = torch.ones(n_windows, dtype=torch.bool)
    mask[unique_batch_pos] = False
    # Indices where the latent is not active
    non_active_indices = mask.nonzero(as_tuple=False).squeeze()
    activations = activation_data.activations
    # per context frequency
    record.per_context_frequency = len(unique_batch_pos) / n_windows
    prep_time = time.perf_counter() - prep_start

    # Add activation examples to the record in place
    pool_start = time.perf_counter()
    if constructor_cfg.center_examples:
        # Time and run OLD implementation
        old_start = time.perf_counter()
        token_windows_old, act_windows_old = pool_centered_activation_windows_old(
            activations=activations,
            tokens=reshaped_tokens,
            n_windows_per_batch=n_windows_per_batch,
            ctx_indices=ctx_indices,
            index_within_ctx=index_within_ctx,
            ctx_len=example_ctx_len,
        )
        old_time = time.perf_counter() - old_start

        # Time and run NEW implementation
        new_start = time.perf_counter()
        token_windows_new, act_windows_new = pool_centered_activation_windows_new(
            activations=activations,
            tokens=reshaped_tokens,
            n_windows_per_batch=n_windows_per_batch,
            ctx_indices=ctx_indices,
            index_within_ctx=index_within_ctx,
            ctx_len=example_ctx_len,
        )
        new_time = time.perf_counter() - new_start

        # === Compare outputs ===
        shape_match = (
            token_windows_old.shape == token_windows_new.shape
            and act_windows_old.shape == act_windows_new.shape
        )

        if not shape_match:
            logger.error(
                f"[POOL CENTERED] Shape mismatch! "
                f"old_tokens={token_windows_old.shape}, "
                f"new_tokens={token_windows_new.shape}, "
                f"old_acts={act_windows_old.shape}, "
                f"new_acts={act_windows_new.shape}"
            )
            raise AssertionError(
                f"pool_centered_activation_windows old vs new shape mismatch: "
                f"old_tokens={token_windows_old.shape}, "
                f"new_tokens={token_windows_new.shape}"
            )
        else:
            # Check if tensors are equal
            tokens_equal = torch.equal(token_windows_old, token_windows_new)
            acts_close = torch.allclose(
                act_windows_old, act_windows_new, rtol=1e-5, atol=1e-8
            )

            if not tokens_equal:
                diff_count = (token_windows_old != token_windows_new).sum().item()
                logger.error(
                    f"[POOL CENTERED] Token windows differ! "
                    f"{diff_count} elements different out of "
                    f"{token_windows_old.numel()}"
                )
                raise AssertionError(
                    f"pool_centered_activation_windows old vs new token mismatch: "
                    f"{diff_count} elements differ"
                )
            if not acts_close:
                diff = (act_windows_old - act_windows_new).abs()
                max_diff = diff.max().item()
                mean_diff = diff.mean().item()
                logger.error(
                    f"[POOL CENTERED] Activation windows differ! "
                    f"max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}"
                )
                raise AssertionError(
                    f"pool_centered_activation_windows old vs new activation mismatch: "
                    f"max_diff={max_diff:.6f}"
                )

            # Log timing comparison (always, since we're always comparing)
            logger.warning(
                f"[POOL CENTERED TIMING] old={old_time:.4f}s, new={new_time:.4f}s, "
                f"speedup={old_time/new_time if new_time > 0 else float('inf'):.2f}x, "
                f"n_activations={len(activations)}, outputs_match=True"
            )

        # Use new implementation's output (it's faster and verified to match)
        token_windows, act_windows = token_windows_new, act_windows_new
    else:
        token_windows, act_windows = pool_max_activation_windows(
            activations=activations,
            tokens=reshaped_tokens,
            ctx_indices=ctx_indices,
            index_within_ctx=index_within_ctx,
            ctx_len=example_ctx_len,
        )
    pool_time = time.perf_counter() - pool_start

    examples_start = time.perf_counter()
    record.examples = [
        ActivatingExample(
            tokens=toks,
            activations=acts,
        )
        for toks, acts in zip(token_windows, act_windows)
    ]
    examples_time = time.perf_counter() - examples_start

    if len(record.examples) < min_examples:
        logger.warning(
            f"Not enough examples to explain the latent: {len(record.examples)}"
        )
        # Not enough examples to explain the latent
        return None

    non_act_start = time.perf_counter()
    if source_non_activating == "random":
        # Add random non-activating examples to the record in place
        non_activating_examples = random_non_activating_windows(
            available_indices=non_active_indices,
            reshaped_tokens=reshaped_tokens,
            n_not_active=n_not_active,
            seed=seed,
            tokenizer=tokenizer,
        )
    elif source_non_activating == "neighbours":
        assert all_data is not None, "All data is required for neighbour constructor"
        non_activating_examples = neighbour_non_activation_windows(
            record,
            not_active_mask=mask,
            tokens=tokens,
            all_data=all_data,
            ctx_len=example_ctx_len,
            n_not_active=n_not_active,
            seed=seed,
            tokenizer=tokenizer,
        )
    elif source_non_activating == "FAISS":
        non_activating_examples = faiss_non_activation_windows(
            available_indices=non_active_indices,
            record=record,
            tokens=tokens,
            ctx_len=example_ctx_len,
            tokenizer=tokenizer,
            n_not_active=n_not_active,
            embedding_model=constructor_cfg.faiss_embedding_model,
            seed=seed,
            cache_enabled=constructor_cfg.faiss_embedding_cache_enabled,
            cache_dir=constructor_cfg.faiss_embedding_cache_dir,
        )
    elif source_non_activating == "quantile":
        non_activating_examples = quantile_non_activating_windows(
            activation_data=activation_data,
            reshaped_tokens=reshaped_tokens,
            cache_ctx_len=cache_ctx_len,
            example_ctx_len=example_ctx_len,
            n_windows=n_windows,
            n_not_active=n_not_active,
            quantile=constructor_cfg.non_activating_quantile,
            tokenizer=tokenizer,
            seed=seed,
        )
    elif source_non_activating == "threshold":
        non_activating_examples = threshold_non_activating_windows(
            activation_data=activation_data,
            reshaped_tokens=reshaped_tokens,
            cache_ctx_len=cache_ctx_len,
            example_ctx_len=example_ctx_len,
            n_windows=n_windows,
            n_not_active=n_not_active,
            threshold=constructor_cfg.non_activating_threshold,
            tokenizer=tokenizer,
            seed=seed,
        )
    else:
        raise ValueError(f"Invalid non-activating source: {source_non_activating}")
    non_act_time = time.perf_counter() - non_act_start
    record.not_active = non_activating_examples

    total_time = time.perf_counter() - total_start
    n_active = len(activation_data.activations)
    n_unique_windows = len(unique_batch_pos)

    logger.debug(
        f"[DEBUG TIMING] constructor: total={total_time:.2f}s, "
        f"prep={prep_time:.2f}s, pool={pool_time:.2f}s, "
        f"examples={examples_time:.2f}s, non_act={non_act_time:.2f}s, "
        f"n_active={n_active}, n_unique_windows={n_unique_windows}"
    )

    return record


def create_token_key(tokens_tensor, ctx_len):
    """
    Create a file key based on token tensors without detokenization.

    Args:
        tokens_tensor: Tensor of tokens
        ctx_len: Context length

    Returns:
        A string key
    """
    h = hashlib.md5()
    total_tokens = 0

    # Process a sample of elements (first, middle, last)
    num_samples = len(tokens_tensor)
    indices_to_hash = (
        [0, num_samples // 2, -1] if num_samples >= 3 else range(num_samples)
    )

    for idx in indices_to_hash:
        if 0 <= idx < num_samples or (idx == -1 and num_samples > 0):
            # Convert tensor to bytes and hash it
            token_bytes = tokens_tensor[idx].cpu().numpy().tobytes()
            h.update(token_bytes)
            total_tokens += len(tokens_tensor[idx])

    # Add collection shape to make collisions less likely
    shape_str = f"{tokens_tensor.shape}"
    h.update(shape_str.encode())

    return f"{h.hexdigest()[:12]}_items{num_samples}_{ctx_len}"


def faiss_non_activation_windows(
    available_indices: Float[Tensor, "windows"],
    record: LatentRecord,
    tokens: Float[Tensor, "batch seq"],
    ctx_len: int,
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    n_not_active: int,
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    seed: int = 42,
    cache_enabled: bool = True,
    cache_dir: str = ".embedding_cache",
) -> list[NonActivatingExample]:
    """
    Generate hard negative examples using FAISS similarity search based
    on text embeddings.

    This function builds a FAISS index over non-activating examples and
    finds examples that are semantically similar to the activating examples
    based on text embeddings.

    Args:
        available_indices: Indices of windows where the latent is not active
        record: The latent record containing activating examples
        tokens: The input tokens
        ctx_len: The context length for examples
        tokenizer: The tokenizer to decode tokens
        n_not_active: Number of non-activating examples to generate
        embedding_model: Model used for text embeddings
        seed: Random seed
        cache_enabled: Whether to cache embeddings
        cache_dir: Directory to store cached embeddings

    Returns:
        A list of non-activating examples that are semantically similar to
            activating examples
    """
    torch.manual_seed(seed)
    if n_not_active == 0:
        return []

    # Check if we have enough non-activating examples
    if available_indices.numel() < n_not_active:
        logger.warning("Not enough non-activating examples available")
        return []

    # Reshape tokens to get context windows
    reshaped_tokens = tokens.reshape(-1, ctx_len)

    # Get non-activating token windows
    non_activating_tokens = reshaped_tokens[available_indices]

    # Define cache directory structure
    cache_dir = os.environ.get("DELPHI_CACHE_DIR", cache_dir)
    embedding_model_name = embedding_model.split("/")[-1]
    cache_path = Path(cache_dir) / embedding_model_name

    # Get activating example texts

    activating_texts = [
        "".join(tokenizer.batch_decode(example.tokens))
        for example in record.examples[: min(10, len(record.examples))]
    ]

    if not activating_texts:
        logger.warning("No activating examples available")
        return []

    # Create unique cache keys for both activating and non-activating texts
    # Use the hash of the concatenated texts to ensure uniqueness
    non_activating_cache_key = create_token_key(non_activating_tokens, ctx_len)
    activating_cache_key = create_token_key(
        torch.stack(
            [
                example.tokens
                for example in record.examples[: min(10, len(record.examples))]
            ]
        ),
        ctx_len,
    )

    # Cache files for activating and non-activating embeddings
    non_activating_cache_file = cache_path / f"{non_activating_cache_key}.faiss"
    activating_cache_file = cache_path / f"{activating_cache_key}.npy"

    # Try to load cached non-activating embeddings
    index = None
    if cache_enabled and non_activating_cache_file.exists():
        try:
            index = faiss.read_index(str(non_activating_cache_file), faiss.IO_FLAG_MMAP)
            logger.info(f"Loaded non-activating index from {non_activating_cache_file}")
        except Exception as e:
            logger.warning(f"Error loading cached embeddings: {repr(e)}")

    if index is None:
        logger.info("Decoding non-activating tokens...")
        non_activating_texts = [
            "".join(tokenizer.batch_decode(tokens)) for tokens in non_activating_tokens
        ]

        logger.info("Computing non-activating embeddings...")
        non_activating_embeddings = get_model(embedding_model).encode(
            non_activating_texts, show_progress_bar=False
        )
        dim = non_activating_embeddings.shape[1]
        index = faiss.IndexFlatL2(dim)

        index.add(non_activating_embeddings)  # type: ignore
        if cache_enabled:
            os.makedirs(cache_path, exist_ok=True)
            faiss.write_index(index, str(non_activating_cache_file))
            logger.info(
                f"Cached non-activating embeddings to {non_activating_cache_file}"
            )

    activating_embeddings = None
    if cache_enabled and activating_cache_file.exists():
        try:
            activating_embeddings = np.load(activating_cache_file)
            logger.info(
                f"Loaded cached activating embeddings from {activating_cache_file}"
            )
        except Exception as e:
            logger.warning(f"Error loading cached embeddings: {repr(e)}")
    # Compute embeddings for activating examples if not cached
    if activating_embeddings is None:
        logger.info("Computing activating embeddings...")
        activating_embeddings = get_model(embedding_model).encode(
            activating_texts, show_progress_bar=False
        )
        # Cache the embeddings
        if cache_enabled:
            os.makedirs(cache_path, exist_ok=True)
            np.save(activating_cache_file, activating_embeddings)
            logger.info(f"Cached activating embeddings to {activating_cache_file}")

    # Search for the nearest neighbors to each activating example
    collected_indices = set()
    hard_negative_indices = []

    # For each activating example, find the closest non-activating examples
    for embedding in activating_embeddings:
        # Skip if we already have enough examples
        if len(hard_negative_indices) >= n_not_active:
            break

        # Search for similar non-activating examples
        distances, indices = index.search(embedding.reshape(1, -1), n_not_active)  # type: ignore

        # Add new indices that haven't been collected yet
        for idx in indices[0]:
            if (
                idx not in collected_indices
                and len(hard_negative_indices) < n_not_active
            ):
                hard_negative_indices.append(idx)
                collected_indices.add(idx)

    # Get the token windows for the selected hard negatives
    selected_tokens = non_activating_tokens[hard_negative_indices]
    # Create non-activating examples
    return prepare_non_activating_examples(
        selected_tokens,
        torch.zeros_like(selected_tokens),
        -1.0,  # Using -1.0 as the distance since these are not neighbour-based
        tokenizer,
    )


def neighbour_non_activation_windows(
    record: LatentRecord,
    not_active_mask: Bool[Tensor, "windows"],
    tokens: Int[Tensor, "batch seq"],
    all_data: dict[int, ActivationData],
    ctx_len: int,
    n_not_active: int,
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    seed: int = 42,
):
    """
    Generate random activation windows and update the latent record.

    Args:
        record (LatentRecord): The latent record to update.
        not_active_mask (TensorType["n_windows"]): The mask of the non-active windows.
        tokens (TensorType["batch", "seq"]): The input tokens.
        all_data (AllData): The all data containing activations and locations.
        ctx_len (int): The context length.
        n_not_active (int): The number of non-activating examples per latent.
        tokenizer (PreTrainedTokenizer | PreTrainedTokenizerFast): The tokenizer.
        seed (int): The random seed.
    """
    torch.manual_seed(seed)
    if n_not_active == 0:
        return []

    assert (
        record.neighbours is not None
    ), "Neighbours are not set, please precompute them"

    cache_token_length = tokens.shape[1]
    reshaped_tokens = tokens.reshape(-1, ctx_len)
    n_windows = reshaped_tokens.shape[0]
    # TODO: For now we use at most 10 examples per neighbour, we may want to allow a
    # variable number of examples per neighbour
    n_examples_per_neighbour = 10

    number_examples = 0
    all_examples = []
    for neighbour in record.neighbours:
        if number_examples >= n_not_active:
            break
        # get the locations of the neighbour
        if neighbour.latent_index not in all_data:
            continue
        locations = all_data[neighbour.latent_index].locations
        activations = all_data[neighbour.latent_index].activations
        # get the active window indices
        flat_indices = locations[:, 0] * cache_token_length + locations[:, 1]
        ctx_indices = flat_indices // ctx_len
        index_within_ctx = flat_indices % ctx_len
        # Set the mask to True for the unique locations
        unique_batch_pos_active = ctx_indices.unique()

        mask = torch.zeros(n_windows, dtype=torch.bool)
        mask[unique_batch_pos_active] = True

        # Get the indices where mask and not_active_mask are True
        mask = mask & not_active_mask

        available_indices = mask.nonzero().flatten()

        mask_ctx = torch.isin(ctx_indices, available_indices)
        available_ctx_indices = ctx_indices[mask_ctx]
        available_index_within_ctx = index_within_ctx[mask_ctx]
        activations = activations[mask_ctx]
        # If there are no available indices, skip this neighbour
        if activations.numel() == 0:
            continue
        token_windows, token_activations = pool_max_activation_windows(
            activations=activations,
            tokens=reshaped_tokens,
            ctx_indices=available_ctx_indices,
            index_within_ctx=available_index_within_ctx,
            ctx_len=ctx_len,
        )
        token_windows = token_windows[:n_examples_per_neighbour]
        token_activations = token_activations[:n_examples_per_neighbour]
        # use the first n_examples_per_neighbour examples,
        # which will be the most active examples
        examples_used = len(token_windows)
        all_examples.extend(
            prepare_non_activating_examples(
                token_windows,
                token_activations,  # activations of neighbour
                -neighbour.distance,
                tokenizer,
            )
        )
        number_examples += examples_used
    if len(all_examples) == 0:
        logger.warning(
            "No examples found, falling back to random non-activating examples"
        )
        non_active_indices = not_active_mask.nonzero(as_tuple=False).squeeze()

        return random_non_activating_windows(
            available_indices=non_active_indices,
            reshaped_tokens=reshaped_tokens,
            n_not_active=n_not_active,
            tokenizer=tokenizer,
        )
    return all_examples


def compute_per_window_activations(
    activation_data: ActivationData,
    cache_ctx_len: int,
    example_ctx_len: int,
    n_windows: int,
) -> Float[Tensor, "n_windows"]:
    """
    Compute per-window maximum activation values from stored activation data.

    Args:
        activation_data: Activation data containing locations and activations.
        cache_ctx_len: Context length of the cached data.
        example_ctx_len: Length of each example window.
        n_windows: Total number of windows.

    Returns:
        Tensor of shape (n_windows,) with maximum activation per window.
    """
    import time

    start_time = time.perf_counter()

    # Initialize all windows with zero activation
    window_activations = torch.zeros(n_windows, dtype=torch.float32)

    if activation_data.locations.numel() == 0:
        return window_activations

    # Compute which window each activation belongs to
    flat_indices = (
        activation_data.locations[:, 0] * cache_ctx_len
        + activation_data.locations[:, 1]
    )
    window_indices = flat_indices // example_ctx_len

    n_activations = len(activation_data.activations)

    # === ASSERTIONS for correctness ===
    assert window_indices.dtype in (
        torch.int32,
        torch.int64,
    ), f"window_indices must be integer type, got {window_indices.dtype}"
    assert n_activations == len(
        window_indices
    ), f"Mismatch: {n_activations} activations vs {len(window_indices)} indices"

    # Check for out-of-bounds indices before clamping
    min_idx, max_idx = window_indices.min().item(), window_indices.max().item()
    if min_idx < 0 or max_idx >= n_windows:
        logger.warning(
            f"[DEBUG] window_indices out of bounds: min={min_idx}, max={max_idx}, "
            f"n_windows={n_windows}. Clamping to valid range."
        )

    scatter_start = time.perf_counter()

    # OPTIMIZED: Use scatter_reduce for O(n_activations) instead of O(n_windows)
    # This is ~1000x faster than the O(n_windows) loop
    activations = activation_data.activations.float()

    # Ensure index dtype is int64 (required by scatter_reduce)
    window_indices = window_indices.to(torch.int64)

    # Clamp window_indices to valid range (safety check for edge cases)
    window_indices = window_indices.clamp(0, n_windows - 1)

    # scatter_reduce with "amax" computes max per window in one vectorized op
    # include_self=False means: for indices that appear, compute max(src_values_only)
    # for indices that don't appear, keep initial_value=0
    # This exactly matches the original loop behavior
    window_activations.scatter_reduce_(
        dim=0,
        index=window_indices,
        src=activations,
        reduce="amax",
        include_self=False,  # Don't include initial zeros in max computation
    )

    scatter_end = time.perf_counter()
    total_time = scatter_end - start_time
    scatter_time = scatter_end - scatter_start

    # === VERIFICATION (run on first few calls to ensure correctness) ===
    # Check a sample of windows to verify scatter_reduce gives correct results
    _VERIFY_SAMPLE_SIZE = int(os.environ.get("DELPHI_VERIFY_SCATTER", "0"))
    if _VERIFY_SAMPLE_SIZE > 0:
        verify_start = time.perf_counter()
        # Sample some windows that have activations
        unique_windows = torch.unique(window_indices)
        sample_size = min(_VERIFY_SAMPLE_SIZE, len(unique_windows))
        sample_windows = unique_windows[
            torch.randperm(len(unique_windows))[:sample_size]
        ]

        for win_idx in sample_windows:
            mask = window_indices == win_idx
            expected = activations[mask].max().item()
            actual = window_activations[win_idx.item()].item()
            if not np.isclose(expected, actual, rtol=1e-5):
                logger.error(
                    f"[VERIFICATION FAILED] window {win_idx.item()}: "
                    f"expected={expected}, actual={actual}"
                )
                raise AssertionError(
                    f"scatter_reduce verification failed for window {win_idx.item()}"
                )

        verify_time = time.perf_counter() - verify_start
        logger.debug(
            f"[DEBUG] Verified {sample_size} windows in {verify_time:.4f}s - "
            "all correct"
        )

    logger.warning(
        f"[DEBUG TIMING] compute_per_window_activations (OPTIMIZED): "
        f"total={total_time:.4f}s, scatter={scatter_time:.4f}s, "
        f"n_windows={n_windows}, n_activations={n_activations}"
    )

    return window_activations


def quantile_non_activating_windows(
    activation_data: ActivationData,
    reshaped_tokens: Int[Tensor, "windows ctx_len"],
    cache_ctx_len: int,
    example_ctx_len: int,
    n_windows: int,
    n_not_active: int,
    quantile: float,
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    seed: int = 42,
) -> list[NonActivatingExample]:
    """
    Generate non-activating sequence windows based on activation quantile.

    Args:
        activation_data: Activation data for the latent.
        reshaped_tokens: The tokens reshaped to the context length.
        cache_ctx_len: Context length of the cached data.
        example_ctx_len: Length of each example window.
        n_windows: Total number of windows.
        n_not_active: The number of non-activating examples to generate.
        quantile: Quantile threshold (e.g., 0.1 for bottom 10%).
        tokenizer: Tokenizer for decoding tokens.
        seed: Random seed.

    Returns:
        List of non-activating examples.
    """
    torch.manual_seed(seed)
    if n_not_active == 0:
        return []

    # Compute per-window activations
    window_activations = compute_per_window_activations(
        activation_data, cache_ctx_len, example_ctx_len, n_windows
    )

    # Compute quantile threshold
    threshold = torch.quantile(window_activations, quantile).item()

    # Find windows below the threshold
    below_threshold = (
        (window_activations <= threshold).nonzero(as_tuple=False).squeeze()
    )

    if below_threshold.numel() == 0:
        logger.warning(
            f"No windows found below quantile {quantile} (threshold: {threshold:.4f})"
        )
        return []

    # If we have fewer windows than needed, use all available
    if below_threshold.numel() < n_not_active:
        logger.warning(
            f"Only {below_threshold.numel()} windows below quantile, "
            f"requested {n_not_active}"
        )
        selected_indices = below_threshold
    else:
        # Randomly sample from windows below threshold
        random_indices = torch.randint(
            0, below_threshold.shape[0], size=(n_not_active,)
        )
        selected_indices = below_threshold[random_indices]

    toks = reshaped_tokens[selected_indices]
    # Get actual activation values for selected windows
    selected_activations = (
        window_activations[selected_indices].unsqueeze(1).expand(-1, example_ctx_len)
    )

    return prepare_non_activating_examples(
        toks,
        selected_activations,
        -1.0,
        tokenizer,
    )


def threshold_non_activating_windows(
    activation_data: ActivationData,
    reshaped_tokens: Int[Tensor, "windows ctx_len"],
    cache_ctx_len: int,
    example_ctx_len: int,
    n_windows: int,
    n_not_active: int,
    threshold: float | None,
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    seed: int = 42,
) -> list[NonActivatingExample]:
    """
    Generate non-activating sequence windows based on activation threshold.

    Args:
        activation_data: Activation data for the latent.
        reshaped_tokens: The tokens reshaped to the context length.
        cache_ctx_len: Context length of the cached data.
        example_ctx_len: Length of each example window.
        n_windows: Total number of windows.
        n_not_active: The number of non-activating examples to generate.
        threshold: Absolute threshold. If None, uses 10th percentile.
        tokenizer: Tokenizer for decoding tokens.
        seed: Random seed.

    Returns:
        List of non-activating examples.
    """
    torch.manual_seed(seed)
    if n_not_active == 0:
        return []

    # Compute per-window activations
    window_activations = compute_per_window_activations(
        activation_data, cache_ctx_len, example_ctx_len, n_windows
    )

    # If threshold is None, use 10th percentile
    if threshold is None:
        threshold = torch.quantile(window_activations, 0.1).item()
        logger.debug(f"Using computed threshold: {threshold:.4f} (10th percentile)")

    # Find windows below the threshold
    below_threshold = (
        (window_activations <= threshold).nonzero(as_tuple=False).squeeze()
    )

    if below_threshold.numel() == 0:
        logger.warning(f"No windows found below threshold {threshold:.4f}")
        return []

    # If we have fewer windows than needed, use all available
    if below_threshold.numel() < n_not_active:
        logger.warning(
            f"Only {below_threshold.numel()} windows below threshold, "
            f"requested {n_not_active}"
        )
        selected_indices = below_threshold
    else:
        # Randomly sample from windows below threshold
        random_indices = torch.randint(
            0, below_threshold.shape[0], size=(n_not_active,)
        )
        selected_indices = below_threshold[random_indices]

    toks = reshaped_tokens[selected_indices]
    # Get actual activation values for selected windows
    selected_activations = (
        window_activations[selected_indices].unsqueeze(1).expand(-1, example_ctx_len)
    )

    return prepare_non_activating_examples(
        toks,
        selected_activations,
        -1.0,
        tokenizer,
    )


def random_non_activating_windows(
    available_indices: Int[Tensor, "windows"],
    reshaped_tokens: Int[Tensor, "windows ctx_len"],
    n_not_active: int,
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    seed: int = 42,
) -> list[NonActivatingExample]:
    """
    Generate random non-activating sequence windows and update the latent record.

    Args:
        record (LatentRecord): The latent record to update.
        available_indices (TensorType["n_windows"]): The indices of the windows where
        the latent is not active.
        reshaped_tokens (TensorType["n_windows", "ctx_len"]): The tokens reshaped
        to the context length.
        n_not_active (int): The number of non activating examples to generate.
    """
    torch.manual_seed(seed)
    if n_not_active == 0:
        return []

    # If this happens it means that the latent is active in every window,
    # so it is a bad latent
    if available_indices.numel() < n_not_active:
        logger.warning("No available randomly sampled non-activating sequences")
        return []
    else:
        random_indices = torch.randint(
            0, available_indices.shape[0], size=(n_not_active,)
        )
        selected_indices = available_indices[random_indices]

    toks = reshaped_tokens[selected_indices]

    return prepare_non_activating_examples(
        toks,
        torch.zeros_like(toks),  # there is no way to define these activations
        -1.0,
        tokenizer,
    )
