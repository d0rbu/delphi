import asyncio
from collections.abc import AsyncIterable, Awaitable, Callable
from functools import wraps
from typing import Any

from tqdm.asyncio import tqdm


def process_wrapper(
    function: Callable[..., Awaitable],
    preprocess: Callable | None = None,
    postprocess: Callable | None = None,
) -> Callable[..., Awaitable]:
    """
    Wraps a function with optional preprocessing and postprocessing steps.

    Args:
        function (Callable): The main function to be wrapped.
        preprocess (Callable, optional): A function to preprocess the input.
            Defaults to None.
        postprocess (Callable, optional): A function to postprocess the output.
            Defaults to None.

    Returns:
        Callable: The wrapped function.
    """

    @wraps(function)
    async def wrapped(input: Any):
        if preprocess is not None:
            input = preprocess(input)

        results = await function(input)

        if postprocess is not None:
            results = postprocess(results)

        return results

    return wrapped


class Pipe:
    """
    Represents a pipe of functions to be executed with the same input.
    """

    def __init__(self, *functions: Callable):
        """
        Initialize the Pipe with a list of functions.

        Args:
            *functions (list[Callable]): Functions to be executed in the pipe.
        """
        self.functions = functions

    async def __call__(self, input: Any) -> list[Any]:
        """
        Execute all functions in the pipe with the given input.

        Args:
            input (Any): The input to be processed by all functions.

        Returns:
            list[Any]: The results of all functions.
        """
        tasks = [function(input) for function in self.functions]

        non_awaitable_tasks = [
            (task_idx, task)
            for task_idx, task in enumerate(tasks)
            if not isinstance(task, Awaitable)
        ]

        awaitable_tasks = [
            (task_idx, task)
            for task_idx, task in enumerate(tasks)
            if isinstance(task, Awaitable)
        ]
        awaitable_task_results = await asyncio.gather(
            *[task for _, task in awaitable_tasks]
        )

        results = [None] * len(tasks)

        for task_idx, result in non_awaitable_tasks:
            results[task_idx] = result
        for (task_idx, _task), result in zip(awaitable_tasks, awaitable_task_results):
            results[task_idx] = result

        return results


class Pipeline:
    """
    Manages the execution of multiple pipes, handling concurrency and progress tracking.
    """

    def __init__(self, loader: AsyncIterable | Callable, *pipes: Pipe | Callable):
        """
        Initialize the Pipeline with a list of pipes.

        Args:
            loader (Callable): The loader to be executed first.
            *pipes (list[Pipe | Callable]): Pipes to be executed in the pipeline.
        """

        self.loader = loader
        self.pipes = pipes

    async def run(self, max_concurrent: int = 10) -> list[Any]:
        """
        Run the pipeline with a maximum number of concurrent tasks.

        Args:
            max_concurrent: Maximum number of concurrent tasks. Defaults to 10.

        Returns:
            list[Any]: The results of all processed items.
        """
        results = []
        semaphore = asyncio.Semaphore(max_concurrent)
        tasks = set()

        progress_bar = tqdm(desc="Processing items")
        number_of_items = 0

        async def process_and_update(item, semaphore):
            result = await self.process_item(item, semaphore)
            progress_bar.update(1)
            return result

        async for item in self.generate_items():
            number_of_items += 1
            task = asyncio.create_task(process_and_update(item, semaphore))
            tasks.add(task)

            if len(tasks) >= max_concurrent:
                done, pending = await asyncio.wait(
                    tasks, return_when=asyncio.FIRST_COMPLETED
                )
                results.extend(task.result() for task in done)
                tasks = pending

        if tasks:
            done, _ = await asyncio.wait(tasks)
            results.extend(task.result() for task in done)

        progress_bar.close()
        return results

    async def generate_items(self) -> AsyncIterable[Any]:
        """
        Generates items from the first pipe, which can be an async iterable or callable

        Yields:
            Any: Items generated from the first pipe.

        Raises:
            TypeError: If the first pipe is neither an async iterable nor a callable.
        """
        if isinstance(self.loader, AsyncIterable):
            async for item in self.loader:
                yield item
        elif callable(self.loader):
            for item in self.loader():
                yield item
                await asyncio.sleep(0)  # Allow other coroutines to run
        else:
            raise TypeError("The first pipe must be an async iterable or a callable")

    async def process_item(self, item: Any, semaphore: asyncio.Semaphore) -> Any:
        """
        Processes a single item through all pipes except the first one.

        Args:
            item (Any): The item to be processed.
            semaphore (asyncio.Semaphore): Semaphore for controlling concurrency.

        Returns:
            Any: The processed item.
        """
        import logging
        import time

        start_time = time.perf_counter()
        item_name = (
            getattr(item, "latent", item) if hasattr(item, "latent") else str(item)[:50]
        )

        async with semaphore:
            result = item
            pipe_times = []
            for pipe_idx, pipe in enumerate(self.pipes):
                if result is None:
                    return None

                pipe_start = time.perf_counter()
                result = await pipe(result)
                pipe_time = time.perf_counter() - pipe_start
                pipe_times.append(pipe_time)

        total_time = time.perf_counter() - start_time
        logging.warning(
            f"[DEBUG TIMING] Pipeline.process_item for {item_name}: "
            f"total={total_time:.2f}s, pipe_times={[f'{t:.2f}s' for t in pipe_times]}"
        )

        return result
