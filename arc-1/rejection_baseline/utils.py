import asyncio
import json
import math
from pathlib import Path
from typing import Any, Callable, Iterator


async def process_queue(
    job_generator: Iterator[Any], worker_func: Callable, max_concurrent: int = 3
) -> list:
    """
    Process jobs with limited concurrency.

    Args:
        job_generator: Iterator yielding jobs to process
        worker_func: Async function to process each job
        max_concurrent: Maximum number of concurrent jobs
    """
    active_tasks = set()
    results = []

    async def run_job(job):
        try:
            result = await worker_func(**job)
            results.append(result)
        finally:
            if task in active_tasks:
                active_tasks.remove(task)

    # Process jobs until generator is exhausted
    try:
        for job in job_generator:
            # If we're at capacity, wait for a task to complete
            while len(active_tasks) >= max_concurrent:
                done, _ = await asyncio.wait(
                    active_tasks, return_when=asyncio.FIRST_COMPLETED
                )
                # Remove completed tasks
                active_tasks.difference_update(done)

            # Create new task
            task = asyncio.create_task(run_job(job))
            active_tasks.add(task)

        # Wait for remaining tasks to complete
        if active_tasks:
            await asyncio.wait(active_tasks)

    except Exception as e:
        # Cancel any remaining tasks on error
        for task in active_tasks:
            task.cancel()
        raise e

    return results


def rfind_token_index(tokens: list[str], sub: str) -> int:
    combined = "".join(tokens)
    found_index = combined.rfind(sub)
    if found_index < 0:
        return -1
    l = 0
    i = 0
    while l < found_index:
        l += len(tokens[i])
        i += 1
    return i


def range_perplexity_per_token(
    begin_pos: int, end_pos: int, tokens: list[str], logprobs: list[float]
) -> float:
    assert len(tokens) == len(logprobs)
    assert end_pos > begin_pos
    s = 0
    t = 0
    for i in range(begin_pos, end_pos):
        if tokens[i].isspace() or tokens[i] in (",", "[", "]"):
            continue
        s += logprobs[i]
        t += 1
    return math.exp(-s / t)


def write_jsonl(file_name: str | Path, lines: list, mode: str = "a") -> None:
    file_path = Path(file_name)
    with file_path.open(mode, encoding="utf-8") as f:
        for l in lines:
            json.dump(l, f)
            f.write("\n")


def read_jsonl(file_name: str | Path) -> Iterator:
    file_path = Path(file_name)
    with file_path.open(mode="r", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


def write_json(file_name: str | Path, data: Any) -> None:
    file_path = Path(file_name)
    with file_path.open("w", encoding="utf-8") as f:
        json.dump(data, f)


def read_json(file_name: str | Path) -> Any:
    file_path = Path(file_name)
    with file_path.open(mode="r", encoding="utf-8") as f:
        return json.load(f)
