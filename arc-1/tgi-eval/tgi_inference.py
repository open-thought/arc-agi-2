import aiohttp
import asyncio
import re
import argparse
from typing import Any, AsyncIterator, Awaitable, Callable, Iterator, Optional
from pathlib import Path
import json
import time

from arc.utils import dataset
from arc.interface import Riddle, Board, BoardPair


async def sample_tgi(
    session: aiohttp.ClientSession,
    prompt: str,
    sampling_params: dict[str, Any],
    generate_url: str = "http://127.0.0.1:8080/generate",
) -> Awaitable[str]:
    data = {"inputs": prompt, "parameters": sampling_params}
    async with session.post(generate_url, json=data) as r:
        r.raise_for_status()
        response_json = await r.json()
        return response_json["generated_text"]


async def get_models(base_url: str = "http://127.0.0.1:8080") -> Awaitable[list[dict]]:
    url = base_url + "/v1/models"
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as r:
            r.raise_for_status()
            response_json = await r.json()
            return response_json["data"]


def format_board(
    board: Board,
    alphabet: list[str],
    col_delimiter: str = ",",
    row_delimiter: str = ",\n",
    with_board_dim: bool = True,
) -> str:
    h, w = board.shape
    buffer = []
    if with_board_dim:
        buffer.append(f"{h}x{w}\n")
    buffer.append(f"[")
    for row in range(h):
        if row > 0 and row_delimiter:
            buffer.append(row_delimiter)
        buffer.append("[")
        for col in range(w):
            if col > 0 and col_delimiter:
                buffer.append(col_delimiter)
            value = board.data[row][col]
            buffer.append(alphabet[value])
        buffer.append("]")
    buffer.append("]")
    return "".join(buffer)


def format_board_pair(
    index: int,
    pair: BoardPair,
    alphabet: list[str],
    col_delimiter: str = ",",
    row_delimiter: str = ",\n",
) -> str:
    input = format_board(
        pair.input,
        alphabet=alphabet,
        col_delimiter=col_delimiter,
        row_delimiter=row_delimiter,
    )
    output = format_board(
        pair.output,
        alphabet=alphabet,
        col_delimiter=col_delimiter,
        row_delimiter=row_delimiter,
    )
    return f"input{index}: {input}\n\noutput{index}: {output}\n\n"


def format_training_examples(
    riddle: Riddle,
    alphabet: list[str],
    col_delimiter: str = ",",
    row_delimiter: str = ",\n",
) -> str:
    buffer = []
    for i, board_pair in enumerate(riddle.train):
        s = format_board_pair(
            index=i,
            pair=board_pair,
            alphabet=alphabet,
            col_delimiter=col_delimiter,
            row_delimiter=row_delimiter,
        )
        buffer.append(s)

    return "".join(buffer)


def format_riddle_input(
    riddle: Riddle,
    alphabet: list[str],
    col_delimiter: str = ",",
    row_delimiter: str = ",\n",
    chatml: bool = True,
) -> str:
    input_examples = format_training_examples(
        riddle,
        alphabet=alphabet,
        col_delimiter=col_delimiter,
        row_delimiter=row_delimiter,
    )
    test_input = format_board(
        riddle.test[0].input,
        alphabet=alphabet,
        col_delimiter=col_delimiter,
        row_delimiter=row_delimiter,
    )
    test_output = format_board(
        riddle.test[0].output,
        alphabet=alphabet,
        col_delimiter=col_delimiter,
        row_delimiter=row_delimiter,
        with_board_dim=False,
    )

    test_index = len(riddle.train) + 1

    if chatml:

        # earlier version of prompt:
        # buffer = [
        #     "<|im_start|>user\n"
        #     "You like to solve puzzles. Find the underlying abstract input and output transformation. Look at the following input-output pairs:\n",
        #     input_examples,
        #     f"Now consider the last input examples and deduce its output. Think deeply! Directly generate output board values.\ninput{test_index}: ",
        #     test_input,
        #     f"<|im_end|>\n<|im_start|>assistant\noutput{test_index}: ",
        # ]

        buffer = [
            "<|im_start|>user\n"
            "As an experienced mathematician you are presented with a set of 2D input/output board pairs. Find the input-to-output transformation rule:\n",
            input_examples,
            f"Now you consider the last input example. You must deduce the corresponding output for it.\ninput{test_index}: ",
            test_input,
            f"<|im_end|>\n<|im_start|>assistant\n",
        ]
    else:
        buffer = [
            "As a math genius you are presented with the following 2D board input/output pairs:\n",
            input_examples,
            f"\nNow you consider the last input example. Your task is to deduce the corresponding output.\ninput{test_index}: ",
            test_input,
            "\nAfter thinking thoroughly about the abstract transformation you come to the conclusion that it must be:",
        ]

    cue_output = False
    if cue_output:
        buffer.append(f"output{test_index}: ")

    x = "".join(buffer)
    return x, test_output


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_url", type=str, default="http://127.0.0.1:8080")
    parser.add_argument("--max_new_tokens", type=int, default=600)
    parser.add_argument("--min_new_tokens", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument(
        "--alphabet", type=str, default='["0","1","2","3","4","5","6","7","8","9"]'
    )
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--col_delimiter", type=str, default=",")
    parser.add_argument("--row_delimiter", type=str, default=",\n")
    parser.add_argument("--cutoff_length", type=int, default=3200)
    parser.add_argument("--jsonl_out", type=str)
    parser.add_argument("--format", type=str, default="chatml", help="legacy, chatml")

    args = parser.parse_args()
    return args


def dump_jsonl(file_name: str | Path, lines: list):
    file_name = Path(file_name)
    with file_name.open("w", encoding="UTF-8") as f:
        for l in lines:
            json.dump(l, f)
            f.write("\n")


async def process_queue(
    job_generator: AsyncIterator[Any], worker_func: Callable, max_concurrent: int = 3
) -> list:
    """
    Process a large number of jobs with limited concurrency.

    Args:
        job_generator: Async iterator yielding jobs to process
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


class Stats:
    def __init__(self):
        self.correct = 0
        self.skipped = 0
        self.correct_ids = []


async def sample_concurrent(
    *,
    riddle_ids: list[str],
    alphabet: list[str],
    col_delimiter: str,
    row_delimiter: str,
    chatml: bool,
    generate_url: str,
    generate_args: dict,
    cutoff_length: int,
    jsonl_out: Optional[str],
    num_trials: int = 1,
    max_concurrent: int = 2,
):
    stats = Stats()
    log_lines = []

    async with aiohttp.ClientSession() as session:

        def generate_sampling_jobs() -> Iterator[dict]:
            for i, id in enumerate(riddle_ids):
                riddle = dataset.load_riddle_from_id(id)
                x, y = format_riddle_input(
                    riddle,
                    alphabet,
                    col_delimiter=col_delimiter,
                    row_delimiter=row_delimiter,
                    chatml=chatml,
                )
                yield ({"index": i, "id": id, "input": x, "target": y})

        async def sampling_worker(
            *, index: int, id: str, input: str, target: str
        ) -> str:
            if len(input) > cutoff_length:
                print(f"skipping {id} ...")
                stats.skipped += 1
                return

            for j in range(num_trials):
                if num_trials > 1:
                    print(f"[{id}] Try {j+1} of {num_trials}")

                try:
                    output = await sample_tgi(
                        session, input, generate_args, generate_url=generate_url
                    )
                except Exception as e:
                    print("Sampling failed:", e)
                    continue

                try:
                    output_ = re.sub(r"\s+", "", output)
                    target_ = re.sub(r"\s+", "", target)
                    pos = output_.index(
                        target_
                    )  # compare ignoring whitespaces (e.g. spaces and newlines)
                except:
                    pos = -1

                log_lines.append(
                    {
                        "id": id,
                        "trial": j,
                        "input": input,
                        "output": output,
                        "ground_truth": target,
                        "found": pos,
                        "solved": (pos > -1),
                    }
                )

                if pos >= 0:
                    print(f"[{id}] SOLUTION found at index {pos}.")
                    print("output:", output)
                    print("ground_truth:", target)
                    stats.correct += 1
                    stats.correct_ids.append(id)
                    break

            print(
                f"[{id}] solved: {stats.correct}/{len(riddle_ids)} (skipped: {stats.skipped})"
            )
            if jsonl_out:
                dump_jsonl(jsonl_out, log_lines)

        results = await process_queue(
            job_generator=generate_sampling_jobs(),
            worker_func=sampling_worker,
            max_concurrent=max_concurrent,
        )

    print(f"\nSolved: {stats.correct}/{len(riddle_ids)}")
    print(f"Skipped: {stats.skipped}")
    print("IDs of solved riddles: ", stats.correct_ids)


async def main():
    args = parse_args()

    model_id = (await get_models(base_url=args.base_url))[0]["id"]
    print(f"Model ID: {model_id}")

    generate_url = args.base_url + "/generate"
    generate_args = {
        "max_new_tokens": args.max_new_tokens,
        "min_new_tokens": args.min_new_tokens,
        "do_sample": args.temperature > 0,
    }
    if args.temperature > 0:
        generate_args["temperature"] = args.temperature
        generate_args["top_p"] = args.top_p

    riddle_directories = ["evaluation", "training"]
    riddle_ids = dataset.get_riddle_ids(riddle_directories)
    print(f"Total number of riddles: {len(riddle_ids)}")

    num_trials = args.trials

    alphabet = json.loads(args.alphabet)
    assert isinstance(alphabet, list)
    assert len(alphabet) >= 10

    col_delimiter = args.col_delimiter
    row_delimiter = args.row_delimiter

    print("alphabet: ", alphabet)
    print("col_delimiter: ", col_delimiter)
    print("row_delimiter: ", row_delimiter)
    print("num_trails:", num_trials)

    chatml = args.format == "chatml"

    print(f"input example {riddle_ids[0]}: ")
    riddle = dataset.load_riddle_from_id(riddle_ids[0])
    x, y = format_riddle_input(
        riddle,
        alphabet,
        col_delimiter=col_delimiter,
        row_delimiter=row_delimiter,
        chatml=chatml,
    )
    print(x)
    print()

    await sample_concurrent(
        riddle_ids=riddle_ids,
        alphabet=alphabet,
        col_delimiter=col_delimiter,
        row_delimiter=row_delimiter,
        chatml=chatml,
        generate_url=generate_url,
        generate_args=generate_args,
        cutoff_length=args.cutoff_length,
        jsonl_out=args.jsonl_out,
        num_trials=num_trials,
        max_concurrent=32,
    )


if __name__ == "__main__":
    asyncio.run(main())
