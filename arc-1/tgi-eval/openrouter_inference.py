import asyncio
import os
import re
import argparse
from typing import Any, AsyncIterator, Awaitable, Callable, Iterable, Iterator, Optional
from pathlib import Path
import json
import time
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam, ChatCompletion

import arckit  # `pip install arckit`, https://github.com/mxbi/arckit


async def llm_generate(
    client: AsyncOpenAI,
    messages: Iterable[ChatCompletionMessageParam],
    sampling_params: dict[str, Any],
) -> Awaitable[str]:
    response = await client.chat.completions.create(
        extra_headers={"X-Title": "open-thought"}, messages=messages, **sampling_params
    )
    return response.choices[0].message.content


def format_board(
    board: list[list[int]],
    alphabet: list[str],
    col_delimiter: str = ",",
    row_delimiter: str = ",\n",
    with_board_dim: bool = True,
) -> str:
    h, w = len(board), len(board[0])
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
            value = board[row][col]
            buffer.append(alphabet[value])
        buffer.append("]")
    buffer.append("]")
    return "".join(buffer)


def format_board_pair(
    index: int,
    pair: dict[str, list[list[int]]],
    alphabet: list[str],
    col_delimiter: str = ",",
    row_delimiter: str = ",\n",
) -> str:
    input = format_board(
        pair["input"],
        alphabet=alphabet,
        col_delimiter=col_delimiter,
        row_delimiter=row_delimiter,
    )
    output = format_board(
        pair["output"],
        alphabet=alphabet,
        col_delimiter=col_delimiter,
        row_delimiter=row_delimiter,
    )
    return f"input{index}: {input}\n\noutput{index}: {output}\n\n"


def format_training_examples(
    riddle: dict,
    alphabet: list[str],
    col_delimiter: str = ",",
    row_delimiter: str = ",\n",
) -> str:
    buffer = []
    for i, board_pair in enumerate(riddle["train"]):
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
    riddle: dict,
    alphabet: list[str],
    col_delimiter: str = ",",
    row_delimiter: str = ",\n",
) -> tuple[list[ChatCompletionMessageParam], str]:
    input_examples = format_training_examples(
        riddle,
        alphabet=alphabet,
        col_delimiter=col_delimiter,
        row_delimiter=row_delimiter,
    )
    test_input = format_board(
        riddle["test"][0]["input"],
        alphabet=alphabet,
        col_delimiter=col_delimiter,
        row_delimiter=row_delimiter,
    )
    test_output = format_board(
        riddle["test"][0]["output"],
        alphabet=alphabet,
        col_delimiter=col_delimiter,
        row_delimiter=row_delimiter,
        with_board_dim=False,
    )

    test_index = len(riddle["train"]) + 1

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
        "As an experienced mathematician you are presented with a set of 2D input/output board pairs. Find the input-to-output transformation rule:\n",
        input_examples,
        f"Now you consider the last input example. You must deduce the corresponding output for it.\ninput{test_index}: ",
        test_input,
    ]

    user_request = "".join(buffer)
    messages = [
        {
            "role": "user",
            "content": user_request,
        },
    ]

    return messages, test_output


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_url", type=str, default="https://openrouter.ai/api/v1")
    parser.add_argument(
        "--model", type=str, default="google/gemini-2.0-flash-thinking-exp:free"
    )
    parser.add_argument("--max_new_tokens", type=int, default=8096)
    parser.add_argument("--temperature", type=float, default=0.85)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument(
        "--alphabet", type=str, default='["0","1","2","3","4","5","6","7","8","9"]'
    )
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--col_delimiter", type=str, default=",")
    parser.add_argument("--row_delimiter", type=str, default=",\n")
    parser.add_argument("--cutoff_length", type=int, default=8096)
    parser.add_argument("--jsonl_out", type=str)
    parser.add_argument("--max_concurrent", type=int, default=1)

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
    dataset: dict,
    alphabet: list[str],
    col_delimiter: str,
    row_delimiter: str,
    open_router_client: AsyncOpenAI,
    generate_args: dict,
    cutoff_length: int,
    jsonl_out: Optional[str],
    num_trials: int = 1,
    max_concurrent: int = 2,
):
    stats = Stats()
    log_lines = []

    riddle_ids = sorted(dataset.keys())

    def generate_sampling_jobs() -> Iterator[dict]:
        for i, id in enumerate(riddle_ids):
            riddle = dataset[id]
            x, y = format_riddle_input(
                riddle,
                alphabet,
                col_delimiter=col_delimiter,
                row_delimiter=row_delimiter,
            )
            yield ({"index": i, "id": id, "input": x, "target": y})

    async def sampling_worker(*, index: int, id: str, input: str, target: str) -> str:
        if len(input) > cutoff_length:
            print(f"skipping {id} ...")
            stats.skipped += 1
            return

        for j in range(num_trials):
            if num_trials > 1:
                print(f"[{id}] Try {j+1} of {num_trials}")

            try:
                output = await llm_generate(
                    open_router_client,
                    input,
                    generate_args,
                )
            except Exception as e:
                print("Sampling failed:", e)
                time.sleep(1.0)
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

    model_id = args.model
    print(f"Model ID: {model_id}")

    open_router_client = AsyncOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
    )
    generate_args = {
        "model": model_id,
        "max_tokens": args.max_new_tokens,
    }
    if args.temperature > 0:
        generate_args["temperature"] = args.temperature
        generate_args["top_p"] = args.top_p

    dataset = {}
    train_set, eval_set = arckit.load_data()
    for x in train_set:
        dataset[x.id] = x.to_dict()
    for x in eval_set:
        dataset[x.id] = x.to_dict()

    riddle_ids = sorted(dataset.keys())
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

    print(f"input example {riddle_ids[0]}: ")
    riddle = dataset[riddle_ids[0]]
    x, y = format_riddle_input(
        riddle,
        alphabet,
        col_delimiter=col_delimiter,
        row_delimiter=row_delimiter,
    )
    print(x)
    print()

    await sample_concurrent(
        dataset=dataset,
        alphabet=alphabet,
        col_delimiter=col_delimiter,
        row_delimiter=row_delimiter,
        open_router_client=open_router_client,
        generate_args=generate_args,
        cutoff_length=args.cutoff_length,
        jsonl_out=args.jsonl_out,
        num_trials=num_trials,
        max_concurrent=args.max_concurrent,
    )


if __name__ == "__main__":
    asyncio.run(main())
