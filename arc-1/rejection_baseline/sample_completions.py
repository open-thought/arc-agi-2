import argparse
import asyncio
from itertools import chain
import json
import re
import time
from typing import Iterator, Optional

import aiohttp
from utils import (
    process_queue,
    range_perplexity,
    read_json,
    rfind_token_index,
    write_jsonl,
)
import tgi
import arckit
from board_formatting import (
    BoardFormattingOptions,
    format_board,
    format_training_examples,
)

from transformers import AutoTokenizer, PreTrainedTokenizerBase
from measure_ground_truth_perplexity import determine_oracle_perplexity


class Stats:
    def __init__(self):
        self.tried = 0
        self.solved = 0
        self.skipped = 0
        self.solved_ids = []


def format_riddle_input(
    riddle: dict,
    formatting_options: BoardFormattingOptions,
) -> tuple[dict, str]:
    input_examples = format_training_examples(
        riddle,
        formatting_options=formatting_options,
    )
    test_input = format_board(
        riddle["test"][0]["input"],
        formatting_options=formatting_options,
    )
    test_output = format_board(
        riddle["test"][0]["output"],
        formatting_options=formatting_options,
        with_board_shape=False,
    )

    # use simple o3 prompt with instruction to place the final answer in <output> tags
    buffer = [
        "Find the common rule that maps an input grid to an output grid, given the examples below.\n\n",
        input_examples,
        "Below is a test input grid. Predict the corresponding output grid by applying the rule you found. Describe how you derived the rule and your overall reasoning process in detail before you submit your answer. Your final answer must be placed in <output></output> tags and should be just be the text output grid itself.\n\nInput:\n",
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


async def sample_concurrent(
    *,
    session: aiohttp.ClientSession,
    tokenizer: PreTrainedTokenizerBase,
    dataset: dict,
    riddle_ids: list[str],
    board_formatting_options: BoardFormattingOptions,
    sampling_params: dict,
    cutoff_length: int,
    jsonl_out: Optional[str],
    generate_url: str,
    num_trials: int = 1,
    max_concurrent: int = 2,
    no_retry_on_solution: bool = False,
):
    stats = Stats()

    def generate_sampling_jobs() -> Iterator[dict]:
        for i, id in enumerate(riddle_ids):
            riddle = dataset[id]
            x, y = format_riddle_input(
                riddle, formatting_options=board_formatting_options
            )
            yield ({"index": i, "id": id, "input": x, "target": y})

    async def sampling_worker(*, index: int, id: str, input: str, target: str) -> str:
        if len(input) > cutoff_length:
            print(f"skipping {id} ...")
            stats.skipped += 1
            return

        stats.tried += 1
        solved = False
        log_lines = []
        for j in range(num_trials):
            if num_trials > 1:
                print(f"[{index}:{id}] Try {j+1} of {num_trials}")

            try:
                chat_prompt = tokenizer.apply_chat_template(input, tokenize=False)
                generate_result = await tgi.generate(
                    session=session,
                    prompt=chat_prompt,
                    sampling_params=sampling_params,
                    generate_url=generate_url,
                )
                output = generate_result["generated_text"].removeprefix("assistant\n\n")
            except TimeoutError:
                print("TimeoutError")
                continue
            except Exception as e:
                print("Sampling failed:", e, type(e))
                time.sleep(1.0)
                continue

            output_match = False
            output_tag_found = False
            try:
                output_ = re.search(
                    r"<output>(.*?)</output>", output, flags=re.DOTALL
                ).group(1)
                output_tag_found = True
                # compare ignoring horizontal whitespaces but with newlines
                output_ = re.sub(r"[^\S\n]", "", output_).strip()
                target_ = re.sub(r"[^\S\n]", "", target).strip()

                if output_ == target_:
                    output_match = True
            except:
                pass

            if output_tag_found:
                ppl = await determine_oracle_perplexity(
                    session=session,
                    tokenizer=tokenizer,
                    input=input,
                    output=output,
                    ground_truth=target,
                    generate_url=generate_url,
                )
            else:
                ppl = None

            log_lines.append(
                {
                    "id": id,
                    "trial": j,
                    "solved": output_match,
                    "output_tag_found": output_tag_found,
                    "input": input,
                    "output": output,
                    "ground_truth": target,
                    "ground_truth_ppl": ppl,
                }
            )

            if output_match:
                print(f"[{index}:{id}:{j}] SOLUTION found.")
                print("output:", output)
                print("ground_truth:", target)
                solved = True
                if no_retry_on_solution:
                    break

        if solved:
            stats.solved += 1
            stats.solved_ids.append(id)
        print(
            f"[{id}] solved: {stats.solved}/{stats.tried} (skipped: {stats.skipped}, total: {len(riddle_ids)})"
        )
        if jsonl_out:
            write_jsonl(jsonl_out, log_lines, "a")

    results = await process_queue(
        job_generator=generate_sampling_jobs(),
        worker_func=sampling_worker,
        max_concurrent=max_concurrent,
    )

    print(f"\nSolved: {stats.solved}/{len(riddle_ids)}")
    print(f"Skipped: {stats.skipped}")
    print("IDs of solved riddles: ", stats.solved_ids)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-ids-json", type=str, default="eval_ids.json")
    parser.add_argument("--result-output-path", type=str, default="eval_result.json")
    parser.add_argument("--jsonl-output-path", type=str)
    parser.add_argument("--num-shard", type=int, default=1)
    parser.add_argument(
        "--model-id", type=str, default="meta-llama/Llama-3.2-3B-Instruct"
    )
    parser.add_argument("--max_concurrent", type=int, default=16)
    parser.add_argument("--cutoff_length", type=int, default=8096)
    parser.add_argument("--max_new_tokens", type=int, default=8096)
    parser.add_argument("--temperature", type=float, default=0.85)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument(
        "--alphabet", type=str, default='["0","1","2","3","4","5","6","7","8","9"]'
    )
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--col_delimiter", type=str, default=",")
    parser.add_argument("--row_delimiter", type=str, default=",\n")
    parser.add_argument(
        "--no_array_brackets", action="store_false", dest="array_brackets"
    )
    args = parser.parse_args()
    return args


async def main() -> None:
    args = parse_args()
    model_id = args.model_id
    num_shard = args.num_shard
    if args.eval_ids_json is not None:
        eval_ids = read_json(args.eval_ids_json)
    else:
        eval_ids = None
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    docker_container_name = "tgi_inference"

    http_timeout = aiohttp.ClientTimeout(total=600, sock_connect=60)
    async with aiohttp.ClientSession(timeout=http_timeout) as session:

        # launch tgi docker container and wait until it becomes ready
        port = tgi.start_container(
            container_name=docker_container_name,
            model_id=model_id,
            num_shard=num_shard,
        )
        base_url = f"http://localhost:{port}"
        await tgi.until_ready(session, base_url, max_tries=90)

        dataset = {}
        train_set, eval_set = arckit.load_data()
        for x in chain(train_set, eval_set):
            if eval_ids is None or x.id in eval_ids:
                dataset[x.id] = x.to_dict()

        riddle_ids = sorted(dataset.keys())
        print(f"Total number of riddles: {len(riddle_ids)}")

        num_trials = args.trials

        alphabet = json.loads(args.alphabet)
        assert isinstance(alphabet, list)
        assert len(alphabet) >= 10

        board_formatting_options = BoardFormattingOptions(
            alphabet=alphabet,
            col_delimiter=args.col_delimiter,
            row_delimiter=args.row_delimiter,
            array_brackets=args.array_brackets,
        )

        print(board_formatting_options)
        print("num_trails:", num_trials)

        sampling_params = {
            "max_new_tokens": args.max_new_tokens,
        }
        if args.temperature > 0:
            sampling_params["do_sample"] = True
            sampling_params["temperature"] = args.temperature
            sampling_params["top_p"] = args.top_p
        else:
            sampling_params["do_sample"] = False

        generate_url = f"{base_url}/generate"
        await sample_concurrent(
            session=session,
            tokenizer=tokenizer,
            dataset=dataset,
            riddle_ids=riddle_ids,
            board_formatting_options=board_formatting_options,
            sampling_params=sampling_params,
            cutoff_length=args.cutoff_length,
            jsonl_out=args.jsonl_output_path,
            generate_url=generate_url,
            num_trials=num_trials,
            max_concurrent=args.max_concurrent,
        )


if __name__ == "__main__":
    asyncio.run(main())
