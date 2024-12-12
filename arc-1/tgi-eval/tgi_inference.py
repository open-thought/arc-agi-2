import argparse
from typing import Any
from pathlib import Path
import json
import requests

from arc.utils import dataset
from arc.interface import Riddle, Board, BoardPair


def sample_tgi(
    prompt: str,
    sampling_params: dict[str, Any],
    generate_url: str = "http://127.0.0.1:8080/generate",
) -> str:
    data = {"inputs": prompt, "parameters": sampling_params}
    r = requests.post(generate_url, json=data)
    r.raise_for_status()
    response_json = r.json()
    return response_json["generated_text"]


def get_models(base_url: str = "http://127.0.0.1:8080") -> list[dict]:
    url = base_url + "/v1/models"
    r = requests.get(url)
    r.raise_for_status()
    response_json = r.json()
    return response_json["data"]


def format_board(
    board: Board,
    alphabet: list[str],
    col_delimiter: str = ",",
    row_delimiter: str = ",\n",
) -> str:
    h, w = board.shape
    buffer = [f"{h}x{w}\n["]
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
    )

    test_index = len(riddle.train) + 1

    if chatml:
        buffer = [
            "<|im_start|>user\n"
            "You like to solve puzzles. Find the underlying abstract input and output transformation. Look at the following input-output pairs:\n",
            input_examples,
            f"Now consider the last input examples and deduce its output. Think deeply! Directly generate output board values.\ninput{test_index}: ",
            test_input,
            f"<|im_end|>\n<|im_start|>assistant\noutput{test_index}: ",
        ]
    else:
        buffer = [
            "As a math genius you are presented with the following 2D board input/output pairs:\n",
            input_examples,
            f"\nNow you consider the last input example. Your task is to deduce the corresponding output.\ninput{test_index}: ",
            test_input,
            "\nAfter thinking thoroughly about the abstract transformation you come to the conclusion that it must be:"
            f"\n\noutput{test_index}: ",
        ]

    x = "".join(buffer)
    return x, test_output


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_url", type=str, default="http://127.0.0.1:8080"
    )
    parser.add_argument("--max_new_tokens", type=int, default=600)
    parser.add_argument("--min_new_tokens", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--alphabet", type=str, default='["0","1","2","3","4","5","6","7","8","9"]')
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--col_delimiter", type=str, default=",")
    parser.add_argument("--row_delimiter", type=str, default=",\n")
    parser.add_argument("--cutoff_length", type=int, default=3200)
    parser.add_argument("--jsonl_out", type=str)
    parser.add_argument("--format", type=str, default="chatml", help="legacy, chatml")

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    model_id = get_models(base_url=args.base_url)[0]["id"]
    print(f"Model ID: {model_id}")

    generate_url = args.base_url + "/generate"
    generate_args = {
        "max_new_tokens": args.max_new_tokens,
        "min_new_tokens": args.min_new_tokens,
        "do_sample": args.temperature > 0
    }
    if args.temperature > 0:
        generate_args["temperature"] =  args.temperature
        generate_args["top_p"] = args.top_p

    riddle_directories = ["evaluation", "training"]
    riddle_ids = dataset.get_riddle_ids(riddle_directories)
    print(f"Total number of riddles: {len(riddle_ids)}")

    num_trials = args.trials

    correct = 0
    skipped = 0
    correct_ids = []

    print(f"Alphabet: {args.alphabet}")
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

    log_lines = []

    def dump_jsonl(file_name: str | Path, lines: list):
        file_name = Path(file_name)
        with file_name.open("w", encoding="UTF-8") as f:
            for l in lines:
                json.dump(l, f)
                f.write("\n")

    for i, id in enumerate(riddle_ids):
        riddle = dataset.load_riddle_from_id(id)

        x, y = format_riddle_input(
            riddle,
            alphabet,
            col_delimiter=col_delimiter,
            row_delimiter=row_delimiter,
            chatml=chatml,
        )
        if len(x) > args.cutoff_length:
            print(f"skipping {id} ...")
            skipped += 1
            continue

        print(f"[{i}] checking {id} (solved: {correct}/{len(riddle_ids)})")

        for j in range(num_trials):
            if num_trials > 1:
                print(f"Try {j+1} of {num_trials}")

            try:
                output = sample_tgi(x, generate_args, generate_url=generate_url)
            except Exception as e:
                print("Sampling failed:", e)
                continue

            #print("output:", output)
            #print("search:", y)

            try:
                index = output.replace(" ", "").index(y)    # compare without spaces
            except:
                index = -1

            log_lines.append(
                {
                    "id": id,
                    "trial": j,
                    "input": x,
                    "output": output,
                    "ground_truth": y,
                    "found": index,
                    "solved": (index > -1),
                }
            )

            if index >= 0:
                print(f"SOLUTION found for {id} at index {index}.")
                print("output:", output)
                print("expected:", y)
                correct += 1
                correct_ids.append(id)
                break

        print()
        if args.jsonl_out:
            dump_jsonl(args.jsonl_out, log_lines)

    print(f"\nSolved: {correct}/{len(riddle_ids)}")
    print(f"Skipped: {skipped}")
    print("IDs of solved riddles: ", correct_ids)


if __name__ == "__main__":
    main()
