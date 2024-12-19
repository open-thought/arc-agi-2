from pathlib import Path
import random
from together import Together
from formatting import format_training_examples, format_board

from loader import load_remix_arc_dataset


def find_token_index(tokens: list[str], sub: str) -> int:
    combined = "".join(tokens)
    found_index = combined.index(sub)
    l = 0
    i = 0
    while l < found_index:
        l += len(tokens[i])
        i += 1
    return i


def range_sum_logprobs(
    begin_pos: int, end_pos: int, tokens: list[str], logprobs: list[float]
) -> float:
    assert len(tokens) == len(logprobs)
    s = 0
    for i in range(begin_pos, end_pos):
        if tokens[i].isspace():
            continue
        s += logprobs[i]
    return s


def main():
    client = Together()

    remix_arc_dateset_path = Path("~/data/remix-arc-1.3k/").expanduser()
    riddles = load_remix_arc_dataset(remix_arc_dateset_path, max_count=10)

    riddle_ids = sorted(riddles.keys())
    print(riddle_ids)

    riddle_id = riddle_ids[0]

    riddle_pairs = riddles[riddle_id]["board_pairs"]

    num_examples = 5 + 1
    alphabet = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
    col_delimiter: str = ","
    row_delimiter: str = ",\n"

    board_pairs = random.sample(riddle_pairs, k=num_examples)

    input_examples = format_training_examples(
        board_pairs[:-1],
        alphabet=alphabet,
        col_delimiter=col_delimiter,
        row_delimiter=row_delimiter,
    )
    test_input = format_board(
        board_pairs[-1]["input"],
        alphabet=alphabet,
        col_delimiter=col_delimiter,
        row_delimiter=row_delimiter,
    )
    test_output = format_board(
        board_pairs[-1]["output"],
        alphabet=alphabet,
        col_delimiter=col_delimiter,
        row_delimiter=row_delimiter,
        with_board_dim=False,
    )

    messages = [
        {
            "role": "system",
            "content": "You an experienced mathematician specialized in intelligence tests and abstract riddles.",
        },
        {
            "role": "user",
            "content": f"""Analyze the following ARC-AGI riddle and try to infer the rule that is used to transform input matrices into output matrices.:
{input_examples}

<test_input>
{test_input}
</test_input>

Please directly generate the result in a <test_output> tag.
""",
        },
        {"role": "assistant", "content": f"<test_output>{test_output}</test_output>"},
    ]

    completion = client.chat.completions.create(
        model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
        messages=messages,
        max_tokens=1,
        echo=True,
        logprobs=1,
    )

    prompt_logprobs = completion.prompt[0].logprobs.token_logprobs
    prompt_tokens = completion.prompt[0].logprobs.tokens

    begin_pos = find_token_index(prompt_tokens, "<test_output>")
    end_pos = find_token_index(prompt_tokens, "</test_output>")

    print(prompt_logprobs, prompt_tokens)
    print(
        begin_pos,
        end_pos,
        end_pos - begin_pos,
        range_sum_logprobs(begin_pos, end_pos, prompt_tokens, prompt_logprobs),
    )
    print(completion.usage)


if __name__ == "__main__":
    main()
