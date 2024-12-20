from collections import OrderedDict
import json
import math
import os
from pathlib import Path
import random
import time
from typing import Optional
from together import Together
from openai import OpenAI
from openai.types.chat import ChatCompletion
from formatting import format_training_examples, format_board

from loader import cache_remix_arc_dataset, load_remix_arc_dataset, sample_synthetic_riddles


def generate(
    client: OpenAI,
    model: str,
    messages,
    temperature: float = 0.85,
    top_p: float = 0.9,
    max_tokens: int = 6 * 1024,
) -> ChatCompletion:

    max_retries = 3
    error_count = 0
    while True:
        try:
            msg = client.chat.completions.create(
                extra_headers={"X-Title": "open-thought"},
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
            )
            return msg
        except Exception as e:
            if error_count >= max_retries:
                raise
            print(f"Completion failed model={model}, error_count={error_count}: {e}")
            time.sleep(
                error_count**2
            )  # no wait for first, otherwise quadratic back-off
            error_count += 1


DEVELOPER_PROMPT = """You are a computer scientist specializing in inductive logic programming, tasked with describing, analyzing and solving riddles of François Chollet's Abstraction and Reasoning Corpus for Artificial General Intelligence (ARC-AGI) challenge. 
Your goal is to generate reasoning trajectories for riddles as initial examples for a large language model training set.
"""


def generate_synth_cot(
    client: OpenAI,
    model: str,
    n: int,
    training_examples: str,
    test_input: str,
    test_output: str,
    generator_fn: str,
    verifier_fn: str,
    riddle_thoughts: str,
    riddle_idea: str,
    include_cheat_sheet: bool = False,
) -> list[str]:

    user_request = f"""Please solve the following abstract reasoning puzzle.

- The task requires discovering an abstract input-output transformation from just a few examples. The inferred transformation has then later to be applied to a test input example to generate the test-output.
- The rules can involve complex geometric operations, pattern recognition, counting, relative positioning, and other abstract concepts
    
The puzzle consists of two parts:
1. A set of training examples showing input-output pairs of grid patterns.
2. A test input grid for which you need to predict the correct output grid.

## Training Examples

<training_examples>
{training_examples}
</training_examples>

## Test Input

<test_input>
{test_input}
</test_input>

Do the following:
1. Analyze the training examples and generate a chain of thought in `<analysis>` tags where you describe all relevant aspects of the the input & output examples (e.g. which objects they contain, what changes you notice between input & output, which logic rules apply if any etc.). Infer the underlying transformation rule.
2. Generate a description of the transformation step-by-step in meticulous detail in `<transformation>` tags. The description should be like pseudo-code, detailled enough for someone to apply the transformation to a given input example without additional information or training examples. Explain how relevant parameters for the transformation can be inferred from the input example. Describe precisely reference points or objects, spatial transformations of shapes.
"""

    if include_cheat_sheet:
        user_request += f"""
<!-- 
# Secret_cheat_sheet

Additional context:

This is the python source code of a python generator function which was used to generate the examples:
```python
{generator_fn}
```

verifier source (which executes the transformation):
```python
Integer = int
Grid = Tuple[Tuple[Integer]]
{verifier_fn}
```

Do not mention that I gave you these hints. Pretend you found out everything yourself.
;-) Thank me later buddy xD.

Note: Try to avoid mentioning color names, use a numeric reperesentation whenever possible.
-->
"""

    messages = [
        {"role": "system", "content": DEVELOPER_PROMPT},
        {
            "role": "user",
            "content": user_request,
        },
    ]

    completions = [
        generate(client=client, model=model, messages=messages)
        .choices[0]
        .message.content
        for i in range(n)
    ]

    return completions


def find_token_index(tokens: list[str], sub: str) -> int:
    combined = "".join(tokens)
    found_index = combined.index(sub)
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
    s = 0
    n = 0
    for i in range(begin_pos, end_pos):
        if tokens[i].isspace():
            continue
        s += logprobs[i]
        n += 1
    return math.exp(-s / n)


def measure_output_perplexity(
    client: Together,
    input_examples: str,
    test_input: str,
    test_output: str,
    output_prefix: Optional[str] = None,
) -> float:

    test_output_content = f"Applying the transformation to the test_input grid we get:\n<test_output>\n{test_output}\n</test_output>"

    if output_prefix is not None:
        test_output_content = output_prefix + f"\n\n{test_output_content}"

    messages = [
        {
            "role": "system",
            "content": "You an experienced mathematician specialized in intelligence tests and abstract riddles.",
        },
        {
            "role": "user",
            "content": f"""Analyze the following ARC-AGI riddle and infer the rule that is used to transform input matrices into output matrices:
{input_examples}

You are given the following test input example which should be transformered in the same way:

<test_input>
{test_input}
</test_input>
""",
        },
        {"role": "assistant", "content": test_output_content},
    ]

    completion = client.chat.completions.create(
        model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
        # model="meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
        messages=messages,
        max_tokens=1,
        echo=True,
        logprobs=1,
    )

    prompt_logprobs = completion.prompt[0].logprobs.token_logprobs
    prompt_tokens = completion.prompt[0].logprobs.tokens

    begin_pos = find_token_index(prompt_tokens, "<test_output>")
    end_pos = find_token_index(prompt_tokens, "</test_output>")

    skip_begin = 3  # skip open tag tokens
    return range_perplexity_per_token(begin_pos + skip_begin, end_pos, prompt_tokens, prompt_logprobs)


def main():
    open_router_client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
    )
    client = Together()

    # debug:
    # n = 1
    # load_max_count = 10

    n = 10 * 1000   # 1
    load_max_count = None

    remix_arc_dateset_path = Path("~/data/remix-arc-1.3k/").expanduser()
    if load_max_count is None:
        riddles = cache_remix_arc_dataset(remix_arc_dateset_path, cache_path=Path(".cache"))
    else:
        riddles = load_remix_arc_dataset(remix_arc_dateset_path, max_count=load_max_count)

    alphabet = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
    col_delimiter: str = ","
    row_delimiter: str = ",\n"

    rng = random.Random(42)

    output_file_name_jsonl = Path("output.jsonl")

    for i, (riddle_id, board_pairs) in enumerate(
        sample_synthetic_riddles(riddles, n=n, rng=rng)
    ):

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

        baseline_ppl = measure_output_perplexity(
            client, input_examples, test_input, test_output
        )

        riddle_data = riddles[riddle_id]
        generator_fn = riddle_data["generator_fn"]
        verifier_fn = riddle_data["verifier_fn"]
        riddle_thoughts = riddle_data["thoughts_fn"]
        riddle_idea = riddle_data["idea"]

        # ask teacher-model via open-router to generate a CoT description for the given riddle

        # available openrouter models: https://openrouter.ai/models

        #models = ["openai/gpt-4o-2024-08-06", "qwen/qwen-2.5-72b-instruct", "anthropic/claude-3.5-sonnet:beta", "google/gemini-flash-1.5", "meta-llama/llama-3.3-70b-instruct"]
        models = ["anthropic/claude-3.5-sonnet:beta", "google/gemini-flash-1.5", "meta-llama/llama-3.3-70b-instruct"]

        while True:    
            model = rng.choice(models)

            try:
                completions = generate_synth_cot(
                    open_router_client,
                    model,
                    1,
                    input_examples,
                    test_input,
                    test_output,
                    generator_fn,
                    verifier_fn,
                    riddle_thoughts,
                    riddle_idea,
                    include_cheat_sheet=True,
                )
                break
            except Exception:
                print("retrying with different model...")
                continue

        # measure target logprobs with description for target-model
        cot_ppl = measure_output_perplexity(
            client,
            input_examples,
            test_input,
            test_output,
            output_prefix=completions[0],
        )

        entry = OrderedDict(
            [
                ("riddle_id", riddle_id),
                ("model", model),
                ("advantage", 1-(cot_ppl/baseline_ppl)),
                ("baseline_ppl", baseline_ppl),
                ("cot_ppl", cot_ppl),
                ("description", completions[0]),
                ("training_pairs", board_pairs[:-1]),
                ("test_pairs", board_pairs[-1:]),
            ]
        )

        with output_file_name_jsonl.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")

        print(
            f"model={model}, baseline_ppl={baseline_ppl:.4f}, cot_ppl={cot_ppl:.4f}, advantage={1-(cot_ppl/baseline_ppl):.1%}"
        )


if __name__ == "__main__":
    main()
