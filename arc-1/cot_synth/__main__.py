import argparse
import random
from pathlib import Path
import json
import pickle
from typing import Optional
from tqdm import tqdm
from datasets import Dataset, Image, Sequence

from loader import load_remix_arc_dataset, cache_remix_arc_dataset
from visu import plot_task
from formatting import format_training_examples, format_board

# Llama 3.2-Vision 11B, 128k context
# native image tile size of vision adapter: 448 px

# Llama 3.2-Vision: https://huggingface.co/docs/transformers/main/model_doc/mllama
# Unsloth llama vision finetuning notebook: https://colab.research.google.com/drive/1j0N4XTY1zXXy7mPAhOC1_gMYZ2F2EBlk?usp=sharing
# Axolotl example: https://github.com/axolotl-ai-cloud/axolotl/blob/effc4dc4097af212432c9ebaba7eb9677d768467/examples/llama-3-vision/lora-11b.yaml


# ask Sonnet to generate an analyse and instruction to for an ARC riddle based on generator code
# 1. Analysis of the riddle, first impression, riddle class and core idea, concepts used objects
# 2. Detailed natural language program which explains the transformation


# Ask Sonnet to verify and possibly improve its own description when shown without the corresponding generator/verifier code.
# try generation with image input


# 1. load riddle examples + generator & verifier
# 2. create text representation of riddle
# 3. generate riddle visualization representation png image
#    - generate single image with overview of transformation
#    - generate multiple (specify color palette, optionally write numbers into fields)
# prepare data and send prompt to sonnet



def generate_transduction_dataset(riddles: dict[str, dict], n: int):
    color_codes = [
        "#000",
        "#0074D9",
        "#FF4136",
        "#2ECC40",
        "#FFDC00",
        "#AAAAAA",
        "#F012BE",
        "#FF851B",
        "#7FDBFF",
        "#870C25",
    ]
    color_names = [
        "black",
        "blue",
        "red",
        "green",
        "yellow",
        "gray",
        "magenta",
        "orange",
        "sky",
        "brown",
    ]

    system_message = "You are a world-class puzzle solver with exceptional pattern recognition skills. Your task is to analyze puzzles, spot patterns, and provide direct solutions."
    alphabet = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
    col_delimiter: str = ","
    row_delimiter: str = ",\n"

    riddle_ids = list(riddles.keys())
    random.shuffle(riddle_ids)

    img_dir = Path("./images/")
    img_dir.mkdir(exist_ok=True)

    messages: list[list] = []
    images: list[list] = []
    
    for i in range(n):
        riddle_id = riddle_ids[i % len(riddle_ids)]

        riddle_pairs = riddles[riddle_id]["board_pairs"]

        # pick random riddle, select 4 to 7 random example boards
        num_examples =  random.randint(4, 7)
        board_pairs = random.sample(riddle_pairs, k=num_examples)

        image_path = img_dir / f"{i}.png"
        plot_task(board_pairs, filename=image_path, hide_last_output=True)

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

        
        conversation = []
        # # add system prompt
        # seems axolotl doesn't like system prompt with images .. got:
        # jinja2.exceptions.TemplateError: Prompting with images is incompatible with system messages.
        # conversation.append(
        #     {"content": [{"text": system_message, "type": "text"}], "role": "system"}
        # )

        buffer = []
        buffer.append("The following color mapping is used: ")
        buffer.append(
            ", ".join([f"{alphabet[i]}: {name}" for i, name in enumerate(color_names)])
        )
        buffer.append("\n\n")

        buffer.append("Input examples:\n")
        buffer.append(input_examples)

        buffer.append("Test input:\n")
        buffer.append(test_input)

        question = "".join(buffer)

        conversation.append(
            {"content": [{"text": question, "type": "text"}, { "index": 0, "type": "image" }], "role": "user"}
        )

        answer = test_output
        conversation.append(
            {"content": [{"text": answer, "type": "text"}, ], "role": "assistant"}
        )

        messages.append(conversation)
        images.append([str(image_path)])
        
    # generate hf dataset
    ds = Dataset.from_dict({"messages": messages})
    ds = ds.add_column("images", images) 
    ds = ds.cast_column("images", Sequence(Image()))  
    ds.save_to_disk("~/data/dummy_vlm_train")

    print("done")


def parse_args():
    parser = argparse.ArgumentParser()

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    print("importing dataset")
    remix_arc_dateset_path = Path("~/data/remix-arc-1.3k/").expanduser()
    data_cache_path = Path(".cache")
    # x = cache_remix_arc_dataset(remix_arc_dateset_path, data_cache_path)
    x = load_remix_arc_dataset(remix_arc_dateset_path, max_count=10)

    generate_transduction_dataset(x, n=50)



if __name__ == "__main__":
    main()
