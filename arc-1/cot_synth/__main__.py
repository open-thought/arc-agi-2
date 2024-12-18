import argparse
from pathlib import Path
import json
import pickle
from tqdm import tqdm

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


def parse_args():
    parser = argparse.ArgumentParser()

    args = parser.parse_args()
    return args


def load_remix_arc_dataset(dataset_path: Path) -> dict[str, dict]:
    riddle_ids = [p.stem for p in dataset_path.glob("*.png")]

    riddles = {}

    # load riddles & their meta-data
    for rid in tqdm(riddle_ids):
        riddle_path = dataset_path / (rid + ".json")
        riddle_meta_path = dataset_path / (rid + ".meta.json")

        board_pairs = json.loads(riddle_path.read_text())
        riddle_meta_data = json.loads(riddle_meta_path.read_text())

        riddles[rid] = {
            "generator_fn": riddle_meta_data["generator_fn"],
            "verifier_fn": riddle_meta_data["verifier_fn"],
            "thoughts_fn": riddle_meta_data["thoughts"],
            "idea": riddle_meta_data["idea"],
            "board_pairs": board_pairs,
        }

    return riddles


def cache_remix_arc_dataset(dataset_path: Path, cache_path: Path) -> dict[str, dict]:
    remix_arc_pickle_path = cache_path / "remix-arc-1.3k.pickle"
    if remix_arc_pickle_path.exists():
        with remix_arc_pickle_path.open("rb") as f:
            remix_arc_data = pickle.load(f)
    else:
        remix_arc_data = load_remix_arc_dataset(dataset_path)
        cache_path.mkdir(exist_ok=True)
        with remix_arc_pickle_path.open("wb") as f:
            pickle.dump(remix_arc_data, f)
    return remix_arc_data


def main():
    args = parse_args()

    print("importing dataset")
    remix_arc_dateset_path = Path("/data/remix-arc-1.3k/")
    data_cache_path = Path(".cache")

    x = cache_remix_arc_dataset(remix_arc_dateset_path, data_cache_path)
    print(len(x))


if __name__ == "__main__":
    main()
