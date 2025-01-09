import random
from itertools import chain
import argparse
import json
from pathlib import Path
import arckit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--output-path", type=str, default="eval_ids.json")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    train_set, eval_set = arckit.load_data()
    ids = set(x.id for x in chain(train_set, eval_set))

    rng = random.Random(args.seed)
    eval_set = sorted(rng.sample(tuple(ids), k=args.n))

    with Path(args.output_path).open("w", encoding="utf-8") as f:
        json.dump(eval_set, f)
