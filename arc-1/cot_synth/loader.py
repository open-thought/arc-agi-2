import json
from pathlib import Path
import pickle
import random
from typing import Iterator, Optional

from tqdm import tqdm


def load_remix_arc_dataset(
    dataset_path: Path, max_count: Optional[int] = None, show_progress: bool = True
) -> dict[str, dict]:
    riddle_ids = [p.stem for p in dataset_path.glob("*.png")]
    riddle_ids.sort()

    if max_count is not None:
        riddle_ids = riddle_ids[:max_count]

    riddles = {}

    # load riddles & their meta-data
    for rid in tqdm(riddle_ids, disable=not show_progress):
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


def cache_remix_arc_dataset(
    dataset_path: Path, cache_path: Path, show_progress: bool = True
) -> dict[str, dict]:
    remix_arc_pickle_path = cache_path / "remix-arc-1.3k.pickle"
    if remix_arc_pickle_path.exists():
        with remix_arc_pickle_path.open("rb") as f:
            remix_arc_data = pickle.load(f)
    else:
        remix_arc_data = load_remix_arc_dataset(
            dataset_path, max_count=None, show_progress=show_progress
        )
        cache_path.mkdir(exist_ok=True)
        with remix_arc_pickle_path.open("wb") as f:
            pickle.dump(remix_arc_data, f)
    return remix_arc_data


def sample_synthetic_riddles(
    riddles: dict[str, dict],
    n: int,
    min_examples: int = 4,
    max_examples: int = 7,
    rng: Optional[random.Random] = random,
) -> Iterator[tuple[str, list[tuple]]]:
    assert max_examples >= min_examples

    riddle_ids = list(riddles.keys())
    rng.shuffle(riddle_ids)

    for i in range(n):
        riddle_id = riddle_ids[i % len(riddle_ids)]
        riddle_pairs = riddles[riddle_id]["board_pairs"]

        num_examples = rng.randint(4, 7)
        board_pairs = rng.sample(riddle_pairs, k=num_examples)

        yield riddle_id, board_pairs
