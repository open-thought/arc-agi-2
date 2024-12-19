import json
from pathlib import Path
import pickle
from typing import Optional

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
