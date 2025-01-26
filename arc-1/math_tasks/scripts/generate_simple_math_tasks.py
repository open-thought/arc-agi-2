import json
from pathlib import Path
from random import Random
import sys

def generate_math_task(
    rng: Random, num_terms: int, num_digits: int, op: list[str] = ["+", "-", "*"]
) -> tuple[str, int]:
    parts = []

    def add_terms(remaining: int):
        num_left = rng.randint(1, remaining)
        num_right = remaining - num_left

        if num_left > 1 and rng.random() > 0.5:
            if rng.random() > 0.5:
                parts.append("-(")
            else:
                parts.append("(")
            add_terms(num_left)
            parts.append(")")
        else:
            for i in range(num_left):
                c = rng.randint(-(10**num_digits) + 1, 10**num_digits - 1)
                parts.append(str(c))
                if i + 1 < num_left:
                    parts.append(rng.choice(op))

        if num_right > 0:
            parts.append(rng.choice(op))
            add_terms(num_right)

    add_terms(num_terms)

    space_parts = []
    for p in parts:
        while rng.random() < 0.15:
            space_parts.append(" ")
        space_parts.append(p)

    term = " ".join(space_parts)
    ground_truth = eval(term)

    return term, ground_truth

def main():
    rng = Random(42)

    num_tasks = int(sys.argv[1]) if len(sys.argv) > 1 else 100_000

    try:
        dir = Path(__file__).parent.parent
    except:
        dir = Path("..")
    output_filename = str(dir / "data" / "math_tasks.jsonl")
    Path(output_filename).parent.mkdir(parents=True, exist_ok=True)
    file_path = Path(output_filename)

    # If num_tasks is provided via arg, create uniform distributions
    combinations = None
    if len(sys.argv) > 1:
        # Create all possible combinations of num_terms (2-6) and num_digits (1-6)
        combinations = [(t, d) for t in range(2, 7) for d in range(1, 7)]
        tasks_per_combo = max(1, num_tasks // len(combinations))

    with file_path.open("w", encoding="utf-8") as f:
        for i in range(num_tasks):
            if combinations:
                combo_idx = i // tasks_per_combo
                if combo_idx >= len(combinations):
                    num_terms = rng.randint(2, 6)
                    num_digits = rng.randint(1, 6)
                else:
                    num_terms, num_digits = combinations[combo_idx]
            else:
                # Original random distribution
                num_terms = rng.randint(2, 6)
                num_digits = rng.randint(1, 6)

            term, ground_truth = generate_math_task(
                rng, num_terms=num_terms, num_digits=num_digits
            )
            if abs(ground_truth) > 10**8 or abs(ground_truth) < 10:
                continue

            question_templates = [
                "{0}",
                "{0} =",
                "{0} = ?",
                "What is {0}?",
                "Solve {0}",
            ]

            template = rng.choice(question_templates)
            formatted_task = template.format(term)

            entry = {
                "id": str(i),
                "question": formatted_task,
                "answer": str(ground_truth),
                "num_terms": num_terms,
                "num_digits": num_digits,
            }

            json.dump(entry, f)
            f.write("\n")

if __name__ == "__main__":
    main()