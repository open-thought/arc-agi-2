import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Optional, Union

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Process JSONL file and group PPL values by ID')
    parser.add_argument(
        '--input', 
        type=str,
        default='./output/llama-3.2-3B-instruct-stage0-simple-t5-64.jsonl',
        help='Input JSONL file path'
    )
    return parser.parse_args()

def process_jsonl(file_path: str) -> dict[str, list[Optional[float]]]:
    """Process JSONL file and group PPL values by ID."""
    ppl_by_id = defaultdict(list)
    
    with open(file_path, 'r') as f:
        for line in f:
            entry = json.loads(line)
            ppl_by_id[entry['id']].append(entry['ground_truth_ppl'])
    
    return dict(ppl_by_id)

def main():
    args = parse_args()
    input_path = Path(args.input)
    
    if not input_path.exists():
        print(f"Error: Input file {input_path} does not exist")
        return
    
    ppl_by_id = process_jsonl(args.input)
    
    # Print summary statistics
    print(f"Found {len(ppl_by_id)} unique IDs")
    for id, ppls in ppl_by_id.items():
        valid_ppls = [p for p in ppls if p is not None]
        print(f"ID {id}: {len(ppls)} entries, {len(valid_ppls)} with valid PPL values")

if __name__ == '__main__':
    main()
