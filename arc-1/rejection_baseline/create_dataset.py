import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Optional, Union, List
import matplotlib.pyplot as plt
import numpy as np

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

def plot_ppl_distribution(ppl_by_id: dict[str, list[Optional[float]]]):
    """Create a histogram of PPL values across all entries."""
    # Collect all valid PPL values
    all_ppls = [
        ppl 
        for ppls in ppl_by_id.values() 
        for ppl in ppls 
        if ppl is not None
    ]
    
    if not all_ppls:
        print("No valid PPL values found to plot")
        return
    
    plt.figure(figsize=(10, 6))
    plt.hist(all_ppls, bins=50, edgecolor='black')
    plt.title('Distribution of Perplexity Values')
    plt.xlabel('Perplexity')
    plt.ylabel('Count')
    plt.grid(True, alpha=0.3)
    
    # Add summary statistics
    mean_ppl = np.mean(all_ppls)
    median_ppl = np.median(all_ppls)
    plt.axvline(mean_ppl, color='r', linestyle='dashed', label=f'Mean: {mean_ppl:.2f}')
    plt.axvline(median_ppl, color='g', linestyle='dashed', label=f'Median: {median_ppl:.2f}')
    plt.legend()
    
    plt.show()

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
    
    # Create and show the PPL distribution plot
    plot_ppl_distribution(ppl_by_id)

if __name__ == '__main__':
    main()
