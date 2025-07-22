#!/usr/bin/env python3
"""
Build factorization dataset for HRM training.

This script converts the factorization dataset from JSON format to the HRM-compatible
format with proper tokenization and sequence formatting.
"""

from typing import Optional, List, Tuple
import os
import json
import numpy as np
from pathlib import Path

from argdantic import ArgParser
from pydantic import BaseModel
from tqdm import tqdm

from common import PuzzleDatasetMetadata


cli = ArgParser()


class DataProcessConfig(BaseModel):
    input_file: str = "dataset/factorization_dataset.json"
    output_dir: str = "data/factorization-1k"
    
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1
    
    max_number_digits: int = 8  # Maximum digits in input number
    max_factors: int = 10       # Maximum number of prime factors
    max_factor_digits: int = 6  # Maximum digits in each factor
    
    seed: int = 42


def tokenize_factorization_problem(
    number: int, 
    prime_factors: List[int], 
    factor_powers: List[int],
    config: DataProcessConfig
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Tokenize a factorization problem into input and label sequences.
    
    Format:
    Input:  [number_digits] [SEP] [PAD] ... [PAD]
    Label:  [factor1] [^] [power1] [x] [factor2] [^] [power2] [x] ... [END] [PAD] ... [PAD]
    
    Vocabulary:
    0-9: digits
    10: SEP (separator)
    11: ^ (power indicator)  
    12: x (multiplication)
    13: END (end of factorization)
    14: PAD (padding)
    """
    
    # Vocabulary mapping
    DIGIT_OFFSET = 0  # 0-9 for digits
    SEP_TOKEN = 10
    POWER_TOKEN = 11
    MULT_TOKEN = 12
    END_TOKEN = 13
    PAD_TOKEN = 14
    
    # Convert number to digit sequence
    number_str = str(number)
    number_tokens = [int(d) + DIGIT_OFFSET for d in number_str]
    
    # Build input sequence: [number_digits] [SEP] [PAD] ... [PAD]
    input_seq = number_tokens + [SEP_TOKEN]
    
    # Build label sequence: factorization
    label_seq = []
    for i, (factor, power) in enumerate(zip(prime_factors, factor_powers)):
        if i > 0:
            label_seq.append(MULT_TOKEN)  # x between factors
        
        # Add factor digits
        factor_str = str(factor)
        factor_tokens = [int(d) + DIGIT_OFFSET for d in factor_str]
        label_seq.extend(factor_tokens)
        
        # Add power if > 1
        if power > 1:
            label_seq.append(POWER_TOKEN)  # ^
            power_str = str(power)
            power_tokens = [int(d) + DIGIT_OFFSET for d in power_str]
            label_seq.extend(power_tokens)
    
    label_seq.append(END_TOKEN)
    
    # Calculate sequence length (input + label)
    max_input_len = config.max_number_digits + 1  # +1 for SEP
    max_label_len = config.max_factors * (config.max_factor_digits + 3)  # factor + ^ + power + x
    seq_len = max_input_len + max_label_len
    
    # Pad sequences
    input_padded = input_seq + [PAD_TOKEN] * (max_input_len - len(input_seq))
    label_padded = label_seq + [PAD_TOKEN] * (max_label_len - len(label_seq))
    
    # Combine input and label for full sequence
    full_input = input_padded + [PAD_TOKEN] * max_label_len
    full_label = [PAD_TOKEN] * max_input_len + label_padded
    
    return np.array(full_input, dtype=np.uint8), np.array(full_label, dtype=np.uint8)


def load_and_split_data(config: DataProcessConfig) -> Tuple[List, List, List]:
    """Load and split the factorization dataset."""
    
    with open(config.input_file, 'r') as f:
        data = json.load(f)
    
    print(f"Loaded {len(data)} factorization examples")
    
    # Set random seed for reproducible splits
    np.random.seed(config.seed)
    
    # Shuffle data
    indices = np.random.permutation(len(data))
    
    # Calculate split sizes
    n_train = int(len(data) * config.train_split)
    n_val = int(len(data) * config.val_split)
    n_test = len(data) - n_train - n_val
    
    # Split data
    train_data = [data[i] for i in indices[:n_train]]
    val_data = [data[i] for i in indices[n_train:n_train + n_val]]
    test_data = [data[i] for i in indices[n_train + n_val:]]
    
    print(f"Split: {len(train_data)} train, {len(val_data)} val, {len(test_data)} test")
    
    return train_data, val_data, test_data


def convert_subset(data: List, set_name: str, config: DataProcessConfig) -> dict:
    """Convert a data subset to HRM format."""

    results = {
        "inputs": [],
        "labels": [],
        "puzzle_identifiers": [],
        "puzzle_indices": [],
        "group_indices": []
    }

    # Initialize indices - following the Sudoku pattern
    results["puzzle_indices"].append(0)
    results["group_indices"].append(0)

    for i, item in enumerate(tqdm(data, desc=f"Converting {set_name}")):
        # Tokenize the problem
        input_seq, label_seq = tokenize_factorization_problem(
            item["number"],
            item["prime_factors"],
            item["factor_powers"],
            config
        )

        results["inputs"].append(input_seq)
        results["labels"].append(label_seq)
        # Use 0 as puzzle identifier for all examples (like Sudoku)
        results["puzzle_identifiers"].append(0)

        # Each example is treated as a separate puzzle/group
        results["puzzle_indices"].append(i + 1)
        results["group_indices"].append(i + 1)

    return results


@cli.command(singleton=True)
def main(config: DataProcessConfig):
    """Main function to build the factorization dataset."""
    
    print("🔢 Building Factorization Dataset for HRM")
    print("=" * 50)
    
    # Create output directory
    os.makedirs(config.output_dir, exist_ok=True)
    
    # Load and split data
    train_data, val_data, test_data = load_and_split_data(config)
    
    # Convert each subset
    datasets = {}
    for data, set_name in [(train_data, "train"), (val_data, "val"), (test_data, "test")]:
        if len(data) > 0:
            datasets[set_name] = convert_subset(data, set_name, config)
    
    # Calculate metadata
    vocab_size = 15  # 0-9 digits + SEP + ^ + x + END + PAD
    max_input_len = config.max_number_digits + 1
    max_label_len = config.max_factors * (config.max_factor_digits + 3)
    seq_len = max_input_len + max_label_len
    
    metadata = PuzzleDatasetMetadata(
        pad_id=14,  # PAD_TOKEN
        ignore_label_id=14,  # Ignore PAD tokens in loss
        blank_identifier_id=0,  # Use puzzle_id 0 for blank
        vocab_size=vocab_size,
        seq_len=seq_len,
        num_puzzle_identifiers=1,  # Only one puzzle identifier (like Sudoku)
        total_groups=len(train_data) if train_data else 1,  # Each example is a group
        mean_puzzle_examples=1.0,  # One example per puzzle
        sets=list(datasets.keys())
    )
    
    # Save datasets and metadata in the expected format
    for set_name, dataset in datasets.items():
        # Create subdirectory for each split
        split_dir = Path(config.output_dir) / set_name
        split_dir.mkdir(parents=True, exist_ok=True)

        # Save data files in the expected format (all__*.npy)
        for key, data in dataset.items():
            if key in ["inputs", "labels"]:
                # Stack arrays for inputs and labels
                data_array = np.stack(data, axis=0)
            else:
                # Convert to numpy array for indices
                data_array = np.array(data, dtype=np.int32)

            np.save(split_dir / f"all__{key}.npy", data_array)

        # Save metadata for each split
        split_metadata = metadata.model_copy()
        split_metadata.sets = ["all"]  # Each split has one subset called "all"

        with open(split_dir / "dataset.json", 'w') as f:
            json.dump(split_metadata.model_dump(), f, indent=2)

        print(f"✅ Saved {set_name} set: {len(dataset['inputs'])} examples")
    
    print(f"✅ Dataset ready at: {config.output_dir}")
    print(f"\nDataset info:")
    print(f"  Vocabulary size: {vocab_size}")
    print(f"  Sequence length: {seq_len}")
    print(f"  Examples: {sum(len(d['inputs']) for d in datasets.values())}")

    # Create identifiers.json for visualization
    identifiers_file = Path(config.output_dir) / "identifiers.json"
    with open(identifiers_file, 'w') as f:
        json.dump(["<blank>"], f)
    print(f"✅ Saved identifiers: {identifiers_file}")


if __name__ == "__main__":
    cli()
