#!/usr/bin/env python3
"""
Build a simple factorization dataset with smaller numbers for faster training.
This focuses on numbers 2-100 to make the learning task easier and faster.
"""

import json
import numpy as np
import argparse
from pathlib import Path
from typing import List, Tuple, Dict
import random


def get_prime_factorization(n: int) -> List[Tuple[int, int]]:
    """Get prime factorization as list of (factor, power) pairs."""
    factors = []
    d = 2
    while d * d <= n:
        count = 0
        while n % d == 0:
            n //= d
            count += 1
        if count > 0:
            factors.append((d, count))
        d += 1
    if n > 1:
        factors.append((n, 1))
    return factors


def factorization_to_string(factors: List[Tuple[int, int]]) -> str:
    """Convert factorization to string format."""
    if not factors:
        return "1 END"
    
    parts = []
    for i, (factor, power) in enumerate(factors):
        if i > 0:
            parts.append("x")
        
        parts.append(str(factor))
        if power > 1:
            parts.extend(["^", str(power)])
    
    parts.append("END")
    return " ".join(parts)


def create_simple_factorization_dataset(
    min_num: int = 2,
    max_num: int = 100,
    output_file: str = "dataset/simple_factorization_dataset.json"
) -> None:
    """Create a simple factorization dataset."""
    
    dataset = []
    
    # Generate all numbers in range
    for number in range(min_num, max_num + 1):
        factors = get_prime_factorization(number)
        factorization_str = factorization_to_string(factors)
        
        # Create input/output pair
        input_str = f"{number} SEP"
        output_str = factorization_str
        
        # Determine complexity level
        if number <= 10:
            complexity = "simple"
        elif number <= 50:
            complexity = "medium"
        else:
            complexity = "hard"
        
        # Separate factors and powers for the expected format
        prime_factors = [f[0] for f in factors]  # Extract factors
        factor_powers = [f[1] for f in factors]  # Extract powers

        dataset.append({
            "input": input_str,
            "output": output_str,
            "number": number,
            "prime_factors": prime_factors,
            "factor_powers": factor_powers,
            "complexity_level": complexity
        })
    
    # Shuffle the dataset
    random.shuffle(dataset)
    
    # Save to file
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(dataset, f, indent=2)
    
    print(f"✅ Created simple factorization dataset with {len(dataset)} examples")
    print(f"📁 Saved to: {output_file}")
    print(f"📊 Number range: {min_num}-{max_num}")
    
    # Show some examples
    print("\n📝 Sample examples:")
    for i, example in enumerate(dataset[:5]):
        factors_with_powers = list(zip(example['prime_factors'], example['factor_powers']))
        print(f"  {example['number']} → {factors_with_powers} → {example['output']}")


def main():
    parser = argparse.ArgumentParser(description="Build simple factorization dataset")
    parser.add_argument("--min-num", type=int, default=2, help="Minimum number to factorize")
    parser.add_argument("--max-num", type=int, default=100, help="Maximum number to factorize")
    parser.add_argument("--output-file", default="dataset/simple_factorization_dataset.json", help="Output file path")
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    create_simple_factorization_dataset(
        min_num=args.min_num,
        max_num=args.max_num,
        output_file=args.output_file
    )


if __name__ == "__main__":
    main()
