#!/usr/bin/env python3
"""
Evaluate the trained HRM model on factorization tasks.

This script loads the trained model and tests its ability to factorize numbers,
providing detailed analysis of its performance.
"""

import torch
import numpy as np
import json
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import argparse
from tqdm import tqdm

from puzzle_dataset import PuzzleDataset, PuzzleDatasetConfig
from utils.device_utils import get_optimal_device, get_device_name
from models.hrm.hrm_act_v1 import HierarchicalReasoningModel_ACTV1
from models.losses import ACTLossHead
import hydra
from omegaconf import DictConfig, OmegaConf


class FactorizationEvaluator:
    def __init__(self, model_path: str, dataset_path: str, device: torch.device):
        self.device = device
        self.dataset_path = dataset_path
        
        # Load model configuration and weights
        self.model, self.metadata = self._load_model(model_path)
        
        # Vocabulary for decoding
        self.vocab = {
            0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9',
            10: 'SEP', 11: '^', 12: 'x', 13: 'END', 14: 'PAD'
        }
        self.reverse_vocab = {v: k for k, v in self.vocab.items()}
        
    def _load_model(self, model_path: str) -> Tuple[torch.nn.Module, Dict]:
        """Load the trained model and metadata."""
        print(f"Loading model from: {model_path}")
        
        # Load metadata
        metadata_path = Path(self.dataset_path) / "train" / "dataset.json"
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Create model configuration (matching training config)
        # Check if this is the simple model based on path
        if "simple" in model_path:
            # Simple model configuration (smaller)
            model_config = {
                "batch_size": 16,  # This will be overridden during inference
                "seq_len": metadata["seq_len"],
                "vocab_size": metadata["vocab_size"],
                "num_puzzle_identifiers": 1,  # Single puzzle type
                "hidden_size": 128,  # Smaller for simple model
                "num_heads": 4,
                "expansion": 4,
                "H_layers": 1,  # Fewer layers for simple model
                "L_layers": 1,  # Fewer layers for simple model
                "H_cycles": 1,  # Fewer cycles for simple model
                "L_cycles": 1,  # Fewer cycles for simple model
                "puzzle_emb_ndim": 128,  # Smaller for simple model
                "pos_encodings": "rope",
                "halt_max_steps": 4,  # Fewer steps for simple model
                "halt_exploration_prob": 0.05,  # Lower exploration
                "rms_norm_eps": 1e-5,
                "rope_theta": 10000.0,
                "forward_dtype": "bfloat16"
            }
        else:
            # Original model configuration
            model_config = {
                "batch_size": 16,  # This will be overridden during inference
                "seq_len": metadata["seq_len"],
                "vocab_size": metadata["vocab_size"],
                "num_puzzle_identifiers": 1,  # Single puzzle type
                "hidden_size": 256,
                "num_heads": 4,
                "expansion": 4,
                "H_layers": 2,
                "L_layers": 2,
                "H_cycles": 2,
                "L_cycles": 2,
                "puzzle_emb_ndim": 256,
                "pos_encodings": "rope",
                "halt_max_steps": 8,
                "halt_exploration_prob": 0.1,
                "rms_norm_eps": 1e-5,
                "rope_theta": 10000.0,
                "forward_dtype": "bfloat16"
            }

        # Create model
        model = HierarchicalReasoningModel_ACTV1(model_config)
        
        # Load weights if available
        if Path(model_path).exists() and model_path != "no_checkpoint":
            try:
                checkpoint = torch.load(model_path, map_location=self.device)

                # Handle different checkpoint formats
                if 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                elif 'model' in checkpoint:
                    state_dict = checkpoint['model']
                else:
                    state_dict = checkpoint

                # Remove "model." prefix if present
                if any(key.startswith('model.') for key in state_dict.keys()):
                    state_dict = {key.replace('model.', ''): value for key, value in state_dict.items()}

                model.load_state_dict(state_dict)
                print("✅ Loaded trained model weights")
            except Exception as e:
                print(f"⚠️  Failed to load checkpoint: {e}")
                print("⚠️  Using randomly initialized model")
        else:
            print("⚠️  No checkpoint found, using randomly initialized model")
        
        model = model.to(self.device)
        model.eval()
        
        return model, metadata
    
    def decode_tokens(self, tokens: torch.Tensor) -> str:
        """Decode token sequence to human-readable string."""
        decoded = []
        for token in tokens:
            if token.item() == 14:  # PAD token
                break
            decoded.append(self.vocab.get(token.item(), f'UNK({token.item()})'))
        return ' '.join(decoded)
    
    def encode_number(self, number: int) -> List[int]:
        """Encode a number as digit tokens."""
        return [int(d) for d in str(number)]
    
    def parse_factorization(self, tokens: List[int]) -> List[Tuple[int, int]]:
        """Parse factorization tokens into (factor, power) pairs."""
        factors = []
        i = 0
        
        while i < len(tokens):
            if tokens[i] == 13:  # END token
                break
            if tokens[i] == 14:  # PAD token
                break
            if tokens[i] == 12:  # x (multiplication)
                i += 1
                continue
                
            # Parse factor
            factor_digits = []
            while i < len(tokens) and 0 <= tokens[i] <= 9:
                factor_digits.append(tokens[i])
                i += 1
            
            if not factor_digits:
                break
                
            factor = int(''.join(map(str, factor_digits)))
            power = 1
            
            # Check for power
            if i < len(tokens) and tokens[i] == 11:  # ^ token
                i += 1
                power_digits = []
                while i < len(tokens) and 0 <= tokens[i] <= 9:
                    power_digits.append(tokens[i])
                    i += 1
                if power_digits:
                    power = int(''.join(map(str, power_digits)))
            
            factors.append((factor, power))
        
        return factors
    
    def verify_factorization(self, number: int, factors: List[Tuple[int, int]]) -> bool:
        """Verify if the factorization is correct."""
        if not factors:
            return False
        
        product = 1
        for factor, power in factors:
            product *= factor ** power
        
        return product == number
    
    def generate_factorization(self, number: int) -> Tuple[str, List[Tuple[int, int]], bool]:
        """Generate factorization for a given number."""
        # Encode input
        input_tokens = self.encode_number(number) + [10]  # number + SEP
        max_seq_len = self.metadata["seq_len"]
        
        # Pad input
        input_padded = input_tokens + [14] * (max_seq_len - len(input_tokens))
        input_tensor = torch.tensor(input_padded, dtype=torch.long, device=self.device).unsqueeze(0)
        
        # Create labels (all PAD for generation)
        labels = torch.full_like(input_tensor, 14)
        
        # Create batch
        batch = {
            "inputs": input_tensor,
            "labels": labels,
            "puzzle_identifiers": torch.tensor([0], device=self.device)
        }
        
        # Generate
        with torch.no_grad():
            # Initialize carry for the inner model
            batch_size = input_tensor.shape[0]
            carry = self.model.inner.empty_carry(batch_size)

            # Move carry to device
            carry.z_H = carry.z_H.to(self.device)
            carry.z_L = carry.z_L.to(self.device)

            # Forward pass through inner model
            new_carry, logits, (z_H, z_L) = self.model.inner(carry=carry, batch=batch)
            
            # Get predictions (greedy decoding)
            predictions = torch.argmax(logits, dim=-1)
            
            # Extract the generated part (after input)
            input_len = len(input_tokens)
            generated_tokens = predictions[0, input_len:].cpu().tolist()
            
            # Decode
            decoded = self.decode_tokens(predictions[0])
            factors = self.parse_factorization(generated_tokens)
            is_correct = self.verify_factorization(number, factors)
            
            return decoded, factors, is_correct
    
    def evaluate_dataset(self, split: str = "test") -> Dict:
        """Evaluate on the dataset."""
        print(f"\n🧪 Evaluating on {split} set...")
        
        # Load dataset
        config = PuzzleDatasetConfig(
            seed=42,
            dataset_path=self.dataset_path,
            global_batch_size=1,
            test_set_mode=True,
            epochs_per_iter=1,
            rank=0,
            num_replicas=1
        )
        
        dataset = PuzzleDataset(config, split)
        
        results = {
            "total": 0,
            "correct": 0,
            "partial_correct": 0,
            "examples": []
        }
        
        # Evaluate each example
        for i, (set_name, batch, batch_size) in enumerate(tqdm(dataset, desc=f"Evaluating {split}")):
            if i >= 50:  # Limit to first 50 examples for speed
                break
                
            input_tokens = batch["inputs"][0].cpu().tolist()
            
            # Extract the number from input
            number_tokens = []
            for token in input_tokens:
                if token == 10:  # SEP token
                    break
                if 0 <= token <= 9:
                    number_tokens.append(token)
            
            if not number_tokens:
                continue
                
            number = int(''.join(map(str, number_tokens)))
            
            # Generate factorization
            decoded, factors, is_correct = self.generate_factorization(number)
            
            results["total"] += 1
            if is_correct:
                results["correct"] += 1
            
            # Store example
            example = {
                "number": number,
                "predicted_factors": factors,
                "is_correct": is_correct,
                "decoded": decoded
            }
            results["examples"].append(example)
        
        # Calculate accuracy
        accuracy = results["correct"] / results["total"] if results["total"] > 0 else 0
        results["accuracy"] = accuracy
        
        return results
    
    def test_manual_examples(self) -> None:
        """Test on some manual examples."""
        print("\n🔢 Testing Manual Examples:")
        print("=" * 50)
        
        test_numbers = [12, 15, 21, 35, 42, 60, 77, 91, 143, 221]
        
        for number in test_numbers:
            decoded, factors, is_correct = self.generate_factorization(number)
            
            # Calculate actual factorization for comparison
            actual_factors = self._get_prime_factorization(number)
            
            status = "✅" if is_correct else "❌"
            print(f"{status} {number:3d} → Predicted: {factors}")
            print(f"      Actual: {actual_factors}")
            print(f"      Decoded: {decoded[:80]}...")
            print()
    
    def _get_prime_factorization(self, n: int) -> List[Tuple[int, int]]:
        """Get the actual prime factorization of a number."""
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


def main():
    parser = argparse.ArgumentParser(description="Evaluate factorization model")
    parser.add_argument("--model-path", default="outputs", help="Path to model checkpoint")
    parser.add_argument("--dataset-path", default="data/factorization-1k", help="Path to dataset")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"], help="Dataset split to evaluate")
    
    args = parser.parse_args()
    
    # Setup device
    device = get_optimal_device()
    print(f"🖥️  Using device: {get_device_name()}")
    
    # Check if model path is a direct file or directory
    model_path = Path(args.model_path)

    if model_path.is_file():
        latest_checkpoint = model_path
        print(f"📁 Using checkpoint: {latest_checkpoint}")
    elif model_path.is_dir():
        checkpoint_files = list(model_path.glob("**/*.pt"))
        if checkpoint_files:
            latest_checkpoint = max(checkpoint_files, key=lambda x: x.stat().st_mtime)
            print(f"📁 Found checkpoint: {latest_checkpoint}")
        else:
            latest_checkpoint = args.model_path  # Pass the path as-is
            print(f"📁 Using checkpoint path: {latest_checkpoint}")
    else:
        latest_checkpoint = args.model_path  # Pass the path as-is
        print(f"📁 Using checkpoint path: {latest_checkpoint}")
    
    # Create evaluator
    evaluator = FactorizationEvaluator(
        model_path=str(latest_checkpoint),
        dataset_path=args.dataset_path,
        device=device
    )
    
    # Run evaluations
    print("🚀 Starting Factorization Model Evaluation")
    print("=" * 60)
    
    # Test manual examples
    evaluator.test_manual_examples()
    
    # Evaluate on dataset
    results = evaluator.evaluate_dataset(args.split)
    
    # Print results
    print("\n📊 Evaluation Results:")
    print("=" * 30)
    print(f"Dataset: {args.split}")
    print(f"Total examples: {results['total']}")
    print(f"Correct: {results['correct']}")
    print(f"Accuracy: {results['accuracy']:.2%}")
    
    # Show some examples
    print(f"\n📝 Sample Results:")
    for i, example in enumerate(results['examples'][:10]):
        status = "✅" if example['is_correct'] else "❌"
        print(f"{status} {example['number']} → {example['predicted_factors']}")


if __name__ == "__main__":
    main()
