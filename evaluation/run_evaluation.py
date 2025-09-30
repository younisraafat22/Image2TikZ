#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Main CLI script for knowledge distillation evaluation.

Usage example:
  python evaluation/run_evaluation.py \
    --teacher_model nllg/detikzify-cl-7b \
    --student_model nllg/detikzify-tl-1.1b \
    --distilled_model ./outputs/model \
    --dataset nllg/datikz-v3 --split test --max_samples 1000 \
    --batch_size 2 --temperature 3.0 --device cuda
"""

import argparse
import os
import sys
import torch

# Add parent directory to path for imports
parent_dir = os.path.join(os.path.dirname(__file__), '..')
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from model_utils import load_teacher_model, load_student_model
from eval_utils import load_model_any, build_dataloader, _bf16_supported
from evaluator import evaluate_against_teacher, evaluate_teacher_self
import json


def parse_args():
    """Parse command line arguments."""
    p = argparse.ArgumentParser(description="Comprehensive evaluation of KD metrics against teacher")
    p.add_argument("--teacher_model", type=str, default="nllg/detikzify-cl-7b")
    p.add_argument("--student_model", type=str, default="nllg/detikzify-tl-1.1b")
    p.add_argument("--distilled_model", type=str, default=None, help="Path or hub id for distilled model")
    p.add_argument("--dataset", type=str, default="nllg/datikz-v3")
    p.add_argument("--split", type=str, default="test")
    p.add_argument("--max_samples", type=int, default=1000)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--temperature", type=float, default=3.0)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--output_file", type=str, default=None, help="Save detailed results to JSON file")
    p.add_argument("--compare_all", action="store_true", help="Compare teacher, student, and distilled models")
    p.add_argument("--num_workers", type=int, default=2, help="Dataloader worker processes (set 0 for Windows safety)")
    p.add_argument("--no_bf16", action='store_true', help="Disable bfloat16 casting even if supported")

    return p.parse_args()


def print_results(results: dict, model_name: str):
    """Print evaluation results for a model."""
    print(f"\n{'='*80}")
    print(f"{model_name.upper()} EVALUATION RESULTS")
    print(f"{'='*80}")
    print(json.dumps(results, indent=2))


def print_simple_comparison(all_results: dict):
    """Print a simple comparison table."""
    print(f"\n{'='*80}")
    print("SUMMARY COMPARISON")
    print(f"{'='*80}")
    
    key_metrics = [
        "CE", "Perplexity", "KL_to_teacher", "ArgmaxAgreementWithTeacher", 
        "NextTokenAccuracy", "ECE", "JS_Divergence", "RankCorrelation"
    ]
    
    print(f"{'Metric':<30} ", end="")
    for model_name in all_results.keys():
        print(f"{model_name:<20}", end="")
    print()
    print("-" * (30 + 20 * len(all_results)))
    
    for metric in key_metrics:
        print(f"{metric:<30} ", end="")
        for model_name, results in all_results.items():
            value = results.get(metric, "N/A")
            if isinstance(value, float):
                print(f"{value:<20.4f}", end="")
            else:
                print(f"{str(value):<20}", end="")
        print()


def save_results_simple(results: dict, output_file: str):
    """Save evaluation results to JSON file."""
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_file}")


def load_and_prepare_teacher(teacher_model_name: str, device: str, use_bf16: bool) -> tuple:
    """Load and prepare teacher model."""
    print("Loading teacher model...")
    teacher_model, teacher_tokenizer, teacher_config = load_teacher_model(teacher_model_name, device=device)
    
    if use_bf16 and _bf16_supported(device):
        teacher_model = teacher_model.to(device).to(dtype=torch.bfloat16)
    else:
        teacher_model = teacher_model.to(device)
    
    teacher_model.eval()
    if hasattr(teacher_model, 'config') and hasattr(teacher_model.config, 'use_cache'):
        teacher_model.config.use_cache = True
    
    return teacher_model, teacher_tokenizer, teacher_config


def evaluate_baseline_student(args, teacher_model):
    """Evaluate baseline student model."""
    print("\nLoading and evaluating baseline student model...")
    student_model, student_tokenizer, student_config = load_student_model(args.student_model, device=args.device)
    
    if not args.no_bf16 and _bf16_supported(args.device):
        student_model = student_model.to(args.device).to(dtype=torch.bfloat16)
    else:
        student_model = student_model.to(args.device)
    
    student_model.eval()
    if hasattr(student_model, 'config') and hasattr(student_model.config, 'use_cache'):
        student_model.config.use_cache = True

    dataloader = build_dataloader(
        args.dataset, args.split, student_tokenizer, student_config, 
        args.batch_size, args.max_samples, num_workers=args.num_workers
    )
    
    return evaluate_against_teacher(teacher_model, student_model, dataloader, args.temperature, args.device)


def evaluate_distilled_model(args, teacher_model):
    """Evaluate distilled model."""
    print(f"\nLoading and evaluating distilled model: {args.distilled_model}")
    distilled_model, distilled_tokenizer, distilled_config = load_model_any(
        args.distilled_model, device=args.device, use_bf16=(not args.no_bf16)
    )
    
    dataloader = build_dataloader(
        args.dataset, args.split, distilled_tokenizer, distilled_config,
        args.batch_size, args.max_samples, num_workers=args.num_workers
    )
    
    return evaluate_against_teacher(teacher_model, distilled_model, dataloader, args.temperature, args.device)


def main():
    """Main evaluation pipeline."""
    args = parse_args()
    
    # Load teacher model
    teacher_model, teacher_tokenizer, teacher_config = load_and_prepare_teacher(
        args.teacher_model, args.device, not args.no_bf16
    )
    
    all_results = {}
    
    # Evaluate baseline student if requested
    if args.compare_all:
        student_results = evaluate_baseline_student(args, teacher_model)
        all_results["baseline_student"] = student_results
        print_results(student_results, "baseline student vs teacher")
    
    # Evaluate distilled model if provided
    if args.distilled_model:
        distilled_results = evaluate_distilled_model(args, teacher_model)
        all_results["distilled_model"] = distilled_results
        print_results(distilled_results, "distilled model vs teacher")
    
    # Teacher self-evaluation for reference
    print("\nEvaluating teacher model (self-evaluation)...")
    reference_tokenizer = teacher_tokenizer
    reference_config = teacher_config
    
    if 'baseline_student' in all_results:
        # Use student tokenizer if available for consistency
        _, reference_tokenizer, reference_config = load_student_model(args.student_model, device=args.device)
    elif 'distilled_model' in all_results:
        # Use distilled model tokenizer if available
        _, reference_tokenizer, reference_config = load_model_any(args.distilled_model, device=args.device, use_bf16=False)
    
    dataloader = build_dataloader(
        args.dataset, args.split, reference_tokenizer, reference_config,
        args.batch_size, args.max_samples, num_workers=args.num_workers
    )
    
    teacher_results = evaluate_teacher_self(teacher_model, dataloader, args.device)
    all_results["teacher_self"] = teacher_results
    print_results(teacher_results, "teacher self-evaluation")
    
    # Generate comparison table if multiple models evaluated
    if len(all_results) > 1:
        print_simple_comparison(all_results)
    
    # Save results if requested
    if args.output_file:
        save_results_simple(all_results, args.output_file)
    
    return all_results


if __name__ == "__main__":
    main()