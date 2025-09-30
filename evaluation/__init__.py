#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Knowledge Distillation Evaluation Package

A comprehensive evaluation framework for analyzing knowledge distillation
performance in multimodal sequence-to-sequence models.

Modules:
    core_metrics: Fundamental evaluation metrics (ECE, JS divergence, etc.)
    advanced_analysis: Sophisticated analysis functions (confidence alignment, diversity)
    eval_utils: Model loading and data preparation utilities
    evaluator: Main evaluation orchestrator
    run_evaluation: CLI interface for running evaluations

Usage:
    from evaluation import run_evaluation
    # or
    python -m evaluation.run_evaluation --help
"""

from .core_metrics import (
    compute_ece_and_accuracy,
    compute_js_divergence, 
    compute_rank_correlation,
    mean_entropy
)

from .advanced_analysis import (
    compute_confidence_alignment,
    topk_overlap,
    analyze_topk_quality,
    analyze_prediction_diversity
)

from .evaluator import (
    evaluate_against_teacher,
    evaluate_teacher_self
)

# Results analysis functions removed - simplified evaluation

__version__ = "1.0.0"
__author__ = "Your Name"
__description__ = "Knowledge Distillation Evaluation Framework"

__all__ = [
    # Core metrics
    'compute_ece_and_accuracy',
    'compute_js_divergence', 
    'compute_rank_correlation',
    'mean_entropy',
    
    # Advanced analysis
    'compute_confidence_alignment',
    'topk_overlap',
    'analyze_topk_quality',
    'analyze_prediction_diversity',
    
    # Main evaluator
    'evaluate_against_teacher',
    'evaluate_teacher_self',
    
    # Results analysis functions removed
]