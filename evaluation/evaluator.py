#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Main evaluation orchestrator for knowledge distillation.

This module coordinates the evaluation process and aggregates results
from various metric modules.
"""

import os
import sys
from typing import Dict
import torch
from tqdm import tqdm

# Add parent directory to path for imports
parent_dir = os.path.join(os.path.dirname(__file__), '..')
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from loss_functions import compute_ce_loss, compute_logits_distillation_loss
from core_metrics import (
    compute_ece_and_accuracy, 
    compute_js_divergence, 
    compute_rank_correlation, 
    mean_entropy
)
from advanced_analysis import (
    compute_confidence_alignment,
    topk_overlap,
    analyze_topk_quality,
    analyze_prediction_diversity
)
from eval_utils import prepare_images_for_models


def evaluate_against_teacher(
    teacher_model,
    student_model,
    dataloader,
    temperature: float,
    device: str,
) -> Dict[str, float]:
    """
    Comprehensive evaluation of student model against teacher model.
    
    Args:
        teacher_model: The teacher model
        student_model: The student model to evaluate
        dataloader: DataLoader for evaluation data
        temperature: Temperature for knowledge distillation
        device: Device to run evaluation on
        
    Returns:
        Dictionary containing all evaluation metrics
    """
    # Initialize accumulators
    total_positions = 0
    sum_ce = 0.0
    sum_kl = 0.0
    agree_matches = 0
    ece_sum = 0.0
    acc_sum = 0.0
    batches = 0
    entropy_sum = 0.0
    topk_overlap_sum = 0.0
    
    # Advanced metrics accumulators
    js_div_sum = 0.0
    rank_corr_sum = 0.0
    confidence_metrics_sum = {"confidence_mse": 0.0, "confidence_correlation": 0.0}
    topk_quality_sum = {}
    diversity_metrics_sum = {}

    for batch in tqdm(dataloader, desc="Evaluating", total=len(dataloader)):
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device) if "labels" in batch else None
        images = batch.get("images")
        
        # Prepare images with appropriate dtypes for each model
        images_for_teacher, images_for_model = prepare_images_for_models(
            images, teacher_model, student_model, device
        )

        with torch.no_grad():
            teacher_outputs = teacher_model(
                input_ids=input_ids, 
                images=images_for_teacher, 
                output_hidden_states=False, 
                output_attentions=False
            )
            student_outputs = student_model(
                input_ids=input_ids, 
                images=images_for_model, 
                output_hidden_states=False, 
                output_attentions=False
            )

        # Core metrics
        ce = compute_ce_loss(student_outputs.logits, labels)
        sum_ce += float(ce.item())

        kl = compute_logits_distillation_loss(
            student_outputs.logits, 
            teacher_outputs.logits, 
            temperature=temperature, 
            labels=labels
        )
        sum_kl += float(kl.item())

        # Argmax agreement with teacher on next-token
        if student_outputs.logits.size(1) >= 2 and labels is not None and labels.size(1) >= 2:
            student_pred = student_outputs.logits[:, :-1, :].argmax(dim=-1)
            teacher_pred = teacher_outputs.logits[:, :-1, :].argmax(dim=-1)
            valid = (labels[:, 1:] != -100)
            agree_matches += int(((student_pred == teacher_pred) & valid).sum().item())
            total_positions += int(valid.sum().item())

        # ECE and accuracy vs labels
        ece, acc = compute_ece_and_accuracy(student_outputs.logits, labels)
        ece_sum += ece
        acc_sum += acc

        # Entropy and top-k overlap
        entropy_sum += mean_entropy(student_outputs.logits)
        topk_overlap_sum += topk_overlap(teacher_outputs.logits, student_outputs.logits, k=5)
        
        # Advanced metrics
        js_div_sum += compute_js_divergence(teacher_outputs.logits, student_outputs.logits, temperature)
        rank_corr_sum += compute_rank_correlation(teacher_outputs.logits, student_outputs.logits)
        
        conf_metrics = compute_confidence_alignment(teacher_outputs.logits, student_outputs.logits)
        for key, value in conf_metrics.items():
            confidence_metrics_sum[key] += value
            
        topk_metrics = analyze_topk_quality(teacher_outputs.logits, student_outputs.logits, labels)
        for key, value in topk_metrics.items():
            if key not in topk_quality_sum:
                topk_quality_sum[key] = 0.0
            topk_quality_sum[key] += value
            
        diversity_metrics = analyze_prediction_diversity(teacher_outputs.logits, student_outputs.logits, temperature)
        for key, value in diversity_metrics.items():
            if key not in diversity_metrics_sum:
                diversity_metrics_sum[key] = 0.0
            diversity_metrics_sum[key] += value

        batches += 1

    # Aggregate results
    results = {
        # Core metrics
        "CE": sum_ce / max(1, batches),
        "Perplexity": float(torch.exp(torch.tensor(sum_ce / max(1, batches))).item()),
        "KL_to_teacher": sum_kl / max(1, batches),
        "ArgmaxAgreementWithTeacher": (agree_matches / max(1, total_positions)) if total_positions > 0 else 0.0,
        "ECE": ece_sum / max(1, batches),
        "NextTokenAccuracy": acc_sum / max(1, batches),
        "MeanEntropy": entropy_sum / max(1, batches),
        "Top5OverlapWithTeacher": topk_overlap_sum / max(1, batches),
        
        # Advanced metrics
        "JS_Divergence": js_div_sum / max(1, batches),
        "RankCorrelation": rank_corr_sum / max(1, batches),
    }
    
    # Add confidence alignment metrics
    for key, value in confidence_metrics_sum.items():
        results[key] = value / max(1, batches)
    
    # Add top-k quality metrics
    for key, value in topk_quality_sum.items():
        results[key] = value / max(1, batches)
        
    # Add diversity metrics
    for key, value in diversity_metrics_sum.items():
        results[key] = value / max(1, batches)
    
    return results


def evaluate_teacher_self(teacher_model, dataloader, device: str) -> Dict[str, float]:
    """
    Self-evaluation of teacher model for reference baseline.
    
    Args:
        teacher_model: The teacher model
        dataloader: DataLoader for evaluation data
        device: Device to run evaluation on
        
    Returns:
        Dictionary containing teacher self-evaluation metrics
    """
    ce_sum = 0.0
    acc_sum = 0.0
    ece_sum = 0.0
    batches = 0
    
    for batch in tqdm(dataloader, desc="Teacher self-eval", total=len(dataloader)):
        input_ids = batch["input_ids"].to(device)
        labels = batch.get("labels")
        if labels is not None:
            labels = labels.to(device)
        images = batch.get("images")
        if images is not None:
            images = images.to(device).to(torch.bfloat16)
            
        with torch.no_grad():
            teacher_outputs = teacher_model(input_ids=input_ids, images=images)
        
        ce_sum += float(compute_ce_loss(teacher_outputs.logits, labels).item())
        ece, acc = compute_ece_and_accuracy(teacher_outputs.logits, labels)
        ece_sum += ece
        acc_sum += acc
        batches += 1
    
    return {
        "CE": ce_sum / max(1, batches),
        "Perplexity": float(torch.exp(torch.tensor(ce_sum / max(1, batches))).item()),
        "NextTokenAccuracy": acc_sum / max(1, batches),
        "ECE": ece_sum / max(1, batches),
    }