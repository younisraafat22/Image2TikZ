#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Advanced analysis metrics for knowledge distillation evaluation.

This module provides sophisticated metrics for analyzing prediction quality,
confidence alignment, and distribution characteristics.
"""

from typing import Dict, List
import torch
import torch.nn.functional as F


@torch.no_grad()
def compute_confidence_alignment(
    teacher_logits: torch.Tensor,
    model_logits: torch.Tensor,
) -> Dict[str, float]:
    """Compute how well student confidence aligns with teacher confidence."""
    if teacher_logits.size(1) < 2 or model_logits.size(1) < 2:
        return {"confidence_mse": 0.0, "confidence_correlation": 0.0}
    
    t_logits = teacher_logits[:, :-1, :].contiguous()
    m_logits = model_logits[:, :-1, :].contiguous()
    
    # Get max probabilities (confidence)
    t_conf = F.softmax(t_logits, dim=-1).max(dim=-1)[0]
    m_conf = F.softmax(m_logits, dim=-1).max(dim=-1)[0]
    
    # MSE between confidences
    conf_mse = F.mse_loss(m_conf, t_conf).item()
    
    # Correlation between confidences
    t_conf_flat = t_conf.view(-1)
    m_conf_flat = m_conf.view(-1)
    
    t_centered = t_conf_flat - t_conf_flat.mean()
    m_centered = m_conf_flat - m_conf_flat.mean()
    
    numerator = (t_centered * m_centered).sum()
    denominator = torch.sqrt((t_centered ** 2).sum() * (m_centered ** 2).sum())
    
    conf_corr = (numerator / denominator).item() if denominator > 1e-8 else 0.0
    
    return {"confidence_mse": conf_mse, "confidence_correlation": conf_corr}


@torch.no_grad()
def topk_overlap(
    teacher_logits: torch.Tensor,
    model_logits: torch.Tensor,
    k: int = 5,
) -> float:
    """Compute mean |TopK_teacher n TopK_student| / K over all next-token positions."""
    if teacher_logits.size(1) < 2 or model_logits.size(1) < 2:
        return 0.0
    t_logits = teacher_logits[:, :-1, :].contiguous()
    m_logits = model_logits[:, :-1, :].contiguous()
    min_vocab = min(t_logits.size(-1), m_logits.size(-1))
    t_logits = t_logits[..., :min_vocab]
    m_logits = m_logits[..., :min_vocab]
    t_topk = t_logits.topk(k, dim=-1).indices  # [B, S-1, K]
    m_topk = m_logits.topk(k, dim=-1).indices  # [B, S-1, K]
    # Broadcast compare -> [B, S-1, K, K]
    overlap_matrix = t_topk.unsqueeze(-1) == m_topk.unsqueeze(-2)
    # For each teacher token (axis -2) any match across model top-k (axis -1)
    teacher_token_matched = overlap_matrix.any(dim=-1).float()  # [B, S-1, K]
    intersection_size = teacher_token_matched.sum(dim=-1)       # [B, S-1]
    frac = (intersection_size / float(k)).mean()
    return float(frac.item())


@torch.no_grad()
def analyze_topk_quality(
    teacher_logits: torch.Tensor,
    model_logits: torch.Tensor,
    labels: torch.Tensor,
    k_values: List[int] = [1, 3, 5, 10],
) -> Dict[str, float]:
    """Analyze how well student's top-k predictions align with teacher and ground truth."""
    if teacher_logits.size(1) < 2 or model_logits.size(1) < 2:
        return {f"teacher_topk_{k}_recall": 0.0 for k in k_values}
    
    t_logits = teacher_logits[:, :-1, :].contiguous()
    m_logits = model_logits[:, :-1, :].contiguous()
    shifted_labels = labels[:, 1:].contiguous()
    
    # Align vocab sizes
    min_vocab = min(t_logits.size(-1), m_logits.size(-1))
    t_logits = t_logits[..., :min_vocab]
    m_logits = m_logits[..., :min_vocab]
    
    valid_mask = (shifted_labels != -100)
    results = {}
    
    for k in k_values:
        # Teacher's top-k
        t_topk = t_logits.topk(k, dim=-1).indices
        # Model's top-k  
        m_topk = m_logits.topk(k, dim=-1).indices
        
        # How often does student's top-k contain teacher's top choice?
        t_top1 = t_logits.argmax(dim=-1)
        student_contains_teacher_top1 = (m_topk == t_top1.unsqueeze(-1)).any(dim=-1)
        teacher_top1_recall = (student_contains_teacher_top1 & valid_mask).float().mean().item()
        results[f"teacher_top1_in_student_top{k}"] = teacher_top1_recall
        
        # How often does student's top-k contain the ground truth?
        gt_in_student_topk = (m_topk == shifted_labels.unsqueeze(-1)).any(dim=-1)
        gt_recall = (gt_in_student_topk & valid_mask).float().mean().item()
        results[f"gt_in_student_top{k}"] = gt_recall
        
        # Overlap ratio between teacher and student top-k sets
        overlap_matrix = (t_topk.unsqueeze(-1) == m_topk.unsqueeze(-2))
        overlap_count = overlap_matrix.any(dim=-1).float().sum(dim=-1)
        overlap_ratio = (overlap_count / k).mean().item()
        results[f"topk_{k}_overlap_ratio"] = overlap_ratio
    
    return results


@torch.no_grad()
def analyze_prediction_diversity(
    teacher_logits: torch.Tensor,
    model_logits: torch.Tensor,
    temperature: float = 1.0,
) -> Dict[str, float]:
    """Analyze prediction diversity and uncertainty characteristics."""
    if teacher_logits.size(1) < 2 or model_logits.size(1) < 2:
        return {"teacher_entropy": 0.0, "student_entropy": 0.0, "entropy_ratio": 0.0}
    
    t_logits = teacher_logits[:, :-1, :].contiguous() / temperature
    m_logits = model_logits[:, :-1, :].contiguous() / temperature
    
    # Compute entropies
    t_probs = F.softmax(t_logits, dim=-1)
    m_probs = F.softmax(m_logits, dim=-1)
    
    t_entropy = (-t_probs * t_probs.clamp_min(1e-12).log()).sum(dim=-1).mean().item()
    m_entropy = (-m_probs * m_probs.clamp_min(1e-12).log()).sum(dim=-1).mean().item()
    
    # Effective vocabulary size (inverse participation ratio)
    t_eff_vocab = (1.0 / (t_probs ** 2).sum(dim=-1)).mean().item()
    m_eff_vocab = (1.0 / (m_probs ** 2).sum(dim=-1)).mean().item()
    
    # Gini coefficient for probability distribution inequality
    def gini_coefficient(probs):
        # Sort probabilities
        sorted_probs = torch.sort(probs, dim=-1)[0]
        n = sorted_probs.size(-1)
        index = torch.arange(1, n + 1, device=probs.device).float()
        gini = (2 * index * sorted_probs).sum(dim=-1) / (n * sorted_probs.sum(dim=-1)) - (n + 1) / n
        return gini.mean().item()
    
    t_gini = gini_coefficient(t_probs)
    m_gini = gini_coefficient(m_probs)
    
    return {
        "teacher_entropy": t_entropy,
        "student_entropy": m_entropy, 
        "entropy_ratio": m_entropy / (t_entropy + 1e-8),
        "teacher_effective_vocab": t_eff_vocab,
        "student_effective_vocab": m_eff_vocab,
        "teacher_gini": t_gini,
        "student_gini": m_gini,
    }