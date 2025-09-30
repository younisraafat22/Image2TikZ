#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Core evaluation metrics for knowledge distillation analysis.

This module contains the fundamental metrics for evaluating
knowledge transfer quality between teacher and student models.
"""

from typing import Dict, Tuple
import torch
import torch.nn.functional as F


@torch.no_grad()
def compute_ece_and_accuracy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100,
    num_bins: int = 10,
) -> Tuple[float, float]:
    """Compute ECE (expected calibration error) and accuracy for next-token predictions.

    Aligns to next-token: uses logits[:, :-1] vs labels[:, 1:].
    """
    if logits.size(1) < 2 or labels.size(1) < 2:
        return 0.0, 0.0

    shifted_logits = logits[:, :-1, :].contiguous()
    shifted_labels = labels[:, 1:].contiguous()

    valid_mask = (shifted_labels != ignore_index)
    if not valid_mask.any():
        return 0.0, 0.0

    probs = F.softmax(shifted_logits, dim=-1)
    confidences, preds = probs.max(dim=-1)
    correct = (preds == shifted_labels) & valid_mask

    # Accuracy over valid positions
    accuracy = correct.sum().item() / valid_mask.sum().item()

    # ECE
    ece = 0.0
    bin_boundaries = torch.linspace(0, 1, steps=num_bins + 1, device=logits.device)
    for i in range(num_bins):
        lo = bin_boundaries[i]
        hi = bin_boundaries[i + 1]
        in_bin = valid_mask & (confidences >= lo) & (confidences < hi if i < num_bins - 1 else confidences <= hi)
        if in_bin.any():
            bin_acc = correct[in_bin].float().mean().item()
            bin_conf = confidences[in_bin].float().mean().item()
            ece += (in_bin.sum().item() / valid_mask.sum().item()) * abs(bin_acc - bin_conf)
    return float(ece), float(accuracy)


@torch.no_grad()
def compute_js_divergence(
    teacher_logits: torch.Tensor,
    model_logits: torch.Tensor,
    temperature: float = 1.0,
    eps: float = 1e-8,
) -> float:
    """Compute Jensen-Shannon divergence between teacher and model distributions with numerical stability."""
    if teacher_logits.size(1) < 2 or model_logits.size(1) < 2:
        return 0.0

    t_logits = teacher_logits[:, :-1, :].contiguous() / max(temperature, eps)
    m_logits = model_logits[:, :-1, :].contiguous() / max(temperature, eps)

    min_vocab = min(t_logits.size(-1), m_logits.size(-1))
    t_logits = t_logits[..., :min_vocab]
    m_logits = m_logits[..., :min_vocab]

    t_log_probs = torch.log_softmax(t_logits, dim=-1)
    m_log_probs = torch.log_softmax(m_logits, dim=-1)
    t_probs = t_log_probs.exp()
    m_probs = m_log_probs.exp()
    avg_probs = 0.5 * (t_probs + m_probs)
    log_avg = (avg_probs + eps).log()
    js_div = 0.5 * F.kl_div(log_avg, t_probs, reduction='none').sum(dim=-1) + \
             0.5 * F.kl_div(log_avg, m_probs, reduction='none').sum(dim=-1)
    return float(js_div.mean().item())


@torch.no_grad()
def compute_rank_correlation(
    teacher_logits: torch.Tensor,
    model_logits: torch.Tensor,
) -> float:
    """Compute Spearman rank correlation between teacher and model probability rankings."""
    if teacher_logits.size(1) < 2 or model_logits.size(1) < 2:
        return 0.0
    
    t_logits = teacher_logits[:, :-1, :].contiguous()
    m_logits = model_logits[:, :-1, :].contiguous()
    
    # Align vocab sizes
    min_vocab = min(t_logits.size(-1), m_logits.size(-1))
    t_logits = t_logits[..., :min_vocab]
    m_logits = m_logits[..., :min_vocab]
    
    # Get ranks (argsort of argsort)
    t_ranks = t_logits.argsort(dim=-1).argsort(dim=-1).float()
    m_ranks = m_logits.argsort(dim=-1).argsort(dim=-1).float()
    
    # Compute correlation coefficient
    t_ranks_flat = t_ranks.view(-1, min_vocab)
    m_ranks_flat = m_ranks.view(-1, min_vocab)
    
    correlations = []
    for i in range(t_ranks_flat.size(0)):
        t_r = t_ranks_flat[i]
        m_r = m_ranks_flat[i]
        
        # Pearson correlation of ranks (Spearman)
        t_centered = t_r - t_r.mean()
        m_centered = m_r - m_r.mean()
        
        numerator = (t_centered * m_centered).sum()
        denominator = torch.sqrt((t_centered ** 2).sum() * (m_centered ** 2).sum())
        
        if denominator > 1e-8:
            corr = numerator / denominator
            correlations.append(corr.item())
    
    return float(sum(correlations) / len(correlations)) if correlations else 0.0


@torch.no_grad()
def mean_entropy(logits: torch.Tensor) -> float:
    """Mean entropy of next-token predictive distribution (temperature=1)."""
    if logits.size(1) < 2:
        return 0.0
    probs = F.softmax(logits[:, :-1, :].contiguous(), dim=-1)
    ent = (-probs.clamp_min(1e-12).log() * probs).sum(dim=-1)  # [batch, seq-1]
    return float(ent.mean().item())