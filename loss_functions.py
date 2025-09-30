#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SIMPLIFIED Loss functions for LOGITS-ONLY knowledge distillation.
Based on analysis showing 40x better compatibility in output space vs hidden states.
"""

import torch
import torch.nn.functional as F


def compute_ce_loss(student_logits, labels, ignore_index=-100):
    """
    Compute autoregressive Cross-Entropy loss with next-token shift.

    Args:
        student_logits: [batch, seq_len, vocab]
        labels: [batch, seq_len]
        ignore_index: Index to ignore in loss computation

    Returns:
        Cross-entropy loss over next-token prediction
    """
    # Require at least two tokens to form a next-token pair
    if student_logits.size(1) < 2 or labels.size(1) < 2:
        return F.cross_entropy(
            student_logits.view(-1, student_logits.size(-1)),
            labels.view(-1),
            ignore_index=ignore_index,
        )

    shifted_logits = student_logits[:, :-1, :].contiguous()
    shifted_labels = labels[:, 1:].contiguous()

    ce_loss = F.cross_entropy(
        shifted_logits.reshape(-1, shifted_logits.size(-1)),
        shifted_labels.reshape(-1),
        ignore_index=ignore_index,
    )
    return ce_loss


def compute_logits_distillation_loss(
    student_logits,
    teacher_logits,
    temperature=4.0,
    ignore_index=-100,
    labels=None,
    # Entropy gating & correctness weighting
    entropy_min=None,
    entropy_max=None,
    correctness_downweight=None,
    teacher_conf_threshold=0.75,
    return_stats=False,
):
    """
    Compute logits-based knowledge distillation loss using KL divergence.
    
    FIXED: Proper KL divergence calculation with MASKING to exclude padding tokens.
    Based on debug analysis showing 215x difference between masked and unmasked KL loss.
    
    Args:
        student_logits: Student model logits [batch_size, seq_len, vocab_size]
        teacher_logits: Teacher model logits [batch_size, seq_len, vocab_size]
        temperature: Temperature for softmax (higher = softer distributions)
        ignore_index: Index to ignore in loss computation 
        labels: Ground truth labels for masking padded positions (REQUIRED for correct loss)
    
    Returns:
        KL divergence loss between student and teacher logits (MASKED to valid positions only)
    """
    # Handle vocabulary size differences (teacher=32024, student=32008)
    min_vocab = min(student_logits.size(-1), teacher_logits.size(-1))
    student_logits_truncated = student_logits[..., :min_vocab]
    teacher_logits_truncated = teacher_logits[..., :min_vocab]

    # If labels provided, perform next-token shift to match CE objective
    if labels is not None and labels.size(1) >= 2:
        student_logits_truncated = student_logits_truncated[:, :-1, :].contiguous()
        teacher_logits_truncated = teacher_logits_truncated[:, :-1, :].contiguous()
        labels_for_mask = labels[:, 1:].contiguous()
    else:
        labels_for_mask = labels

    # Apply temperature scaling
    scaled_student = student_logits_truncated / temperature
    scaled_teacher = teacher_logits_truncated / temperature
    student_log_probs = F.log_softmax(scaled_student, dim=-1)
    teacher_probs = F.softmax(scaled_teacher, dim=-1)

    # Compute per-token teacher entropy (natural log base)
    with torch.no_grad():
        # Avoid log(0) via clamp
        tp_clamped = teacher_probs.clamp_min(1e-9)
        teacher_entropy = -(tp_clamped * tp_clamped.log()).sum(dim=-1)  # [batch, seq]

    if labels_for_mask is not None:
        # Valid positions mask [batch, seq-1]
        valid_pos_mask = (labels_for_mask != ignore_index)
        gating_mask = valid_pos_mask

        # Entropy gating: keep tokens whose entropy in (min,max)
        if entropy_min is not None or entropy_max is not None:
            proposed_mask = gating_mask
            if entropy_min is not None:
                proposed_mask = proposed_mask & (teacher_entropy >= entropy_min)
            if entropy_max is not None:
                proposed_mask = proposed_mask & (teacher_entropy <= entropy_max)
            # Fallback: if nothing kept OR ratio below floor, revert (stability)
            if proposed_mask.any():
                # tentative ratio check after applying
                tentative_ratio = proposed_mask.sum().float() / gating_mask.sum().float().clamp_min(1)
                if tentative_ratio >= 0.45:  # gating floor
                    gating_mask = proposed_mask

        if valid_pos_mask.any():
            # Compute per-token KL without corrupting distributions by masking inputs
            # kl_per_element: [batch, seq, vocab]
            kl_per_element = F.kl_div(
                student_log_probs, teacher_probs, reduction='none'
            )
            # Sum over vocab to get per-position KL: [batch, seq]
            kl_per_position = kl_per_element.sum(dim=-1)
            # Correctness down-weighting
            if correctness_downweight is not None and correctness_downweight < 1.0 and labels_for_mask is not None:
                with torch.no_grad():
                    teacher_argmax = teacher_probs.argmax(dim=-1)  # [batch, seq]
                    # Shift labels already aligned as labels_for_mask
                    correct_mask = (teacher_argmax == labels_for_mask)
                    if teacher_conf_threshold is not None and teacher_conf_threshold > 0:
                        # Teacher confidence = max prob
                        max_probs = teacher_probs.max(dim=-1).values
                        confident = max_probs >= teacher_conf_threshold
                        correct_mask = correct_mask & confident
                # Apply scaling to KL positions where teacher already confidently correct
                if correct_mask.any():
                    kl_per_position = torch.where(
                        correct_mask, kl_per_position * correctness_downweight, kl_per_position
                    )
            # Apply gating mask (already includes valid positions)
            kl_per_position = kl_per_position * gating_mask
            used_tokens = gating_mask.sum().clamp_min(1)
            kl_loss = kl_per_position.sum() / used_tokens
            gating_ratio = (gating_mask.sum().float() / valid_pos_mask.sum().float()).item()
        else:
            kl_loss = torch.tensor(0.0, device=student_logits.device, requires_grad=True)
            gating_ratio = 0.0
    else:
        # Fallback to unmasked batchmean
        kl_loss = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean')
        gating_ratio = 1.0

    # Apply temperature squared scaling (standard KD practice)
    kl_loss = kl_loss * (temperature ** 2)
    if return_stats:
        return kl_loss, {
            "gating_ratio": gating_ratio,
            "avg_teacher_entropy": float(teacher_entropy.mean().item()),
        }
    return kl_loss


def compute_distillation_loss(
    student_logits,
    teacher_logits,
    labels,
    temperature: float = 4.0,
    alpha: float = 0.003,
    entropy_min=None,
    entropy_max=None,
    correctness_downweight=None,
    teacher_conf_threshold=0.75,
):
    """
    LOGITS-ONLY knowledge distillation loss function (shifted CE + masked KL).
    
    Args:
        student_logits: Student model logits
        teacher_logits: Teacher model logits  
        labels: Ground truth labels
        temperature: Temperature for logits distillation (4.0 confirmed working)
        alpha: Balance between KD loss and CE loss (FIXED: 0.003 = ~0.3% KD, 99.7% CE)
        verbose: Whether to print debug info
        use_advanced_logits: IGNORED - simplified approach only uses basic logits
    
    Returns:
        Tuple of (total_loss, ce_loss, kd_loss)
    """
    # Cross-Entropy loss (supervised learning)
    ce_loss = compute_ce_loss(student_logits, labels)
    
    # Logits-based knowledge distillation (masked, shifted)
    kd_loss, kd_stats = compute_logits_distillation_loss(
        student_logits,
        teacher_logits,
        temperature,
        labels=labels,
        entropy_min=entropy_min,
        entropy_max=entropy_max,
        correctness_downweight=correctness_downweight,
        teacher_conf_threshold=teacher_conf_threshold,
        return_stats=True,
    )
    
    # KD dominance clamp: soften KD if overwhelming CE early
    with torch.no_grad():
        if kd_loss > 2.0 * ce_loss:
            kd_loss = kd_loss * 0.85

    # Combined loss
    total_loss = alpha * kd_loss + (1.0 - alpha) * ce_loss
    
    return (total_loss, ce_loss, kd_loss, kd_stats)