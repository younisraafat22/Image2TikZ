#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Training utilities for knowledge distillation.
"""

import torch
from loss_functions import compute_distillation_loss


def _compute_dynamic_kd_params(args, optimizer_step):
    """Derive dynamic alpha, temperature, and gating thresholds based on progress.
    Updated schedule (stabilize CE, prevent early KD dominance):
      * Temperature: hold high longer (0-25%), then gentle anneal.
      * Alpha: start decaying earlier (from 10%), deeper floor (40% of base).
      * Entropy gating: OFF until 40% progress to avoid premature sparsity; then gradually narrow band.
    Returns (alpha, temp, entropy_min, entropy_max, correctness_downweight).
    """
    total_opt_steps = getattr(args, 'total_optimizer_steps', None)
    if not total_opt_steps or total_opt_steps <= 0:
        return args.alpha, args.temperature, None, None, None

    progress = optimizer_step / total_opt_steps
    base_temp = args.temperature

    # Temperature schedule (prolonged soft guidance)
    if progress < 0.25:
        temp = base_temp
    elif progress < 0.50:
        temp = base_temp * 0.85
    elif progress < 0.75:
        temp = base_temp * 0.70
    else:
        temp = max(2.0, base_temp * 0.60)

    base_alpha = args.alpha
    if progress < 0.10:
        alpha = base_alpha
    elif progress < 0.40:
        frac = (progress - 0.10) / 0.30  # to ~0.55*base by 40%
        alpha = base_alpha * (1.0 - 0.45 * frac)
    elif progress < 0.70:
        frac = (progress - 0.40) / 0.30  # to 0.40*base by 70%
        alpha = base_alpha * (0.55 - 0.15 * frac)
    else:
        alpha = base_alpha * 0.40

    # Entropy gating (deferred until 40% progress)
    entropy_min = entropy_max = None
    if progress >= 0.40:
        if progress < 0.55:
            entropy_min, entropy_max = 6.0, 10.6
        elif progress < 0.70:
            entropy_min, entropy_max = 6.2, 10.2
        elif progress < 0.85:
            entropy_min, entropy_max = 6.4, 9.9
        else:
            entropy_min, entropy_max = 6.6, 9.6

    correctness_downweight = 0.30
    return alpha, temp, entropy_min, entropy_max, correctness_downweight


def train_step(student_model, teacher_model, batch, optimizer, args):
    """Single training step for LOGITS-ONLY distillation."""
    student_model.train()
    teacher_model.eval()

    # Optional split across two GPUs: student on one, teacher on another
    use_dual_gpu = getattr(args, 'use_dual_gpu', False)
    teacher_device = getattr(args, 'teacher_device', 'cuda:0') if use_dual_gpu else args.device
    student_device = getattr(args, 'student_device', args.device)
    device_for_batch = student_device
    
    # Move batch to appropriate device
    input_ids = batch["input_ids"].to(device_for_batch)
    labels = batch["labels"].to(device_for_batch) if "labels" in batch else None
    
    # CRITICAL FIX: Create proper attention mask for pad tokens
    # Use STUDENT tokenizer's pad token since we're training the student model
    if hasattr(args, 'student_tokenizer'):
        if hasattr(args.student_tokenizer, 'pad_token_id'):
            pad_token_id = args.student_tokenizer.pad_token_id
        elif hasattr(args.student_tokenizer, 'text') and hasattr(args.student_tokenizer.text, 'pad_token_id'):
            pad_token_id = args.student_tokenizer.text.pad_token_id
        else:
            pad_token_id = 32000  # DetikzifyTokenizer student default
    elif hasattr(args, 'teacher_tokenizer'):
        # Fallback to teacher if student not available
        if hasattr(args.teacher_tokenizer, 'pad_token_id'):
            pad_token_id = args.teacher_tokenizer.pad_token_id
        elif hasattr(args.teacher_tokenizer, 'text') and hasattr(args.teacher_tokenizer.text, 'pad_token_id'):
            pad_token_id = args.teacher_tokenizer.text.pad_token_id
        else:
            pad_token_id = 32016  # DetikzifyTokenizer teacher default
    else:
        pad_token_id = 32000  # Default to student pad token
    
    # Create attention mask: 1 for non-padding tokens, 0 for padding tokens
    attention_mask = (input_ids != pad_token_id).long().to(device_for_batch)
    
    # Handle images if available
    images = None
    if "images" in batch:
        images = batch["images"].to(device_for_batch)
        images = images.to(torch.bfloat16)
    
    # Prepare inputs for each device
    if use_dual_gpu:
        teacher_input_ids = input_ids.to(teacher_device)
        teacher_attention_mask = attention_mask.to(teacher_device)
        teacher_images = images.to(teacher_device) if images is not None else None

        student_input_ids = input_ids  # already on student_device
        student_attention_mask = attention_mask
        student_images = images
    else:
        teacher_input_ids = input_ids
        teacher_attention_mask = attention_mask
        teacher_images = images
        student_input_ids = input_ids
        student_attention_mask = attention_mask
        student_images = images
    
    # Forward pass through student (logits only - no hidden states needed)
    student_outputs = student_model(
        input_ids=student_input_ids,
        attention_mask=student_attention_mask,
        images=student_images,
        output_hidden_states=False,  # LOGITS-ONLY: No hidden states needed
        output_attentions=False      # No attention distillation
    )
    
    # Forward pass through teacher (logits only - no hidden states needed)  
    with torch.no_grad():
        teacher_outputs = teacher_model(
            input_ids=teacher_input_ids,
            attention_mask=teacher_attention_mask,
            images=teacher_images,
            output_hidden_states=False,  # LOGITS-ONLY: No hidden states needed
            output_attentions=False      # No attention distillation
        )
    
    # LOGITS-ONLY approach: Only extract logits, ignore hidden states
    
    teacher_logits = teacher_outputs.logits
    if use_dual_gpu:
        teacher_logits = teacher_logits.to(student_device)
    
    # Compute LOGITS-ONLY distillation loss (based on 0.82+ similarity analysis)
    # Dynamic KD parameters
    dyn_alpha, dyn_temp, e_min, e_max, corr_dw = _compute_dynamic_kd_params(args, getattr(args, 'current_optimizer_step', 0))
    loss, ce_loss, kd_loss, kd_stats = compute_distillation_loss(
        student_logits=student_outputs.logits,
        teacher_logits=teacher_logits,
        labels=labels,
        temperature=dyn_temp,
        alpha=dyn_alpha,
        entropy_min=e_min,
        entropy_max=e_max,
        correctness_downweight=corr_dw,
        teacher_conf_threshold=0.75,
    )
    
    # MEMORY CLEANUP after loss computation
    del student_outputs, teacher_outputs
    
    # Backward pass
    loss.backward()
    
    # Gradient clipping for stability
    if hasattr(args, 'max_grad_norm') and args.max_grad_norm > 0:
        torch.nn.utils.clip_grad_norm_(student_model.parameters(), args.max_grad_norm)
    
    return {
        "loss": loss.item(),
        "ce_loss": ce_loss.item(),
        "kd_loss": kd_loss.item(),
        "kd_gating_ratio": kd_stats.get("gating_ratio", 1.0),
        "kd_teacher_entropy": kd_stats.get("avg_teacher_entropy", 0.0),
        "dyn_alpha": float(dyn_alpha),
        "dyn_temp": float(dyn_temp),
    }


def evaluate_step(student_model, teacher_model, batch, args):
    """Single evaluation step for LOGITS-ONLY distillation."""
    student_model.eval()
    teacher_model.eval()
    
    with torch.no_grad():
        use_dual_gpu = getattr(args, 'use_dual_gpu', False)
        teacher_device = getattr(args, 'teacher_device', 'cuda:0') if use_dual_gpu else args.device
        student_device = getattr(args, 'student_device', args.device)
        device_for_batch = student_device
        
        # Move batch to appropriate device
        input_ids = batch["input_ids"].to(device_for_batch)
        labels = batch["labels"].to(device_for_batch) if "labels" in batch else None
        
        # CRITICAL FIX: Create proper attention mask for pad tokens
        # Use STUDENT tokenizer's pad token since we're training the student model
        if hasattr(args, 'student_tokenizer'):
            if hasattr(args.student_tokenizer, 'pad_token_id'):
                pad_token_id = args.student_tokenizer.pad_token_id
            elif hasattr(args.student_tokenizer, 'text') and hasattr(args.student_tokenizer.text, 'pad_token_id'):
                pad_token_id = args.student_tokenizer.text.pad_token_id
            else:
                pad_token_id = 32000  # DetikzifyTokenizer student default
        elif hasattr(args, 'teacher_tokenizer'):
            # Fallback to teacher if student not available
            if hasattr(args.teacher_tokenizer, 'pad_token_id'):
                pad_token_id = args.teacher_tokenizer.pad_token_id
            elif hasattr(args.teacher_tokenizer, 'text') and hasattr(args.teacher_tokenizer.text, 'pad_token_id'):
                pad_token_id = args.teacher_tokenizer.text.pad_token_id
            else:
                pad_token_id = 32016  # DetikzifyTokenizer teacher default
        else:
            pad_token_id = 32000  # Default to student pad token
        
        # Create attention mask: 1 for non-padding tokens, 0 for padding tokens
        attention_mask = (input_ids != pad_token_id).long().to(device_for_batch)
        
        # Handle images if available
        images = None
        if "images" in batch:
            images = batch["images"].to(device_for_batch)
            images = images.to(torch.bfloat16)
        
        if use_dual_gpu:
            teacher_input_ids = input_ids.to(teacher_device)
            teacher_attention_mask = attention_mask.to(teacher_device)
            teacher_images = images.to(teacher_device) if images is not None else None
            student_input_ids = input_ids
            student_attention_mask = attention_mask
            student_images = images
        else:
            teacher_input_ids = input_ids
            teacher_attention_mask = attention_mask
            teacher_images = images
            student_input_ids = input_ids
            student_attention_mask = attention_mask
            student_images = images
        
        # Forward pass through student (only hidden states, no attention)
        student_outputs = student_model(
            input_ids=student_input_ids,
            attention_mask=student_attention_mask,
            images=student_images,
            output_hidden_states=False,  # LOGITS-ONLY: No hidden states needed
            output_attentions=False      # No attention distillation
        )
        
        # Forward pass through teacher (logits only)
        teacher_outputs = teacher_model(
            input_ids=teacher_input_ids,
            attention_mask=teacher_attention_mask,
            images=teacher_images,
            output_hidden_states=False,  # LOGITS-ONLY: No hidden states needed
            output_attentions=False      # No attention distillation
        )
        
        # LOGITS-ONLY approach: Only use logits
        
        teacher_logits = teacher_outputs.logits
        if use_dual_gpu:
            teacher_logits = teacher_logits.to(student_device)
        
        # Compute LOGITS-ONLY distillation loss
        dyn_alpha, dyn_temp, e_min, e_max, corr_dw = _compute_dynamic_kd_params(args, getattr(args, 'current_optimizer_step', 0))
        loss, ce_loss, kd_loss, kd_stats = compute_distillation_loss(
            student_logits=student_outputs.logits,
            teacher_logits=teacher_logits,
            labels=labels,
            temperature=dyn_temp,
            alpha=dyn_alpha,
            entropy_min=e_min,
            entropy_max=e_max,
            correctness_downweight=corr_dw,
            teacher_conf_threshold=0.75,
        )
        
        return {
            "eval_loss": loss.item(),
            "eval_ce_loss": ce_loss.item(),
            "eval_kd_loss": kd_loss.item(),
            "eval_dyn_alpha": float(dyn_alpha),
            "eval_dyn_temp": float(dyn_temp),
            "eval_kd_gating_ratio": kd_stats.get("gating_ratio", 1.0),
            "eval_kd_teacher_entropy": kd_stats.get("avg_teacher_entropy", 0.0),
        } 