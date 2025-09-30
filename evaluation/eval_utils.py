#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Model loading and utility functions for evaluation.

This module handles model loading, device management, and data preparation
for knowledge distillation evaluation.
"""

import os
import sys
import torch

# Add parent directory to path for imports
parent_dir = os.path.join(os.path.dirname(__file__), '..')
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from model_utils import load_teacher_model, load_student_model
from data_utils import (
    KnowledgeDistillationDataset,
    KnowledgeDistillationCollator,
    create_detikzify_dataset_with_split,
)


def _bf16_supported(device: str) -> bool:
    """Check if bfloat16 is supported on the given device."""
    try:
        if device.startswith('cuda') and torch.cuda.is_available():
            return torch.cuda.is_bf16_supported()
        return False
    except Exception:
        return False


def load_model_any(path_or_name: str, device: str, use_bf16: bool = True):
    """Load a model/tokenizer via existing helpers, supporting local checkpoints and hub ids."""
    if os.path.isdir(path_or_name):
        model, tokenizer, config = load_student_model(path_or_name, device=device)
    else:
        if 'cl-7b' in path_or_name or 'teacher' in path_or_name:
            model, tokenizer, config = load_teacher_model(path_or_name, device=device)
        else:
            model, tokenizer, config = load_student_model(path_or_name, device=device)
    
    model = model.to(device)
    if use_bf16 and _bf16_supported(device):
        model = model.to(dtype=torch.bfloat16)
    
    model.eval()
    if hasattr(model, 'config') and hasattr(model.config, 'use_cache'):
        model.config.use_cache = True
    
    return model, tokenizer, config


def build_dataloader(
    dataset_name: str, 
    split: str, 
    tokenizer, 
    model_config, 
    batch_size: int, 
    max_samples: int, 
    num_workers: int = 4
):
    """Build a dataloader for evaluation."""
    raw = create_detikzify_dataset_with_split(dataset_name, split, max_samples, 512)
    dataset = KnowledgeDistillationDataset(
        dataset=raw,
        training_tokenizer=tokenizer,
        model_config=model_config,
        dtype=torch.bfloat16 if _bf16_supported('cuda') else torch.float32,
    )
    collator_tok = tokenizer.text if hasattr(tokenizer, 'text') else tokenizer
    collator = KnowledgeDistillationCollator(collator_tok)
    
    # On Windows / constrained envs avoid multi-workers by default
    if os.name == 'nt' and num_workers > 0:
        num_workers = min(num_workers, 0)
    
    loader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        collate_fn=collator, 
        num_workers=num_workers, 
        pin_memory=True
    )
    return loader


def prepare_images_for_models(images, teacher_model, student_model, device):
    """Prepare image tensors with appropriate dtypes for different models."""
    images_for_teacher = None
    images_for_model = None
    
    if images is not None:
        try:
            teacher_dtype = next(teacher_model.parameters()).dtype
        except Exception:
            teacher_dtype = None
        try:
            model_dtype = next(student_model.parameters()).dtype
        except Exception:
            model_dtype = None
            
        if teacher_dtype is not None:
            images_for_teacher = images.to(device).to(teacher_dtype)
        if model_dtype is not None:
            images_for_model = images.to(device).to(model_dtype)
            
        # Fallbacks
        if images_for_teacher is None:
            images_for_teacher = images.to(device)
        if images_for_model is None:
            images_for_model = images.to(device)
    
    return images_for_teacher, images_for_model