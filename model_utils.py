#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Model utilities for standard DeTikZify knowledge distillation.
"""

import torch
import sys
import os

# Add the parent directory to the path to import detikzify
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

try:
    from detikzify.model import load as load_detikzify_model
    from detikzify.model import register as register_detikzify
    register_detikzify()
except ImportError:
    print("⚠️ Warning: Could not import detikzify. Make sure you're in the correct directory.")
    load_detikzify_model = None


def load_teacher_model(model_name: str, device: str = "cuda", torch_dtype: torch.dtype = torch.bfloat16):
    """
    Load teacher model for distillation.
    
    Args:
        model_name: Name of the teacher model
        device: Device to load the model on
        torch_dtype: Data type for the model
        
    Returns:
        Tuple of (model, tokenizer, config)
    """
    print(f"Loading teacher model: {model_name}")
    
    try:
        # Load model and tokenizer
        model, tokenizer = load_detikzify_model(model_name)
        # Check if model has proper config
        if hasattr(model, 'config'):
            config = model.config
        else:
            config = None
        # Move model to device and set dtype
        if device is not None:
            model = model.to(device)
        if torch_dtype is not None:
            model = model.to(dtype=torch_dtype)
        # Do not override teacher pad_token_id. We rely on explicit attention_mask built from the student pad token.
        # This avoids mutating teacher config/tokenizer unnecessarily.
        print("Teacher model loaded")
        return model, tokenizer, config
    except Exception as e:
        print(f"Error loading teacher model: {e}")
        return None, None, None
        


def load_student_model(model_name: str, device: str = "cuda", torch_dtype: torch.dtype = torch.bfloat16):
    """
    Load student model for distillation.
    
    Args:
        model_name: Name of the student model
        device: Device to load the model on
        torch_dtype: Data type for the model
        
    Returns:
        Tuple of (model, tokenizer, config)
    """
    print(f"Loading student model: {model_name}")
    
    try:
        # Load model and tokenizer (without device and torch_dtype parameters)
        model, tokenizer = load_detikzify_model(model_name)
        
        # Move model to device and set dtype after loading
        if device is not None:
            model = model.to(device)
        if torch_dtype is not None:
            model = model.to(dtype=torch_dtype)
        
        # Get configuration
        config = model.config
        
        # Set student to training mode
        model.train()
        print("Student model loaded")
        return model, tokenizer, config
        
    except Exception as e:
        print(f"Failed to load student model: {e}")
        raise




def load_distilled_model(model_path: str, device: str = "cuda", torch_dtype: torch.dtype = torch.bfloat16):
    """
    Load a distilled model for inference.
    
    Args:
        model_path: Path to the distilled model
        device: Device to load the model on
        torch_dtype: Data type for the model
        
    Returns:
        Tuple of (model, tokenizer)
    """
    print(f"Loading distilled model from: {model_path}")
    
    try:
        # Load model and tokenizer (without device and torch_dtype parameters)
        model, tokenizer = load_detikzify_model(model_path)
        
        # Move model to device and set dtype after loading
        if device is not None:
            model = model.to(device)
        if torch_dtype is not None:
            # Convert string dtype to torch dtype if needed
            if isinstance(torch_dtype, str):
                if torch_dtype == "bfloat16":
                    torch_dtype = torch.bfloat16
                elif torch_dtype == "float16":
                    torch_dtype = torch.float16
                elif torch_dtype == "float32":
                    torch_dtype = torch.float32
                else:
                    torch_dtype = torch.bfloat16  # default fallback
            model = model.to(dtype=torch_dtype)
        
        # Set to evaluation mode
        model.eval()
        print("Distilled model loaded")
        return model, tokenizer
        
    except Exception as e:
        print(f"Failed to load distilled model: {e}")
        raise 