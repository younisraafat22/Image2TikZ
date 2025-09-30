#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Checkpoint management utilities for knowledge distillation training.
"""

import os
import json
import glob
import torch
import datetime


def find_latest_checkpoint(output_dir):
    """Find the latest checkpoint in the output directory."""
    checkpoint_pattern = os.path.join(output_dir, "checkpoint-*")
    checkpoints = glob.glob(checkpoint_pattern)
    if not checkpoints:
        return None
    
    # Extract step numbers and find the latest
    step_numbers = []
    for checkpoint in checkpoints:
        try:
            step = int(checkpoint.split("-")[-1])
            step_numbers.append((step, checkpoint))
        except (ValueError, IndexError):
            continue
    
    if not step_numbers:
        return None
    
    # Return the checkpoint with the highest step number
    latest_checkpoint = max(step_numbers, key=lambda x: x[0])[1]
    return latest_checkpoint


def save_checkpoint(output_dir, model, tokenizer, optimizer, scheduler,
                   global_step, epoch, best_eval_loss, args=None):
    """Save a training checkpoint."""
    checkpoint_dir = os.path.join(output_dir, f"checkpoint-{global_step}")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Save model and tokenizer
    model.save_pretrained(checkpoint_dir)
    if hasattr(tokenizer, 'text'):
        tokenizer.text.save_pretrained(checkpoint_dir)
    else:
        tokenizer.save_pretrained(checkpoint_dir)
    
    # Save optimizer and scheduler state
    torch.save(optimizer.state_dict(), os.path.join(checkpoint_dir, "optimizer.pt"))
    torch.save(scheduler.state_dict(), os.path.join(checkpoint_dir, "scheduler.pt"))
    
    # Save training state (filter out non-serializable objects)
    if args is not None:
        # Filter out non-serializable objects like tokenizers, models, etc.
        serializable_args = {}
        for key, value in vars(args).items():
            try:
                # Test if value is JSON serializable
                json.dumps(value)
                serializable_args[key] = value
            except (TypeError, ValueError):
                # Skip non-serializable objects
                # Non-serializable arg skipped
                pass
    else:
        serializable_args = {}
    
    training_state = {
        "global_step": global_step,
        "epoch": epoch,
        "best_eval_loss": best_eval_loss,
        "args": serializable_args
    }
    
    with open(os.path.join(checkpoint_dir, "training_state.json"), 'w') as f:
        json.dump(training_state, f, indent=2)

    # (RNG state & adapter saving removed for simplicity)
    
    print(f"?? Checkpoint saved: {checkpoint_dir}")
    return checkpoint_dir


def save_best_model(output_dir, model, tokenizer, eval_loss, epoch, global_step):
    """Save the best model based on evaluation loss."""
    best_model_dir = os.path.join(output_dir, "best_model")
    os.makedirs(best_model_dir, exist_ok=True)
    
    # Save model and tokenizer
    model.save_pretrained(best_model_dir)
    if hasattr(tokenizer, 'text'):
        tokenizer.text.save_pretrained(best_model_dir)
    else:
        tokenizer.save_pretrained(best_model_dir)
    
    # Save best model info
    best_model_info = {
        "eval_loss": eval_loss,
        "epoch": epoch,
        "global_step": global_step,
        "timestamp": datetime.datetime.now().isoformat()
    }
    
    with open(os.path.join(best_model_dir, "best_model_info.json"), 'w') as f:
        json.dump(best_model_info, f, indent=2)
    
    print(f"?? Best model saved: eval_loss={eval_loss:.6f}, epoch={epoch+1}, step={global_step}")
    return best_model_dir 