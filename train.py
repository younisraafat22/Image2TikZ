#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SIMPLIFIED LOGITS Knowledge Distillation Training Script
Clean, focused implementation without progressive distillation or curriculum learning.
"""

import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
import os
import sys
from tqdm import tqdm

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

# Essential imports only
from model_utils import load_teacher_model, load_student_model
from checkpoint_utils import save_checkpoint, save_best_model, find_latest_checkpoint
from data_utils import (
    KnowledgeDistillationDataset, KnowledgeDistillationCollator, 
    create_detikzify_dataset_with_split
)
from training_utils import train_step, evaluate_step


def parse_args():
    """Parse essential command line arguments."""
    parser = argparse.ArgumentParser(description="Simplified LOGITS Knowledge Distillation")
    
    # Essential arguments
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--teacher_model", type=str, default="nllg/detikzify-cl-7b", help="Teacher model")
    parser.add_argument("--student_model", type=str, default="nllg/detikzify-tl-1.1b", help="Student model")
    parser.add_argument("--dataset", type=str, default="nllg/datikz-v2", help="Dataset")
    
    # Training parameters
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16, help="Gradient accumulation")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Warmup ratio for LR scheduler")
    
    # Distillation parameters
    parser.add_argument("--temperature", type=float, default=4.0, help="Temperature for logits distillation")
    parser.add_argument("--alpha", type=float, default=0.7, help="KD vs CE loss balance (0.7 = 70% KD)")
    
    # Optional parameters
    parser.add_argument("--max_samples", type=int, default=None, help="Max samples (None = all)")
    parser.add_argument("--logging_steps", type=int, default=25, help="Logging frequency")
    parser.add_argument("--save_steps", type=int, default=500, help="Save frequency")
    parser.add_argument("--eval_steps", type=int, default=500, help="Evaluation frequency")
    parser.add_argument("--eval_batches", type=int, default=100, help="Number of batches to use for each evaluation")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    # Optional multi-GPU (teacher/student split)
    parser.add_argument("--use_dual_gpu", action="store_true", help="Run teacher on one GPU and student on another")
    parser.add_argument("--teacher_device", type=str, default="cuda:0", help="Device for teacher when using dual GPU")
    parser.add_argument("--student_device", type=str, default="cuda:1", help="Device for student when using dual GPU")
    
    # Resume options
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--resume_if_possible", action="store_true", help="Resume from latest checkpoint in output_dir if available")
    
    return parser.parse_args()


def setup_training_data(args, student_tokenizer, student_config):
    """Setup training and test datasets (no shuffling for deterministic reproducibility)."""
    print("?? Setting up datasets (deterministic order, no shuffle)...")
    
    # Load training data
    train_raw_dataset = create_detikzify_dataset_with_split(
        args.dataset, "train", args.max_samples, 512
    )
    
    # Load test data (smaller sample)
    test_max_samples = min(1000, args.max_samples) if args.max_samples else 1000
    test_raw_dataset = create_detikzify_dataset_with_split(
        args.dataset, "test", test_max_samples, 512
    )
    
    print(f"   Training samples: {len(train_raw_dataset):,}")
    print(f"   Test samples: {len(test_raw_dataset):,}")
    
    # Create datasets
    train_dataset = KnowledgeDistillationDataset(
        dataset=train_raw_dataset,
        training_tokenizer=student_tokenizer,
        model_config=student_config,
        dtype=torch.bfloat16
    )
    
    test_dataset = KnowledgeDistillationDataset(
        dataset=test_raw_dataset,
        training_tokenizer=student_tokenizer,
        model_config=student_config,
        dtype=torch.bfloat16
    )
    
    # Create collator
    tokenizer_for_collator = student_tokenizer.text if hasattr(student_tokenizer, 'text') else student_tokenizer
    collator = KnowledgeDistillationCollator(tokenizer_for_collator)
    
    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=collator, num_workers=4, pin_memory=True
    )
    
    test_dataloader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=collator, num_workers=4, pin_memory=True
    )
    
    print(f"   ?? Simple KD setup complete (deterministic order)")
    
    return train_dataloader, test_dataloader


def main():
    """Main training function - simplified."""
    args = parse_args()
    
    # Basic setup
    torch.manual_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("?? SIMPLIFIED LOGITS Knowledge Distillation Training")
    print(f"   Teacher: {args.teacher_model}")
    print(f"   Student: {args.student_model}")
    print(f"   ?? Using proven 0.82+ logits compatibility")
    print(f"   ?? Loss balance: {args.alpha:.1f}*KD + {1.0-args.alpha:.1f}*CE")

    # Load models
    print("\n?? Loading models...")
    teacher_model, teacher_tokenizer, teacher_config = load_teacher_model(
        args.teacher_model, device=args.device
    )
    student_model, student_tokenizer, student_config = load_student_model(
        args.student_model, device=args.device
    )

    # Place models on correct devices for dual-GPU setup
    if getattr(args, 'use_dual_gpu', False):
        teacher_model = teacher_model.to(args.teacher_device)
        student_model = student_model.to(args.student_device)
    else:
        teacher_model = teacher_model.to(args.device)
        student_model = student_model.to(args.device)

    # Use bfloat16 for speed/memory and disable cache
    teacher_model = teacher_model.to(dtype=torch.bfloat16)
    student_model = student_model.to(dtype=torch.bfloat16)
    for m in (teacher_model, student_model):
        if hasattr(m, 'config') and hasattr(m.config, 'use_cache'):
            m.config.use_cache = False
    
    # Store tokenizers for training_utils
    args.teacher_tokenizer = teacher_tokenizer
    args.student_tokenizer = student_tokenizer
    
    # Enable gradient checkpointing and disable cache (memory reduction)
    if hasattr(teacher_model, 'gradient_checkpointing_enable'):
        teacher_model.gradient_checkpointing_enable()
    if hasattr(student_model, 'gradient_checkpointing_enable'):
        student_model.gradient_checkpointing_enable()
    
    # Freeze vision encoders
    if hasattr(teacher_model, 'vision_tower'):
        for param in teacher_model.vision_tower.parameters():
            param.requires_grad = False
    if hasattr(student_model, 'vision_tower'):
        for param in student_model.vision_tower.parameters():
            param.requires_grad = False
    
    print("Models loaded")
    # Ensure teacher runs without dropout for stable KD targets
    try:
        teacher_model.eval()
    except Exception:
        pass
    
    # Initialize training state defaults BEFORE potential resume
    start_epoch = 0
    global_step = 0
    optimizer_step = 0
    best_eval_loss = float('inf')

    # We'll build data AFTER potential resume so that skipping aligns with current dataset order
    pending_resume_optimizer_state = None
    resume_path_loaded = None
    if args.resume_from_checkpoint or args.resume_if_possible:
        resume_path = args.resume_from_checkpoint or find_latest_checkpoint(args.output_dir)
        if resume_path and os.path.isdir(resume_path):
            print(f"?? Resuming from checkpoint: {resume_path}")
            resume_path_loaded = resume_path
            # Load student model weights fresh to avoid residual grads / fragmentation
            try:
                # Use class method to load to ensure clean instance
                ModelClass = type(student_model)
                del student_model
                torch.cuda.empty_cache()
                student_model = ModelClass.from_pretrained(resume_path)
                # Check BatchNorm stats after loading
                if getattr(args, 'use_dual_gpu', False):
                    student_model = student_model.to(args.student_device)
                else:
                    student_model = student_model.to(args.device)
                student_model = student_model.to(dtype=torch.bfloat16)
                if hasattr(student_model, 'gradient_checkpointing_enable'):
                    student_model.gradient_checkpointing_enable()
                if hasattr(student_model, 'config') and hasattr(student_model.config, 'use_cache'):
                    student_model.config.use_cache = False
                print("? Student model reloaded for resume")
            except Exception as e:
                print(f"?? Could not reload student model: {e}. Continuing with existing instance.")

            # Capture optimizer / scheduler states to apply after recreation
            try:
                opt_state = torch.load(os.path.join(resume_path, "optimizer.pt"), map_location='cpu')
                sched_state = torch.load(os.path.join(resume_path, "scheduler.pt"), map_location='cpu')
                pending_resume_optimizer_state = (opt_state, sched_state)
            except Exception as e:
                print(f"?? Could not load optimizer/scheduler state: {e}")

            # Load training state
            try:
                import json
                with open(os.path.join(resume_path, "training_state.json"), "r") as f:
                    training_state = json.load(f)
                global_step = int(training_state.get("global_step", 0))
                start_epoch = int(training_state.get("epoch", 0))
                best_eval_loss = float(training_state.get("best_eval_loss", float('inf')))
                optimizer_step = max(0, global_step // max(1, args.gradient_accumulation_steps))
                # Hyperparameter consistency check
                saved_args = training_state.get("args", {})
                critical = [
                    ("batch_size", args.batch_size, saved_args.get("batch_size")),
                    ("gradient_accumulation_steps", args.gradient_accumulation_steps, saved_args.get("gradient_accumulation_steps")),
                    ("temperature", args.temperature, saved_args.get("temperature")),
                    ("alpha", args.alpha, saved_args.get("alpha")),
                ]
                mismatches = []
                for name, current, saved in critical:
                    if saved is not None and current != saved:
                        mismatches.append((name, current, saved))
                if mismatches:
                    print(" Critical hyperparameter mismatches detected - aborting resume:")
                    for name, current, saved in mismatches:
                        print(f"   {name}: current={current} saved={saved}")
                    print(" Resume halted. Restart with matching arguments or start a fresh output_dir.")
                    return
                print(f"Resume state: epoch={start_epoch}, global_step={global_step}, optimizer_step={optimizer_step}, best_eval={best_eval_loss:.4f}")

            except Exception as e:
                print(f"⚠️ Could not load training_state.json: {e}")

    # Setup data AFTER resume so we know len(train_dataloader)
    train_dataloader, test_dataloader = setup_training_data(args, student_tokenizer, student_config)

    # Setup optimizer and scheduler (must be after potential model reload)
    optimizer = optim.AdamW(student_model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    total_steps = (len(train_dataloader) * args.num_train_epochs) // args.gradient_accumulation_steps
    warmup_steps = int(total_steps * args.warmup_ratio)
    # Expose total optimizer steps to training utils for scheduling
    args.total_optimizer_steps = total_steps
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    if pending_resume_optimizer_state:
        opt_state, sched_state = pending_resume_optimizer_state
        try:
            optimizer.load_state_dict(opt_state)
            scheduler.load_state_dict(sched_state)
            print("✅ Optimizer & scheduler states restored")
        except Exception as e:
            print(f"⚠️ Failed to restore optimizer/scheduler: {e}")
    # Derive per-optimizer-step intervals for logging/eval/save
    log_every_opt_steps = max(1, args.logging_steps // max(1, args.gradient_accumulation_steps))
    eval_every_opt_steps = max(1, args.eval_steps // max(1, args.gradient_accumulation_steps))
    save_every_opt_steps = max(1, args.save_steps // max(1, args.gradient_accumulation_steps))
    print(f"Training setup: {total_steps} steps, {warmup_steps} warmup steps")
    
    # (resume diagnostics removed for brevity)

    # Training loop
    print(f"\nStarting training...")
    print(f"Fixed parameters: a={args.alpha}, temp={args.temperature}")
    
    # Track last processed optimizer step for side-effects
    last_logged_optimizer_step = -1
    last_eval_optimizer_step = -1
    last_saved_optimizer_step = -1
    
    for epoch in range(start_epoch, args.num_train_epochs):
        print(f"\nEpoch {epoch + 1}/{args.num_train_epochs}")
        
        student_model.train()
        epoch_losses = []

        # If resuming mid-epoch, skip already-processed micro-steps within this epoch
        # The saved global_step represents the last COMPLETED micro-step, so start from the next one
        if epoch == start_epoch and global_step > 0:
            steps_to_skip = (global_step + 1) % len(train_dataloader)
        else:
            steps_to_skip = 0

        if steps_to_skip > 0:
            print(f"Skipping {steps_to_skip} batches (resume global_step={global_step})")
        
        dataloader_iter = iter(train_dataloader)
        # Pre-skip consumed batches deterministically
        for _ in range(steps_to_skip):
            try:
                next(dataloader_iter)
            except StopIteration:
                break
        
        progress_bar = tqdm(
            range(steps_to_skip, len(train_dataloader)),
            desc=f"Training Epoch {epoch + 1}",
            total=len(train_dataloader),
            initial=steps_to_skip,
        )

        for step in progress_bar:
            try:
                batch = next(dataloader_iter)
            except StopIteration:
                break

            # Training step
            step_losses = train_step(student_model, teacher_model, batch, optimizer, args)
            epoch_losses.append(step_losses["loss"])
            
            # Optimizer step (with gradient accumulation)
            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(student_model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                optimizer_step += 1
                # Make current optimizer step visible to training_utils dynamic scheduler
                args.current_optimizer_step = optimizer_step
                
                # Memory cleanup
                torch.cuda.empty_cache()
                
                # Logging (once per optimizer step)
                if optimizer_step % log_every_opt_steps == 0 and last_logged_optimizer_step != optimizer_step:
                    ce_loss = step_losses['ce_loss']
                    kd_loss = step_losses['kd_loss']
                    dyn_alpha = step_losses.get('dyn_alpha', args.alpha)
                    dyn_temp = step_losses.get('dyn_temp', args.temperature)
                    gating_ratio = step_losses.get('kd_gating_ratio', 1.0)
                    teacher_entropy = step_losses.get('kd_teacher_entropy', 0.0)
                    total_loss = step_losses['loss']
                    if torch.cuda.is_available():
                        try:
                            allocated = torch.cuda.memory_allocated() / 1024**3
                            reserved = torch.cuda.memory_reserved() / 1024**3
                            print(f"GPU Mem (alloc/res): {allocated:.2f}G/{reserved:.2f}G")
                        except Exception:
                            pass
                    print(
                        f"OptStep {optimizer_step:6d} | Micro {global_step:6d} | "
                        f"Loss: {total_loss:.4f} | CE: {ce_loss:.4f} | KD: {kd_loss:.4f} | "
                        f"a:{dyn_alpha:.3f} T:{dyn_temp:.2f} gate:{gating_ratio:.2f} Ht:{teacher_entropy:.2f} | "
                        f"LR: {scheduler.get_last_lr()[0]:.2e}"
                    )
                    last_logged_optimizer_step = optimizer_step

                # Evaluation (once per optimizer step)
                if optimizer_step % eval_every_opt_steps == 0 and last_eval_optimizer_step != optimizer_step:
                    student_model.eval()
                    eval_losses = []
                    eval_ce_losses = []
                    eval_kd_losses = []
                    with torch.no_grad():
                        batches_used = 0
                        for eval_batch in test_dataloader:
                            eval_step_losses = evaluate_step(student_model, teacher_model, eval_batch, args)
                            eval_losses.append(eval_step_losses["eval_loss"])
                            eval_ce_losses.append(eval_step_losses["eval_ce_loss"])
                            eval_kd_losses.append(eval_step_losses["eval_kd_loss"])
                            batches_used += 1
                            if batches_used >= max(1, getattr(args, 'eval_batches', 100)):
                                break
                    avg_eval_loss = sum(eval_losses) / len(eval_losses)
                    avg_eval_ce = sum(eval_ce_losses) / len(eval_ce_losses)
                    avg_eval_kd = sum(eval_kd_losses) / len(eval_kd_losses)
                    # dynamic params at eval time
                    dyn_alpha = step_losses.get('dyn_alpha', args.alpha)
                    dyn_temp = step_losses.get('dyn_temp', args.temperature)
                    print(f"Eval Loss: {avg_eval_loss:.4f} | CE: {avg_eval_ce:.4f} | KD: {avg_eval_kd:.4f} | a:{dyn_alpha:.3f} T:{dyn_temp:.2f}")
                    if avg_eval_loss < best_eval_loss:
                        best_eval_loss = avg_eval_loss
                        print(f"New best model: {best_eval_loss:.4f}")
                        save_best_model(
                            args.output_dir, student_model, student_tokenizer, avg_eval_loss, epoch, global_step
                        )
                    student_model.train()
                    last_eval_optimizer_step = optimizer_step

                # Save checkpoint (once per optimizer step)
                if optimizer_step % save_every_opt_steps == 0 and last_saved_optimizer_step != optimizer_step:
                    save_checkpoint(
                        args.output_dir,
                        student_model,
                        student_tokenizer,
                        optimizer,
                        scheduler,
                        global_step,
                        epoch,
                        best_eval_loss,
                        args,
                    )
                    last_saved_optimizer_step = optimizer_step
            
            global_step += 1
        
        # End of epoch summary
        avg_epoch_loss = sum(epoch_losses) / len(epoch_losses)
    print(f"Epoch {epoch + 1} avg loss: {avg_epoch_loss:.4f}")
    
    # Save final model
    final_model_dir = os.path.join(args.output_dir, "final_model")
    os.makedirs(final_model_dir, exist_ok=True)
    student_model.save_pretrained(final_model_dir)
    if hasattr(student_tokenizer, 'text'):
        student_tokenizer.text.save_pretrained(final_model_dir)
    else:
        student_tokenizer.save_pretrained(final_model_dir)
    
    print(f"\nTraining completed")
    print(f"Final model saved to: {final_model_dir}")
    print(f"Best eval loss: {best_eval_loss:.4f}")


if __name__ == "__main__":
    main()
