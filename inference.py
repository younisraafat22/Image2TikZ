#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Simple inference script for distilled DeTikZify models - like usage.py
"""

import torch
from PIL import Image
import sys
import os
import time
from operator import itemgetter

# Add paths for detikzify
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from detikzify.infer import DetikzifyPipeline, TikzDocument
from model_utils import load_distilled_model

# Configuration - Replace with your actual paths
model_path = "outputs/your_model/checkpoint-best"  # Path to your distilled model checkpoint
image_path = "path/to/your/image.png"  # Path to your input image

device = "cuda"
torch_dtype = torch.bfloat16

# Define output directory
output_dir = "./inference_output"
os.makedirs(output_dir, exist_ok=True)

print("Loading Distilled DeTikZify Model...")

# Start timing for pipeline initialization
start_time = time.time()
# Load the distilled model
model, tokenizer = load_distilled_model(model_path, device, torch_dtype)

# Create pipeline with distilled model (using default parameters)
pipeline = DetikzifyPipeline(
    model=model, 
    tokenizer=tokenizer, 
    device=device,
)

init_time = time.time() - start_time
print(f"Pipeline initialization time: {init_time:.2f} seconds")

# Load image
image = Image.open(image_path).convert("RGB")
print(f"ðŸ“· Image loaded: {image.size}")

# Measure time for generating a single TikZ program
start_time = time.time()
# Generate single TikZ program with deterministic settings
print("\n🎯 Generating DETERMINISTIC TikZ code...")
fig = pipeline.sample(image=image)

single_tikz_time = time.time() - start_time
print(f"Single TikZ generation time: {single_tikz_time:.2f} seconds")

generated_code = fig.code

# Display results
print("\nGenerated TikZ Code:")
print("=" * 50)
print(generated_code)
print("=" * 50)
print("Inference completed!")
