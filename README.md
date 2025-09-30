# Image2TikZ: Knowledge Distillation for DeTikZify

This repository contains the implementation of a knowledge distillation framework for **DeTikZify**, a model that converts raster images (PNG, JPG) into TikZ LaTeX code for generating vector graphics of scientific figures and sketches.

## 🎯 Project Overview

This project implements knowledge distillation techniques to create smaller, faster versions of the DeTikZify model while maintaining high-quality TikZ code generation. The main contributions include:

- **Knowledge Distillation Framework**: A clean implementation for distilling knowledge from larger teacher models to smaller student models
- **Custom Loss Functions**: Specialized distillation losses optimized for sequence-to-sequence generation tasks
- **Evaluation Metrics**: Comprehensive knowledge distillation evaluation suite with KL divergence, confidence alignment, and prediction quality metrics
- **Training Pipeline**: Simplified training scripts with checkpoint management

## 🏗️ Architecture

The project is built on top of the original [DeTikZify](https://github.com/potamides/DeTikZify) framework and includes:

- **Teacher Model**: Large pre-trained DeTikZify model (e.g., `nllg/detikzify-cl-7b`)
- **Student Model**: Smaller model to be trained via distillation (e.g., `nllg/detikzify-tl-1.1b`)
- **Knowledge Distillation**: Logits-based distillation with temperature scaling and loss balancing

## 📁 Repository Structure

```
├── detikzify/                  # Core DeTikZify framework
├── evaluation/                 # Modular evaluation framework
│   ├── core_metrics.py         # Core KL and distillation metrics
│   ├── advanced_analysis.py    # Top-K and confidence analysis
│   ├── eval_utils.py           # Model loading and data utilities
│   ├── evaluator.py            # Main evaluation orchestrator
│   └── run_evaluation.py       # CLI evaluation interface
├── train.py                    # Main training script for knowledge distillation
├── inference.py                # Simple inference example
├── model_utils.py              # Model loading and management utilities
├── data_utils.py               # Data processing for knowledge distillation
├── training_utils.py           # Training step implementations
├── loss_functions.py           # Custom loss functions for distillation
├── checkpoint_utils.py         # Checkpoint saving and loading
└── requirements.txt            # Python dependencies
```

## 🚀 Getting Started

### Prerequisites

1. **Hardware Requirements**: 
   - **2+ High-end CUDA GPUs** (required for dual GPU setup)
   - Tested on 2x 48GB GPUs (high VRAM required for 7B parameter models)
2. **Python Environment**: Python 3.8+
3. **LaTeX Distribution**: TeX Live 2023+ with pdflatex, lualatex, xelatex
4. **System Dependencies**:
   - Ghostscript (for PDF processing)
   - Poppler utils (for PDF to image conversion)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/younisraafat22/Image2TikZ.git
cd Image2TikZ
```

2. Install Python dependencies:
```bash
pip install -r requirements.txt
```

3. Install system dependencies:

**Option A: Using conda (recommended - no sudo required):**
```bash
conda install -c conda-forge texlive-core ghostscript poppler
```

**Option B: Using system package manager (Ubuntu/Debian):**
```bash
sudo apt-get update
sudo apt-get install texlive-full ghostscript poppler-utils
```

## 📊 Usage

### Knowledge Distillation Training

Train a student model using knowledge distillation:

```bash
python train.py \\
    --output_dir ./outputs/my_distilled_model \\
    --teacher_model nllg/detikzify-cl-7b \\
    --student_model nllg/detikzify-tl-1.1b \\
    --dataset nllg/datikz-v3 \\
    --num_train_epochs 2 \\
    --batch_size 1 \\
    --gradient_accumulation_steps 32 \\
    --learning_rate 1e-5 \\
    --temperature 4.0 \\
    --alpha 0.2 \\
    --use_dual_gpu
```

**Important**: The `--use_dual_gpu` flag is **required** for proper operation and is enabled by default. It places the teacher model on GPU 0 and student model on GPU 1 to avoid memory conflicts and device mismatch errors.

### Simple Inference

Generate TikZ code from an image using a trained model:

```python
from inference import load_distilled_model
from detikzify.infer import DetikzifyPipeline
from PIL import Image

# Load your trained model
model, tokenizer = load_distilled_model("path/to/checkpoint", device="cuda")

# Create pipeline
pipeline = DetikzifyPipeline(model=model, tokenizer=tokenizer)

# Generate TikZ code
image = Image.open("your_image.png")
tikz_doc = pipeline.sample(image=image)

print("Generated TikZ code:")
print(tikz_doc.code)

# Save as PDF if compilation successful
if tikz_doc.pdf:
    tikz_doc.save("output.pdf")
```

### Evaluation

Evaluate knowledge distillation performance using the modular evaluation framework:

```bash
cd evaluation
python run_evaluation.py \\
    --teacher_model nllg/detikzify-cl-7b \\
    --distilled_model ../outputs/my_distilled_model/checkpoint-best \\
    --dataset nllg/datikz-v3 \\
    --compare_all \\
    --output_file evaluation_results.json \\
    --max_samples 1000 \\
    --temperature 3.0
```

## 📈 Key Features

### Knowledge Distillation
- **Temperature Scaling**: Softens teacher logits for better knowledge transfer
- **Loss Balancing**: Combines distillation loss with task-specific cross-entropy loss
- **Gradient Accumulation**: Enables training with larger effective batch sizes

### Evaluation Metrics
- **KL Divergence**: Measures knowledge transfer quality from teacher to student
- **Jensen-Shannon Divergence**: Symmetric measure of distribution similarity
- **Confidence Alignment**: How well student confidence matches teacher confidence
- **Top-K Analysis**: Overlap between teacher and student top predictions
- **Rank Correlation**: Spearman correlation of probability rankings
- **Calibration**: Expected Calibration Error (ECE) for prediction reliability

### Evaluation Framework
- **Modular Design**: Clean separation of metrics, analysis, and utilities
- **Core Metrics**: ECE, JS divergence, rank correlation, entropy analysis
- **Advanced Analysis**: Confidence alignment, top-K quality, prediction diversity
- **Comprehensive Comparison**: Teacher, baseline student, and distilled model evaluation

### Training Features
- **Checkpoint Management**: Automatic saving and loading of best models
- **Console Logging**: Training progress and evaluation metrics tracking
- **Mixed Precision**: Efficient training with bfloat16 precision
- **Gradient Clipping**: Stable training with gradient norm clipping

## 🔬 Research Context

This implementation is part of a Master's thesis exploring knowledge distillation techniques for multimodal sequence-to-sequence models. The work focuses on:

1. **Model Compression**: Reducing model size while maintaining performance
2. **Efficiency Optimization**: Faster inference for practical deployment
3. **Quality Preservation**: Maintaining high-quality TikZ code generation

## 📚 Citation

If you use this code in your research, please cite the original DeTikZify paper:

```bibtex
@misc{belouadi2024detikzify,
    title={DeTi{k}Zify: Synthesizing Graphics Programs for Scientific Figures and Sketches with Ti{k}Z}, 
    author={Jonas Belouadi and Simone Paolo Ponzetto and Steffen Eger},
    year={2024},
    eprint={2405.15306},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests.

## 📝 License

This project builds upon the [DeTikZify](https://github.com/potamides/DeTikZify) framework. Please refer to the original repository for licensing information.

The MCTS implementation is licensed under the MIT License (see `detikzify/mcts/LICENSE`).

## 🔗 Related Resources

- [Original DeTikZify Repository](https://github.com/potamides/DeTikZify)
- [DeTikZify Paper](https://arxiv.org/abs/2405.15306)
- [TikZ Documentation](https://ctan.org/pkg/pgf)
- [Hugging Face Models](https://huggingface.co/nllg)

## 🛠️ Troubleshooting

### Training Issues
- **Dual GPU Required**: This framework requires 2+ GPUs and uses `--use_dual_gpu` by default
- **Device Mismatch Error**: If you get "Expected all tensors to be on the same device" error, ensure you have 2+ GPUs available
- **Single GPU Not Supported**: The current implementation requires separate GPUs for teacher and student models
- **GPU Memory**: If running out of memory, reduce `--batch_size` or increase `--gradient_accumulation_steps`

### System Dependencies  
- Ensure LaTeX is properly installed and accessible via PATH
- For compilation errors, check that all required LaTeX packages are installed