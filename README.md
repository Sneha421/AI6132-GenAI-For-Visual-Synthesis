# AI6132 — Generative AI for Visual Synthesis: Instruction-Based Facial Makeup Transfer

> **Course:** AI6132 – Generative AI for Visual Synthesis  
> **Project:** Instruction-based facial makeup transfer using InstructPix2Pix, ControlNet, and Vision-Language Models

## Overview

This repository contains the code for a group project that explores **instruction-driven facial makeup transfer**. Given a bare-face image and a natural-language makeup instruction (e.g., *"Apply light foundation, red lips, and natural makeup"*), the system generates a realistic makeup-applied version of the face.

The pipeline consists of three stages:

1. **Dataset Construction** — Pair before/after makeup images from existing datasets and generate text instructions with Vision-Language Models (LLaVA, InstructBLIP, LLaMA).
2. **Model Training** — Fine-tune InstructPix2Pix (with full fine-tuning and LoRA) and ControlNet on the constructed dataset.
3. **Evaluation** — Assess output quality using FID, CLIP (text–image and image–image similarity), and LPIPS metrics.

## Repository Structure

```
.
├── models/
│   ├── controlnet.py                  # MultiStyleControlNet architecture (6 makeup styles)
│   └── controlNet_fusion.py           # StyleFusionBlock — cross-attention fusion of style features
├── scripts/
│   ├── controlNet_train.py            # ControlNet training script
│   ├── controlNet_inference.py        # ControlNet inference script
│   ├── dataset_process.py             # Dataset preprocessing utilities
│   └── evaluate_with_bare.py          # Evaluation metrics (FID, CLIP, LPIPS)
├── Instruction-FFHQ/                  # Placeholder for InstructBLIP-generated instructions
├── LLAVA Caption Generator.py         # Generate captions from makeup images using LLaVA
├── LLAMA Shorten Caption.py           # Shorten verbose captions using LLaMA
├── instruction-by-blip.py             # Generate editing instructions using InstructBLIP
├── ffhq-makeup-generated-by-instructblip.py  # FFHQ-Makeup instruction generation via InstructBLIP
├── Sort-dataset.py                    # Organise and sort dataset image pairs
├── ip2p_full_training.py              # InstructPix2Pix full fine-tuning (Colab script)
├── ip2p full training.ipynb           # InstructPix2Pix full fine-tuning (notebook)
├── instructpix2pix_lora_and_eval_functions.py  # InstructPix2Pix LoRA training + eval
├── instructpix2pix_LORA_and_eval_functions.ipynb  # Same as above (notebook)
├── ip2p_run_on_test_set.py            # Batch inference on test set
├── IP2P_run_on_test_set.ipynb         # Same as above (notebook)
├── captions_transformed*.json         # Generated caption files (multiple shards)
├── image_pairs-2.json                 # Paired image metadata
├── sample_makeup05.jsonl              # Sample makeup prompts/data
└── README.md
```

## Datasets

| Dataset | Source |
|---------|--------|
| **FFHQ-Makeup** — Paired synthetic makeup with facial consistency across multiple styles | [HuggingFace](https://huggingface.co/datasets/cyberagent/FFHQ-Makeup) |
| **BeautyBank (Bare-Makeup-Synthesis)** — Encoding facial makeup in latent space | [HuggingFace](https://huggingface.co/datasets/lulululululululululu/Bare-Makeup-Synthesis-Dataset) |

Text instructions for each image pair are automatically generated using **LLaVA**, **InstructBLIP**, and **LLaMA** (for caption condensation).

## Methods

### 1. Instruction Generation (VLM Pipeline)

- **LLaVA Caption Generator** (`LLAVA Caption Generator.py`) — Produces detailed makeup descriptions from images via few-shot prompting.
- **InstructBLIP** (`instruction-by-blip.py`, `ffhq-makeup-generated-by-instructblip.py`) — Generates editing instructions from before/after image pairs.
- **LLaMA Caption Shortener** (`LLAMA Shorten Caption.py`) — Condenses verbose captions into concise editing prompts.

### 2. InstructPix2Pix

- **Full Fine-Tuning** (`ip2p_full_training.py`) — Fine-tunes Stable Diffusion 2.1 via the InstructPix2Pix training pipeline using HuggingFace Diffusers + Accelerate (FP16, xformers).
- **LoRA Fine-Tuning** (`instructpix2pix_lora_and_eval_functions.py`) — Parameter-efficient fine-tuning with LoRA adapters, plus built-in evaluation functions.

### 3. ControlNet

- **Multi-Style ControlNet** (`models/controlnet.py`) — Custom architecture with 6 parallel style encoders and learnable style fusion blocks.
- **Style Fusion** (`models/controlNet_fusion.py`) — Cross-attention mechanism that fuses multiple makeup style features into a single conditioning signal.
- **Training & Inference** (`scripts/controlNet_train.py`, `scripts/controlNet_inference.py`)

### 4. Evaluation

- **FID** — Fréchet Inception Distance between generated and ground-truth makeup images.
- **CLIP Score** — Text–image similarity (does the output match the instruction?) and image–image similarity.
- **LPIPS** — Learned Perceptual Image Patch Similarity between generated and reference images.

Evaluation code: `scripts/evaluate_with_bare.py`

## Getting Started

### Prerequisites

- Python 3.8+
- PyTorch ≥ 1.13 with CUDA support
- A GPU with ≥ 16 GB VRAM (A100 recommended)

### Installation

```bash
pip install torch torchvision
pip install diffusers transformers accelerate xformers ftfy datasets tensorboard
pip install lpips clip-benchmark
```

### Training InstructPix2Pix (Full Fine-Tuning)

```bash
accelerate launch --mixed_precision="fp16" \
  diffusers/examples/instruct_pix2pix/train_instruct_pix2pix.py \
  --pretrained_model_name_or_path=stabilityai/stable-diffusion-2-1 \
  --dataset_name="<your-hf-dataset>" \
  --output_dir="./output" \
  --resolution=512 --random_flip \
  --train_batch_size=4 --gradient_accumulation_steps=4 --gradient_checkpointing \
  --max_train_steps=2200 --learning_rate=1e-05 \
  --mixed_precision=fp16 --seed=42
```

### Inference

```python
import torch
from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler
from PIL import Image

pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
    "<your-model-path>", torch_dtype=torch.float16, safety_checker=None
)
pipe.to("cuda")
pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

image = Image.open("bare_face.jpg").convert("RGB")
result = pipe("Apply light foundation, red lips, and natural makeup", image=image).images[0]
result.save("makeup_result.png")
```

## References

1. **FFHQ-Makeup**: Paired Synthetic Makeup Dataset with Facial Consistency Across Multiple Styles — [Dataset](https://huggingface.co/datasets/cyberagent/FFHQ-Makeup)
2. **BeautyBank**: Encoding Facial Makeup in Latent Space — [Dataset](https://huggingface.co/datasets/lulululululululululu/Bare-Makeup-Synthesis-Dataset)
3. **Face-MakeUp**: Multimodal Facial Prompts for Text-to-Image Generation — [Paper](https://arxiv.org/pdf/2501.02523)
4. **InstructPix2Pix**: Learning to Follow Image Editing Instructions — [Paper](https://arxiv.org/abs/2312.04780)

## Team

| Task | Member |
|------|--------|
| ControlNet + CLIP (text–image & image–image) | Hedi |
| InstructPix2Pix + LPIPS + FID | Xinyang |
| LLaVA + Scripting for BeautyBank & FFHQ-Makeup | Sneha |
| InstructBLIP/LLaMA + Scripting for BeautyBank & FFHQ-Makeup | Deng |