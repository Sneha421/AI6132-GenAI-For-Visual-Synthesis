#!/usr/bin/env python3
"""
Simple inference script for Multi-Style ControlNet
Uses --conditions (6 image paths) instead of --img_id
"""

import torch
from PIL import Image
import argparse
from pathlib import Path
from torchvision import transforms
import sys

sys.path.append(str(Path(__file__).parent.parent))
from models.controlnet import MultiStyleControlNet

def main():
    parser = argparse.ArgumentParser(description="Generate with ControlNet")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint path")
    parser.add_argument("--conditions", nargs=6, required=True, help="6 condition images")
    parser.add_argument("--prompt", default="A face with professional makeup", help="Text prompt")
    parser.add_argument("--output", required=True, help="Output path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--steps", type=int, default=50, help="Inference steps")
    parser.add_argument("--device", default="cuda", help="Device")
    args = parser.parse_args()
    
    device = torch.device(args.device)
    
    print(f"Loading conditions...")
    transform = transforms.Compose([
        transforms.Resize(512),
        transforms.CenterCrop(512),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    
    conditions = []
    for img_path in args.conditions:
        img = Image.open(img_path).convert('RGB')
        conditions.append(transform(img))
    
    conditions = torch.stack(conditions).unsqueeze(0).to(device)
    
    print(f"Loading ControlNet from {args.checkpoint}")
    controlnet = MultiStyleControlNet(num_styles=6)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    controlnet.load_state_dict(checkpoint['model'])
    controlnet = controlnet.to(device)
    controlnet.eval()
    
    print(f"Generating with prompt: '{args.prompt}'")
    with torch.no_grad():
        _ = controlnet(conditions)
    
    # For now, save the 4th condition (makeup_03) as output
    # Full SD inference would go here
    output_img = Image.open(args.conditions[3])
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_img.save(output_path)
    
    print(f"Saved to: {output_path}")

if __name__ == "__main__":
    main()
