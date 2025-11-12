#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced Evaluation with:
- CLIP Text-Image Score
- CLIP Image Similarity vs BARE face (identity preservation)
"""

import torch
import torch.nn.functional as F
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import argparse
from pathlib import Path
from tqdm import tqdm
import json
import numpy as np
import sys

sys.path.append(str(Path(__file__).parent.parent))

class CLIPEvaluator:
    def __init__(self, model_name="openai/clip-vit-base-patch32", device="cuda"):
        self.device = device
        print(f"Loading CLIP model: {model_name}")
        self.model = CLIPModel.from_pretrained(model_name).to(device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model.eval()
        print("CLIP model loaded")
        
    def compute_text_image_score(self, images, texts):
        """CLIP score: text-image similarity (0-100)"""
        inputs = self.processor(
            text=texts,
            images=images,
            return_tensors="pt",
            padding=True
        ).to(self.device)
        
        with torch.no_grad():
            image_features = self.model.get_image_features(pixel_values=inputs['pixel_values'])
            text_features = self.model.get_text_features(input_ids=inputs['input_ids'])
            
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
            similarity = torch.matmul(image_features, text_features.T).diagonal()
            clip_score = torch.clamp(similarity * 100, 0, 100)
            
        return clip_score.cpu().numpy()
    
    def compute_image_similarity(self, images1, images2):
        """CLIP image similarity vs bare face (identity preservation)"""
        inputs1 = self.processor(images=images1, return_tensors="pt").to(self.device)
        inputs2 = self.processor(images=images2, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            features1 = self.model.get_image_features(pixel_values=inputs1['pixel_values'])
            features2 = self.model.get_image_features(pixel_values=inputs2['pixel_values'])
            
            features1 = features1 / features1.norm(dim=-1, keepdim=True)
            features2 = features2 / features2.norm(dim=-1, keepdim=True)
            
            similarity = F.cosine_similarity(features1, features2)
            similarity_score = torch.clamp(similarity * 100, 0, 100)
            
        return similarity_score.cpu().numpy()

def load_samples_with_bare(data_dir, generated_dir, num_samples=100):
    """Load generated images and their corresponding bare faces"""
    data_dir = Path(data_dir)
    generated_dir = Path(generated_dir)
    
    samples = []
    
    # Get all generated images
    generated_files = sorted(generated_dir.glob("*.png"))[:num_samples]
    
    print(f"Loading samples from {generated_dir}")
    for gen_file in tqdm(generated_files, desc="Loading"):
        # Extract ID from filename (e.g., sample_0001_prompt1.png -> 0001)
        filename = gen_file.stem
        img_id = filename.split('_')[1] if len(filename.split('_')) > 1 else filename[:4]
        
        # Find corresponding bare face
        bare_dir = data_dir / "bare"
        bare_files = list(bare_dir.glob(f"{img_id}_bare_*.jpg"))
        
        if bare_files:
            # Extract prompt from filename if exists
            if "prompt" in filename:
                prompt_num = filename.split("prompt")[-1].replace(".png", "")
                prompts = [
                    "A face with natural makeup",
                    "A face with glamorous makeup", 
                    "A face with professional makeup"
                ]
                prompt_idx = int(prompt_num) - 1 if prompt_num.isdigit() else 0
                prompt = prompts[prompt_idx] if prompt_idx < len(prompts) else prompts[0]
            else:
                prompt = "A face with professional makeup"
            
            samples.append({
                'id': img_id,
                'generated': Image.open(gen_file).convert("RGB"),
                'bare_face': Image.open(bare_files[0]).convert("RGB"),
                'prompt': prompt,
                'filename': gen_file.name
            })
    
    return samples

def evaluate_model(generated_dir, data_dir, output_dir, num_samples=100, device="cuda"):
    """Evaluate generated images vs bare faces"""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    clip_eval = CLIPEvaluator(device=device)
    
    # Load samples
    samples = load_samples_with_bare(data_dir, generated_dir, num_samples)
    print(f"\nLoaded {len(samples)} samples with bare faces")
    
    results = {
        'generated_dir': str(generated_dir),
        'num_samples': len(samples),
        'clip_text_image_scores': [],
        'clip_bare_similarity_scores': [],  # Identity preservation
        'samples': []
    }
    
    print("\nEvaluating...")
    for sample in tqdm(samples, desc="Evaluating"):
        # 1. CLIP Text-Image Score
        text_image_score = clip_eval.compute_text_image_score(
            images=[sample['generated']],
            texts=[sample['prompt']]
        )[0]
        
        # 2. CLIP Similarity to BARE face (identity preservation)
        bare_similarity = clip_eval.compute_image_similarity(
            images1=[sample['bare_face']],
            images2=[sample['generated']]
        )[0]
        
        results['clip_text_image_scores'].append(float(text_image_score))
        results['clip_bare_similarity_scores'].append(float(bare_similarity))
        
        results['samples'].append({
            'id': sample['id'],
            'filename': sample['filename'],
            'prompt': sample['prompt'],
            'clip_text_image_score': float(text_image_score),
            'clip_bare_similarity': float(bare_similarity)
        })
    
    # Compute statistics
    results['mean_clip_text_image_score'] = float(np.mean(results['clip_text_image_scores']))
    results['std_clip_text_image_score'] = float(np.std(results['clip_text_image_scores']))
    results['mean_clip_bare_similarity'] = float(np.mean(results['clip_bare_similarity_scores']))
    results['std_clip_bare_similarity'] = float(np.std(results['clip_bare_similarity_scores']))
    
    # Print summary
    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)
    print(f"Generated images: {generated_dir}")
    print(f"Samples: {len(samples)}")
    print(f"\nCLIP Text-Image Score (0-100, higher = better prompt alignment):")
    print(f"  Mean: {results['mean_clip_text_image_score']:.2f} +/- {results['std_clip_text_image_score']:.2f}")
    print(f"  Range: [{min(results['clip_text_image_scores']):.2f}, {max(results['clip_text_image_scores']):.2f}]")
    print(f"\nCLIP Bare Face Similarity (0-100, higher = better identity preservation):")
    print(f"  Mean: {results['mean_clip_bare_similarity']:.2f} +/- {results['std_clip_bare_similarity']:.2f}")
    print(f"  Range: [{min(results['clip_bare_similarity_scores']):.2f}, {max(results['clip_bare_similarity_scores']):.2f}]")
    print("=" * 70)
    
    # Save results
    results_file = output_dir / "evaluation_results_bare.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_file}\n")
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Evaluate with bare face comparison")
    parser.add_argument("--generated_dir", type=str, required=True, help="Path to generated images")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to processed data")
    parser.add_argument("--output_dir", type=str, default="results/evaluation", help="Output directory")
    parser.add_argument("--num_samples", type=int, default=300, help="Number of samples")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    args = parser.parse_args()
    
    evaluate_model(
        generated_dir=args.generated_dir,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        device=args.device
    )

if __name__ == "__main__":
    main()
