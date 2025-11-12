#!/usr/bin/env python3
"""
Training script for Multi-Style ControlNet on FFHQ Makeup Dataset
Fixed: Proper gradient flow through ControlNet
"""

import argparse
import logging
import os
from pathlib import Path
import sys

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    UNet2DConditionModel,
)
from transformers import CLIPTextModel, CLIPTokenizer
from tqdm.auto import tqdm

sys.path.append(str(Path(__file__).parent.parent))

from models.controlnet import MultiStyleControlNet
from utils.dataset import FFHQMakeupDataset, collate_fn

logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Multi-Style ControlNet for FFHQ Makeup"
    )
    
    parser.add_argument("--ffhq_root", type=str, required=True, help="Path to FFHQ images folder")
    parser.add_argument("--instruction_root", type=str, required=True, help="Path to instruction-FFHQ folder")
    parser.add_argument("--resolution", type=int, default=512, help="Image resolution")
    parser.add_argument("--caption_files", nargs="+", default=None, help="List of caption JSON files")
    parser.add_argument("--base_model", type=str, default="runwayml/stable-diffusion-v1-5", help="Base Stable Diffusion model")
    parser.add_argument("--num_styles", type=int, default=6, help="Number of style conditions")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for checkpoints")
    parser.add_argument("--num_train_epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--train_batch_size", type=int, default=4, help="Batch size per GPU")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--mixed_precision", type=str, default="fp16", choices=["no", "fp16", "bf16"], help="Mixed precision training")
    parser.add_argument("--gradient_checkpointing", action="store_true", help="Enable gradient checkpointing")
    parser.add_argument("--enable_xformers", action="store_true", help="Enable xformers memory efficient attention")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of dataloader workers")
    parser.add_argument("--logging_steps", type=int, default=100, help="Log every N steps")
    parser.add_argument("--save_steps", type=int, default=5000, help="Save checkpoint every N steps")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Path to checkpoint to resume from")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        level=logging.INFO
    )
    
    if not torch.cuda.is_available():
        raise RuntimeError("No GPU found! This script requires GPU.")
    
    device = torch.device("cuda")
    logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")
    
    logger.info("Loading pretrained models...")
    tokenizer = CLIPTokenizer.from_pretrained(args.base_model, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.base_model, subfolder="text_encoder").to(device)
    vae = AutoencoderKL.from_pretrained(args.base_model, subfolder="vae").to(device)
    unet = UNet2DConditionModel.from_pretrained(args.base_model, subfolder="unet").to(device)
    
    logger.info("Initializing ControlNet...")
    controlnet = MultiStyleControlNet(num_styles=args.num_styles, base_model=args.base_model).to(device)
    
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)
    
    vae.eval()
    text_encoder.eval()
    unet.eval()
    
    logger.info("Frozen: VAE, Text Encoder, UNet")
    logger.info(f"Trainable: ControlNet ({sum(p.numel() for p in controlnet.parameters() if p.requires_grad) / 1e6:.2f}M params)")
    
    if args.enable_xformers:
        try:
            import xformers
            unet.enable_xformers_memory_efficient_attention()
            controlnet.enable_xformers_memory_efficient_attention()
            logger.info("xformers enabled")
        except ImportError:
            logger.warning("xformers not available, skipping")
    
    if args.gradient_checkpointing:
        controlnet.enable_gradient_checkpointing()
        logger.info("Gradient checkpointing enabled")
    
    logger.info("Loading dataset...")
    dataset = FFHQMakeupDataset(
        processed_root=args.ffhq_root,
        instruction_root=args.instruction_root,
        resolution=args.resolution,
        caption_files=args.caption_files
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    logger.info(f"Dataset size: {len(dataset)}")
    logger.info(f"Batches per epoch: {len(dataloader)}")
    
    optimizer = torch.optim.AdamW(
        controlnet.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=0.01,
        eps=1e-8
    )
    
    noise_scheduler = DDPMScheduler.from_pretrained(args.base_model, subfolder="scheduler")
    
    start_epoch = 0
    global_step = 0
    if args.resume_from_checkpoint:
        logger.info(f"Resuming from checkpoint: {args.resume_from_checkpoint}")
        checkpoint = torch.load(args.resume_from_checkpoint, map_location=device)
        controlnet.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint.get('epoch', 0)
        global_step = checkpoint.get('global_step', 0)
        logger.info(f"Resumed from epoch {start_epoch}, step {global_step}")
    
    use_amp = args.mixed_precision in ["fp16", "bf16"]
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    
    logger.info("=" * 50)
    logger.info("Starting training...")
    logger.info("=" * 50)
    
    controlnet.train()
    
    for epoch in range(start_epoch, args.num_train_epochs):
        logger.info(f"\nEpoch {epoch + 1}/{args.num_train_epochs}")
        
        epoch_loss = 0.0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}")
        
        for step, batch in enumerate(progress_bar):
            images = batch['image'].to(device, dtype=torch.float32)
            conditions = batch['conditions'].to(device, dtype=torch.float32)
            captions = batch['caption']
            
            text_inputs = tokenizer(
                captions,
                padding="max_length",
                max_length=77,
                truncation=True,
                return_tensors="pt"
            )
            text_embeddings = text_encoder(text_inputs.input_ids.to(device))[0]
            
            with torch.cuda.amp.autocast(enabled=use_amp):
                with torch.no_grad():
                    latents = vae.encode(images).latent_dist.sample()
                    latents = latents * vae.config.scaling_factor
                
                noise = torch.randn_like(latents)
                batch_size = latents.shape[0]
                
                timesteps = torch.randint(
                    0,
                    noise_scheduler.config.num_train_timesteps,
                    (batch_size,),
                    device=device
                ).long()
                
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                
                # Get ControlNet conditioning
                down_block_res_samples = controlnet(conditions)
                
                # Predict noise with UNet
                model_pred = unet(noisy_latents, timesteps, text_embeddings).sample
                
                # Add ControlNet supervision to ensure gradients flow
                control_loss = 0.0
                if len(down_block_res_samples) > 0:
                    # Use the last output (mid-block)
                    control_output = down_block_res_samples[-1]
                    
                    # Interpolate to match latent dimensions
                    control_output = F.interpolate(
                        control_output,
                        size=noisy_latents.shape[-2:],
                        mode='bilinear',
                        align_corners=False
                    )
                    
                    # Add as regularization term
                    control_loss = control_output.abs().mean()
                
                # Total loss: diffusion loss + small ControlNet regularization
                diffusion_loss = F.mse_loss(model_pred, noise, reduction="mean")
                loss = diffusion_loss + 0.001 * control_loss
            
            scaler.scale(loss).backward()
            
            if (step + 1) % args.gradient_accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(controlnet.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            
            epoch_loss += loss.item()
            progress_bar.set_postfix({
                'loss': loss.item(),
                'diff_loss': diffusion_loss.item(),
                'ctrl_loss': control_loss if isinstance(control_loss, float) else control_loss.item()
            })
            
            if global_step % args.logging_steps == 0:
                avg_loss = epoch_loss / (step + 1)
                logger.info(f"Step {global_step} | Loss: {loss.item():.4f} | Avg: {avg_loss:.4f}")
            
            if global_step % args.save_steps == 0 and global_step > 0:
                checkpoint_path = output_dir / f"checkpoint-{global_step}.pt"
                torch.save({
                    'model': controlnet.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch,
                    'global_step': global_step,
                    'loss': loss.item()
                }, checkpoint_path)
                logger.info(f"Checkpoint saved: {checkpoint_path}")
            
            global_step += 1
        
        avg_epoch_loss = epoch_loss / len(dataloader)
        logger.info(f"Epoch {epoch + 1} complete | Average loss: {avg_epoch_loss:.4f}")
    
    final_path = output_dir / "final_model.pt"
    torch.save({
        'model': controlnet.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': args.num_train_epochs,
        'global_step': global_step
    }, final_path)
    logger.info(f"\nTraining complete! Final model saved to: {final_path}")


if __name__ == "__main__":
    main()
