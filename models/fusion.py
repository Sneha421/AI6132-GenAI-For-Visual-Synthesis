"""
Multi-style fusion module for combining 6 makeup conditions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class StyleFusionBlock(nn.Module):
    def __init__(self, num_styles: int = 6, feature_dim: int = 320):
        super().__init__()
        self.num_styles = num_styles
        self.feature_dim = feature_dim
        
        # Learnable style importance weights
        self.style_weights = nn.Parameter(
            torch.ones(num_styles) / num_styles
        )
        
        # Per-style normalization
        self.style_norms = nn.ModuleList([
            nn.GroupNorm(32, feature_dim) 
            for _ in range(num_styles)
        ])
        
        # Cross-attention between styles
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=8,
            batch_first=True
        )
        
        # Final fusion convolution
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim, 3, padding=1),
            nn.GroupNorm(32, feature_dim),
            nn.SiLU()
        )
    
    def forward(
        self, 
        style_features: torch.Tensor,  # [B, 6, C, H, W]
        weights: Optional[torch.Tensor] = None
    ):
        """
        Fuse 6 style features into single conditioning signal
        
        Args:
            style_features: [B, num_styles, C, H, W]
            weights: Optional manual weights [num_styles]
        
        Returns:
            fused: [B, C, H, W]
        """
        B, num_styles, C, H, W = style_features.shape
        
        # Use provided weights or learned weights
        if weights is None:
            weights = F.softmax(self.style_weights, dim=0)
        
        # Normalize each style
        normalized = []
        for i in range(num_styles):
            norm_feat = self.style_norms[i](style_features[:, i])
            normalized.append(norm_feat)
        
        # Weighted sum fusion
        fused = sum(w * feat for w, feat in zip(weights, normalized))
        
        # Optional: Apply cross-attention for style mixing
        # Reshape for attention: [B, H*W, C]
        flat_features = [
            feat.flatten(2).transpose(1, 2) 
            for feat in normalized
        ]
        
        # Attend from fused to all styles
        fused_flat = fused.flatten(2).transpose(1, 2)
        style_keys = torch.cat(flat_features, dim=1)  # [B, 6*H*W, C]
        
        attended, _ = self.cross_attention(
            fused_flat, style_keys, style_keys
        )
        
        # Reshape back and apply final conv
        attended = attended.transpose(1, 2).reshape(B, C, H, W)
        output = self.fusion_conv(attended)
        
        return output
