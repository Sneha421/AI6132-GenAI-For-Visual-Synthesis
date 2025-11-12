"""
Multi-Style ControlNet for FFHQ Makeup Transfer
with multi-scale feature extraction
"""

import torch
import torch.nn as nn
from typing import Optional, List

class MultiStyleControlNet(nn.Module):
    def __init__(
        self,
        num_styles: int = 6,
        base_model: str = "runwayml/stable-diffusion-v1-5"
    ):
        super().__init__()
        self.num_styles = num_styles
        
        # 6 parallel style encoders (one for each makeup style + bare)
        self.style_encoders = nn.ModuleList([
            self._create_multi_scale_encoder()
            for _ in range(num_styles)
        ])
        
        # Style fusion modules (one per scale)
        self.style_fusions = nn.ModuleList([
            StyleFusionBlock(num_styles, 320),   # Scale 0
            StyleFusionBlock(num_styles, 320),   # Scale 1
            StyleFusionBlock(num_styles, 320),   # Scale 2
            StyleFusionBlock(num_styles, 320),   # Scale 3
            StyleFusionBlock(num_styles, 640),   # Scale 4
            StyleFusionBlock(num_styles, 640),   # Scale 5
            StyleFusionBlock(num_styles, 640),   # Scale 6
            StyleFusionBlock(num_styles, 1280),  # Scale 7
            StyleFusionBlock(num_styles, 1280),  # Scale 8
            StyleFusionBlock(num_styles, 1280),  # Scale 9
            StyleFusionBlock(num_styles, 1280),  # Scale 10
            StyleFusionBlock(num_styles, 1280),  # Scale 11
            StyleFusionBlock(num_styles, 1280),  # Scale 12
            StyleFusionBlock(num_styles, 1280),  # Scale 13 (mid)
        ])
        
        # Zero convolution layers (14 for SD 1.5)
        self.zero_convs = self._create_zero_convs()
        
    def _create_multi_scale_encoder(self) -> nn.ModuleDict:
        """Create multi-scale encoder for one style"""
        return nn.ModuleDict({
            # Scale 0-3: 320 channels
            'block_0': nn.Sequential(
                nn.Conv2d(3, 16, 3, padding=1),
                nn.SiLU(),
                nn.Conv2d(16, 320, 3, padding=1),
            ),
            'block_1': nn.Sequential(
                nn.Conv2d(320, 320, 3, padding=1),
                nn.SiLU(),
            ),
            'block_2': nn.Sequential(
                nn.Conv2d(320, 320, 3, padding=1),
                nn.SiLU(),
            ),
            'block_3': nn.Sequential(
                nn.Conv2d(320, 320, 3, padding=1, stride=2),
                nn.SiLU(),
            ),
            # Scale 4-6: 640 channels
            'block_4': nn.Sequential(
                nn.Conv2d(320, 640, 3, padding=1),
                nn.SiLU(),
            ),
            'block_5': nn.Sequential(
                nn.Conv2d(640, 640, 3, padding=1),
                nn.SiLU(),
            ),
            'block_6': nn.Sequential(
                nn.Conv2d(640, 640, 3, padding=1, stride=2),
                nn.SiLU(),
            ),
            # Scale 7-13: 1280 channels
            'block_7': nn.Sequential(
                nn.Conv2d(640, 1280, 3, padding=1),
                nn.SiLU(),
            ),
            'block_8': nn.Sequential(
                nn.Conv2d(1280, 1280, 3, padding=1),
                nn.SiLU(),
            ),
            'block_9': nn.Sequential(
                nn.Conv2d(1280, 1280, 3, padding=1),
                nn.SiLU(),
            ),
            'block_10': nn.Sequential(
                nn.Conv2d(1280, 1280, 3, padding=1),
                nn.SiLU(),
            ),
            'block_11': nn.Sequential(
                nn.Conv2d(1280, 1280, 3, padding=1),
                nn.SiLU(),
            ),
            'block_12': nn.Sequential(
                nn.Conv2d(1280, 1280, 3, padding=1),
                nn.SiLU(),
            ),
            'block_13': nn.Sequential(
                nn.Conv2d(1280, 1280, 3, padding=1),
                nn.SiLU(),
            ),
        })
    
    def _create_zero_convs(self) -> nn.ModuleList:
        """Create zero-initialized convolution blocks"""
        dims = [320, 320, 320, 320, 640, 640, 640, 1280, 
                1280, 1280, 1280, 1280, 1280, 1280]
        
        zero_convs = nn.ModuleList()
        for dim in dims:
            conv = nn.Conv2d(dim, dim, 1)
            nn.init.zeros_(conv.weight)
            nn.init.zeros_(conv.bias)
            zero_convs.append(conv)
        
        return zero_convs
    
    def forward(self, conditions: torch.Tensor) -> List[torch.Tensor]:
        """
        Forward pass with multi-style conditioning
        
        Args:
            conditions: [B, 6, 3, H, W] - 6 style conditioning images
        
        Returns:
            List of 14 control features for UNet injection
        """
        batch_size = conditions.shape[0]
        
        # Process each style through multi-scale encoder
        all_scale_features = []  # Will contain [6 styles, 14 scales, features]
        
        for style_idx in range(self.num_styles):
            encoder = self.style_encoders[style_idx]
            x = conditions[:, style_idx]  # [B, 3, H, W]
            
            scale_features = []
            for scale_idx in range(14):
                x = encoder[f'block_{scale_idx}'](x)
                scale_features.append(x)
            
            all_scale_features.append(scale_features)
        
        # Fuse styles at each scale
        fused_scales = []
        for scale_idx in range(14):
            # Gather features from all styles at this scale
            style_features_at_scale = torch.stack([
                all_scale_features[style_idx][scale_idx]
                for style_idx in range(self.num_styles)
            ], dim=1)  # [B, 6, C, H, W]
            
            # Fuse
            fused = self.style_fusions[scale_idx](style_features_at_scale)
            fused_scales.append(fused)
        
        # Apply zero convolutions
        outputs = [
            zero_conv(fused) 
            for zero_conv, fused in zip(self.zero_convs, fused_scales)
        ]
        
        return outputs
    
    def enable_gradient_checkpointing(self):
        """Enable gradient checkpointing to save memory"""
        pass  # Implement if needed
    
    def enable_xformers_memory_efficient_attention(self):
        """Placeholder for xformers integration"""
        pass


class StyleFusionBlock(nn.Module):
    """Fusion module to combine 6 style features"""
    
    def __init__(self, num_styles: int = 6, feature_dim: int = 320):
        super().__init__()
        self.num_styles = num_styles
        
        # Learnable weights for each style
        self.style_weights = nn.Parameter(
            torch.ones(num_styles) / num_styles
        )
        
        # Per-style normalization
        num_groups = min(32, feature_dim // 4)  # Ensure valid group count
        self.style_norms = nn.ModuleList([
            nn.GroupNorm(num_groups, feature_dim) 
            for _ in range(num_styles)
        ])
        
        # Final fusion convolution
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim, 3, padding=1),
            nn.GroupNorm(num_groups, feature_dim),
            nn.SiLU()
        )
    
    def forward(
        self, 
        style_features: torch.Tensor,
        weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Fuse 6 style features
        
        Args:
            style_features: [B, 6, C, H, W]
            weights: Optional manual weights [6]
        
        Returns:
            fused: [B, C, H, W]
        """
        if weights is None:
            weights = torch.softmax(self.style_weights, dim=0)
        
        # Normalize each style
        normalized = []
        for i in range(self.num_styles):
            norm_feat = self.style_norms[i](style_features[:, i])
            normalized.append(norm_feat)
        
        # Weighted fusion
        fused = sum(w * feat for w, feat in zip(weights, normalized))
        
        # Apply fusion convolution
        output = self.fusion_conv(fused)
        
        return output
