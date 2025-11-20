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
        
        self.style_encoders = nn.ModuleList([
            self._create_multi_scale_encoder()
            for _ in range(num_styles)
        ])
        
        self.style_fusions = nn.ModuleList([
            StyleFusionBlock(num_styles, 320),
            StyleFusionBlock(num_styles, 320),
            StyleFusionBlock(num_styles, 320),
            StyleFusionBlock(num_styles, 320),
            StyleFusionBlock(num_styles, 640),
            StyleFusionBlock(num_styles, 640),
            StyleFusionBlock(num_styles, 640),
            StyleFusionBlock(num_styles, 1280),
            StyleFusionBlock(num_styles, 1280),
            StyleFusionBlock(num_styles, 1280),
            StyleFusionBlock(num_styles, 1280),
            StyleFusionBlock(num_styles, 1280),
            StyleFusionBlock(num_styles, 1280),
            StyleFusionBlock(num_styles, 1280),
        ])
        
        self.zero_convs = self._create_zero_convs()
        
    def _create_multi_scale_encoder(self) -> nn.ModuleDict:
        return nn.ModuleDict({
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
        dims = [320, 320, 320, 320, 640, 640, 640, 1280, 
                1280, 1280, 1280, 1280, 1280, 1280]
        
        zero_convs = nn.ModuleList()
        for dim in dims:
            conv = nn.Conv2d(dim, dim, 1)
            nn.init.zeros_(conv.weight)
            nn.init.zeros_(conv.bias)
            zero_convs.append(conv)
        
        return zero_convs
    
    def forward(self, conditions: torch.Tensor, caption_transformed: Optional[str] = None) -> List[torch.Tensor]:
        batch_size = conditions.shape[0]
        
        all_scale_features = [] 
        
        for style_idx in range(self.num_styles):
            encoder = self.style_encoders[style_idx]
            x = conditions[:, style_idx] 
            
            scale_features = []
            for scale_idx in range(14):
                x = encoder[f'block_{scale_idx}'](x)
                scale_features.append(x)
            
            all_scale_features.append(scale_features)
        
        fused_scales = []
        for scale_idx in range(14):
            style_features_at_scale = torch.stack([
                all_scale_features[style_idx][scale_idx]
                for style_idx in range(self.num_styles)
            ], dim=1) 
            
            fused = self.style_fusions[scale_idx](style_features_at_scale)
            fused_scales.append(fused)
        
        outputs = [
            zero_conv(fused) 
            for zero_conv, fused in zip(self.zero_convs, fused_scales)
        ]
        
        return outputs
    
    def enable_gradient_checkpointing(self):
        pass  
    
    def enable_xformers_memory_efficient_attention(self):
        pass


class StyleFusionBlock(nn.Module):
    def __init__(self, num_styles: int = 6, feature_dim: int = 320):
        super().__init__()
        self.num_styles = num_styles
        
        self.style_weights = nn.Parameter(
            torch.ones(num_styles) / num_styles
        )
        
        num_groups = min(32, feature_dim // 4)  
        self.style_norms = nn.ModuleList([
            nn.GroupNorm(num_groups, feature_dim) 
            for _ in range(num_styles)
        ])
        
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
        if weights is None:
            weights = torch.softmax(self.style_weights, dim=0)
        
        normalized = []
        for i in range(self.num_styles):
            norm_feat = self.style_norms[i](style_features[:, i])
            normalized.append(norm_feat)
        
        fused = sum(w * feat for w, feat in zip(weights, normalized))
        
        output = self.fusion_conv(fused)
        
        return output


caption_transformed = "Multi-Style ControlNet for FFHQ Makeup Transfer with multi-scale feature extraction"

model = MultiStyleControlNet(num_styles=6)
conditions = torch.randn(1, 6, 3, 512, 512)
control_features = model(conditions, caption_transformed=caption_transformed)
