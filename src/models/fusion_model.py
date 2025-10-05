import torch
import torch.nn as nn


from typing import Optional


from src.models.multi_modal_gpt_config import MultiModalGPTConfig


class FusionModel(nn.Module):
    def __init__(self, config: MultiModalGPTConfig, fusion_type: str = "concat"):
        super().__init__()
        self.fusion_type = fusion_type
        self.config = config

        if fusion_type == "concat":
            if config.vision_embed_dim != config.n_embd:
                self.vision_projection = nn.Linear(
                    config.vision_embed_dim, config.n_embd
                )
            else:
                self.vision_projection = nn.Identity()
        elif fusion_type == "projection":
            self.vision_projection = nn.Linear(
                config.vision_embed_dim, config.vision_projection_dim
            )
            self.projection_norm = nn.LayerNorm(config.vision_projection_dim)
        elif fusion_type == "cross_attention":
            self.cross_attention = nn.MultiheadAttention(
                config.n_embd, config.n_head, batch_first=True
            )
            self.norm = nn.LayerNorm(config.n_embd)

    def forward(
        self,
        vision_features: torch.Tensor,
        text_embeddings: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        if self.fusion_type == "concat":
            vision_features = self.vision_projection(vision_features)
            combined = torch.cat([vision_features, text_embeddings], dim=1)
            return combined
        elif self.fusion_type == "projection":
            vision_projection = self.vision_projection(vision_features)
            vision_projection = self.projection_norm(vision_projection)
            combined = torch.cat([vision_projection, text_embeddings], dim=1)
            return combined
        elif self.fusion_type == "cross_attention":
            attended = self.cross_attention(
                text_embeddings, vision_features, vision_features
            )
            return self.norm(text_embeddings + attended)
        else:
            raise ValueError(f"Unknown fusion type: {self.fusion_type}")
            
