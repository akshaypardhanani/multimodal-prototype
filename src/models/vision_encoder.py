import torch
import torch.nn as nn


from src.models.patch_embeddings import PatchEmbeddings
from src.models.vision_transformer_block import VisionTransformerBlock

class VisionEncoder(nn.Module):
    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 768,
        n_layers: int = 12,
        n_heads: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ): 
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_embeddings = PatchEmbeddings(
            image_size=image_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
        )

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        n_patches = self.patch_embeddings.num_patches
        self.pos_embeddings = nn.Parameter(torch.zeros(1, n_patches + 1, embed_dim))

        self.dropout = nn.Dropout(dropout)

        self.transformer_blocks = nn.ModuleList([
            VisionTransformerBlock(
                embed_dim=embed_dim,
                n_heads=n_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
            )
            for _ in range(n_layers)
        ])

        self.norm = nn.LayerNorm(embed_dim)
        self._init_weights()

    def _init_weights(self):
        # trunc_normal_ will modify the nn.Parameters pos_embeddings and cls_token in place.
        torch.nn.init.trunc_normal_(self.pos_embeddings, std=0.02)
        torch.nn.init.trunc_normal_(self.cls_token, std=0.02)
        
    def forward(self, x: torch.Tensor, return_all_tokens: bool = True):
        batch_size = x.shape[0]
        x = self.patch_embeddings(x)

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        x = x + self.pos_embeddings
        x = self.dropout(x)

        for block in self.transformer_blocks:
            x = block(x)

        x = self.norm(x)

        if return_all_tokens:
            return x
        else:
            return x[:, 0]
