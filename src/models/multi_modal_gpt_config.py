from transformers import GPT2Config


class MultiModalGPTConfig(GPT2Config):
    def __init__(
        self,
        vocab_size=50257,
        n_positions=1024,
        n_embd=768,
        n_layer=12,
        n_head=12,
        vision_embed_dim=768,
        vision_layers=6,
        vision_heads=12,
        img_size=224,
        patch_size=16,
        cross_attention_layers=None,
        vision_projection_dim=None,
        **kwargs,
    ):
        super().__init__(
            vocab_size,
            n_positions,
            n_embd,
            n_layer,
            n_head,
            **kwargs,
        )

        self.vision_embed_dim = vision_embed_dim
        self.vision_layers = vision_layers
        self.vision_heads = vision_heads
        self.img_size = img_size
        self.patch_size = patch_size

        self.cross_attention_layers = cross_attention_layers or []
        self.vision_projection_dim = vision_projection_dim or n_embd
        
