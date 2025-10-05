import torch


from src.gpts.multi_modal_gpt import MultiModalGPT
from src.models.multi_modal_gpt_config import MultiModalGPTConfig


text_config = {
        'vocab_size': 50257,
        'n_positions': 512,
        'n_embd': 768,
        'n_layer': 6,  # Smaller for testing
        'n_head': 12
    }
    
vision_config = {
    'embed_dim': 768,
    'n_layers': 6,
    'n_heads': 12,
    'img_size': 224,
    'patch_size': 16
}

fusion_config = {
    'type': 'projection',  # or 'concat' or 'cross_attention'
    'vision_projection_dim': 768,
    'cross_attention_layers': []
}


def test_create_multimodal_gpt():
    config = MultiModalGPTConfig(
        vocab_size=text_config.get('vocab_size', 50257),
        n_positions=text_config.get('n_positions', 1024),
        n_embd=text_config.get('n_embd', 768),
        n_layer=text_config.get('n_layer', 12),
        n_head=text_config.get('n_head', 12),
        vision_embed_dim=vision_config.get('embed_dim', 768),
        vision_layers=vision_config.get('n_layers', 6),
        vision_heads=vision_config.get('n_heads', 12),
        img_size=vision_config.get('img_size', 224),
        patch_size=vision_config.get('patch_size', 16),
        cross_attention_layers=fusion_config.get('cross_attention_layers', []) if fusion_config else [],
        vision_projection_dim=fusion_config.get('vision_projection_dim') if fusion_config else None
    )

    assert config is not None

    model = MultiModalGPT(config)
    assert model is not None
    
    batch_size = 2
    images = torch.randn(batch_size, 3, 224, 224)
    input_ids = torch.randint(0, 1000, (batch_size, 20))
    attention_mask = torch.ones(batch_size, 20)
    labels = input_ids.clone()
    
    outputs = model(
        images=images,
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels
    )

    generated = model.generate(images, input_ids[:, :5], max_length=10)