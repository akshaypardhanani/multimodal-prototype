import os
import torch


from PIL import Image


from src.models.vision_encoder import VisionEncoder
from src.utils.image_processor import ImageProcessor


def test_image_encoder():
    image_path = os.path.expanduser("tests/assets/horse.jpg")
    image = Image.open(image_path)

    processor = ImageProcessor()
    vision_encoder = VisionEncoder(
        image_size=224,
        patch_size=16, 
        embed_dim=768,
        n_layers=6,
        n_heads=12
    )

    image = processor(image)
    image = image.unsqueeze(0)

    with torch.no_grad():
        features = vision_encoder(image)
    
    expected_features_shape = (1, 197, 768)
    assert features.shape == expected_features_shape
