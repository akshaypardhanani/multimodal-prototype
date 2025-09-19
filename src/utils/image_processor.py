from torchvision import transforms

class ImageProcessor:
    def __init__(self, img_size: int = 224) -> None:
        self.img_size = img_size
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    
    def __call__(self, image):
        return self.transform(image)