import requests
from PIL import Image
from io import BytesIO


from datasets import load_dataset
from torch.utils.data import Dataset as TorchDataset
from transformers import PreTrainedTokenizerBase
from urllib.parse import urlparse


from src.utils.image_processor import ImageProcessor


class MultimodalDataset(TorchDataset):
    def __init__(
        self,
        dataset_name: str,
        split: str,
        tokenizer: PreTrainedTokenizerBase,
        image_processor: ImageProcessor,
        max_text_length: int = 512,
        task_type: str = "vqa",
        image_token: str = "<image>",
        question_token: str = "<question>",
        answer_token: str = "<answer>",
        cache_images: bool = True
    ):
        self.dataset = load_dataset(dataset_name, split=split)
        self.split = split
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.max_text_length = max_text_length
        self.task_type = task_type
        self.image_token = image_token
        self.question_token = question_token
        self.answer_token = answer_token
        self.cache_images = cache_images
        self.tokenizer_vocab = self.tokenizer.get_vocab()

        for tok in [image_token, question_token, answer_token]:
            if tok not in self.tokenizer_vocab:
                self.tokenizer.add_tokens([tok])

        self.image_token_id = self.tokenizer.convert_tokens_to_ids(image_token)
        self.question_token_id = self.tokenizer.convert_tokens_to_ids(question_token)
        self.answer_token_id = self.tokenizer.convert_tokens_to_ids(answer_token)

        self.image_cache = {} if self.cache_images else None

    def __len__(self):
        return len(self.dataset)
        

    def __getitem__(self, idx):
        item = self.dataset[idx]
        processed_item = self._process_item(item)

        image, question, answer, question_type = (processed_item[x] for x in ["image", "question", "answer", "question_type"])

        image_tensor = None
        if self.cache_images:
            image_tensor = self.image_cache[idx]
        if image_tensor is None:
            image_tensor = self.image_processor(image)
            if self.cache_images:
                self.image_cache[idx] = image_tensor
        
        input_text = self._create_input_sequence(question, answer, include_answer=True)

        encoding = self.tokenizer(
            input_text,
            truncation=True,
            max_length=self.max_text_length,
            padding="max_length",
            return_tensors="pt",
        )

        input_ids, attention_mask = (encoding[x] for x in ["input_ids", "attention_mask"])

        question_only = self._create_input_sequence(question, "", include_answer=False)
        question_encoding = self.tokenizer(question_only, add_special_tokens=False)
        question_length = len(question_encoding["input_ids"])

        labels = self._create_labels(input_ids, question_length)
        
        return {
            'image': image_tensor,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'question': question,
            'answer': answer,
            'question_type': question_type,
            'image_id': processed_item.get('image_id', 'unknown'),
            'question_id': processed_item.get('question_id', 'unknown')
        }

    def _create_labels(self, input_ids, question_length):
        labels = input_ids.clone()

        for i, token_id in enumerate(input_ids):
            if token_id == self.answer_token_id:
                labels[: i + 1] = -100
                return labels

        labels[:] = -100
        return labels
        
    
    def _create_input_sequence(self, question, answer, include_answer=True):
        if include_answer:
            return f"{self.image_token} {question} {self.answer_token} {answer}"
        else:
            return f"{self.image_token} {question} {self.answer_token}"
    
    def _load_image(self, image_source):
        img = urlparse(image_source)
        try:
            if img.scheme in ['http', 'https']:
                response = requests.get(image_source) # TODO: Look into switching this to httpx
                return Image.open(BytesIO(response.content)).convert("RGB")
            elif img.scheme == '':
                return Image.open(image_source).convert("RGB")
            else:
                raise ValueError(f"Unsupported image source: {image_source}")
        except ValueError:
            print(f"Unsupported Image source {img}")
            return None

    
    def _process_item(self, item):
        if 'image_url' in item:
            image_source = item['image_url']
        elif 'image' in item:
            if isinstance(item['image'], str):
                image_source = item['image']
            else:
                image = item['image']
                image_source =  None
        else:
            return None

        if image_source:
            image = self._load_image(image_source)
            if image is None:
                return None

        question = item.get('question')
        answer = item.get('answer')

        if question is None or answer is None:
            return None

        question_type = item.get('question_type', 'unknown')
        image_id = item.get('image_id', 'unknown')
        question_id = item.get('question_id', 'unknown')

        return {
            'image': image,
            'question': question,
            'answer': answer,
            'question_type': question_type,
            'image_id': image_id,
            'question_id': question_id,
            'image_source': image_source
        }
        
        