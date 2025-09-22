import torch


from typing import Dict, List


class Collator:

    def __init__(self, pad_token_id: int) -> None:
        self.pad_token_id = pad_token_id

    def __call__(self, batch: List[Dict[str, torch.Tensor]]) -> Any:
        for item in batch:
            images = torch.stack(item['image'])
            input_ids = torch.stack(item['input_ids'])
            attention_mask = torch.stack(item['attention_mask'])
            labels = torch.stack(item['labels'])
            
            questions = [item["question"]]
            answers = [item["answer"]]
            question_type = [item["question_type"]]
            image_ids = [item["image_id"]]
            question_ids = [item["question_id"]]
            
        return {
            "images": images,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "questions": questions,
            "answers": answers,
            "question_type": question_type,
            "image_ids": image_ids,
            "question_ids": question_ids
        }
            