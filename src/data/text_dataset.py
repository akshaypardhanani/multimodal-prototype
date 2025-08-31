import multiprocessing

from datasets import load_dataset, Dataset
from torch.utils import data
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase


class TextDataset(Dataset):
    def __init__(self, dataset_name: str | Dataset, split: str, tokenizer: PreTrainedTokenizerBase, block_size: int = 512):
        self.dataset = load_dataset(dataset_name, split=split) if isinstance(dataset_name, str) else dataset_name

        def tokenize_fn(examples):
            return tokenizer(examples["text"], truncation=False)

        tokenized = self.dataset.map(
            tokenize_fn,
            batched=True,
            num_proc=multiprocessing.cpu_count(),
            remove_columns=["text"],
            desc="Tokenizing dataset"
        )
        
        def group_texts(examples):
            concatenated_ids = sum(examples["input_ids"], [])
            concatenated_attn_masks = sum(examples["attention_mask"], [])

            total_length = (len(concatenated_ids) // block_size) * block_size
            return {
                "input_ids": [
                    concatenated_ids[i : i + block_size]
                    for i in range(0, total_length, block_size)
                ],
                "attention_mask": [
                    concatenated_attn_masks[i : i + block_size]
                    for i in range(0, total_length, block_size)
                ]
            }
        
        lm_dataset = tokenized.map(
            group_texts,
            batched=True,
            num_proc=multiprocessing.cpu_count(),
            desc=f"Grouping texts into blocks of {block_size}"
        )

        self.dataset = lm_dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        return {
            "input_ids": item["input_ids"],
            "labels": item["input_ids"]
        }