import multiprocessing

from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase


class TextDataset(Dataset):
    def __init__(self, dataset_name: str, split: str, tokenizer: PreTrainedTokenizerBase, block_size: int = 512):
        self.dataset = load_dataset(dataset_name, split=split)

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
            concatenated = sum(examples["input_ids"], [])
            total_length = (len(concatenated) // block_size) * block_size
            return {
                "input_ids": [
                    concatenated[i : i + block_size]
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