from pathlib import Path
from transformers import (
    DataCollatorForLanguageModeling,
    GPT2Config,
    GPT2LMHeadModel,
    PreTrainedTokenizerFast,
    Trainer,
    TrainingArguments,
)


from src.data.prepare_data import dataset_to_artifact, load_dataset_from_artifact
from src.tokenizers.byte_pair_encoding import load_tokenizer


class TinyGpt:
    def __init__(
        self,
        data_set: str,
        tokenizer: PreTrainedTokenizerFast = None,
        config: dict = {},
    ):
        self._split = config.pop("split", "train[:1%]")
        if Path(data_set).exists():
            self._data_set = load_dataset_from_artifact(data_set, self._split)
        else:
            self._data_set = dataset_to_artifact(
                data_set,
                split=self._split,
            )
        self._tokenizer = (
            tokenizer
            if tokenizer is not None
            else load_tokenizer("artifacts/tokenizer.json")
        )
        self._config = GPT2Config(
            vocab_size=self._tokenizer.vocab_size,
            n_layer=config.get("n_layer", 2),
            n_head=config.get("n_head", 2),
            n_embd=config.get("n_embd", 128),
            n_positions=config.get("n_positions", 128),
        )
        self.model = GPT2LMHeadModel(self._config)

    def _tokenize_fn(self, batch):
        return self._tokenizer(
            batch["text"], truncation=True, max_length=self._config.n_positions
        )

    def train(self, num_epochs: int = 1, batch_size: int = 8):
        tokenized_dataset = self._data_set.map(
            self._tokenize_fn, batched=True, remove_columns=["text"]
        )
        data_collator = DataCollatorForLanguageModeling(self._tokenizer, mlm=False)

        training_args = TrainingArguments(
            output_dir="artifacts/training",
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            save_steps=10,
            save_strategy="epoch",
            logging_dir="artifacts/logs",
            logging_steps=10,
        )

        Trainer(
            model=self.model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=tokenized_dataset,
        ).train()
        self.model.save_pretrained("artifacts/models/toy_gpt")
