from tokenizers import Tokenizer, models, trainers, pre_tokenizers
from transformers import PreTrainedTokenizerFast


def tokenize(
    vocabulary_size: int, source_dataset: str, artifacts_directory: str = "artifacts"
):
    tok = Tokenizer(models.BPE(unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=vocabulary_size,
        show_progress=True,
        special_tokens=["<unk>", "<pad>", "<bos>", "<eos>"],
    )
    tok.train([source_dataset], trainer)
    tok.save(f"{artifacts_directory}/tokenizer.json")


def load_tokenizer(
    pre_trained_tokenizer_path: str = "artifacts/tokenizer.json",
    bos_token: str = "<bos>",
    eos_token: str = "<eos>",
    pad_token: str = "<pad>",
) -> PreTrainedTokenizerFast:
    return PreTrainedTokenizerFast(
        tokenizer_file=pre_trained_tokenizer_path,
        bos_token=bos_token,
        eos_token=eos_token,
        pad_token=pad_token,
    )
