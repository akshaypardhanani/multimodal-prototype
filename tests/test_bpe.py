from pathlib import Path
from src.tokenizers.byte_pair_encoding import tokenize, load_tokenizer


def test_tokenize():
    vocabulary_size = 2000
    source_dataset = "artifacts/TinyStories.jsonl"

    tokenize(vocabulary_size, source_dataset)
    assert Path("artifacts/tokenizer.json").exists()


def test_load_tokenizer():
    pre_trained_tokenizer_path = "artifacts/tokenizer.json"
    tok = load_tokenizer(pre_trained_tokenizer_path)
    assert tok is not None
