import pytest


from pathlib import Path
from src.gpts.tiny_gpt import TinyGpt

@pytest.fixture
def tiny_gpt():
    return TinyGpt(data_set="./artifacts/TinyStories.jsonl")
    
def test_train(tiny_gpt):
    tiny_gpt.train()
    assert Path("artifacts/models/toy_gpt").exists()