from pathlib import Path
from src.data.prepare_data import dataset_to_artifact


def test_dataset_to_artifact():
    hf_dataset = "roneneldan/TinyStories"
    split = "train[:1%]"

    dataset_to_artifact(hf_dataset, split)
    assert Path("artifacts/TinyStories.jsonl").exists()
    