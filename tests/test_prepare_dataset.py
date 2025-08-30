from pathlib import Path
from src.data.prepare_data import dataset_to_artifact, load_dataset_from_artifact


def test_dataset_to_artifact():
    hf_dataset = "roneneldan/TinyStories"
    split = "train[:1%]"

    dataset_to_artifact(hf_dataset, split)
    assert Path("artifacts/TinyStories.jsonl").exists()


def test_load_dataset_from_artifact():
    artifact_path = "artifacts/TinyStories.jsonl"
    dataset = load_dataset_from_artifact(artifact_path, "train[:1%]")
    assert dataset is not None
