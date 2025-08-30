import os


from datasets import load_dataset, Dataset
from pathlib import Path


def _artifact_dir(directory: str) -> bool:
    """
    Takes a directory which corresponds to a path relative to the src directory.
    A different parent may be specified however an absolute path must be passed in then.
    """
    try:
        Path(directory).mkdir(parents=True, exist_ok=True)
    except Exception as e:
        raise e
    return True


def dataset_to_artifact(
    hf_dataset: str, split: str, artifacts_directory: str = "artifacts"
):
    data_set = load_dataset(hf_dataset, split=split)
    if _artifact_dir(artifacts_directory):
        data_set.to_json(f"{artifacts_directory}/{Path(hf_dataset).stem}.jsonl")
    else:
        raise RuntimeError("Artifacts directory not created")


def load_dataset_from_artifact(artifact_path: str, split: str) -> Dataset:
    _, ext = os.path.splitext(artifact_path)
    if ext.startswith(".json"):
        loader_type = "json"
    else:
        loader_type = ext.replace(".", "")
    return load_dataset(loader_type, data_files=artifact_path, split=split)
