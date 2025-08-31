import hydra
import multiprocessing
import torch


from datasets import load_dataset
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from transformers import GPT2Config, GPT2LMHeadModel, GPT2TokenizerFast

from src.data.text_dataset import TextDataset
from src.utils.parameters import count_parameters, compute_model_size, save_model_attributes

@hydra.main(config_path="../configs", config_name="gpt2.yml")
def train(cfg: DictConfig):
    print(f"""
    Training GPT-2 with the following configuration:
    {cfg}
    """)

    dataset = load_dataset(cfg.dataset.name, cfg.dataset.variant, split=cfg.dataset.split)
    texts = dataset["text"]
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    data_set = TextDataset(
        dataset_name=dataset,
        split=cfg.dataset.split,
        tokenizer=tokenizer,
        block_size=cfg.model.n_positions,
    )

    data_loader = DataLoader(
        data_set,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=multiprocessing.cpu_count(),
    )

    model_cfg = GPT2Config(
        vocab_size=len(tokenizer),
        n_layer=cfg.model.n_layer,
        n_head=cfg.model.n_head,
        n_embd=cfg.model.n_embd,
        n_positions=cfg.model.n_positions,
    )

    model = GPT2LMHeadModel(model_cfg)

    model.resize_token_embeddings(len(tokenizer))
    if torch.accelerator.is_available():
        model.to(torch.accelerator.current_accelerator().type)
    else:
        model.to("cpu")

    # num_params = count_parameters(model)
    # metrics = compute_model_size(cfg, num_params)
    save_model_attributes(model, cfg, cfg.model.save_dir)

if __name__ == "__main__":
    train()
    