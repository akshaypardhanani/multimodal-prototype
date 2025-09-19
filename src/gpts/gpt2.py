import hydra
import os
import multiprocessing
import time
import torch


from accelerate import Accelerator
from datasets import load_dataset
from omegaconf import DictConfig
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import GPT2Config, GPT2LMHeadModel, GPT2TokenizerFast, get_scheduler

from src.data.text_dataset import TextDataset
from src.utils.parameters import (
    count_parameters,
    compute_model_size,
    save_model_attributes,
)


def collate_fn(batch):
    input_ids = torch.tensor(
        [example["input_ids"] for example in batch], dtype=torch.long
    )
    labels = torch.tensor([example["labels"] for example in batch], dtype=torch.long)
    return {"input_ids": input_ids, "labels": labels}


@hydra.main(config_path="../configs", config_name="gpt2.yml")
def train(cfg: DictConfig):
    accelerate = Accelerator()
    print(f"""
    Training GPT-2 with the following configuration:
    {cfg}
    """)

    print("-------------------------------------")

    print(
        f"[Process {accelerate.process_index}] Starting training on device {accelerate.device} "
        f"(total processes = {accelerate.num_processes})"
    )

    dataset = load_dataset(
        cfg.dataset.name, cfg.dataset.variant, split=cfg.dataset.split
    )
    device = (
        torch.accelerator.current_accelerator().type
        if torch.accelerator.is_available()
        else "cpu"
    )
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
        collate_fn=collate_fn,
    )

    # Peek one batch to check shapes
    batch = next(iter(data_loader))
    print(
        f"[Process {accelerate.process_index}] Per-process batch shape: {batch['input_ids'].shape}"
    )

    global_batch_size = cfg.training.batch_size * accelerate.num_processes
    print(
        f"[Process {accelerate.process_index}] Effective global batch size: {global_batch_size}"
    )

    steps_per_epoch = len(data_loader)  # length per process
    print(
        f"[Process {accelerate.process_index}] Steps per epoch (per process): {steps_per_epoch}"
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

    save_model_attributes(model, cfg, cfg.model.save_dir)

    optimiser = AdamW(model.parameters(), lr=cfg.training.lr)
    num_training_steps = cfg.training.num_epochs * len(data_loader)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimiser,
        num_warmup_steps=int(steps_per_epoch * 0.1),
        num_training_steps=num_training_steps,
    )

    model, optimiser, data_loader, lr_scheduler = accelerate.prepare(
        model, optimiser, data_loader, lr_scheduler
    )

    loss_fn = torch.nn.CrossEntropyLoss()

    model.train()
    for epoch in range(cfg.training.num_epochs):
        epoch_start_time = time.time()
        for step, batch in enumerate(data_loader):
            step_start_time = time.time()
            inputs = batch["input_ids"].to(device)
            labels = inputs.clone().to(device)
            optimiser.zero_grad()
            with accelerate.autocast():
                outputs = model(inputs)
                logits = outputs.logits

                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = labels[:, 1:].contiguous()
                loss = loss_fn(
                    shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
                )

            accelerate.backward(loss)
            optimiser.step()
            lr_scheduler.step()

            if step % 50 == 0:
                print(
                    f"Epoch {epoch}, Step {step}, Loss {loss.item()}, step took {time.time() - step_start_time}"
                )
                print(
                    f"[Process {accelerate.process_index}] Step 0 batch shape: {batch['input_ids'].shape}"
                )
        print(f"Epoch took: {time.time() - epoch_start_time}")
        accelerate.save_state(f"{os.getcwd()}/outputs/checkpoints")
    print(f"Epoch {epoch}, Step {step}, Loss {loss.item()}")
    accelerate.wait_for_everyone()
    accelerate.unwrap_model(model).save_pretrained(cfg.model.save_dir)
    tokenizer.save_pretrained(cfg.model.save_dir)


if __name__ == "__main__":
    train()
