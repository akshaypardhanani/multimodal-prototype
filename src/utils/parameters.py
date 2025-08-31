import json

from pathlib import Path
from omegaconf import DictConfig, OmegaConf
from torch import nn

def count_parameters(model: nn.Module):
    return sum(p.numel() for p in model.parameters())

def compute_model_size(cfg: DictConfig, num_parameters: int):
    n_layer = cfg.model.n_layer
    n_embd = cfg.model.n_embd
    n_head = cfg.model.n_head
    seq_len = cfg.model.n_positions
    batch_size = cfg.training.batch_size

    params_per_block = estimate_params_per_block(n_embd, n_head)

    flops_per_token_per_layer = 2 * (n_embd ** 2) + 8 * (n_embd ** 2)
    flops_forward = flops_per_token_per_layer * seq_len * n_layer

    weight_mem_bytes = num_parameters * 2  # FP16
    optimizer_mem_bytes = weight_mem_bytes * 3
    act_mem_bytes = batch_size * seq_len * n_embd * n_layer * 2

    def to_mb(x): return round(x / (1024 ** 2), 2)
    def to_gb(x): return round(x / (1024 ** 3), 2)

    return {
        "params_total": num_parameters,
        "params_per_layer_est": params_per_block,
        "hidden_size": n_embd,
        "num_heads": n_head,
        "flops_per_forward_pass_est": flops_forward,
        "memory_estimates": {
            "weights_mb": to_mb(weight_mem_bytes),
            "optimizer_states_mb": to_mb(optimizer_mem_bytes),
            "activations_mb": to_mb(act_mem_bytes),
            "total_gb": to_gb(weight_mem_bytes + optimizer_mem_bytes + act_mem_bytes)
        }
    }

def estimate_params_per_block(n_embd: int, n_head: int):
    attention = (3 * n_embd * n_embd) + (n_embd * n_embd)

    mlp = (n_embd * 4 * n_embd) + (4 * n_embd * n_embd)

    layer_norms = 2 * n_embd

    return attention + mlp + layer_norms

def save_model_attributes(model, cfg, output_dir: str = "artifacts/models"):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    num_params = count_parameters(model)
    metrics = compute_model_size(cfg, num_params)

    attributes = {
        "config": OmegaConf.to_container(cfg, resolve=True),
        "metrics": metrics
    }
    with open(Path(output_dir) / "model_attributes.json", "w") as f:
        json.dump(attributes, f, indent=2)
    return metrics