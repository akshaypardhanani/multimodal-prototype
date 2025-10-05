from typing import Optional
import torch
import torch.nn as nn


from transformers import GPT2LMHeadModel


from src.models.vision_encoder import VisionEncoder
from src.models.multi_modal_gpt_config import MultiModalGPTConfig
from src.models.fusion_model import FusionModel


class MultiModalGPT(nn.Module):
    def __init__(self, config: MultiModalGPTConfig):
        super().__init__()
        self.config = config

        self.vision_encoder = VisionEncoder(
            image_size=config.img_size,
            patch_size=config.patch_size,
            embed_dim=config.vision_embed_dim,
            n_layers=config.vision_layers,
            n_heads=config.vision_heads,
        )

        self.text_model = GPT2LMHeadModel(config)

        self.fusion = FusionModel(config, fusion_type="projection")
        self.max_vision_tokens = (config.img_size // config.patch_size) ** 2 + 1
        self.extended_position_embeddings = nn.Parameter(
            torch.zeros(1, self.max_vision_tokens + config.n_positions, config.n_embd)
        )
        self._init_extended_positions()

    def _init_extended_positions(self):
        original_pos_embeddings = self.text_model.transformer.wpe.weight.data
        seq_length, embed_dim = original_pos_embeddings.shape

        with torch.no_grad():
            torch.nn.init.normal_(
                self.extended_position_embeddings[:, : self.max_vision_tokens, :],
                std=0.02,
            )
            self.extended_position_embeddings[
                :, self.max_vision_tokens : self.max_vision_tokens + seq_length, :
            ] = original_pos_embeddings

    def get_vision_features(self, images: torch.Tensor):
        return self.vision_encoder(images, return_all_tokens=True)

    def prepare_multimodal_inputs(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        batch_size = images.size(0)
        vision_features = self.get_vision_features(images)
        text_embeddings = self.text_model.transformer.wte(input_ids)
        combined_embeddings = self.fusion(vision_features, text_embeddings)

        if attention_mask is not None:
            vision_mask = torch.ones(
                batch_size, vision_features.size(1), device=images.device
            )
            extended_attention_mask = torch.cat([vision_mask, attention_mask], dim=1)
        else:
            extended_attention_mask = None

        seq_len = combined_embeddings.size(1)
        position_embeddings = self.extended_position_embeddings[:, :seq_len, :]
        combined_embeddings = combined_embeddings + position_embeddings

        return combined_embeddings, extended_attention_mask

    def forward(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        combined_embeddings, extended_attention_mask = self.prepare_multimodal_inputs(images, input_ids, attention_mask)

        transformer_output = self.text_model.transformer(inputs_embeds=combined_embeddings, attention_mask=extended_attention_mask, **kwargs)
        hidden_states = transformer_output[0]

        # vision_tokens = (self.config.img_size // self.config.patch_size) ** 2 + 1
        text_hidden_states = hidden_states[:, self.max_vision_tokens:, :]

        lm_logits = self.text_model.lm_head(text_hidden_states)

        loss = None

        if labels is not None:
            if lm_logits.size(1) > 0:
                shift_logits = lm_logits[..., :-1, :].contiguous()
                shift_labels = labels [..., 1:].contiguous()

                loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
                loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            else:
                loss = torch.tensor(0.0, device=images.device, requires_grad=True)

        return {
            "loss": loss,
            "logits": lm_logits,
            "hidden_states": hidden_states,
            "vision_tokens": self.get_vision_features(images),
        }

    def generate(
        self, 
        images: torch.Tensor,
        input_ids: torch.Tensor,
        max_length: int = 50,
        temperature: float = 1.0,
        do_sample: bool = True,
        top_k: int = 50,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        self.eval()
        batch_size = images.size(0)
        device = images.device

        with torch.no_grad():
            generated_ids = input_ids.clone()
            for _ in range(max_length):
                outputs = self.forward(
                    images=images,
                    input_ids=generated_ids,
                    attention_mask=attention_mask,
                )
                next_token_logits = outputs["logits"][:, -1, :] / temperature

                if do_sample:
                    if top_k > 0:
                        top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k)
                        probs = torch.softmax(top_k_logits, dim=-1)
                        next_token_indices = torch.multinomial(probs, num_samples=1)
                        next_token = top_k_indices.gather(-1, next_token_indices)
                    else:
                        probs = torch.softmax(next_token_logits, dim=-1)
                        next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                generated_ids = torch.cat([generated_ids, next_token], dim=-1)

                if attention_mask is not None:
                    attention_mask = torch.cat([
                        attention_mask,
                        torch.ones(batch_size, 1, device=device),
                    ], dim=-1)

        return generated_ids
    
    def resize_token_embeddings(self, new_num_tokens: int):
        self.text_model.resize_token_embeddings(new_num_tokens)

    def get_inputs_embeddings(self):
        return self.text_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.text_model.set_input_embeddings(value)
