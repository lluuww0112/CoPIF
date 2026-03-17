from typing import Any

from omegaconf import DictConfig

import torch
import torch.nn as nn

from transformers import CLIPVisionModelWithProjection, CLIPTextModelWithProjection



class CLIPVisionModel(nn.Module):
    def __init__(self, pretrained_name: str):
        super().__init__()

        original_model = CLIPVisionModelWithProjection.from_pretrained(pretrained_name)

        self.model = original_model
        self.vision_model = original_model.vision_model
        self.visual_projection = original_model.visual_projection
        self.config = original_model.config

    def forward(
        self,
        pixel_values: torch.Tensor,
    ) -> torch.Tensor:
        """
        Dense patch embeddings adapted from the last CLIP ViT block.

        The final block is truncated to the value pathway used by MaskCLIP:
        apply the block input LayerNorm, keep only the value projection, then
        pass it through the attention output projection. The last block's
        second LayerNorm and MLP are skipped for dense patch extraction.
        Patch tokens are finally projected to the CLIP latent space.
        """
        
        hidden_states = self.vision_model.embeddings(pixel_values=pixel_values)
        hidden_states = self.vision_model.pre_layrnorm(hidden_states)

        encoder_layers = self.vision_model.encoder.layers
        for encoder_layer in encoder_layers[:-1]:
            layer_outputs = encoder_layer(
                hidden_states=hidden_states,
                attention_mask=None,
                causal_attention_mask=None,
                output_attentions=False,
            )
            if isinstance(layer_outputs, tuple):
                hidden_states = layer_outputs[0]
            else:
                hidden_states = layer_outputs

        # For the last ViT block, keep only the value pathway from the final
        # attention layer for dense patch extraction.
        last_layer = encoder_layers[-1]

        hidden_states = last_layer.layer_norm1(hidden_states)
        hidden_states = last_layer.self_attn.v_proj(hidden_states)
        hidden_states = last_layer.self_attn.out_proj(hidden_states)

        # projection to image-text latent space
        projected_tokens = self.visual_projection(hidden_states)
        patch_embeddings = projected_tokens[:, 1:, :]
        return patch_embeddings



class CLIPTextModel(nn.Module):
    def __init__(self, pretrained_name: str):

        super().__init__()

        original_model = CLIPTextModelWithProjection.from_pretrained(pretrained_name)
        self.text_model = original_model.text_model
        self.text_projection = original_model.text_projection
        self.eos_token_id = self.text_model.config.eos_token_id
        self.pad_token_id = self.text_model.config.pad_token_id

    def _resolve_attention_mask(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """Infer valid-token positions from padding when no mask is provided."""
        if attention_mask is not None:
            return attention_mask
        if self.pad_token_id is None:
            return torch.ones_like(input_ids, dtype=torch.long)
        return input_ids.ne(self.pad_token_id).long()

    def _get_eos_token_indices(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        fallback_indices = attention_mask.long().sum(dim=-1).sub(1).clamp(min=0)
        if self.eos_token_id is None:
            return fallback_indices

        eos_mask = input_ids.eq(self.eos_token_id)
        has_eos = eos_mask.any(dim=-1)
        eos_indices = eos_mask.int().argmax(dim=-1)
        return torch.where(has_eos, eos_indices, fallback_indices)

    def _prepare_phrase_inputs(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Normalize text inputs to `(batch, phrase_count, context_length)`.

        A 2D input is treated as one phrase per sample. A 3D input is assumed
        to already contain pre-split word phrases for each sample.
        """
        if input_ids.ndim == 2:
            return input_ids.unsqueeze(1), attention_mask.unsqueeze(1)
        if input_ids.ndim == 3:
            if attention_mask.ndim != 3:
                raise ValueError("attention_mask must have shape (batch, phrase_count, context_length).")
            return input_ids, attention_mask
        raise ValueError("input_ids must have shape (batch, context_length) or (batch, phrase_count, context_length).")

    def _encode_phrase_sequence(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encode each phrase independently and return `(batch, phrase_count, emb_dim)`.

        Internally this flattens `(batch, phrase_count, context_length)` into
        `(batch * phrase_count, context_length)` so every phrase uses the
        standard CLIP EOS embedding path.
        """
        phrase_input_ids, phrase_attention_mask = self._prepare_phrase_inputs(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        batch_size, phrase_count, context_length = phrase_input_ids.shape
        flat_input_ids = phrase_input_ids.reshape(batch_size * phrase_count, context_length)
        flat_attention_mask = phrase_attention_mask.reshape(batch_size * phrase_count, context_length)

        output = self.text_model(
            input_ids=flat_input_ids,
            attention_mask=flat_attention_mask,
        )
        last_hidden_state = output.last_hidden_state
        eos_token_indices = self._get_eos_token_indices(
            input_ids=flat_input_ids,
            attention_mask=flat_attention_mask,
        )
        flat_indices = torch.arange(last_hidden_state.size(0), device=last_hidden_state.device)
        flat_embeddings = self.text_projection(last_hidden_state[flat_indices, eos_token_indices])
        return flat_embeddings.reshape(batch_size, phrase_count, -1)

    def forward(self, **kwargs):
        """
        Encode one phrase per sample and return `(batch, emb_dim)`.

        This path is only for `input_ids` shaped `(batch, context_length)`.
        For pre-split word phrases shaped `(batch, phrase_count, context_length)`,
        use `forward_full_sequence()`.
        """
        input_ids = kwargs["input_ids"]
        attention_mask = self._resolve_attention_mask(
            input_ids=input_ids,
            attention_mask=kwargs.get("attention_mask"),
        )

        phrase_sequence = self._encode_phrase_sequence(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        if input_ids.ndim == 3:
            raise ValueError(
                "forward() expects a single phrase per sample with shape (batch, context_length). "
                "Use forward_full_sequence() for pre-split word inputs with shape "
                "(batch, phrase_count, context_length)."
            )
        return phrase_sequence[:, 0, :]

    def forward_full_sequence(self, **kwargs):
        """
        Encode pre-split word phrases and return `(batch, phrase_count, emb_dim)`.

        The input order is preserved, so if preprocessing supplies phrases like
        `["guy", "on", "cell"]`, the output sequence follows the same order.
        """
        input_ids = kwargs["input_ids"]
        attention_mask = self._resolve_attention_mask(
            input_ids=input_ids,
            attention_mask=kwargs.get("attention_mask"),
        )

        return self._encode_phrase_sequence(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        
