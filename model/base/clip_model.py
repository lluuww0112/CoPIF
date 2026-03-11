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
        """_summary_

        Args:
            pixel_values (torch.Tensor): (batch_size, patch_num, vision_dim)

        Returns:
            patch_embeddins(torch.Tensor): (batch_size, patch_num, clip_latent_dim)
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

        # The final transformer block is reduced to the value projection only.
        last_layer = encoder_layers[-1]
        value_inputs = last_layer.layer_norm1(hidden_states)
        value_states = last_layer.self_attn.v_proj(value_inputs)
        projected_tokens = self.visual_projection(value_states)

        patch_embeddings = projected_tokens[:, 1:, :]
        return patch_embeddings



class CLIPTextModel(nn.Module):
    def __init__(self, pretrained_name: str):

        super().__init__()

        original_model = CLIPTextModelWithProjection.from_pretrained(pretrained_name)
        self.text_model = original_model.text_model
        self.text_projection = original_model.text_projection
         

    def forward(self, **kwargs):
        output = self.text_model(**kwargs)

        last_hidden_state = output.last_hidden_state
        input_ids = kwargs["input_ids"]

        eos_token_indices = input_ids.argmax(dim=-1)
        batch_indices = torch.arange(last_hidden_state.size(0), device=last_hidden_state.device)
        eos_hidden_states = last_hidden_state[batch_indices, eos_token_indices]

        return self.text_projection(eos_hidden_states)


if __name__ == "__main__":
    import hydra
    from hydra.utils import instantiate
    from typing import cast


    @hydra.main(version_base=None, config_path="../../config", config_name="base")
    def main(config : DictConfig)->None:
        visionModel = cast(
            CLIPVisionModel,
            instantiate(config.model.vision_model)
            
        )

        textModel = cast(
            CLIPTextModel,
            instantiate(config.model.text_model)
        )

        visionModel.eval()
        textModel.eval()

        batch_size = 2

        image_size = visionModel.vision_model.config.image_size
        vocab_size = textModel.text_model.config.vocab_size
        context_length = textModel.text_model.config.max_position_embeddings

        pixel_values = torch.randn(batch_size, 3, image_size, image_size)
        input_ids = torch.randint(
            low=0,
            high=vocab_size,
            size=(batch_size, context_length),
        )
        attention_mask = torch.ones(batch_size, context_length, dtype=torch.long)

        with torch.no_grad():
            image_embeddings = visionModel(pixel_values)
            text_embeddings = textModel(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

        print(f"pixel_values shape: {tuple(pixel_values.shape)}")
        print(f"input_ids shape: {tuple(input_ids.shape)}")

        print(f"image_embeddings shape: {tuple(image_embeddings.shape)}")
        print(f"text_embeddings shape: {tuple(text_embeddings.shape)}")



    main()
