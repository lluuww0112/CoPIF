import math
from typing import Optional

import torch
import torch.nn as nn


from model.base.clip_model import CLIPVisionModel, CLIPTextModel


class RECModel(nn.Module):
    def __init__(self,
        emb_dim : int,
        layer_num: int,
        head_num : int,
        dropout: float,
        image_encoder: CLIPVisionModel,
        text_encoder : CLIPTextModel,
    ):
        super().__init__()

        # caching PE
        vision_config = image_encoder.vision_model.config
        patch_size = vision_config.patch_size
        input_res = vision_config.image_size
        total_patch_num = (input_res // patch_size) ** 2
        self.register_buffer('pos_embed', self._get_sinusoidal_pe(total_patch_num + 1, emb_dim))

        # REP token
        self.rep_token = nn.Parameter(torch.zeros(1, 1, emb_dim))
        
        # pretrained backbone
        self.image_encoder = image_encoder
        self.text_encoder = text_encoder
        self._freeze_backbone()

        # x-attn decoder
        layer = nn.TransformerDecoderLayer(
            d_model=emb_dim,
            nhead=head_num,
            dim_feedforward=emb_dim*4,
            dropout=dropout,
            activation="relu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer=layer,
            num_layers=layer_num,
            norm=nn.LayerNorm(emb_dim)
        )

        # bbox header
        self.bbox_mlp = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, 4)
        )

        self._init_weights()
    

    def _get_sinusoidal_pe(self, n_position, d_model):
        pe = torch.zeros(n_position, d_model)
        position = torch.arange(0, n_position, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)

    def _freeze_backbone(self):
        for backbone in (self.image_encoder, self.text_encoder):
            backbone.eval()
            for param in backbone.parameters():
                param.requires_grad = False

    def train(self, mode: bool = True):
        super().train(mode)
        self._freeze_backbone()
        return self
    
    def _init_weights(self):
        nn.init.normal_(self.rep_token, std=0.02)

        trainable_modules = [
            self.decoder,
            self.bbox_mlp,
        ]

        for module in trainable_modules:
            for submodule in module.modules():
                if isinstance(submodule, nn.Linear):
                    nn.init.xavier_uniform_(submodule.weight)
                    if submodule.bias is not None:
                        nn.init.zeros_(submodule.bias)
                elif isinstance(submodule, nn.LayerNorm):
                    nn.init.ones_(submodule.weight)
                    nn.init.zeros_(submodule.bias)


    def _encode_image_features(
        self,
        image_features: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if (image_features is None) == (pixel_values is None):
            raise ValueError("Exactly one of image_features or pixel_values must be provided.")

        if pixel_values is not None:
            with torch.no_grad():
                return self.image_encoder(pixel_values)

        return image_features

    def _encode_text_features(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        with torch.no_grad():
            return self.text_encoder.forward_full_sequence(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
    
    def _decoder_forward(self, image_features: torch.Tensor, text_features: torch.Tensor, attention_mask: torch.Tensor):
        attention_mask = ~attention_mask.bool()
        decoder_output = self.decoder.forward(
            tgt=image_features,
            memory=text_features,
            memory_key_padding_mask=attention_mask
        )

        return decoder_output

    def _bbox_header_forward(self, rep_features : torch.Tensor):
        logits = self.bbox_mlp(rep_features)
        pred_boxes = torch.sigmoid(logits)
        pred_boxes = (pred_boxes * 1.2) - 0.1 
        
        if not self.training:
            pred_boxes = torch.clamp(pred_boxes, 0.0, 1.0)
        return pred_boxes

    def _predict_from_image_features(
        self,
        image_features: torch.Tensor,
        text_features: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        B = image_features.shape[0]

        # add rep token & PE
        image_features = torch.cat([self.rep_token.expand(B, -1, -1), image_features], dim=1)
        image_features += self.pos_embed

        # decoder forward
        decoder_output = self._decoder_forward(
            image_features=image_features,
            text_features=text_features,
            attention_mask=attention_mask,
        )

        # bbox header forward
        rep_features = decoder_output[:, 0, :]
        return self._bbox_header_forward(rep_features)

    def invoke(self, pixel_values: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        return self.forward(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

    
    def forward(
        self,
        image_features: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
    ):
        if input_ids is None or attention_mask is None:
            raise ValueError("input_ids and attention_mask must be provided.")

        encoded_image_features = self._encode_image_features(
            image_features=image_features,
            pixel_values=pixel_values,
        )
        text_features = self._encode_text_features(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        return self._predict_from_image_features(
            image_features=encoded_image_features,
            text_features=text_features,
            attention_mask=attention_mask,
        )


if __name__ == "__main__":
    from types import SimpleNamespace

    class _DummyVisionEncoder(nn.Module):
        def __init__(self, emb_dim: int, image_size: int, patch_size: int):
            super().__init__()
            self.vision_model = SimpleNamespace(
                config=SimpleNamespace(image_size=image_size, patch_size=patch_size)
            )
            self.emb_dim = emb_dim

        def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
            batch_size = pixel_values.shape[0]
            grid_size = self.vision_model.config.image_size // self.vision_model.config.patch_size
            patch_count = grid_size ** 2
            return torch.randn(batch_size, patch_count, self.emb_dim, device=pixel_values.device)

    class _DummyTextEncoder(nn.Module):
        def __init__(self, emb_dim: int):
            super().__init__()
            self.emb_dim = emb_dim

        def forward_full_sequence(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
            batch_size, seq_len = input_ids.shape
            return torch.randn(batch_size, seq_len, self.emb_dim, device=input_ids.device)

    torch.manual_seed(0)

    batch_size = 2
    emb_dim = 32
    image_size = 32
    patch_size = 8
    seq_len = 10

    model = RECModel(
        emb_dim=emb_dim,
        layer_num=2,
        head_num=4,
        dropout=0.1,
        image_encoder=_DummyVisionEncoder(emb_dim=emb_dim, image_size=image_size, patch_size=patch_size),
        text_encoder=_DummyTextEncoder(emb_dim=emb_dim),
    )
    model.eval()

    pixel_values = torch.randn(batch_size, 3, image_size, image_size)
    input_ids = torch.randint(0, 100, (batch_size, seq_len))
    attention_mask = torch.tensor(
        [
            [1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        ],
        dtype=torch.long,
    )

    with torch.no_grad():
        pred_boxes = model.invoke(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

    print(f"pixel_values shape: {tuple(pixel_values.shape)}")
    print(f"input_ids shape: {tuple(input_ids.shape)}")
    print(f"attention_mask shape: {tuple(attention_mask.shape)}")
    print(f"pred_boxes shape: {tuple(pred_boxes.shape)}")
    print(f"pred_boxes sample: {pred_boxes[0].tolist()}")
