import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from transformers import CLIPVisionModel, CLIPTextModel



class REC_model(nn.Module):
    def __init__(self,
            patch_num: int,
            image_emb_dim: int, 
            text_emb_dim: int,
            head_num: int,
            layer_num: int,
            pretrained_clip: str,
            dropout: float = 0.1
        ):
        super().__init__()
        
        self.image_emb_dim = image_emb_dim
        
        self.clip_text_encoder = CLIPTextModel.from_pretrained(pretrained_clip)
        self.clip_image_encoder = CLIPVisionModel.from_pretrained(pretrained_clip)
        # freeze backbone
        for param in self.clip_text_encoder.parameters():
            param.requires_grad = False
        for param in self.clip_image_encoder.parameters():
            param.requires_grad = False

        # text_emb_dim -> image_emb_dim
        self.text_proj = nn.Sequential(
            nn.Linear(text_emb_dim, image_emb_dim),
            nn.LayerNorm(image_emb_dim),
            nn.GELU(),
            nn.Linear(image_emb_dim, image_emb_dim)
        )
        
        # caching PE map
        self.register_buffer('pos_embed', self._get_sinusoidal_pe(patch_num, image_emb_dim))
        # rep token
        self.rep_token = nn.Parameter(torch.zeros(1, 1, image_emb_dim))
        # decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=image_emb_dim,
            nhead=head_num,
            dim_feedforward=image_emb_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer=decoder_layer,
            num_layers=layer_num,
            norm=nn.LayerNorm(image_emb_dim)
        )
        # bbox
        self.box_head = nn.Sequential(
            nn.Linear(image_emb_dim, image_emb_dim),
            nn.ReLU(),
            nn.Linear(image_emb_dim, 4)
        )

        self._init_weights()

    def _get_sinusoidal_pe(self, n_position, d_model):
        pe = torch.zeros(n_position, d_model)
        position = torch.arange(0, n_position, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        
        nn.init.normal_(self.rep_token, std=0.02)
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, 
            contexts: torch.Tensor, # text ids (HF preprocessor)
            context_padding_masks: torch.Tensor, # text attention mask (HF preprocessor)
            image_features: torch.Tensor | None = None, # DiT output (train) | Image Backbone (inference)
            image_free=True
        ):
        
        # image free가 아닌 경우 clip.image_encoder 사용 
        if not image_free:
            with torch.no_grad():
                image_output = self.clip_image_encoder(pixel_values=image_features)
                image_features = image_output.last_hidden_state[:, 1:, :]
        
        # get text feature (referring expression) & padding mask
        with torch.no_grad():
            text_outputs = self.clip_text_encoder(
                input_ids=contexts,
                attention_mask=context_padding_masks
            )
            text_features = text_outputs.last_hidden_state
        context_padding_masks = (context_padding_masks == 0)
        
        # prepare Memory (text)
        memory = self.text_proj(text_features)

        # prepare Target (iamge), add REP & PE
        B = image_features.shape[0]
        tgt = image_features + self.pos_embed 
        rep_tokens = self.rep_token.expand(B, -1, -1)  
        tgt = torch.cat((rep_tokens, tgt), dim=1)      

        # decoder forwarding
        out = self.decoder(
            tgt=tgt, # image
            memory=memory, # referring expression
            memory_key_padding_mask=context_padding_masks
        )
        rep_out = out[:, 0, :]

        # bbox predidction
        logits = self.box_head(rep_out)
        # post-processing (scale & shift)
        pred_boxes = torch.sigmoid(logits)
        pred_boxes = (pred_boxes * 1.2) - 0.1 
        
        if not self.training:
            pred_boxes = torch.clamp(pred_boxes, 0.0, 1.0)
        return pred_boxes