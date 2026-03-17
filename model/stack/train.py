from __future__ import annotations

from pathlib import Path

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

import torch
import torch.nn.functional as F
from transformers import CLIPTokenizer

from model.base.train import (
    GeneratedFeatureMapDataset,
    RECTrainer as BaseRECTrainer,
    TrainingMonitorCallback,
    _build_training_arguments,
    _feature_map_manifest_path,
    _generalized_box_iou_loss,
)
from model.stack.clip_model import CLIPTextModel, CLIPVisionModel
from model.stack.preprocessing import TrainCollator
from model.stack.rec_model import RECModel


def _build_rec_model(
    config: DictConfig,
    vision_model: CLIPVisionModel,
    text_model: CLIPTextModel,
) -> RECModel:
    rec_config = config.model.get("RECModel", {})

    emb_dim = int(vision_model.config.projection_dim)
    layer_num = int(rec_config.get("layer_num", 12))
    head_num = int(rec_config.get("head_num", 8))
    dropout = float(rec_config.get("dropout", 0.1))

    if emb_dim % head_num != 0:
        raise ValueError(f"emb_dim={emb_dim} must be divisible by head_num={head_num}.")

    return RECModel(
        emb_dim=emb_dim,
        layer_num=layer_num,
        head_num=head_num,
        dropout=dropout,
        image_encoder=vision_model,
        text_encoder=text_model,
    )


class RECTrainer(BaseRECTrainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        bbox_targets = inputs["bbox_targets"]
        pred_boxes = model(
            image_features=inputs["image_features"],
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            phrase_mask=inputs["phrase_mask"],
        )
        loss_bbox = F.l1_loss(pred_boxes, bbox_targets)
        loss_giou = _generalized_box_iou_loss(pred_boxes, bbox_targets)
        loss = (self.bbox_loss_coef * loss_bbox) + (self.giou_loss_coef * loss_giou)
        self.latest_loss_components = {
            "loss": float(loss.detach().cpu().item()),
            "loss_bbox": float(loss_bbox.detach().cpu().item()),
            "loss_giou": float(loss_giou.detach().cpu().item()),
        }

        if return_outputs:
            return loss, {
                "pred_boxes": pred_boxes,
                "loss_bbox": loss_bbox.detach(),
                "loss_giou": loss_giou.detach(),
            }
        return loss


@hydra.main(version_base=None, config_path="../../config", config_name="stack")
def main(config: DictConfig) -> None:
    train_config = config.get("train", {})
    preprocessing_config = config.get("preprocessing", {})
    requested_device = str(train_config.get("device", config.generator.device))
    batch_size = int(train_config.get("batch_size", 8))
    num_epochs = int(train_config.get("num_epochs", 1))
    learning_rate = float(train_config.get("learning_rate", 1e-4))
    weight_decay = float(train_config.get("weight_decay", 1e-4))
    num_workers = int(train_config.get("num_workers", 0))
    grad_clip = float(train_config.get("grad_clip", 1.0))
    log_every = int(train_config.get("log_every", 10))
    max_steps = int(train_config.get("max_steps", 0))
    noise_scale = float(preprocessing_config.get("noise_scale", config.shared.get("noise_scale", 0.0)))
    max_phrase_count = int(preprocessing_config.get("max_phrase_count", 8))
    checkpoint_dir = Path(str(train_config.get("checkpoint_dir", "checkpoints")))
    save_steps = int(train_config.get("save_steps", 500))
    save_total_limit = int(train_config.get("save_total_limit", 2))
    warmup_steps = int(train_config.get("warmup_steps", 0))
    gradient_accumulation_steps = int(train_config.get("gradient_accumulation_steps", 1))
    bbox_loss_coef = float(train_config.get("bbox_loss_coef", 5.0))
    giou_loss_coef = float(train_config.get("giou_loss_coef", 2.0))
    seed = int(train_config.get("seed", 42))
    resume_from_checkpoint = train_config.get("resume_from_checkpoint", None)

    manifest_path = _feature_map_manifest_path(config)
    tokenizer = CLIPTokenizer.from_pretrained(config.shared.pretrained_clip)

    vision_model = instantiate(config.model.vision_model)
    text_model = instantiate(config.model.text_model)
    model = _build_rec_model(config, vision_model=vision_model, text_model=text_model)

    expected_patch_count = model.pos_embed.shape[1] - 1
    dataset = GeneratedFeatureMapDataset(manifest_path=manifest_path)
    collator = TrainCollator(
        tokenizer=tokenizer,
        expected_patch_count=expected_patch_count,
        noise_scale=noise_scale,
        max_phrase_count=max_phrase_count,
    )

    print(f"manifest_path: {manifest_path}")
    print(
        "train_data: "
        f"{train_config.get('dataset', config.generator.dataset)}/"
        f"{train_config.get('splitby', config.generator.splitby)}/"
        f"{train_config.get('split', config.generator.split)}"
    )
    print(f"requested_device: {requested_device}")
    print(f"dataset size: {len(dataset)}")
    print(f"batch_size: {batch_size}")
    print(f"num_epochs: {num_epochs}")
    print(f"noise_scale: {noise_scale}")
    print(f"max_phrase_count: {max_phrase_count}")
    print(f"expected_patch_count: {expected_patch_count}")
    print(f"bbox_loss_coef: {bbox_loss_coef}")
    print(f"giou_loss_coef: {giou_loss_coef}")
    if requested_device == "cuda" and not torch.cuda.is_available():
        print("cuda is not available. Trainer will run on cpu.")
    print(f"monitor_log_path: {checkpoint_dir / 'training_log.jsonl'}")
    print(f"monitor_plot_path: {checkpoint_dir / 'loss_curve.png'}")

    training_args = _build_training_arguments(
        checkpoint_dir=checkpoint_dir,
        batch_size=batch_size,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_steps=warmup_steps,
        grad_clip=grad_clip,
        num_workers=num_workers,
        log_every=log_every,
        save_steps=save_steps,
        save_total_limit=save_total_limit,
        gradient_accumulation_steps=gradient_accumulation_steps,
        max_steps=max_steps,
        requested_device=requested_device,
        seed=seed,
    )

    optimizer = torch.optim.AdamW(
        (param for param in model.parameters() if param.requires_grad),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    trainer = RECTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
        optimizers=(optimizer, None),
        bbox_loss_coef=bbox_loss_coef,
        giou_loss_coef=giou_loss_coef,
        callbacks=[TrainingMonitorCallback(checkpoint_dir)],
    )

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    trainer.save_model()
    trainer.save_state()


if __name__ == "__main__":
    main()
