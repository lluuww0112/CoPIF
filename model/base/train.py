from __future__ import annotations

import bisect
import inspect
import json
from pathlib import Path
from typing import Any

import hydra
from hydra.utils import get_original_cwd, instantiate
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from omegaconf import DictConfig

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import CLIPTokenizer
from transformers import Trainer, TrainerCallback, TrainingArguments

from generation.base.generate import SaveModel
from model.base.clip_model import CLIPTextModel, CLIPVisionModel
from model.base.rec_model import RECModel


def _torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _feature_map_manifest_path(config: DictConfig) -> Path:
    project_root = Path(get_original_cwd())
    train_config = config.get("train", {})
    dataset = str(train_config.get("dataset", config.generator.dataset))
    splitby = str(train_config.get("splitby", config.generator.splitby))
    split = str(train_config.get("split", config.generator.split))
    return (
        project_root
        / "data"
        / "generated"
        / dataset
        / splitby
        / split
        / "feature_maps.pt"
    )


class GeneratedFeatureMapDataset(Dataset):
    def __init__(self, manifest_path: Path):
        if not manifest_path.is_file():
            raise FileNotFoundError(f"feature map manifest not found: {manifest_path}")

        manifest = _torch_load(manifest_path)
        self.manifest_path = manifest_path
        self.chunk_paths = [manifest_path.parent / chunk_name for chunk_name in manifest["chunks"]]
        self.total_saved = int(manifest["total_saved"])

        self.chunk_sizes: list[int] = []
        self.chunk_offsets: list[int] = [0]
        for chunk_path in self.chunk_paths:
            chunk_data = _torch_load(chunk_path)
            chunk_size = len(chunk_data)
            self.chunk_sizes.append(chunk_size)
            self.chunk_offsets.append(self.chunk_offsets[-1] + chunk_size)

        if self.chunk_offsets[-1] != self.total_saved:
            raise ValueError(
                f"manifest total_saved={self.total_saved} does not match "
                f"actual chunk size sum={self.chunk_offsets[-1]}"
            )

        self._cached_chunk_index: int | None = None
        self._cached_chunk_data: list[SaveModel] | None = None

    def __len__(self) -> int:
        return self.total_saved

    def _load_chunk(self, chunk_index: int) -> list[SaveModel]:
        if self._cached_chunk_index == chunk_index and self._cached_chunk_data is not None:
            return self._cached_chunk_data

        chunk_data = _torch_load(self.chunk_paths[chunk_index])
        self._cached_chunk_index = chunk_index
        self._cached_chunk_data = chunk_data
        return chunk_data

    def __getitem__(self, index: int) -> dict[str, Any]:
        if index < 0 or index >= self.total_saved:
            raise IndexError(f"index out of range: {index}")

        chunk_index = bisect.bisect_right(self.chunk_offsets, index) - 1
        local_index = index - self.chunk_offsets[chunk_index]
        sample = self._load_chunk(chunk_index)[local_index]

        return {
            "annotation": sample.annotation,
            "feature_map": sample.feature_map.float(),
            "bbox": torch.tensor(
                [sample.bbox.x, sample.bbox.y, sample.bbox.w, sample.bbox.h],
                dtype=torch.float32,
            ),
            "image_path": sample.image_path,
        }


class TrainCollator:
    def __init__(
        self,
        tokenizer: CLIPTokenizer,
        expected_patch_count: int,
        noise_scale: float,
    ):
        self.tokenizer = tokenizer
        self.expected_patch_count = expected_patch_count
        self.noise_scale = noise_scale

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        annotations = [sample["annotation"] for sample in batch]
        tokenized = self.tokenizer(
            annotations,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )

        flattened_feature_maps = []
        bbox_targets = []

        for sample in batch:
            feature_map = sample["feature_map"]
            if feature_map.ndim != 3:
                raise ValueError(
                    "feature_map must have shape (grid_size, grid_size, emb_dim), "
                    f"got {tuple(feature_map.shape)}."
                )

            patch_count = feature_map.shape[0] * feature_map.shape[1]
            if patch_count != self.expected_patch_count:
                raise ValueError(
                    f"flattened patch count must match expected_patch_count={self.expected_patch_count}, "
                    f"got {patch_count} from shape {tuple(feature_map.shape)}."
                )

            flattened_feature_maps.append(feature_map.reshape(patch_count, feature_map.shape[-1]))
            bbox_targets.append(sample["bbox"])

        image_features = torch.stack(flattened_feature_maps, dim=0)
        if self.noise_scale > 0:
            image_features = image_features + (torch.randn_like(image_features) * self.noise_scale)

        return {
            "image_features": image_features,
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "bbox_targets": torch.stack(bbox_targets, dim=0),
        }


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


def _box_xywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    x, y, w, h = boxes.unbind(dim=-1)
    w = torch.clamp(w, min=0.0)
    h = torch.clamp(h, min=0.0)
    return torch.stack((x, y, x + w, y + h), dim=-1)


def _box_area(boxes: torch.Tensor) -> torch.Tensor:
    widths = torch.clamp(boxes[:, 2] - boxes[:, 0], min=0.0)
    heights = torch.clamp(boxes[:, 3] - boxes[:, 1], min=0.0)
    return widths * heights


def _generalized_box_iou_loss(pred_boxes: torch.Tensor, target_boxes: torch.Tensor) -> torch.Tensor:
    pred_xyxy = _box_xywh_to_xyxy(pred_boxes)
    target_xyxy = _box_xywh_to_xyxy(target_boxes)

    inter_x1 = torch.max(pred_xyxy[:, 0], target_xyxy[:, 0])
    inter_y1 = torch.max(pred_xyxy[:, 1], target_xyxy[:, 1])
    inter_x2 = torch.min(pred_xyxy[:, 2], target_xyxy[:, 2])
    inter_y2 = torch.min(pred_xyxy[:, 3], target_xyxy[:, 3])

    inter_w = torch.clamp(inter_x2 - inter_x1, min=0.0)
    inter_h = torch.clamp(inter_y2 - inter_y1, min=0.0)
    inter_area = inter_w * inter_h

    pred_area = _box_area(pred_xyxy)
    target_area = _box_area(target_xyxy)
    union_area = pred_area + target_area - inter_area
    iou = inter_area / union_area.clamp(min=1e-6)

    enc_x1 = torch.min(pred_xyxy[:, 0], target_xyxy[:, 0])
    enc_y1 = torch.min(pred_xyxy[:, 1], target_xyxy[:, 1])
    enc_x2 = torch.max(pred_xyxy[:, 2], target_xyxy[:, 2])
    enc_y2 = torch.max(pred_xyxy[:, 3], target_xyxy[:, 3])
    enc_area = torch.clamp(enc_x2 - enc_x1, min=0.0) * torch.clamp(enc_y2 - enc_y1, min=0.0)

    giou = iou - ((enc_area - union_area) / enc_area.clamp(min=1e-6))
    return (1.0 - giou).mean()


class RECTrainer(Trainer):
    def __init__(self, *args, bbox_loss_coef: float = 5.0, giou_loss_coef: float = 2.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.bbox_loss_coef = bbox_loss_coef
        self.giou_loss_coef = giou_loss_coef
        self.latest_loss_components: dict[str, float] = {}

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        bbox_targets = inputs["bbox_targets"]
        pred_boxes = model(
            image_features=inputs["image_features"],
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
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

    def log(self, logs: dict[str, float], *args, **kwargs) -> None:
        merged_logs = dict(logs)
        for key in ("loss_bbox", "loss_giou"):
            if key in self.latest_loss_components:
                merged_logs[key] = self.latest_loss_components[key]
        super().log(merged_logs, *args, **kwargs)

    def _save(self, output_dir: str | None = None, state_dict=None):
        save_dir = Path(output_dir if output_dir is not None else self.args.output_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        model_to_save = self.model
        if state_dict is None:
            state_dict = model_to_save.state_dict()

        torch.save(state_dict, save_dir / "pytorch_model.bin")
        torch.save(self.args, save_dir / "training_args.bin")


class TrainingMonitorCallback(TrainerCallback):
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.output_dir / "training_log.jsonl"
        self.plot_path = self.output_dir / "loss_curve.png"
        self.history: dict[str, list[tuple[int, float]]] = {
            "loss": [],
            "loss_bbox": [],
            "loss_giou": [],
        }

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return

        record: dict[str, float | int | None] = {
            "step": int(state.global_step),
            "epoch": float(state.epoch) if state.epoch is not None else None,
        }

        for key, value in logs.items():
            if isinstance(value, (int, float)):
                record[key] = float(value)
            else:
                try:
                    record[key] = float(value)
                except (TypeError, ValueError):
                    continue

        with self.log_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record, ensure_ascii=True) + "\n")

        updated = False
        for key in ("loss", "loss_bbox", "loss_giou"):
            if key in record:
                self.history[key].append((int(record["step"]), float(record[key])))
                updated = True

        if updated:
            self._save_loss_plot()

    def _save_loss_plot(self) -> None:
        if not any(self.history.values()):
            return

        figure, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True)
        plot_specs = [
            ("loss", "Total Loss", "#1f77b4"),
            ("loss_bbox", "BBox L1 Loss", "#ff7f0e"),
            ("loss_giou", "GIoU Loss", "#2ca02c"),
        ]

        for axis, (key, title, color) in zip(axes, plot_specs):
            series = self.history[key]
            if series:
                steps = [step for step, _ in series]
                values = [value for _, value in series]
                axis.plot(steps, values, linewidth=2, color=color)
            axis.set_title(title)
            axis.set_ylabel(key)
            axis.grid(True, alpha=0.3)

        axes[-1].set_xlabel("global step")
        plt.tight_layout()
        plt.savefig(self.plot_path, dpi=160)
        plt.close(figure)


def _build_training_arguments(
    checkpoint_dir: Path,
    batch_size: int,
    num_epochs: int,
    learning_rate: float,
    weight_decay: float,
    warmup_steps: int,
    grad_clip: float,
    num_workers: int,
    log_every: int,
    save_steps: int,
    save_total_limit: int,
    gradient_accumulation_steps: int,
    max_steps: int,
    requested_device: str,
    seed: int,
) -> TrainingArguments:
    requested_kwargs = {
        "output_dir": str(checkpoint_dir),
        "overwrite_output_dir": False,
        "per_device_train_batch_size": batch_size,
        "num_train_epochs": num_epochs,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "warmup_steps": warmup_steps,
        "max_grad_norm": grad_clip,
        "dataloader_num_workers": num_workers,
        "logging_steps": log_every,
        "save_steps": save_steps,
        "save_total_limit": save_total_limit,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "remove_unused_columns": False,
        "report_to": "none",
        "max_steps": max_steps if max_steps > 0 else -1,
        "no_cuda": requested_device != "cuda",
        "save_safetensors": False,
        "seed": seed,
    }

    supported_params = set(inspect.signature(TrainingArguments.__init__).parameters)
    filtered_kwargs = {
        key: value
        for key, value in requested_kwargs.items()
        if key in supported_params
    }
    ignored_kwargs = sorted(set(requested_kwargs) - set(filtered_kwargs))
    if ignored_kwargs:
        print(
            "TrainingArguments compatibility: ignored unsupported kwargs: "
            f"{', '.join(ignored_kwargs)}"
        )

    return TrainingArguments(**filtered_kwargs)


@hydra.main(version_base=None, config_path="../../config", config_name="base")
def main(config: DictConfig) -> None:
    train_config = config.get("train", {})
    requested_device = str(train_config.get("device", config.generator.device))
    batch_size = int(train_config.get("batch_size", 8))
    num_epochs = int(train_config.get("num_epochs", 1))
    learning_rate = float(train_config.get("learning_rate", 1e-4))
    weight_decay = float(train_config.get("weight_decay", 1e-4))
    num_workers = int(train_config.get("num_workers", 0))
    grad_clip = float(train_config.get("grad_clip", 1.0))
    log_every = int(train_config.get("log_every", 10))
    max_steps = int(train_config.get("max_steps", 0))
    noise_scale = float(train_config.get("noise_scale", config.shared.get("noise_scale", 0.0)))
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
