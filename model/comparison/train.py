from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import hydra
from hydra.utils import get_original_cwd, instantiate
from omegaconf import DictConfig
from PIL import Image

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import CLIPImageProcessor, CLIPTokenizer

from APIs.refcocoAPI import RefcocoModel, load_refcoco
from generation.base.generate import select_fewshot_image_samples
from model.base.train import (
    RECTrainer,
    TrainingMonitorCallback,
    _build_rec_model,
    _build_training_arguments,
    _generalized_box_iou_loss,
    _torch_load,
)

_BICUBIC_RESAMPLE = (
    Image.Resampling.BICUBIC if hasattr(Image, "Resampling") else Image.BICUBIC
)


def _resolve_project_path(path_like: str | Path) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path
    return Path(get_original_cwd()) / path


def _resolve_device(requested_device: str) -> torch.device:
    if requested_device == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if requested_device == "cuda":
        print("cuda is not available. Falling back to cpu.")
    return torch.device("cpu")


def _find_checkpoint_path(checkpoint_dir: Path) -> Path:
    root_checkpoint = checkpoint_dir / "pytorch_model.bin"
    if root_checkpoint.is_file():
        return root_checkpoint

    checkpoint_candidates = sorted(
        (
            path / "pytorch_model.bin"
            for path in checkpoint_dir.glob("checkpoint-*")
            if path.is_dir() and (path / "pytorch_model.bin").is_file()
        ),
        key=lambda path: int(path.parent.name.split("-")[-1]),
    )

    if checkpoint_candidates:
        return checkpoint_candidates[-1]

    raise FileNotFoundError(
        f"could not find pytorch_model.bin under checkpoint_dir={checkpoint_dir}"
    )


def _normalize_bbox(sample: RefcocoModel) -> torch.Tensor:
    if sample.size is None:
        raise ValueError("sample.size is required.")

    width = float(sample.size.width)
    height = float(sample.size.height)
    if width <= 0 or height <= 0:
        raise ValueError("image size must be positive.")

    return torch.tensor(
        [
            float(sample.bbox.x) / width,
            float(sample.bbox.y) / height,
            float(sample.bbox.w) / width,
            float(sample.bbox.h) / height,
        ],
        dtype=torch.float32,
    )


class ImageTrainingDataset(Dataset):
    def __init__(self, samples: list[RefcocoModel]):
        self.entries: list[dict[str, Any]] = []

        for sample in samples:
            if sample.image_path is None or sample.size is None:
                continue

            image_path = Path(sample.image_path)
            if not image_path.is_file():
                continue

            self.entries.append(
                {
                    "image_id": sample.image_id,
                    "image_path": str(image_path),
                    "annotation": sample.annotation,
                    "bbox": _normalize_bbox(sample),
                }
            )

        if not self.entries:
            raise FileNotFoundError(
                "could not find valid image-backed samples for comparison training."
            )

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.entries[index]


class ComparisonTrainCollator:
    def __init__(
        self,
        image_processor: CLIPImageProcessor,
        tokenizer: CLIPTokenizer,
        input_res: int,
    ):
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.input_res = input_res

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        images = []
        annotations = []
        bbox_targets = []

        for sample in batch:
            with Image.open(sample["image_path"]) as image:
                images.append(
                    image.convert("RGB").resize(
                        (self.input_res, self.input_res),
                        resample=_BICUBIC_RESAMPLE,
                    )
                )
            annotations.append(sample["annotation"])
            bbox_targets.append(sample["bbox"])

        image_inputs = self.image_processor(
            images=images,
            return_tensors="pt",
            do_resize=False,
            do_center_crop=False,
        )
        text_inputs = self.tokenizer(
            annotations,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )

        return {
            "pixel_values": image_inputs["pixel_values"],
            "input_ids": text_inputs["input_ids"],
            "attention_mask": text_inputs["attention_mask"],
            "bbox_targets": torch.stack(bbox_targets, dim=0),
        }


class ImageRECTrainer(RECTrainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        bbox_targets = inputs["bbox_targets"]
        pred_boxes = model(
            pixel_values=inputs["pixel_values"],
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


def _select_training_samples(
    all_samples: list[RefcocoModel],
    num_images: int | None,
    seed: int,
) -> tuple[list[RefcocoModel], list[str]]:
    if num_images is None:
        valid_samples = [
            sample
            for sample in all_samples
            if sample.image_path is not None
            and sample.size is not None
            and Path(sample.image_path).is_file()
        ]
        selected_image_ids = sorted({sample.image_id for sample in valid_samples})
        return valid_samples, selected_image_ids

    return select_fewshot_image_samples(
        refcoco_samples=all_samples,
        num_images=num_images,
        seed=seed,
    )


def _write_selection_summary(
    output_dir: Path,
    payload: dict[str, Any],
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "comparison_selection.json"
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=True, indent=2)
    return output_path


@hydra.main(version_base=None, config_path="../../config", config_name="comparison")
def main(config: DictConfig) -> None:
    train_config = config.get("train", {})

    requested_device = str(train_config.get("device", "cuda"))
    device = _resolve_device(requested_device=requested_device)

    dataset = str(train_config.get("dataset", "refcoco"))
    splitby = str(train_config.get("splitby", "unc"))
    split = str(train_config.get("split", "train"))

    num_images_value = train_config.get("num_images", None)
    num_images = int(num_images_value) if num_images_value is not None else None
    if num_images is not None and num_images <= 0:
        raise ValueError("train.num_images must be a positive integer or null.")

    seed = int(train_config.get("seed", 42))
    batch_size = int(train_config.get("batch_size", 8))
    num_epochs = int(train_config.get("num_epochs", 1))
    learning_rate = float(train_config.get("learning_rate", 1e-4))
    weight_decay = float(train_config.get("weight_decay", 1e-4))
    num_workers = int(train_config.get("num_workers", 0))
    grad_clip = float(train_config.get("grad_clip", 1.0))
    log_every = int(train_config.get("log_every", 10))
    max_steps = int(train_config.get("max_steps", 0))
    save_steps = int(train_config.get("save_steps", 500))
    save_total_limit = int(train_config.get("save_total_limit", 2))
    warmup_steps = int(train_config.get("warmup_steps", 0))
    gradient_accumulation_steps = int(train_config.get("gradient_accumulation_steps", 1))
    bbox_loss_coef = float(train_config.get("bbox_loss_coef", 5.0))
    giou_loss_coef = float(train_config.get("giou_loss_coef", 2.0))
    resume_from_checkpoint = train_config.get("resume_from_checkpoint", None)

    checkpoint_dir = _resolve_project_path(
        str(train_config.get("checkpoint_dir", "history/comparison"))
    )
    init_checkpoint_dir_value = train_config.get("init_checkpoint_dir", None)
    init_checkpoint_dir = (
        _resolve_project_path(str(init_checkpoint_dir_value))
        if init_checkpoint_dir_value is not None
        else None
    )

    tokenizer = CLIPTokenizer.from_pretrained(config.shared.pretrained_clip)
    image_processor = CLIPImageProcessor.from_pretrained(config.shared.pretrained_clip)

    vision_model = instantiate(config.model.vision_model)
    text_model = instantiate(config.model.text_model)
    model = _build_rec_model(config, vision_model=vision_model, text_model=text_model)
    input_res = int(vision_model.vision_model.config.image_size)

    init_checkpoint_path = None
    if init_checkpoint_dir is not None:
        init_checkpoint_path = _find_checkpoint_path(init_checkpoint_dir)
        state_dict = _torch_load(init_checkpoint_path)
        model.load_state_dict(state_dict)

    all_samples = load_refcoco(
        dataset=dataset,
        splitby=splitby,
        split=split,
        include_image_metadata=True,
    )
    selected_samples, selected_image_ids = _select_training_samples(
        all_samples=all_samples,
        num_images=num_images,
        seed=seed,
    )

    dataset_obj = ImageTrainingDataset(samples=selected_samples)
    collator = ComparisonTrainCollator(
        image_processor=image_processor,
        tokenizer=tokenizer,
        input_res=input_res,
    )

    selection_summary_path = _write_selection_summary(
        output_dir=checkpoint_dir,
        payload={
            "dataset": dataset,
            "splitby": splitby,
            "split": split,
            "requested_num_images": num_images,
            "selected_num_images": len(selected_image_ids),
            "selected_num_entries": len(dataset_obj),
            "selected_image_ids": selected_image_ids,
            "init_checkpoint_path": str(init_checkpoint_path) if init_checkpoint_path is not None else None,
        },
    )

    print(f"train_data: {dataset}/{splitby}/{split}")
    print(f"requested_device: {requested_device}")
    print(f"resolved_device: {device}")
    print(f"requested_num_images: {num_images}")
    print(f"selected_num_images: {len(selected_image_ids)}")
    print(f"selected_num_entries: {len(dataset_obj)}")
    print(f"batch_size: {batch_size}")
    print(f"num_epochs: {num_epochs}")
    print(f"input_res: {input_res}")
    print(f"bbox_loss_coef: {bbox_loss_coef}")
    print(f"giou_loss_coef: {giou_loss_coef}")
    print(f"selection_summary_path: {selection_summary_path}")
    if init_checkpoint_path is not None:
        print(f"init_checkpoint_path: {init_checkpoint_path}")

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

    trainer = ImageRECTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset_obj,
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
