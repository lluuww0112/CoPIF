from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import hydra
from hydra.utils import get_original_cwd, instantiate
from omegaconf import DictConfig
from PIL import Image

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import CLIPImageProcessor, CLIPTokenizer

from APIs.refcocoAPI import RefcocoModel, load_refcoco
from model.base.train import _build_rec_model

_BICUBIC_RESAMPLE = (
    Image.Resampling.BICUBIC if hasattr(Image, "Resampling") else Image.BICUBIC
)


def _torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _resolve_device(requested_device: str) -> torch.device:
    if requested_device == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if requested_device == "cuda":
        print("cuda is not available. Falling back to cpu.")
    return torch.device("cpu")


def _resolve_project_path(path_like: str | Path) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path
    return Path(get_original_cwd()) / path


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


def _normalize_bbox(sample: RefcocoModel) -> tuple[float, float, float, float]:
    if sample.size is None:
        raise ValueError("sample.size is required.")

    width = sample.size.width
    height = sample.size.height
    if width <= 0 or height <= 0:
        raise ValueError("image size must be positive.")

    return (
        sample.bbox.x / width,
        sample.bbox.y / height,
        sample.bbox.w / width,
        sample.bbox.h / height,
    )


def _denormalize_bbox(
    bbox: torch.Tensor,
    width: float,
    height: float,
) -> tuple[float, float, float, float]:
    return (
        float(bbox[0].item() * width),
        float(bbox[1].item() * height),
        float(bbox[2].item() * width),
        float(bbox[3].item() * height),
    )


class RefcocoInferenceDataset(Dataset):
    def __init__(self, samples: list[RefcocoModel]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> RefcocoModel:
        return self.samples[index]


class InferenceCollator:
    def __init__(
        self,
        image_processor: CLIPImageProcessor,
        tokenizer: CLIPTokenizer,
        input_res: int,
    ):
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.input_res = input_res

    def __call__(self, batch: list[RefcocoModel]) -> dict[str, Any]:
        images = []
        annotations = []

        for sample in batch:
            if sample.image_path is None:
                raise ValueError("sample.image_path is required.")
            with Image.open(sample.image_path) as image:
                images.append(
                    image.convert("RGB").resize(
                        (self.input_res, self.input_res),
                        resample=_BICUBIC_RESAMPLE,
                    )
                )
            annotations.append(sample.annotation)

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
            "samples": batch,
            "pixel_values": image_inputs["pixel_values"],
            "input_ids": text_inputs["input_ids"],
            "attention_mask": text_inputs["attention_mask"],
        }


def _load_inference_samples(
    dataset: str,
    splitby: str,
    split: str,
    max_samples: int | None,
) -> list[RefcocoModel]:
    all_samples = load_refcoco(
        dataset=dataset,
        splitby=splitby,
        split=split,
        include_image_metadata=True,
    )

    valid_samples = [
        sample
        for sample in all_samples
        if sample.image_path is not None
        and sample.size is not None
        and Path(sample.image_path).is_file()
    ]

    if not valid_samples:
        raise FileNotFoundError(
            "could not find RefCOCO samples with valid image_path and size metadata."
        )

    if max_samples is not None and max_samples > 0:
        valid_samples = valid_samples[:max_samples]

    return valid_samples


def _write_predictions_to_csv(
    output_path: Path,
    rows: list[dict[str, Any]],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "image_id",
        "annotation",
        "image_path",
        "image_width",
        "image_height",
        "pred_x",
        "pred_y",
        "pred_w",
        "pred_h",
        "gt_x",
        "gt_y",
        "gt_w",
        "gt_h",
        "pred_x_norm",
        "pred_y_norm",
        "pred_w_norm",
        "pred_h_norm",
        "gt_x_norm",
        "gt_y_norm",
        "gt_w_norm",
        "gt_h_norm",
    ]

    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


@hydra.main(version_base=None, config_path="../../config", config_name="base")
def main(config: DictConfig) -> None:
    invoke_config = config.get("invoke", {})
    train_config = config.get("train", {})

    dataset = str(invoke_config.get("dataset", train_config.get("dataset", config.generator.dataset)))
    splitby = str(invoke_config.get("splitby", train_config.get("splitby", config.generator.splitby)))
    split = str(invoke_config.get("split", train_config.get("split", config.generator.split)))
    requested_device = str(invoke_config.get("device", train_config.get("device", config.generator.device)))
    batch_size = int(invoke_config.get("batch_size", 8))
    max_samples_value = invoke_config.get("max_samples", None)
    max_samples = int(max_samples_value) if max_samples_value is not None else None
    checkpoint_dir = _resolve_project_path(
        str(invoke_config.get("checkpoint_dir", train_config.get("checkpoint_dir", "history/base")))
    )
    output_csv_value = invoke_config.get("output_csv", None)
    if output_csv_value is None:
        output_csv = checkpoint_dir / f"inference_{dataset}_{splitby}_{split}.csv"
    else:
        output_csv = _resolve_project_path(output_csv_value)

    device = _resolve_device(requested_device=requested_device)
    checkpoint_path = _find_checkpoint_path(checkpoint_dir=checkpoint_dir)

    tokenizer = CLIPTokenizer.from_pretrained(config.shared.pretrained_clip)
    image_processor = CLIPImageProcessor.from_pretrained(config.shared.pretrained_clip)

    vision_model = instantiate(config.model.vision_model)
    text_model = instantiate(config.model.text_model)
    model = _build_rec_model(config, vision_model=vision_model, text_model=text_model)
    input_res = int(vision_model.vision_model.config.image_size)

    state_dict = _torch_load(checkpoint_path)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    samples = _load_inference_samples(
        dataset=dataset,
        splitby=splitby,
        split=split,
        max_samples=max_samples,
    )

    dataloader = DataLoader(
        RefcocoInferenceDataset(samples),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=InferenceCollator(
            image_processor=image_processor,
            tokenizer=tokenizer,
            input_res=input_res,
        ),
    )

    print(f"checkpoint_path: {checkpoint_path}")
    print(f"device: {device}")
    print(f"inference_data: {dataset}/{splitby}/{split}")
    print(f"num_samples: {len(samples)}")
    print(f"batch_size: {batch_size}")
    print(f"input_res: {input_res}")
    print(f"output_csv: {output_csv}")

    rows: list[dict[str, Any]] = []

    with torch.no_grad():
        for batch in dataloader:
            pixel_values = batch["pixel_values"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            pred_boxes = model.invoke(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
            ).detach().cpu()

            for sample, pred_box in zip(batch["samples"], pred_boxes):
                if sample.size is None:
                    raise ValueError("sample.size is required.")

                width = float(sample.size.width)
                height = float(sample.size.height)
                gt_norm = _normalize_bbox(sample)
                pred_abs = _denormalize_bbox(pred_box, width=width, height=height)

                rows.append(
                    {
                        "image_id": sample.image_id,
                        "annotation": sample.annotation,
                        "image_path": sample.image_path,
                        "image_width": width,
                        "image_height": height,
                        "pred_x": pred_abs[0],
                        "pred_y": pred_abs[1],
                        "pred_w": pred_abs[2],
                        "pred_h": pred_abs[3],
                        "gt_x": float(sample.bbox.x),
                        "gt_y": float(sample.bbox.y),
                        "gt_w": float(sample.bbox.w),
                        "gt_h": float(sample.bbox.h),
                        "pred_x_norm": float(pred_box[0].item()),
                        "pred_y_norm": float(pred_box[1].item()),
                        "pred_w_norm": float(pred_box[2].item()),
                        "pred_h_norm": float(pred_box[3].item()),
                        "gt_x_norm": gt_norm[0],
                        "gt_y_norm": gt_norm[1],
                        "gt_w_norm": gt_norm[2],
                        "gt_h_norm": gt_norm[3],
                    }
                )

    _write_predictions_to_csv(output_path=output_csv, rows=rows)
    print(f"saved {len(rows)} predictions to {output_csv}")


if __name__ == "__main__":
    main()
