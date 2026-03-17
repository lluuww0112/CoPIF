from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image

import torch
from torch.utils.data import Dataset
from transformers import CLIPImageProcessor, CLIPTokenizer

from APIs.refcocoAPI import RefcocoModel

_BICUBIC_RESAMPLE = (
    Image.Resampling.BICUBIC if hasattr(Image, "Resampling") else Image.BICUBIC
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


class TrainCollator:
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
