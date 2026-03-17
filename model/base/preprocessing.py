from __future__ import annotations

import bisect
from pathlib import Path
from typing import Any

from hydra.utils import get_original_cwd
from omegaconf import DictConfig

import torch
from torch.utils.data import Dataset
from transformers import CLIPTokenizer

from generation.base.generate import SaveModel


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
