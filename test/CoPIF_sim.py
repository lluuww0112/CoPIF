from __future__ import annotations

import math
import random
import re
import textwrap
from pathlib import Path
from typing import Any

import hydra
from hydra.utils import get_original_cwd, instantiate
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from omegaconf import DictConfig
from PIL import Image
import torch
import torch.nn.functional as F
from transformers import CLIPTokenizer

from generation.base.generate import SaveModel
from model.base.clip_model import CLIPTextModel


OUTPUT_DIR = Path("test") / "res"


def _torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _sanitize_filename(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", value).strip("_") or "sample"


def _feature_map_manifest_path(config: DictConfig) -> Path:
    project_root = Path(get_original_cwd())
    dataset = str(config.generator.dataset)
    splitby = str(config.generator.splitby)
    split = str(config.generator.split)
    return (
        project_root
        / "data"
        / "generated"
        / dataset
        / splitby
        / split
        / "feature_maps.pt"
    )


def _load_generated_samples(manifest_path: Path) -> list[SaveModel]:
    if not manifest_path.is_file():
        raise FileNotFoundError(f"feature map manifest not found: {manifest_path}")

    manifest = _torch_load(manifest_path)
    chunk_names = manifest.get("chunks", [])
    if not chunk_names:
        raise ValueError(f"manifest has no chunks: {manifest_path}")

    save_models: list[SaveModel] = []
    for chunk_name in chunk_names:
        chunk_path = manifest_path.parent / chunk_name
        chunk_data = _torch_load(chunk_path)
        save_models.extend(chunk_data)

    if not save_models:
        raise ValueError(f"no SaveModel entries found under: {manifest_path.parent}")
    return save_models


def _select_samples(samples: list[SaveModel], limit: int) -> list[SaveModel]:
    valid_samples = [
        sample
        for sample in samples
        if sample.image_path
        and Path(sample.image_path).is_file()
        and sample.feature_map is not None
    ]
    if not valid_samples:
        raise FileNotFoundError(
            "Could not find generated samples with a valid image_path and feature_map. "
            f"Check generated data under {samples[:1]}."
        )

    if len(valid_samples) <= limit:
        return valid_samples
    return random.sample(valid_samples, k=limit)


def _bbox_to_grid_coords(sample: SaveModel, grid_size: int) -> tuple[int, int, int, int]:
    image_width = float(sample.size.width)
    image_height = float(sample.size.height)

    x0 = max(0.0, float(sample.bbox.x) * image_width)
    y0 = max(0.0, float(sample.bbox.y) * image_height)
    x1 = min(image_width, (float(sample.bbox.x) + float(sample.bbox.w)) * image_width)
    y1 = min(image_height, (float(sample.bbox.y) + float(sample.bbox.h)) * image_height)

    grid_x0 = min(grid_size - 1, int((x0 / image_width) * grid_size))
    grid_y0 = min(grid_size - 1, int((y0 / image_height) * grid_size))
    grid_x1 = max(grid_x0 + 1, min(grid_size, math.ceil((x1 / image_width) * grid_size)))
    grid_y1 = max(grid_y0 + 1, min(grid_size, math.ceil((y1 / image_height) * grid_size)))
    return grid_x0, grid_y0, grid_x1, grid_y1


def _compute_grid_similarity(
    sample: SaveModel,
    text_model: CLIPTextModel,
    tokenizer: CLIPTokenizer,
    device: torch.device,
    text_noise_scale: float,
    num_text_augmentations: int,
) -> tuple[torch.Tensor, float, float]:
    text_inputs = tokenizer(
        [sample.annotation],
        padding=True,
        truncation=True,
        return_tensors="pt",
    )
    text_inputs = {key: value.to(device) for key, value in text_inputs.items()}

    with torch.no_grad():
        text_embedding = text_model(**text_inputs).squeeze(0)

    if num_text_augmentations < 1:
        raise ValueError(f"num_text_augmentations must be >= 1, got {num_text_augmentations}.")

    if text_noise_scale > 0:
        augmented_embeddings = []
        for _ in range(num_text_augmentations):
            noise = torch.randn_like(text_embedding) * text_noise_scale
            augmented_embeddings.append(text_embedding + noise)
        text_embedding = torch.stack(augmented_embeddings, dim=0).mean(dim=0)

    feature_map = sample.feature_map.to(device=device, dtype=torch.float32)
    grid_size = feature_map.shape[0]
    flattened = feature_map.reshape(-1, feature_map.shape[-1])

    flattened = F.normalize(flattened, dim=-1)
    text_embedding = F.normalize(text_embedding, dim=-1)

    similarity = torch.matmul(flattened, text_embedding)
    similarity_map = similarity.reshape(grid_size, grid_size).detach().cpu()

    grid_x0, grid_y0, grid_x1, grid_y1 = _bbox_to_grid_coords(sample, grid_size)
    inside = similarity_map[grid_y0:grid_y1, grid_x0:grid_x1]
    bbox_mean = inside.mean().item()

    outside_mask = torch.ones_like(similarity_map, dtype=torch.bool)
    outside_mask[grid_y0:grid_y1, grid_x0:grid_x1] = False
    outside_mean = similarity_map[outside_mask].mean().item()

    return similarity_map, bbox_mean, outside_mean


def _load_sample_image(sample: SaveModel) -> Image.Image:
    return Image.open(sample.image_path).convert("RGB")


def _visualize_similarity(
    sample: SaveModel,
    similarity_map: torch.Tensor,
    save_path: Path,
    text_noise_scale: float,
    num_text_augmentations: int,
) -> None:
    image = _load_sample_image(sample)
    grid_size = similarity_map.shape[0]
    grid_x0, grid_y0, grid_x1, grid_y1 = _bbox_to_grid_coords(sample, grid_size)

    image_width = float(sample.size.width)
    image_height = float(sample.size.height)
    bbox_x = float(sample.bbox.x) * image_width
    bbox_y = float(sample.bbox.y) * image_height
    bbox_w = float(sample.bbox.w) * image_width
    bbox_h = float(sample.bbox.h) * image_height

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)

    axes[0].imshow(image)
    axes[0].add_patch(
        Rectangle(
            (bbox_x, bbox_y),
            bbox_w,
            bbox_h,
            fill=False,
            edgecolor="lime",
            linewidth=2,
        )
    )
    axes[0].set_title("Image + bbox")
    axes[0].axis("off")

    heatmap = axes[1].imshow(similarity_map.numpy(), cmap="magma", interpolation="nearest")
    axes[1].add_patch(
        Rectangle(
            (grid_x0 - 0.5, grid_y0 - 0.5),
            grid_x1 - grid_x0,
            grid_y1 - grid_y0,
            fill=False,
            edgecolor="cyan",
            linewidth=2,
        )
    )
    axes[1].set_title("CoPIF grid-text similarity")
    axes[1].set_xlabel("grid x")
    axes[1].set_ylabel("grid y")
    fig.colorbar(heatmap, ax=axes[1], fraction=0.046, pad=0.04)

    axes[2].imshow(image)
    axes[2].imshow(
        similarity_map.numpy(),
        cmap="magma",
        alpha=0.55,
        extent=(0, image_width, image_height, 0),
        interpolation="bilinear",
    )
    axes[2].add_patch(
        Rectangle(
            (bbox_x, bbox_y),
            bbox_w,
            bbox_h,
            fill=False,
            edgecolor="white",
            linewidth=2,
        )
    )
    axes[2].set_title("Overlay")
    axes[2].axis("off")

    wrapped_annotation = "\n".join(textwrap.wrap(f'annotation: "{sample.annotation}"', width=70))
    augmentation_note = (
        f"text_noise_scale={text_noise_scale}, num_text_augmentations={num_text_augmentations}"
    )
    fig.suptitle("CoPIF grid-text similarity", fontsize=14, fontweight="bold")
    fig.text(
        0.5,
        0.02,
        f"{wrapped_annotation}\n{augmentation_note}",
        ha="center",
        va="bottom",
        fontsize=11,
        bbox={"boxstyle": "round,pad=0.4", "facecolor": "white", "alpha": 0.9, "edgecolor": "0.7"},
    )
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


@hydra.main(version_base=None, config_path="../config", config_name="base")
def main(config: DictConfig) -> None:
    requested_device = str(config.generator.device)
    if requested_device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    copif_config = config.get("test", {}).get("copif", {})
    text_noise_scale = float(copif_config.get("text_noise_scale", config.shared.get("noise_scale", 0.0)))
    num_text_augmentations = int(copif_config.get("num_text_augmentations", 1))
    sample_limit = int(copif_config.get("sample_limit", 5))

    manifest_path = _feature_map_manifest_path(config)
    text_model = instantiate(config.model.text_model)
    tokenizer = CLIPTokenizer.from_pretrained(config.shared.pretrained_clip)
    text_model.to(device).eval()

    generated_samples = _load_generated_samples(manifest_path)
    samples = _select_samples(generated_samples, limit=sample_limit)

    for index, sample in enumerate(samples, start=1):
        similarity_map, bbox_mean, outside_mean = _compute_grid_similarity(
            sample=sample,
            text_model=text_model,
            tokenizer=tokenizer,
            device=device,
            text_noise_scale=text_noise_scale,
            num_text_augmentations=num_text_augmentations,
        )

        noise_suffix = ""
        if text_noise_scale > 0:
            scale_tag = str(text_noise_scale).replace(".", "p")
            noise_suffix = f"_noise{scale_tag}_n{num_text_augmentations}"
        output_name = _sanitize_filename(
            f"copif_{index:02d}_{Path(sample.image_path).stem}_{sample.annotation[:40]}{noise_suffix}.png"
        )
        output_path = OUTPUT_DIR / output_name

        _visualize_similarity(
            sample=sample,
            similarity_map=similarity_map,
            save_path=output_path,
            text_noise_scale=text_noise_scale,
            num_text_augmentations=num_text_augmentations,
        )

        print(f"[sample {index}] image_path: {sample.image_path}")
        print(f"[sample {index}] annotation: {sample.annotation}")
        print(
            f"[sample {index}] bbox_norm: "
            f"({sample.bbox.x:.4f}, {sample.bbox.y:.4f}, {sample.bbox.w:.4f}, {sample.bbox.h:.4f})"
        )
        print(f"[sample {index}] grid_size: {similarity_map.shape[0]}x{similarity_map.shape[1]}")
        print(f"[sample {index}] text_noise_scale: {text_noise_scale}")
        print(f"[sample {index}] num_text_augmentations: {num_text_augmentations}")
        print(f"[sample {index}] bbox_grid_mean_similarity: {bbox_mean:.4f}")
        print(f"[sample {index}] outside_grid_mean_similarity: {outside_mean:.4f}")
        print(f"[sample {index}] saved_heatmap: {output_path.resolve()}")


if __name__ == "__main__":
    main()
