from __future__ import annotations

from pathlib import Path
import math
import random
import re

import hydra
from hydra.utils import instantiate
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from omegaconf import DictConfig
from PIL import Image
import torch
import torch.nn.functional as F
import textwrap
from transformers import CLIPImageProcessor, CLIPTokenizer

from APIs.refcocoAPI import RefcocoModel, load_refcoco
from model.base.clip_model import CLIPTextModel, CLIPVisionModel


OUTPUT_DIR = Path("test") / "res"


def _sanitize_filename(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", value).strip("_") or "sample"


def _select_samples(samples: list[RefcocoModel], limit: int) -> list[RefcocoModel]:
    valid_samples: list[RefcocoModel] = []
    seen_paths: list[str] = []
    for sample in samples:
        if sample.image_path is None or sample.size is None:
            continue
        if Path(sample.image_path).is_file():
            valid_samples.append(sample)
        if len(seen_paths) < 5:
            seen_paths.append(sample.image_path)
    if valid_samples:
        if len(valid_samples) <= limit:
            return valid_samples
        return random.sample(valid_samples, k=limit)
    raise FileNotFoundError(
        f"Could not find {limit} RefCOCO samples with a valid image_path. "
        "Checked sample paths such as: "
        f"{seen_paths}. "
        "Check the dataset image directory under APIs/refer_python3_2024/data."
    )


def _bbox_to_patch_coords(sample: RefcocoModel, patch_grid_size: int) -> tuple[int, int, int, int]:
    if sample.size is None:
        raise ValueError("sample.size is required.")

    image_width = sample.size.width
    image_height = sample.size.height

    x0 = max(0.0, sample.bbox.x)
    y0 = max(0.0, sample.bbox.y)
    x1 = min(image_width, sample.bbox.x + sample.bbox.w)
    y1 = min(image_height, sample.bbox.y + sample.bbox.h)

    patch_x0 = min(patch_grid_size - 1, int((x0 / image_width) * patch_grid_size))
    patch_y0 = min(patch_grid_size - 1, int((y0 / image_height) * patch_grid_size))
    patch_x1 = max(
        patch_x0 + 1,
        min(patch_grid_size, math.ceil((x1 / image_width) * patch_grid_size)),
    )
    patch_y1 = max(
        patch_y0 + 1,
        min(patch_grid_size, math.ceil((y1 / image_height) * patch_grid_size)),
    )
    return patch_x0, patch_y0, patch_x1, patch_y1


def _load_sample_image(sample: RefcocoModel) -> Image.Image:
    if sample.image_path is None:
        raise ValueError("sample.image_path is required.")
    return Image.open(sample.image_path).convert("RGB")


def _compute_patch_similarity(
    sample: RefcocoModel,
    vision_model: CLIPVisionModel,
    text_model: CLIPTextModel,
    image_processor: CLIPImageProcessor,
    tokenizer: CLIPTokenizer,
    device: torch.device,
    text_noise_scale: float,
    num_text_augmentations: int,
) -> tuple[torch.Tensor, float, float]:
    image = _load_sample_image(sample)
    image_inputs = image_processor(images=image, return_tensors="pt")
    image_inputs = {key: value.to(device) for key, value in image_inputs.items()}

    text_inputs = tokenizer(
        [sample.annotation],
        padding=True,
        truncation=True,
        return_tensors="pt",
    )
    text_inputs = {key: value.to(device) for key, value in text_inputs.items()}

    with torch.no_grad():
        patch_embeddings = vision_model(image_inputs["pixel_values"]).squeeze(0)
        text_embedding = text_model(**text_inputs).squeeze(0)

    if num_text_augmentations < 1:
        raise ValueError(f"num_text_augmentations must be >= 1, got {num_text_augmentations}.")

    if text_noise_scale > 0:
        augmented_embeddings = []
        for _ in range(num_text_augmentations):
            noise = torch.randn_like(text_embedding) * text_noise_scale
            augmented_embeddings.append(text_embedding + noise)
        text_embedding = torch.stack(augmented_embeddings, dim=0).mean(dim=0)

    patch_embeddings = F.normalize(patch_embeddings, dim=-1)
    text_embedding = F.normalize(text_embedding, dim=-1)
    similarity = torch.matmul(patch_embeddings, text_embedding)

    patch_count = similarity.shape[0]
    patch_grid_size = int(math.sqrt(patch_count))
    if patch_grid_size * patch_grid_size != patch_count:
        raise ValueError(f"Patch count must be a square number, got {patch_count}.")

    similarity_map = similarity.reshape(patch_grid_size, patch_grid_size).detach().cpu()

    patch_x0, patch_y0, patch_x1, patch_y1 = _bbox_to_patch_coords(sample, patch_grid_size)
    inside = similarity_map[patch_y0:patch_y1, patch_x0:patch_x1]

    bbox_mean = inside.mean().item()
    outside_mask = torch.ones_like(similarity_map, dtype=torch.bool)
    outside_mask[patch_y0:patch_y1, patch_x0:patch_x1] = False
    outside_mean = similarity_map[outside_mask].mean().item()

    return similarity_map, bbox_mean, outside_mean


def _visualize_similarity(
    sample: RefcocoModel,
    similarity_map: torch.Tensor,
    save_path: Path,
    text_noise_scale: float,
    num_text_augmentations: int,
) -> None:
    image = _load_sample_image(sample)
    patch_grid_size = similarity_map.shape[0]
    patch_x0, patch_y0, patch_x1, patch_y1 = _bbox_to_patch_coords(sample, patch_grid_size)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)

    axes[0].imshow(image)
    axes[0].add_patch(
        Rectangle(
            (sample.bbox.x, sample.bbox.y),
            sample.bbox.w,
            sample.bbox.h,
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
            (patch_x0 - 0.5, patch_y0 - 0.5),
            patch_x1 - patch_x0,
            patch_y1 - patch_y0,
            fill=False,
            edgecolor="cyan",
            linewidth=2,
        )
    )
    axes[1].set_title("Patch-text similarity")
    axes[1].set_xlabel("patch x")
    axes[1].set_ylabel("patch y")
    fig.colorbar(heatmap, ax=axes[1], fraction=0.046, pad=0.04)

    axes[2].imshow(image)
    image_width, image_height = image.size
    axes[2].imshow(
        similarity_map.numpy(),
        cmap="magma",
        alpha=0.55,
        extent=(0, image_width, image_height, 0),
        interpolation="bilinear",
    )
    axes[2].add_patch(
        Rectangle(
            (sample.bbox.x, sample.bbox.y),
            sample.bbox.w,
            sample.bbox.h,
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
    fig.suptitle("MaskCLIP patch-text similarity", fontsize=14, fontweight="bold")
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
    device = torch.device(config.generator.device)
    text_noise_scale = float(config.test.maskclip.text_noise_scale)
    num_text_augmentations = int(config.test.maskclip.num_text_augmentations)

    vision_model = instantiate(config.model.vision_model)
    text_model = instantiate(config.model.text_model)
    image_processor = CLIPImageProcessor.from_pretrained(config.shared.pretrained_clip)
    tokenizer = CLIPTokenizer.from_pretrained(config.shared.pretrained_clip)

    vision_model.to(device).eval()
    text_model.to(device).eval()

    refcoco_samples = load_refcoco(
        dataset="refcoco",
        split="train",
        splitby="unc",
        include_image_metadata=True,
    )
    samples = _select_samples(refcoco_samples, limit=5)

    for index, sample in enumerate(samples, start=1):
        similarity_map, bbox_mean, outside_mean = _compute_patch_similarity(
            sample=sample,
            vision_model=vision_model,
            text_model=text_model,
            image_processor=image_processor,
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
            f"{index:02d}_{sample.image_id}_{sample.annotation[:40]}{noise_suffix}.png"
        )
        output_path = OUTPUT_DIR / output_name
        _visualize_similarity(
            sample=sample,
            similarity_map=similarity_map,
            save_path=output_path,
            text_noise_scale=text_noise_scale,
            num_text_augmentations=num_text_augmentations,
        )

        print(f"[sample {index}] image_id: {sample.image_id}")
        print(f"[sample {index}] image_path: {sample.image_path}")
        print(f"[sample {index}] annotation: {sample.annotation}")
        print(f"[sample {index}] text_noise_scale: {text_noise_scale}")
        print(f"[sample {index}] num_text_augmentations: {num_text_augmentations}")
        print(f"[sample {index}] bbox: ({sample.bbox.x}, {sample.bbox.y}, {sample.bbox.w}, {sample.bbox.h})")
        print(f"[sample {index}] patch_grid_size: {similarity_map.shape[0]}x{similarity_map.shape[1]}")
        print(f"[sample {index}] bbox_patch_mean_similarity: {bbox_mean:.4f}")
        print(f"[sample {index}] outside_patch_mean_similarity: {outside_mean:.4f}")
        print(f"[sample {index}] saved_heatmap: {output_path.resolve()}")


if __name__ == "__main__":
    main()
