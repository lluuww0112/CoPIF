from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import random
from statistics import median
import sys
import textwrap
from typing import Any

try:
    import hydra
    from hydra.utils import get_original_cwd
    from omegaconf import DictConfig, ListConfig
except ModuleNotFoundError:
    hydra = None
    DictConfig = Any
    ListConfig = tuple

from PIL import Image, ImageDraw, ImageFont


def _resolve_project_path(path_like: str | Path, project_root: Path | None = None) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path
    if project_root is not None:
        return project_root / path
    if hydra is not None:
        return Path(get_original_cwd()) / path
    return Path.cwd() / path


def _resolve_default_input_csv(config: DictConfig) -> Path:
    eval_config = config.get("eval", {})
    invoke_config = config.get("invoke", {})
    train_config = config.get("train", {})

    input_csv_value = eval_config.get("input_csv", None)
    if input_csv_value is not None:
        return _resolve_project_path(str(input_csv_value))

    dataset = str(eval_config.get("dataset", invoke_config.get("dataset", train_config.get("dataset", config.generator.dataset))))
    splitby = str(eval_config.get("splitby", invoke_config.get("splitby", train_config.get("splitby", config.generator.splitby))))
    split = str(eval_config.get("split", invoke_config.get("split", train_config.get("split", config.generator.split))))
    checkpoint_dir = _resolve_project_path(
        str(eval_config.get("checkpoint_dir", invoke_config.get("checkpoint_dir", train_config.get("checkpoint_dir", "history/base"))))
    )
    return checkpoint_dir / f"inference_{dataset}_{splitby}_{split}.csv"


def _find_latest_inference_csv(checkpoint_dir: Path) -> Path:
    candidates = sorted(
        checkpoint_dir.glob("inference_*.csv"),
        key=lambda path: path.stat().st_mtime,
    )
    if not candidates:
        raise FileNotFoundError(
            f"could not find inference_*.csv under checkpoint_dir={checkpoint_dir}"
        )
    return candidates[-1]


def _resolve_thresholds(raw_thresholds: Any) -> list[float]:
    if raw_thresholds is None:
        return [0.25, 0.5, 0.75]

    if isinstance(raw_thresholds, ListConfig):
        values = list(raw_thresholds)
    elif isinstance(raw_thresholds, (list, tuple)):
        values = list(raw_thresholds)
    else:
        values = [raw_thresholds]

    thresholds = [float(value) for value in values]
    if not thresholds:
        raise ValueError("eval.iou_thresholds must not be empty.")
    return thresholds


def _resolve_quantiles(raw_quantiles: Any) -> list[float]:
    if raw_quantiles is None:
        values = [0.0, 25.0, 50.0, 75.0, 100.0]
    elif isinstance(raw_quantiles, ListConfig):
        values = list(raw_quantiles)
    elif isinstance(raw_quantiles, (list, tuple)):
        values = list(raw_quantiles)
    else:
        values = [raw_quantiles]

    quantiles = [float(value) for value in values]
    if not quantiles:
        raise ValueError("eval.bucket_quantiles must not be empty.")

    normalized = [value / 100.0 if value > 1.0 else value for value in quantiles]
    if normalized[0] != 0.0 or normalized[-1] != 1.0:
        raise ValueError("eval.bucket_quantiles must start at 0 and end at 1 (or 0 and 100).")
    if any(value < 0.0 or value > 1.0 for value in normalized):
        raise ValueError("eval.bucket_quantiles must be within [0, 1] or [0, 100].")
    if any(right <= left for left, right in zip(normalized, normalized[1:])):
        raise ValueError("eval.bucket_quantiles must be strictly increasing.")
    return normalized


def _read_rows(input_csv: Path) -> list[dict[str, Any]]:
    if not input_csv.is_file():
        raise FileNotFoundError(f"inference csv not found: {input_csv}")

    with input_csv.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        rows = list(reader)

    if not rows:
        raise ValueError(f"inference csv is empty: {input_csv}")
    return rows


def _parse_float(row: dict[str, Any], key: str) -> float:
    value = row.get(key, None)
    if value in (None, ""):
        raise ValueError(f"missing required column '{key}' in inference csv row.")
    return float(value)


def _xywh_to_xyxy(box: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    x, y, w, h = box
    return x, y, x + max(w, 0.0), y + max(h, 0.0)


def _clip_xyxy(
    box: tuple[float, float, float, float],
    image_width: float,
    image_height: float,
) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = box
    return (
        min(max(x1, 0.0), image_width),
        min(max(y1, 0.0), image_height),
        min(max(x2, 0.0), image_width),
        min(max(y2, 0.0), image_height),
    )


def _box_area(box: tuple[float, float, float, float]) -> float:
    x1, y1, x2, y2 = box
    return max(x2 - x1, 0.0) * max(y2 - y1, 0.0)


def _compute_iou(
    pred_box_xywh: tuple[float, float, float, float],
    gt_box_xywh: tuple[float, float, float, float],
    image_width: float,
    image_height: float,
    clip_to_image: bool,
) -> float:
    pred_xyxy = _xywh_to_xyxy(pred_box_xywh)
    gt_xyxy = _xywh_to_xyxy(gt_box_xywh)

    if clip_to_image:
        pred_xyxy = _clip_xyxy(pred_xyxy, image_width=image_width, image_height=image_height)
        gt_xyxy = _clip_xyxy(gt_xyxy, image_width=image_width, image_height=image_height)

    inter_x1 = max(pred_xyxy[0], gt_xyxy[0])
    inter_y1 = max(pred_xyxy[1], gt_xyxy[1])
    inter_x2 = min(pred_xyxy[2], gt_xyxy[2])
    inter_y2 = min(pred_xyxy[3], gt_xyxy[3])

    inter_area = _box_area((inter_x1, inter_y1, inter_x2, inter_y2))
    pred_area = _box_area(pred_xyxy)
    gt_area = _box_area(gt_xyxy)
    union_area = pred_area + gt_area - inter_area
    if union_area <= 0.0:
        return 0.0
    return inter_area / union_area


def _center_xy(box: tuple[float, float, float, float]) -> tuple[float, float]:
    x, y, w, h = box
    return x + (w / 2.0), y + (h / 2.0)


def _safe_mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _round(value: float, digits: int = 6) -> float:
    return round(float(value), digits)


def _parse_box(row: dict[str, Any], prefix: str) -> tuple[float, float, float, float]:
    return (
        _parse_float(row, f"{prefix}_x"),
        _parse_float(row, f"{prefix}_y"),
        _parse_float(row, f"{prefix}_w"),
        _parse_float(row, f"{prefix}_h"),
    )


def _build_ranked_rows(
    rows: list[dict[str, Any]],
    clip_to_image: bool,
) -> list[dict[str, Any]]:
    ranked_rows: list[dict[str, Any]] = []

    for index, row in enumerate(rows):
        image_width = _parse_float(row, "image_width")
        image_height = _parse_float(row, "image_height")
        pred_box = _parse_box(row, "pred")
        gt_box = _parse_box(row, "gt")
        iou = _compute_iou(
            pred_box_xywh=pred_box,
            gt_box_xywh=gt_box,
            image_width=image_width,
            image_height=image_height,
            clip_to_image=clip_to_image,
        )
        ranked_rows.append(
            {
                "row_index": index,
                "image_id": row.get("image_id"),
                "annotation": row.get("annotation"),
                "image_path": row.get("image_path"),
                "iou": _round(iou),
                "iou_raw": float(iou),
                "pred_box": [_round(value, 4) for value in pred_box],
                "gt_box": [_round(value, 4) for value in gt_box],
                "row": row,
            }
        )

    return ranked_rows


def _format_bucket_label(start: float, end: float) -> str:
    return f"p{int(round(start * 100)):02d}_to_p{int(round(end * 100)):02d}"


def _summarize_ranked_subset(
    bucket_label: str,
    start_percentile: float,
    end_percentile: float,
    ranked_subset: list[dict[str, Any]],
) -> dict[str, Any]:
    ious = [float(entry["iou_raw"]) for entry in ranked_subset]
    return {
        "bucket_label": bucket_label,
        "percentile_range": [_round(start_percentile * 100.0, 2), _round(end_percentile * 100.0, 2)],
        "candidate_count": len(ranked_subset),
        "iou_min": _round(min(ious)) if ious else None,
        "iou_mean": _round(_safe_mean(ious)) if ious else None,
        "iou_max": _round(max(ious)) if ious else None,
    }


def _select_iou_bucket_samples(
    ranked_rows: list[dict[str, Any]],
    quantiles: list[float],
    samples_per_bucket: int,
    seed: int,
) -> list[dict[str, Any]]:
    if samples_per_bucket <= 0 or not ranked_rows:
        return []

    sorted_rows = sorted(ranked_rows, key=lambda row: (row["iou_raw"], row["row_index"]))
    row_count = len(sorted_rows)
    boundaries = [min(row_count, max(0, int(math.floor(row_count * quantile)))) for quantile in quantiles[:-1]]
    boundaries.append(row_count)

    rng = random.Random(seed)
    buckets: list[dict[str, Any]] = []

    for bucket_index, (start, end) in enumerate(zip(quantiles, quantiles[1:])):
        start_index = boundaries[bucket_index]
        end_index = boundaries[bucket_index + 1]
        if end == 1.0:
            end_index = row_count

        ranked_subset = sorted_rows[start_index:end_index]
        bucket_label = _format_bucket_label(start, end)
        summary = _summarize_ranked_subset(
            bucket_label=bucket_label,
            start_percentile=start,
            end_percentile=end,
            ranked_subset=ranked_subset,
        )
        sampled_entries = (
            rng.sample(ranked_subset, k=min(samples_per_bucket, len(ranked_subset)))
            if ranked_subset
            else []
        )
        sampled_entries = sorted(sampled_entries, key=lambda row: (row["iou_raw"], row["row_index"]))
        summary["sampled_count"] = len(sampled_entries)
        summary["samples"] = sampled_entries
        buckets.append(summary)

    return buckets


def _sanitize_slug(value: Any, fallback: str = "sample") -> str:
    cleaned = "".join(character if str(character).isalnum() else "_" for character in str(value))
    cleaned = cleaned.strip("_")
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    if not cleaned:
        return fallback
    return cleaned[:48]


def _load_font(size: int) -> ImageFont.ImageFont | ImageFont.FreeTypeFont:
    for font_name in ("DejaVuSans.ttf", "Arial.ttf"):
        try:
            return ImageFont.truetype(font_name, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _draw_labeled_box(
    draw: ImageDraw.ImageDraw,
    box_xyxy: tuple[float, float, float, float],
    color: tuple[int, int, int],
    label: str,
    font: ImageFont.ImageFont | ImageFont.FreeTypeFont,
    image_width: int,
    image_height: int,
    line_width: int,
) -> None:
    x1, y1, x2, y2 = box_xyxy
    if x2 <= x1 or y2 <= y1:
        return

    draw.rectangle((x1, y1, x2, y2), outline=color, width=line_width)

    padding = max(4, line_width)
    text_bbox = draw.textbbox((0, 0), label, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]

    label_x = max(0.0, min(x1, image_width - text_width - (padding * 2)))
    label_y = y1 - text_height - (padding * 2) - 2
    if label_y < 0:
        label_y = min(y1 + 2, image_height - text_height - (padding * 2))

    label_box = (
        label_x,
        label_y,
        label_x + text_width + (padding * 2),
        label_y + text_height + (padding * 2),
    )
    draw.rounded_rectangle(label_box, radius=6, fill=(*color, 230))
    draw.text((label_x + padding, label_y + padding), label, fill=(255, 255, 255, 255), font=font)


def _build_overlay_lines(
    image_id: Any,
    annotation: str,
    iou: float,
    image_width: int,
    font_size: int,
) -> list[str]:
    wrap_width = max(28, image_width // max(font_size, 12))
    wrapped_annotation = textwrap.wrap(annotation, width=wrap_width) or ["(empty annotation)"]
    image_id_text = str(image_id) if image_id not in (None, "") else "(unknown)"

    lines = [
        f"Image ID: {image_id_text}",
        f"IoU: {iou:.4f} | GT: green | Pred: red",
        f"Ref: {wrapped_annotation[0]}",
    ]
    lines.extend(f"     {line}" for line in wrapped_annotation[1:])
    return lines


def _resolve_heatmap_device(requested_device: str | None) -> Any:
    import torch

    if requested_device == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if requested_device == "cuda":
        print("heatmap cuda is not available. Falling back to cpu.")
    return torch.device("cpu")


def _load_heatmap_components(
    pretrained_name: str,
    requested_device: str | None,
) -> dict[str, Any]:
    import torch
    from transformers import CLIPImageProcessor, CLIPTokenizer

    from model.base.clip_model import CLIPTextModel, CLIPVisionModel

    device = _resolve_heatmap_device(requested_device=requested_device)
    vision_model = CLIPVisionModel(pretrained_name)
    text_model = CLIPTextModel(pretrained_name)
    image_processor = CLIPImageProcessor.from_pretrained(pretrained_name)
    tokenizer = CLIPTokenizer.from_pretrained(pretrained_name)

    vision_model.to(device).eval()
    text_model.to(device).eval()

    return {
        "device": device,
        "vision_model": vision_model,
        "text_model": text_model,
        "image_processor": image_processor,
        "tokenizer": tokenizer,
        "torch": torch,
    }


def _bbox_to_grid_coords(
    box_xywh: tuple[float, float, float, float],
    image_width: float,
    image_height: float,
    grid_size: int,
) -> tuple[int, int, int, int]:
    x0 = max(0.0, box_xywh[0])
    y0 = max(0.0, box_xywh[1])
    x1 = min(image_width, box_xywh[0] + max(box_xywh[2], 0.0))
    y1 = min(image_height, box_xywh[1] + max(box_xywh[3], 0.0))

    grid_x0 = min(grid_size - 1, int((x0 / image_width) * grid_size))
    grid_y0 = min(grid_size - 1, int((y0 / image_height) * grid_size))
    grid_x1 = max(grid_x0 + 1, min(grid_size, math.ceil((x1 / image_width) * grid_size)))
    grid_y1 = max(grid_y0 + 1, min(grid_size, math.ceil((y1 / image_height) * grid_size)))
    return grid_x0, grid_y0, grid_x1, grid_y1


def _compute_similarity_map(
    row: dict[str, Any],
    heatmap_components: dict[str, Any],
) -> tuple[Any, float, float]:
    torch = heatmap_components["torch"]

    image_path = Path(str(row["image_path"]))
    annotation = str(row.get("annotation", ""))
    if not image_path.is_file():
        raise FileNotFoundError(f"image file not found for heatmap: {image_path}")

    with Image.open(image_path) as image_file:
        image = image_file.convert("RGB")

    image_inputs = heatmap_components["image_processor"](images=image, return_tensors="pt")
    image_inputs = {key: value.to(heatmap_components["device"]) for key, value in image_inputs.items()}

    text_inputs = heatmap_components["tokenizer"](
        [annotation],
        padding=True,
        truncation=True,
        return_tensors="pt",
    )
    text_inputs = {key: value.to(heatmap_components["device"]) for key, value in text_inputs.items()}

    with torch.no_grad():
        patch_embeddings = heatmap_components["vision_model"](image_inputs["pixel_values"]).squeeze(0)
        text_embedding = heatmap_components["text_model"](**text_inputs).squeeze(0)

    patch_embeddings = torch.nn.functional.normalize(patch_embeddings, dim=-1)
    text_embedding = torch.nn.functional.normalize(text_embedding, dim=-1)
    similarity = torch.matmul(patch_embeddings, text_embedding)

    patch_count = similarity.shape[0]
    grid_size = int(math.sqrt(patch_count))
    if grid_size * grid_size != patch_count:
        raise ValueError(f"patch count must be a square number, got {patch_count}.")

    similarity_map = similarity.reshape(grid_size, grid_size).detach().cpu()
    gt_box = _parse_box(row, "gt")
    image_width = _parse_float(row, "image_width")
    image_height = _parse_float(row, "image_height")
    grid_x0, grid_y0, grid_x1, grid_y1 = _bbox_to_grid_coords(
        box_xywh=gt_box,
        image_width=image_width,
        image_height=image_height,
        grid_size=grid_size,
    )
    inside = similarity_map[grid_y0:grid_y1, grid_x0:grid_x1]
    bbox_mean = inside.mean().item()

    outside_mask = torch.ones_like(similarity_map, dtype=torch.bool)
    outside_mask[grid_y0:grid_y1, grid_x0:grid_x1] = False
    outside_mean = similarity_map[outside_mask].mean().item()
    return similarity_map, bbox_mean, outside_mean


def _render_bucket_visualization(
    row: dict[str, Any],
    ranked_entry: dict[str, Any],
    bucket_label: str,
    sample_index: int,
    output_dir: Path,
    clip_to_image: bool,
    heatmap_components: dict[str, Any] | None,
) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    image_path = Path(str(row["image_path"]))
    if not image_path.is_file():
        raise FileNotFoundError(f"image file not found: {image_path}")

    image_width = _parse_float(row, "image_width")
    image_height = _parse_float(row, "image_height")
    pred_box = _parse_box(row, "pred")
    gt_box = _parse_box(row, "gt")

    if clip_to_image:
        pred_xyxy = _clip_xyxy(_xywh_to_xyxy(pred_box), image_width=image_width, image_height=image_height)
        gt_xyxy = _clip_xyxy(_xywh_to_xyxy(gt_box), image_width=image_width, image_height=image_height)
        pred_box = (pred_xyxy[0], pred_xyxy[1], pred_xyxy[2] - pred_xyxy[0], pred_xyxy[3] - pred_xyxy[1])
        gt_box = (gt_xyxy[0], gt_xyxy[1], gt_xyxy[2] - gt_xyxy[0], gt_xyxy[3] - gt_xyxy[1])

    with Image.open(image_path) as image_file:
        image = image_file.convert("RGB")

    if heatmap_components is not None:
        similarity_map, bbox_mean, outside_mean = _compute_similarity_map(
            row=row,
            heatmap_components=heatmap_components,
        )
        grid_size = int(similarity_map.shape[0])
        gt_grid = _bbox_to_grid_coords(gt_box, image_width=image_width, image_height=image_height, grid_size=grid_size)
        pred_grid = _bbox_to_grid_coords(pred_box, image_width=image_width, image_height=image_height, grid_size=grid_size)
    else:
        similarity_map = None
        bbox_mean = None
        outside_mean = None
        gt_grid = None
        pred_grid = None

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)

    axes[0].imshow(image)
    axes[0].add_patch(
        Rectangle((gt_box[0], gt_box[1]), gt_box[2], gt_box[3], fill=False, edgecolor="lime", linewidth=2)
    )
    axes[0].add_patch(
        Rectangle((pred_box[0], pred_box[1]), pred_box[2], pred_box[3], fill=False, edgecolor="red", linewidth=2)
    )
    axes[0].set_title("Prediction vs GT")
    axes[0].axis("off")

    if similarity_map is not None and gt_grid is not None and pred_grid is not None:
        heatmap = axes[1].imshow(similarity_map.numpy(), cmap="magma", interpolation="nearest")
        axes[1].add_patch(
            Rectangle(
                (gt_grid[0] - 0.5, gt_grid[1] - 0.5),
                gt_grid[2] - gt_grid[0],
                gt_grid[3] - gt_grid[1],
                fill=False,
                edgecolor="cyan",
                linewidth=2,
            )
        )
        axes[1].add_patch(
            Rectangle(
                (pred_grid[0] - 0.5, pred_grid[1] - 0.5),
                pred_grid[2] - pred_grid[0],
                pred_grid[3] - pred_grid[1],
                fill=False,
                edgecolor="white",
                linewidth=2,
                linestyle="--",
            )
        )
        axes[1].set_title("Patch-text heatmap")
        axes[1].set_xlabel("patch x")
        axes[1].set_ylabel("patch y")
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
            Rectangle((gt_box[0], gt_box[1]), gt_box[2], gt_box[3], fill=False, edgecolor="white", linewidth=2)
        )
        axes[2].add_patch(
            Rectangle((pred_box[0], pred_box[1]), pred_box[2], pred_box[3], fill=False, edgecolor="red", linewidth=2)
        )
        axes[2].set_title("Heatmap overlay")
        axes[2].axis("off")
    else:
        axes[1].imshow(image)
        axes[1].set_title("Heatmap disabled")
        axes[1].axis("off")
        axes[2].imshow(image)
        axes[2].set_title("Overlay skipped")
        axes[2].axis("off")

    wrapped_annotation = "\n".join(textwrap.wrap(f'annotation: "{row.get("annotation", "")}"', width=70))
    stats = [
        f"bucket={bucket_label}",
        f"row={ranked_entry['row_index']}",
        f"image_id={row.get('image_id', '(unknown)')}",
        f"iou={ranked_entry['iou']:.4f}",
    ]
    if bbox_mean is not None and outside_mean is not None:
        stats.append(f"gt_patch_mean={bbox_mean:.4f}")
        stats.append(f"outside_patch_mean={outside_mean:.4f}")

    fig.suptitle("IoU bucket benchmark", fontsize=14, fontweight="bold")
    fig.text(
        0.5,
        0.02,
        f"{wrapped_annotation}\n{' | '.join(stats)}",
        ha="center",
        va="bottom",
        fontsize=11,
        bbox={"boxstyle": "round,pad=0.4", "facecolor": "white", "alpha": 0.92, "edgecolor": "0.7"},
    )

    image_id_slug = _sanitize_slug(row.get("image_id", "image"), fallback="image")
    output_path = output_dir / bucket_label / f"sample_{sample_index:02d}_row_{ranked_entry['row_index']:05d}_{image_id_slug}.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _build_bucket_payload_sample(
    ranked_entry: dict[str, Any],
    visualization_path: str | None,
) -> dict[str, Any]:
    payload = {
        key: value
        for key, value in ranked_entry.items()
        if key != "row"
    }
    payload["visualization_path"] = visualization_path
    return payload


def _visualize_iou_buckets(
    ranked_rows: list[dict[str, Any]],
    output_dir: Path | None,
    quantiles: list[float],
    samples_per_bucket: int,
    seed: int,
    clip_to_image: bool,
    heatmap_enabled: bool,
    pretrained_name: str,
    requested_device: str | None,
) -> dict[str, Any]:
    bucket_payload = {
        "quantiles": [_round(value * 100.0, 2) for value in quantiles],
        "samples_per_bucket": samples_per_bucket,
        "seed": seed,
        "heatmap_enabled": heatmap_enabled,
        "buckets": [],
        "failures": [],
    }

    if output_dir is None or samples_per_bucket <= 0:
        return bucket_payload

    selected_buckets = _select_iou_bucket_samples(
        ranked_rows=ranked_rows,
        quantiles=quantiles,
        samples_per_bucket=samples_per_bucket,
        seed=seed,
    )
    if not selected_buckets:
        return bucket_payload

    heatmap_components = None
    if heatmap_enabled:
        heatmap_components = _load_heatmap_components(
            pretrained_name=pretrained_name,
            requested_device=requested_device,
        )

    saved_count = 0
    for bucket in selected_buckets:
        bucket_summary = {
            key: value
            for key, value in bucket.items()
            if key != "samples"
        }
        bucket_summary["samples"] = []

        for sample_index, ranked_entry in enumerate(bucket["samples"], start=1):
            visualization_path = None
            try:
                saved_path = _render_bucket_visualization(
                    row=ranked_entry["row"],
                    ranked_entry=ranked_entry,
                    bucket_label=str(bucket["bucket_label"]),
                    sample_index=sample_index,
                    output_dir=output_dir,
                    clip_to_image=clip_to_image,
                    heatmap_components=heatmap_components,
                )
                visualization_path = str(saved_path)
                saved_count += 1
            except Exception as error:
                bucket_payload["failures"].append(
                    f"bucket={bucket['bucket_label']}, row={ranked_entry['row_index']}: {error}"
                )

            bucket_summary["samples"].append(
                _build_bucket_payload_sample(
                    ranked_entry=ranked_entry,
                    visualization_path=visualization_path,
                )
            )

        bucket_payload["buckets"].append(bucket_summary)

    if saved_count > 0:
        print(f"saved {saved_count} IoU bucket visualization(s) to {output_dir}")
    if bucket_payload["failures"]:
        print("iou_bucket_visualization_failures:")
        for failure in bucket_payload["failures"]:
            print(f"  {failure}")

    return bucket_payload


def _render_visualization(
    row: dict[str, Any],
    row_index: int,
    sample_index: int,
    output_dir: Path,
    clip_to_image: bool,
) -> Path:
    image_path_value = row.get("image_path", None)
    if image_path_value in (None, ""):
        raise ValueError("missing image_path in inference csv row.")

    image_path = Path(str(image_path_value))
    if not image_path.is_file():
        raise FileNotFoundError(f"image file not found: {image_path}")

    image_width = int(round(_parse_float(row, "image_width")))
    image_height = int(round(_parse_float(row, "image_height")))
    pred_box = _parse_box(row, "pred")
    gt_box = _parse_box(row, "gt")
    iou = _compute_iou(
        pred_box_xywh=pred_box,
        gt_box_xywh=gt_box,
        image_width=float(image_width),
        image_height=float(image_height),
        clip_to_image=clip_to_image,
    )

    pred_xyxy = _xywh_to_xyxy(pred_box)
    gt_xyxy = _xywh_to_xyxy(gt_box)
    if clip_to_image:
        pred_xyxy = _clip_xyxy(pred_xyxy, image_width=float(image_width), image_height=float(image_height))
        gt_xyxy = _clip_xyxy(gt_xyxy, image_width=float(image_width), image_height=float(image_height))

    with Image.open(image_path) as image_file:
        image = image_file.convert("RGBA")

    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    box_font_size = max(14, min(image.size) // 28)
    info_font_size = max(16, min(image.size) // 24)
    box_font = _load_font(box_font_size)
    info_font = _load_font(info_font_size)
    line_width = max(3, min(image.size) // 160)

    _draw_labeled_box(
        draw=draw,
        box_xyxy=gt_xyxy,
        color=(46, 204, 113),
        label="GT",
        font=box_font,
        image_width=image.width,
        image_height=image.height,
        line_width=line_width,
    )
    _draw_labeled_box(
        draw=draw,
        box_xyxy=pred_xyxy,
        color=(231, 76, 60),
        label="Pred",
        font=box_font,
        image_width=image.width,
        image_height=image.height,
        line_width=line_width,
    )

    overlay_lines = _build_overlay_lines(
        image_id=row.get("image_id"),
        annotation=str(row.get("annotation", "")),
        iou=iou,
        image_width=image.width,
        font_size=info_font_size,
    )
    result = Image.alpha_composite(image, overlay).convert("RGB")

    caption_padding = 14
    caption_spacing = 6
    caption_text = "\n".join(overlay_lines)
    caption_probe = Image.new("RGB", (image.width, 1), (248, 246, 240))
    caption_draw = ImageDraw.Draw(caption_probe)
    text_bbox = caption_draw.multiline_textbbox((0, 0), caption_text, font=info_font, spacing=caption_spacing)
    caption_height = max(
        info_font_size + (caption_padding * 2),
        (text_bbox[3] - text_bbox[1]) + (caption_padding * 2),
    )
    canvas = Image.new("RGB", (image.width, image.height + caption_height), (248, 246, 240))
    canvas.paste(result, (0, 0))

    caption_draw = ImageDraw.Draw(canvas)
    caption_top = image.height
    caption_draw.rectangle(
        (0, caption_top, image.width, image.height + caption_height),
        fill=(248, 246, 240),
    )
    caption_draw.line(
        (0, caption_top, image.width, caption_top),
        fill=(210, 206, 198),
        width=max(1, line_width // 2),
    )
    caption_draw.multiline_text(
        (caption_padding, caption_top + caption_padding),
        caption_text,
        fill=(32, 32, 32),
        font=info_font,
        spacing=caption_spacing,
    )

    image_id_slug = _sanitize_slug(row.get("image_id", "image"), fallback="image")
    output_path = output_dir / f"sample_{sample_index:02d}_row_{row_index:05d}_{image_id_slug}.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
    return output_path


def _select_visualization_rows(
    rows: list[dict[str, Any]],
    num_samples: int,
    seed: int,
) -> list[tuple[int, dict[str, Any]]]:
    if num_samples <= 0:
        return []

    grouped_rows: dict[str, list[tuple[int, dict[str, Any]]]] = {}
    for row_index, row in enumerate(rows):
        group_key = str(row.get("image_path", row.get("image_id", f"row-{row_index}")))
        grouped_rows.setdefault(group_key, []).append((row_index, row))

    rng = random.Random(seed)
    group_keys = list(grouped_rows.keys())
    rng.shuffle(group_keys)

    selected_rows: list[tuple[int, dict[str, Any]]] = []
    for group_key in group_keys[:num_samples]:
        candidates = grouped_rows[group_key]
        selected_rows.append(rng.choice(candidates))
    return selected_rows


def _visualize_samples(
    rows: list[dict[str, Any]],
    output_dir: Path | None,
    num_samples: int,
    seed: int,
    clip_to_image: bool,
) -> list[str]:
    if output_dir is None or num_samples <= 0:
        return []

    selected_rows = _select_visualization_rows(rows=rows, num_samples=num_samples, seed=seed)
    if not selected_rows:
        return []

    saved_paths: list[str] = []
    failures: list[str] = []

    for sample_index, (row_index, row) in enumerate(selected_rows, start=1):
        try:
            saved_path = _render_visualization(
                row=row,
                row_index=row_index,
                sample_index=sample_index,
                output_dir=output_dir,
                clip_to_image=clip_to_image,
            )
            saved_paths.append(str(saved_path))
        except Exception as error:
            failures.append(f"row={row_index}: {error}")

    if saved_paths:
        print(f"saved {len(saved_paths)} visualization(s) to {output_dir}")
    if failures:
        print("visualization_failures:")
        for failure in failures:
            print(f"  {failure}")

    return saved_paths


def _build_summary(
    rows: list[dict[str, Any]],
    thresholds: list[float],
    clip_to_image: bool,
    worst_k: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    ranked_rows = _build_ranked_rows(rows=rows, clip_to_image=clip_to_image)
    ious: list[float] = []
    pixel_abs_errors = {"x": [], "y": [], "w": [], "h": []}
    norm_abs_errors = {"x": [], "y": [], "w": [], "h": []}
    center_errors_px: list[float] = []
    center_errors_norm: list[float] = []
    area_ratios: list[float] = []
    unique_images: set[str] = set()
    unique_instances: set[tuple[str, float, float, float, float]] = set()

    for index, row in enumerate(rows):
        image_width = _parse_float(row, "image_width")
        image_height = _parse_float(row, "image_height")
        pred_box = (
            _parse_float(row, "pred_x"),
            _parse_float(row, "pred_y"),
            _parse_float(row, "pred_w"),
            _parse_float(row, "pred_h"),
        )
        gt_box = (
            _parse_float(row, "gt_x"),
            _parse_float(row, "gt_y"),
            _parse_float(row, "gt_w"),
            _parse_float(row, "gt_h"),
        )
        pred_box_norm = (
            _parse_float(row, "pred_x_norm"),
            _parse_float(row, "pred_y_norm"),
            _parse_float(row, "pred_w_norm"),
            _parse_float(row, "pred_h_norm"),
        )
        gt_box_norm = (
            _parse_float(row, "gt_x_norm"),
            _parse_float(row, "gt_y_norm"),
            _parse_float(row, "gt_w_norm"),
            _parse_float(row, "gt_h_norm"),
        )

        iou = _compute_iou(
            pred_box_xywh=pred_box,
            gt_box_xywh=gt_box,
            image_width=image_width,
            image_height=image_height,
            clip_to_image=clip_to_image,
        )
        ious.append(iou)

        for key, pred_value, gt_value in zip(("x", "y", "w", "h"), pred_box, gt_box):
            pixel_abs_errors[key].append(abs(pred_value - gt_value))

        for key, pred_value, gt_value in zip(("x", "y", "w", "h"), pred_box_norm, gt_box_norm):
            norm_abs_errors[key].append(abs(pred_value - gt_value))

        pred_center_x, pred_center_y = _center_xy(pred_box)
        gt_center_x, gt_center_y = _center_xy(gt_box)
        center_errors_px.append(math.hypot(pred_center_x - gt_center_x, pred_center_y - gt_center_y))

        pred_center_x_norm, pred_center_y_norm = _center_xy(pred_box_norm)
        gt_center_x_norm, gt_center_y_norm = _center_xy(gt_box_norm)
        center_errors_norm.append(
            math.hypot(pred_center_x_norm - gt_center_x_norm, pred_center_y_norm - gt_center_y_norm)
        )

        gt_area = max(gt_box[2], 0.0) * max(gt_box[3], 0.0)
        pred_area = max(pred_box[2], 0.0) * max(pred_box[3], 0.0)
        if gt_area > 0.0:
            area_ratios.append(pred_area / gt_area)

        image_key = str(row.get("image_id", row.get("image_path", f"row-{index}")))
        unique_images.add(image_key)
        unique_instances.add(
            (
                image_key,
                round(gt_box[0], 4),
                round(gt_box[1], 4),
                round(gt_box[2], 4),
                round(gt_box[3], 4),
            )
        )

    threshold_metrics = {
        f"acc@{threshold:.2f}": _round(sum(iou >= threshold for iou in ious) / len(ious))
        for threshold in thresholds
    }

    summary = {
        "row_count": len(rows),
        "unique_image_count": len(unique_images),
        "unique_instance_count": len(unique_instances),
        "clip_to_image": clip_to_image,
        "mean_iou": _round(_safe_mean(ious)),
        "median_iou": _round(median(ious)),
        "min_iou": _round(min(ious)),
        "max_iou": _round(max(ious)),
        "pixel_mae": {key: _round(_safe_mean(values), 4) for key, values in pixel_abs_errors.items()},
        "norm_mae": {key: _round(_safe_mean(values), 6) for key, values in norm_abs_errors.items()},
        "mean_center_error_px": _round(_safe_mean(center_errors_px), 4),
        "mean_center_error_norm": _round(_safe_mean(center_errors_norm)),
        "mean_pred_area_over_gt_area": _round(_safe_mean(area_ratios)),
        "threshold_metrics": threshold_metrics,
    }

    example_count = max(worst_k, 0)
    summarized_ranked_rows = [
        {
            key: value
            for key, value in ranked_row.items()
            if key not in {"row", "iou_raw"}
        }
        for ranked_row in ranked_rows
    ]
    worst_examples = sorted(summarized_ranked_rows, key=lambda row: (row["iou"], row["row_index"]))[:example_count]
    best_examples = sorted(summarized_ranked_rows, key=lambda row: (-row["iou"], row["row_index"]))[:example_count]
    return summary, worst_examples, best_examples


def _print_summary(
    input_csv: Path,
    summary: dict[str, Any],
    worst_examples: list[dict[str, Any]],
    best_examples: list[dict[str, Any]],
) -> None:
    print(f"input_csv: {input_csv}")
    print(f"row_count: {summary['row_count']}")
    print(f"unique_image_count: {summary['unique_image_count']}")
    print(f"unique_instance_count: {summary['unique_instance_count']}")
    print(f"clip_to_image: {summary['clip_to_image']}")
    print(
        "iou: "
        f"mean={summary['mean_iou']:.6f}, "
        f"median={summary['median_iou']:.6f}, "
        f"min={summary['min_iou']:.6f}, "
        f"max={summary['max_iou']:.6f}"
    )
    threshold_summary = ", ".join(
        f"{metric}={value:.6f}" for metric, value in summary["threshold_metrics"].items()
    )
    print(f"threshold_metrics: {threshold_summary}")
    print(
        "pixel_mae: "
        f"x={summary['pixel_mae']['x']:.4f}, "
        f"y={summary['pixel_mae']['y']:.4f}, "
        f"w={summary['pixel_mae']['w']:.4f}, "
        f"h={summary['pixel_mae']['h']:.4f}"
    )
    print(
        "norm_mae: "
        f"x={summary['norm_mae']['x']:.6f}, "
        f"y={summary['norm_mae']['y']:.6f}, "
        f"w={summary['norm_mae']['w']:.6f}, "
        f"h={summary['norm_mae']['h']:.6f}"
    )
    print(f"mean_center_error_px: {summary['mean_center_error_px']:.4f}")
    print(f"mean_center_error_norm: {summary['mean_center_error_norm']:.6f}")
    print(f"mean_pred_area_over_gt_area: {summary['mean_pred_area_over_gt_area']:.6f}")

    if worst_examples:
        print("worst_examples:")
        for example in worst_examples:
            print(
                f"  row={example['row_index']}, "
                f"image_id={example['image_id']}, "
                f"iou={example['iou']:.6f}, "
                f"annotation={example['annotation']}"
            )

    if best_examples:
        print("best_examples:")
        for example in best_examples:
            print(
                f"  row={example['row_index']}, "
                f"image_id={example['image_id']}, "
                f"iou={example['iou']:.6f}, "
                f"annotation={example['annotation']}"
            )


def _run_evaluation(
    input_csv: Path,
    output_json: Path | None,
    thresholds: list[float],
    clip_to_image: bool,
    worst_k: int,
    visualization_dir: Path | None,
    visualization_count: int,
    visualization_seed: int,
    bucket_quantiles: list[float],
    bucket_samples_per_bucket: int,
    bucket_seed: int,
    heatmap_enabled: bool,
    heatmap_pretrained_name: str,
    heatmap_device: str | None,
) -> None:
    rows = _read_rows(input_csv=input_csv)
    ranked_rows = _build_ranked_rows(rows=rows, clip_to_image=clip_to_image)
    summary, worst_examples, best_examples = _build_summary(
        rows=rows,
        thresholds=thresholds,
        clip_to_image=clip_to_image,
        worst_k=worst_k,
    )
    _print_summary(
        input_csv=input_csv,
        summary=summary,
        worst_examples=worst_examples,
        best_examples=best_examples,
    )
    visualization_paths = _visualize_samples(
        rows=rows,
        output_dir=visualization_dir,
        num_samples=visualization_count,
        seed=visualization_seed,
        clip_to_image=clip_to_image,
    )
    bucket_visualization_dir = (
        visualization_dir / "iou_buckets"
        if visualization_dir is not None
        else input_csv.parent / f"{input_csv.stem}_iou_buckets"
    )
    iou_bucket_benchmarks = _visualize_iou_buckets(
        ranked_rows=ranked_rows,
        output_dir=bucket_visualization_dir,
        quantiles=bucket_quantiles,
        samples_per_bucket=bucket_samples_per_bucket,
        seed=bucket_seed,
        clip_to_image=clip_to_image,
        heatmap_enabled=heatmap_enabled,
        pretrained_name=heatmap_pretrained_name,
        requested_device=heatmap_device,
    )

    if output_json is not None:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "input_csv": str(input_csv),
            "summary": summary,
            "worst_examples": worst_examples,
            "best_examples": best_examples,
            "visualizations": visualization_paths,
            "iou_bucket_benchmarks": iou_bucket_benchmarks,
        }
        with output_json.open("w", encoding="utf-8") as file:
            json.dump(payload, file, ensure_ascii=False, indent=2)
        print(f"saved evaluation summary to {output_json}")


def _hydra_entry(config: DictConfig) -> None:
    eval_config = config.get("eval", {})
    input_csv = _resolve_default_input_csv(config)
    output_json_value = eval_config.get("output_json", None)
    output_json = _resolve_project_path(str(output_json_value)) if output_json_value is not None else None
    visualization_dir_value = eval_config.get("visualization_dir", None)
    visualization_dir = (
        _resolve_project_path(str(visualization_dir_value))
        if visualization_dir_value is not None
        else input_csv.parent / f"{input_csv.stem}_visualizations"
    )
    thresholds = _resolve_thresholds(eval_config.get("iou_thresholds", None))
    clip_to_image = bool(eval_config.get("clip_to_image", True))
    worst_k = int(eval_config.get("worst_k", 5))
    visualization_count = int(eval_config.get("visualization_count", 10))
    visualization_seed = int(eval_config.get("visualization_seed", 42))
    bucket_quantiles = _resolve_quantiles(eval_config.get("bucket_quantiles", None))
    bucket_samples_per_bucket = int(eval_config.get("bucket_samples_per_bucket", 5))
    bucket_seed = int(eval_config.get("bucket_seed", visualization_seed))
    heatmap_enabled = bool(eval_config.get("heatmap_enabled", True))
    heatmap_pretrained_name = str(eval_config.get("heatmap_pretrained_name", config.shared.pretrained_clip))
    heatmap_device_value = eval_config.get("heatmap_device", None)
    heatmap_device = str(heatmap_device_value) if heatmap_device_value not in (None, "") else None

    _run_evaluation(
        input_csv=input_csv,
        output_json=output_json,
        thresholds=thresholds,
        clip_to_image=clip_to_image,
        worst_k=worst_k,
        visualization_dir=visualization_dir,
        visualization_count=visualization_count,
        visualization_seed=visualization_seed,
        bucket_quantiles=bucket_quantiles,
        bucket_samples_per_bucket=bucket_samples_per_bucket,
        bucket_seed=bucket_seed,
        heatmap_enabled=heatmap_enabled,
        heatmap_pretrained_name=heatmap_pretrained_name,
        heatmap_device=heatmap_device,
    )


def _parse_cli_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize IoU and bbox errors from an inference CSV generated by model/base/invoke.py."
    )
    parser.add_argument("--input-csv", type=str, default=None, help="Path to the inference CSV.")
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="history/base",
        help="Directory to search for the latest inference CSV when --input-csv is omitted.",
    )
    parser.add_argument("--output-json", type=str, default=None, help="Optional path to save the summary as JSON.")
    parser.add_argument(
        "--iou-thresholds",
        type=float,
        nargs="+",
        default=[0.25, 0.5, 0.75],
        help="IoU thresholds used for accuracy metrics.",
    )
    parser.add_argument(
        "--clip-to-image",
        dest="clip_to_image",
        action="store_true",
        help="Clip predicted and GT boxes to image boundaries before IoU computation.",
    )
    parser.add_argument(
        "--no-clip-to-image",
        dest="clip_to_image",
        action="store_false",
        help="Do not clip boxes to image boundaries before IoU computation.",
    )
    parser.set_defaults(clip_to_image=True)
    parser.add_argument(
        "--worst-k",
        type=int,
        default=5,
        help="Number of lowest/highest-IoU rows to print as failure/success examples.",
    )
    parser.add_argument(
        "--visualization-dir",
        type=str,
        default=None,
        help="Directory where sampled visualization PNG files will be saved. Defaults to <input_csv_stem>_visualizations.",
    )
    parser.add_argument(
        "--visualization-count",
        type=int,
        default=10,
        help="Number of random images to visualize. Set to 0 to disable image outputs.",
    )
    parser.add_argument(
        "--visualization-seed",
        type=int,
        default=42,
        help="Random seed used when sampling images for visualization.",
    )
    parser.add_argument(
        "--bucket-quantiles",
        type=float,
        nargs="+",
        dest="bucket_quantiles",
        default=[0.0, 25.0, 50.0, 75.0, 100.0],
        help="IoU quantile bucket boundaries. Defaults to quartiles and accepts either [0, 25, 50, 75, 100] or [0, 0.25, 0.5, 0.75, 1].",
    )
    parser.add_argument(
        "--bucket-samples-per-bucket",
        type=int,
        default=5,
        help="Number of rows to sample from each IoU quantile bucket.",
    )
    parser.add_argument(
        "--bucket-seed",
        type=int,
        default=42,
        help="Random seed used when sampling rows within each IoU quantile bucket.",
    )
    parser.add_argument(
        "--heatmap-device",
        type=str,
        default=None,
        help="Device for patch-text heatmap generation. Defaults to cpu unless cuda is requested and available.",
    )
    parser.add_argument(
        "--heatmap-pretrained-name",
        type=str,
        default="openai/clip-vit-large-patch14",
        help="Pretrained CLIP checkpoint used for MaskCLIP-style heatmap generation.",
    )
    parser.add_argument(
        "--disable-heatmap",
        dest="heatmap_enabled",
        action="store_false",
        help="Disable MaskCLIP-style patch-text heatmap rendering for IoU bucket samples.",
    )
    parser.set_defaults(heatmap_enabled=True)
    return parser.parse_args(argv)


def _cli_entry(argv: list[str]) -> None:
    args = _parse_cli_args(argv)
    project_root = Path.cwd()

    if args.input_csv is not None:
        input_csv = _resolve_project_path(args.input_csv, project_root=project_root)
    else:
        checkpoint_dir = _resolve_project_path(args.checkpoint_dir, project_root=project_root)
        input_csv = _find_latest_inference_csv(checkpoint_dir=checkpoint_dir)

    output_json = (
        _resolve_project_path(args.output_json, project_root=project_root)
        if args.output_json is not None
        else None
    )
    visualization_dir = (
        _resolve_project_path(args.visualization_dir, project_root=project_root)
        if args.visualization_dir is not None
        else input_csv.parent / f"{input_csv.stem}_visualizations"
    )
    _run_evaluation(
        input_csv=input_csv,
        output_json=output_json,
        thresholds=_resolve_thresholds(args.iou_thresholds),
        clip_to_image=bool(args.clip_to_image),
        worst_k=int(args.worst_k),
        visualization_dir=visualization_dir,
        visualization_count=int(args.visualization_count),
        visualization_seed=int(args.visualization_seed),
        bucket_quantiles=_resolve_quantiles(args.bucket_quantiles),
        bucket_samples_per_bucket=int(args.bucket_samples_per_bucket),
        bucket_seed=int(args.bucket_seed),
        heatmap_enabled=bool(args.heatmap_enabled),
        heatmap_pretrained_name=str(args.heatmap_pretrained_name),
        heatmap_device=str(args.heatmap_device) if args.heatmap_device is not None else None,
    )


if hydra is not None:
    main = hydra.main(version_base=None, config_path="../../config", config_name="base")(_hydra_entry)
else:
    main = None


if __name__ == "__main__":
    use_argparse = hydra is None or any(argument.startswith("--") for argument in sys.argv[1:])
    if use_argparse:
        _cli_entry(sys.argv[1:])
    elif main is not None:
        main()
