from __future__ import annotations

from pathlib import Path
import math
import re
import textwrap

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
from transformers import CLIPImageProcessor, CLIPTokenizer

from APIs.refcocoAPI import RefcocoModel, load_refcoco
from model.base.clip_model import CLIPTextModel, CLIPVisionModel


OUTPUT_DIR = Path("test") / "res" / "maskclip_noun_heatmaps"
AVERAGE_HEATMAP_LABEL = "__average_heatmap__"


def _sanitize_filename(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", value).strip("_") or "sample"


def _load_sample_image(sample: RefcocoModel) -> Image.Image:
    if sample.image_path is None:
        raise ValueError("sample.image_path is required.")
    return Image.open(sample.image_path).convert("RGB")


def _resolve_device(config: DictConfig) -> torch.device:
    requested_device = str(config.generator.device)
    if requested_device == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


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


def _group_samples_by_image_id(samples: list[RefcocoModel]) -> dict[str, list[RefcocoModel]]:
    grouped_samples: dict[str, list[RefcocoModel]] = {}
    for sample in samples:
        if sample.image_path is None or sample.size is None:
            continue
        if not Path(sample.image_path).is_file():
            continue
        grouped_samples.setdefault(sample.image_id, []).append(sample)

    for image_samples in grouped_samples.values():
        image_samples.sort(key=lambda sample: sample.annotation)

    return grouped_samples


def _find_reference_sample(
    samples: list[RefcocoModel],
    target_image_id: str,
    target_annotation: str,
) -> tuple[RefcocoModel, list[RefcocoModel], int]:
    grouped_samples = _group_samples_by_image_id(samples)
    image_samples = grouped_samples.get(target_image_id)
    if not image_samples:
        example_ids = list(grouped_samples)[:10]
        raise FileNotFoundError(
            f"No valid RefCOCO samples found for image_id={target_image_id}. "
            f"Example available image_ids: {example_ids}"
        )

    normalized_target = _normalize_term(target_annotation)
    for index, sample in enumerate(image_samples):
        if _normalize_term(sample.annotation) == normalized_target:
            return sample, image_samples, index

    available_annotations = [sample.annotation for sample in image_samples[:10]]
    raise ValueError(
        f"Could not find annotation '{target_annotation}' for image_id={target_image_id}. "
        f"Example annotations: {available_annotations}"
    )


def _normalize_term(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _resolve_target_texts(maskclip_config: DictConfig) -> list[str]:
    target_texts_config = maskclip_config.get("target_texts")
    if target_texts_config in (None, ""):
        raise ValueError(
            "Set test.maskclip.target_texts to a non-empty string list. "
            'Example: test.maskclip.target_texts=["red","luggage","red luggage"]'
        )

    target_texts = [str(text).strip() for text in target_texts_config if str(text).strip()]
    if not target_texts:
        raise ValueError("test.maskclip.target_texts must contain at least one non-empty string.")
    return target_texts


def _compute_bbox_stats(
    similarity_map: torch.Tensor,
    sample: RefcocoModel,
) -> tuple[float, float]:
    patch_grid_size = similarity_map.shape[0]
    patch_x0, patch_y0, patch_x1, patch_y1 = _bbox_to_patch_coords(sample, patch_grid_size)
    inside = similarity_map[patch_y0:patch_y1, patch_x0:patch_x1]

    bbox_mean = inside.mean().item()
    outside_mask = torch.ones_like(similarity_map, dtype=torch.bool)
    outside_mask[patch_y0:patch_y1, patch_x0:patch_x1] = False
    outside_mean = similarity_map[outside_mask].mean().item()
    return bbox_mean, outside_mean


def _softmax_heatmap(similarity_map: torch.Tensor) -> torch.Tensor:
    return F.softmax(similarity_map.reshape(-1), dim=0).reshape_as(similarity_map)


def _compute_heatmaps(
    sample: RefcocoModel,
    target_texts: list[str],
    vision_model: CLIPVisionModel,
    text_model: CLIPTextModel,
    image_processor: CLIPImageProcessor,
    tokenizer: CLIPTokenizer,
    device: torch.device,
) -> tuple[list[tuple[str, torch.Tensor, float, float, float]], int]:
    image = _load_sample_image(sample)
    image_inputs = image_processor(images=image, return_tensors="pt")
    image_inputs = {key: value.to(device) for key, value in image_inputs.items()}

    text_targets = target_texts
    text_inputs = tokenizer(
        text_targets,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )
    text_inputs = {key: value.to(device) for key, value in text_inputs.items()}

    with torch.no_grad():
        patch_embeddings = vision_model(image_inputs["pixel_values"]).squeeze(0)
        text_embeddings = text_model(**text_inputs)

    patch_embeddings = F.normalize(patch_embeddings, dim=-1)
    text_embeddings = F.normalize(text_embeddings, dim=-1)
    similarities = torch.matmul(patch_embeddings, text_embeddings.transpose(0, 1))

    patch_count = similarities.shape[0]
    patch_grid_size = int(math.sqrt(patch_count))
    if patch_grid_size * patch_grid_size != patch_count:
        raise ValueError(f"Patch count must be a square number, got {patch_count}.")

    results: list[tuple[str, torch.Tensor, float, float, float]] = []
    for text_index, text in enumerate(text_targets):
        similarity_map = similarities[:, text_index].reshape(patch_grid_size, patch_grid_size).detach().cpu()
        bbox_mean, outside_mean = _compute_bbox_stats(similarity_map, sample)
        global_mean = similarity_map.mean().item()
        results.append((text, similarity_map, bbox_mean, outside_mean, global_mean))

    stacked_heatmaps = torch.stack([similarity_map for _, similarity_map, _, _, _ in results], dim=0)
    average_heatmap = stacked_heatmaps.mean(dim=0)
    average_heatmap = _softmax_heatmap(average_heatmap)
    average_bbox_mean, average_outside_mean = _compute_bbox_stats(average_heatmap, sample)
    average_global_mean = average_heatmap.mean().item()
    results.append(
        (
            AVERAGE_HEATMAP_LABEL,
            average_heatmap,
            average_bbox_mean,
            average_outside_mean,
            average_global_mean,
        )
    )
    return results, patch_grid_size


def _resolve_heatmap_label(text: str, panel_index: int, text_count: int) -> str:
    if text == AVERAGE_HEATMAP_LABEL:
        return "mean heatmap (softmax)"
    if panel_index < text_count:
        return f'text: "{text}"'
    return 'full expression: "' + text + '"'


def _resolve_heatmap_log_label(text: str, panel_index: int, text_count: int) -> str:
    if text == AVERAGE_HEATMAP_LABEL:
        return "mean_heatmap_softmax"
    if panel_index < text_count:
        return f'text="{text}"'
    return f'full_expression="{text}"'


def _visualize_heatmaps(
    sample: RefcocoModel,
    target_texts: list[str],
    heatmaps: list[tuple[str, torch.Tensor, float, float, float]],
    reference_rank: int,
    total_references: int,
    save_path: Path,
) -> None:
    image = _load_sample_image(sample)
    patch_grid_size = heatmaps[0][1].shape[0]
    patch_x0, patch_y0, patch_x1, patch_y1 = _bbox_to_patch_coords(sample, patch_grid_size)

    panel_count = 2 + len(heatmaps)
    fig, axes = plt.subplots(
        panel_count,
        1,
        figsize=(8.6, 4.2 * panel_count),
        constrained_layout=True,
        gridspec_kw={"height_ratios": [1.2, *([1.0] * len(heatmaps)), 1.35]},
    )

    if panel_count == 1:
        axes = [axes]

    fig.suptitle("MaskCLIP text heatmaps", fontsize=14, fontweight="bold")

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
    axes[0].set_title(
        "\n".join(
            [
                "Image + bbox",
                f"image_id={sample.image_id} | reference {reference_rank + 1}/{total_references}",
                "\n".join(textwrap.wrap(f'annotation: "{sample.annotation}"', width=70)),
            ]
        ),
        fontsize=11,
    )
    axes[0].axis("off")

    for axis, (text, similarity_map, bbox_mean, outside_mean, global_mean), panel_index in zip(
        axes[1:-1],
        heatmaps,
        range(1, len(heatmaps) + 1),
    ):
        heatmap = axis.imshow(
            similarity_map.numpy(),
            cmap="magma",
            interpolation="nearest",
        )
        axis.add_patch(
            Rectangle(
                (patch_x0 - 0.5, patch_y0 - 0.5),
                patch_x1 - patch_x0,
                patch_y1 - patch_y0,
                fill=False,
                edgecolor="cyan",
                linewidth=2,
            )
        )
        panel_label = _resolve_heatmap_label(
            text=text,
            panel_index=panel_index,
            text_count=len(target_texts),
        )
        title_lines = textwrap.wrap(panel_label, width=42)
        title_lines.append(f"avg:{global_mean:.3f}")
        title_lines.append(f"in:{bbox_mean:.3f} / out:{outside_mean:.3f}")
        axis.set_title("\n".join(title_lines), fontsize=10)
        axis.set_xlabel("patch x")
        axis.set_ylabel("patch y")
        fig.colorbar(heatmap, ax=axis, fraction=0.035, pad=0.02)

    summary_axis = axes[-1]
    summary_axis.axis("off")
    text_summary = ", ".join(f'"{text}"' for text in target_texts) if target_texts else "(none)"
    summary_lines = [
        "Heatmap Summary",
        "",
        f"selected reference: {reference_rank + 1}/{total_references}",
        f'image_id: {sample.image_id}',
        f"target texts: {text_summary}",
        "",
    ]
    for panel_index, (text, _, bbox_mean, outside_mean, global_mean) in enumerate(heatmaps, start=1):
        label = _resolve_heatmap_label(
            text=text,
            panel_index=panel_index,
            text_count=len(target_texts),
        )
        summary_lines.extend(
            textwrap.wrap(
                f"{panel_index}. {label}",
                width=70,
                subsequent_indent="   ",
            )
        )
        summary_lines.append(f"   avg={global_mean:.4f}")
        summary_lines.append(f"   bbox={bbox_mean:.4f}")
        summary_lines.append(f"   outside={outside_mean:.4f}")
        summary_lines.append("")
    summary_axis.text(0.0, 1.0, "\n".join(summary_lines), va="top", ha="left", fontsize=10)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


@hydra.main(version_base=None, config_path="../config", config_name="base")
def main(config: DictConfig) -> None:
    maskclip_config = config.test.get("maskclip", {})
    target_image_id = str(maskclip_config.get("target_image_id", "")).strip()
    if not target_image_id:
        raise ValueError("Set test.maskclip.target_image_id to a valid RefCOCO image id.")

    target_texts = _resolve_target_texts(maskclip_config)
    target_annotation = target_texts[-1]

    device = _resolve_device(config)

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
    sample, image_samples, selected_ref_index = _find_reference_sample(
        samples=refcoco_samples,
        target_image_id=target_image_id,
        target_annotation=target_annotation,
    )

    heatmaps, patch_grid_size = _compute_heatmaps(
        sample=sample,
        target_texts=target_texts,
        vision_model=vision_model,
        text_model=text_model,
        image_processor=image_processor,
        tokenizer=tokenizer,
        device=device,
    )

    output_name = _sanitize_filename(
        f"{sample.image_id}_{selected_ref_index:02d}_{target_annotation[:40]}_text_heatmaps.png"
    )
    output_path = OUTPUT_DIR / output_name
    _visualize_heatmaps(
        sample=sample,
        target_texts=target_texts,
        heatmaps=heatmaps,
        reference_rank=selected_ref_index,
        total_references=len(image_samples),
        save_path=output_path,
    )

    print(f"[sample] image_id: {sample.image_id}")
    print(f"[sample] image_path: {sample.image_path}")
    print(f"[sample] annotation: {sample.annotation}")
    print(f"[sample] selected_reference_index: {selected_ref_index}")
    print(f"[sample] total_references_for_image: {len(image_samples)}")
    print(f"[sample] target_texts: {target_texts}")
    print(f"[sample] bbox: ({sample.bbox.x}, {sample.bbox.y}, {sample.bbox.w}, {sample.bbox.h})")
    print(f"[sample] patch_grid_size: {patch_grid_size}x{patch_grid_size}")
    for text_index, (text, _, bbox_mean, outside_mean, global_mean) in enumerate(heatmaps, start=1):
        label = _resolve_heatmap_log_label(
            text=text,
            panel_index=text_index,
            text_count=len(target_texts),
        )
        print(
            f"[sample] {label}: avg={global_mean:.4f}, "
            f"bbox_mean={bbox_mean:.4f}, outside_mean={outside_mean:.4f}"
        )
    print(f"[sample] saved_heatmap: {output_path.resolve()}")


if __name__ == "__main__":
    main()
