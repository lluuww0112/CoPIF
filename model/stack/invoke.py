from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from typing import Any

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

import torch
from torch.utils.data import DataLoader
from transformers import CLIPImageProcessor, CLIPTokenizer

from model.base.invoke import (
    RefcocoInferenceDataset,
    _denormalize_bbox,
    _find_checkpoint_path,
    _load_inference_samples,
    _normalize_bbox,
    _resolve_device,
    _resolve_project_path,
    _torch_load,
    _write_predictions_to_csv,
)
from model.stack.preprocessing import InferenceCollator
from model.stack.train import _build_rec_model


@hydra.main(version_base=None, config_path="../../config", config_name="stack")
def main(config: DictConfig) -> None:
    invoke_config = config.get("invoke", {})
    train_config = config.get("train", {})
    preprocessing_config = config.get("preprocessing", {})

    dataset = str(invoke_config.get("dataset", train_config.get("dataset", config.generator.dataset)))
    splitby = str(invoke_config.get("splitby", train_config.get("splitby", config.generator.splitby)))
    split = str(invoke_config.get("split", train_config.get("split", config.generator.split)))
    requested_device = str(invoke_config.get("device", train_config.get("device", config.generator.device)))
    batch_size = int(invoke_config.get("batch_size", 8))
    max_samples_value = invoke_config.get("max_samples", None)
    max_samples = int(max_samples_value) if max_samples_value is not None else None
    max_phrase_count = int(preprocessing_config.get("max_phrase_count", 8))
    checkpoint_dir = _resolve_project_path(
        str(invoke_config.get("checkpoint_dir", train_config.get("checkpoint_dir", "history/stack")))
    )
    output_csv_value = invoke_config.get("output_csv", None)
    if output_csv_value is None:
        output_csv = checkpoint_dir / f"inference_{dataset}_{splitby}_{split}.csv"
    else:
        output_csv = _resolve_project_path(output_csv_value)

    device = _resolve_device(requested_device=requested_device)
    use_bf16 = device.type == "cuda" and getattr(torch.cuda, "is_bf16_supported", lambda: True)()
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
            max_phrase_count=max_phrase_count,
        ),
    )

    print(f"checkpoint_path: {checkpoint_path}")
    print(f"device: {device}")
    print(f"inference_data: {dataset}/{splitby}/{split}")
    print(f"num_samples: {len(samples)}")
    print(f"batch_size: {batch_size}")
    print(f"input_res: {input_res}")
    print(f"max_phrase_count: {max_phrase_count}")
    print(f"output_csv: {output_csv}")

    rows: list[dict[str, Any]] = []

    with torch.no_grad():
        for batch in dataloader:
            pixel_values = batch["pixel_values"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            phrase_mask = batch["phrase_mask"].to(device)

            autocast_context = (
                torch.autocast(device_type=device.type, dtype=torch.bfloat16)
                if use_bf16 and device.type == "cuda"
                else nullcontext()
            )
            with autocast_context:
                pred_boxes = model.invoke(
                    pixel_values=pixel_values,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    phrase_mask=phrase_mask,
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
