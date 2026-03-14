from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import hydra
from hydra.utils import get_original_cwd, instantiate
from omegaconf import DictConfig

import torch
from transformers import CLIPImageProcessor, CLIPTokenizer

from APIs.refcocoAPI import load_refcoco
from generation.base.generate import (
    generate_and_save_real_image_feature_maps,
    select_fewshot_image_samples,
)
from model.base.train import (
    GeneratedFeatureMapDataset,
    RECTrainer,
    TrainCollator,
    TrainingMonitorCallback,
    _build_rec_model,
    _build_training_arguments,
    _torch_load,
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


def _default_feature_manifest_dir(
    dataset: str,
    splitby: str,
    split: str,
    num_images: int,
    seed: int,
) -> Path:
    return (
        Path(get_original_cwd())
        / "data"
        / "generated"
        / dataset
        / splitby
        / split
        / f"fewshot_{num_images}images_seed{seed}"
    )


def _write_selection_summary(
    output_dir: Path,
    payload: dict[str, Any],
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "fewshot_selection.json"
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=True, indent=2)
    return output_path


@hydra.main(version_base=None, config_path="../../config", config_name="base")
def main(config: DictConfig) -> None:
    fewshot_config = config.get("fewshot", {})
    train_config = config.get("train", {})
    generator_config = config.get("generator", {})

    requested_device = str(fewshot_config.get("device", train_config.get("device", generator_config.device)))
    device = _resolve_device(requested_device=requested_device)

    dataset = str(fewshot_config.get("dataset", train_config.get("dataset", generator_config.dataset)))
    splitby = str(fewshot_config.get("splitby", train_config.get("splitby", generator_config.splitby)))
    split = str(fewshot_config.get("split", "train"))
    num_images = int(fewshot_config.get("num_images", 0))
    if num_images <= 0:
        raise ValueError("fewshot.num_images must be a positive integer.")

    seed = int(fewshot_config.get("seed", train_config.get("seed", 42)))
    batch_size = int(fewshot_config.get("batch_size", 8))
    num_epochs = int(fewshot_config.get("num_epochs", 3))
    learning_rate = float(fewshot_config.get("learning_rate", train_config.get("learning_rate", 1e-4)))
    weight_decay = float(fewshot_config.get("weight_decay", train_config.get("weight_decay", 1e-4)))
    num_workers = int(fewshot_config.get("num_workers", train_config.get("num_workers", 0)))
    grad_clip = float(fewshot_config.get("grad_clip", train_config.get("grad_clip", 1.0)))
    log_every = int(fewshot_config.get("log_every", train_config.get("log_every", 10)))
    max_steps = int(fewshot_config.get("max_steps", 0))
    noise_scale = float(fewshot_config.get("noise_scale", 0.0))
    save_steps = int(fewshot_config.get("save_steps", train_config.get("save_steps", 500)))
    save_total_limit = int(fewshot_config.get("save_total_limit", train_config.get("save_total_limit", 2)))
    warmup_steps = int(fewshot_config.get("warmup_steps", 0))
    gradient_accumulation_steps = int(
        fewshot_config.get("gradient_accumulation_steps", train_config.get("gradient_accumulation_steps", 1))
    )
    bbox_loss_coef = float(fewshot_config.get("bbox_loss_coef", train_config.get("bbox_loss_coef", 5.0)))
    giou_loss_coef = float(fewshot_config.get("giou_loss_coef", train_config.get("giou_loss_coef", 2.0)))
    image_batch_size = int(fewshot_config.get("image_batch_size", batch_size))
    resume_from_checkpoint = fewshot_config.get("resume_from_checkpoint", None)

    init_checkpoint_dir = _resolve_project_path(
        str(fewshot_config.get("init_checkpoint_dir", train_config.get("checkpoint_dir", "history/base")))
    )
    checkpoint_dir = _resolve_project_path(
        str(fewshot_config.get("checkpoint_dir", f"history/fewshot_{num_images}images"))
    )

    feature_manifest_dir_value = fewshot_config.get("feature_manifest_dir", None)
    if feature_manifest_dir_value is None:
        feature_manifest_dir = _default_feature_manifest_dir(
            dataset=dataset,
            splitby=splitby,
            split=split,
            num_images=num_images,
            seed=seed,
        )
    else:
        feature_manifest_dir = _resolve_project_path(str(feature_manifest_dir_value))

    feature_subset_name = fewshot_config.get("feature_subset_name", None)
    if feature_subset_name is None:
        subset_name = f"fewshot_{num_images}images_seed{seed}"
    else:
        subset_name = str(feature_subset_name)

    tokenizer = CLIPTokenizer.from_pretrained(config.shared.pretrained_clip)
    image_processor = CLIPImageProcessor.from_pretrained(config.shared.pretrained_clip)

    vision_model = instantiate(config.model.vision_model)
    vision_model.to(device)
    vision_model.eval()

    all_samples = load_refcoco(
        dataset=dataset,
        splitby=splitby,
        split=split,
        include_image_metadata=True,
    )
    selected_samples, selected_image_ids = select_fewshot_image_samples(
        refcoco_samples=all_samples,
        num_images=num_images,
        seed=seed,
    )

    manifest_path, selected_image_count, selected_entry_count = generate_and_save_real_image_feature_maps(
        refcoco_samples=selected_samples,
        image_model=vision_model,
        image_processor=image_processor,
        device=device,
        image_batch_size=image_batch_size,
        dataset=dataset,
        splitby=splitby,
        split=split,
        subset_name=subset_name,
        output_dir=feature_manifest_dir,
    )

    text_model = instantiate(config.model.text_model)
    model = _build_rec_model(config, vision_model=vision_model, text_model=text_model)
    checkpoint_path = _find_checkpoint_path(init_checkpoint_dir)
    state_dict = _torch_load(checkpoint_path)
    model.load_state_dict(state_dict)
    model.to(device)

    expected_patch_count = model.pos_embed.shape[1] - 1
    dataset_obj = GeneratedFeatureMapDataset(manifest_path=manifest_path)
    collator = TrainCollator(
        tokenizer=tokenizer,
        expected_patch_count=expected_patch_count,
        noise_scale=noise_scale,
    )

    selection_summary_path = _write_selection_summary(
        output_dir=checkpoint_dir,
        payload={
            "dataset": dataset,
            "splitby": splitby,
            "split": split,
            "requested_num_images": num_images,
            "selected_num_images": len(selected_image_ids),
            "selected_num_entries": selected_entry_count,
            "selected_image_ids": selected_image_ids,
            "feature_manifest_path": str(manifest_path),
            "init_checkpoint_path": str(checkpoint_path),
        },
    )

    print(f"init_checkpoint_path: {checkpoint_path}")
    print(f"fewshot_checkpoint_dir: {checkpoint_dir}")
    print(f"fewshot_manifest_path: {manifest_path}")
    print(f"fewshot_selection_summary: {selection_summary_path}")
    print(f"dataset: {dataset}/{splitby}/{split}")
    print(f"requested_device: {requested_device}")
    print(f"resolved_device: {device}")
    print(f"selected_image_count: {selected_image_count}")
    print(f"selected_entry_count: {selected_entry_count}")
    print(f"batch_size: {batch_size}")
    print(f"num_epochs: {num_epochs}")
    print(f"learning_rate: {learning_rate}")
    print(f"noise_scale: {noise_scale}")

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
        requested_device=device.type,
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
