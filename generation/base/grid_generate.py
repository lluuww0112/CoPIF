from pydantic import BaseModel, ConfigDict
import hydra
from omegaconf import DictConfig
from hydra.utils import instantiate
from transformers import CLIPTokenizer
import torch
import torch.nn.functional as F
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import os
from tqdm import tqdm

from APIs.refcocoAPI import RefcocoModel_list, load_refcoco, BboxModel, ImageSizeModel
from generation.base import grid_schema
from model.base.clip_model import CLIPTextModel


def encode_annotations(
    annotations: list[str],
    text_model: CLIPTextModel,
    tokenizer: CLIPTokenizer,
    device: torch.device,
) -> torch.Tensor:
    tokenized = tokenizer(
        annotations,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )
    tokenized = {key: value.to(device) for key, value in tokenized.items()}

    with torch.no_grad():
        text_embeddings = text_model(**tokenized)

    return text_embeddings


def bbox_to_grid_coords(feature_map: grid_schema.FeatureMapModel) -> tuple[int, int, int, int]:
    if feature_map.size is None:
        raise ValueError("size is required to map bbox coordinates to the grid.")

    grid_size = feature_map.grid.grid_size
    if grid_size is None:
        raise ValueError("grid.grid_size is not set.")

    image_width = feature_map.size.width
    image_height = feature_map.size.height
    if image_width <= 0 or image_height <= 0:
        raise ValueError("image size must be positive.")

    x0 = max(0.0, feature_map.bbox.x)
    y0 = max(0.0, feature_map.bbox.y)
    x1 = min(image_width, feature_map.bbox.x + feature_map.bbox.w)
    y1 = min(image_height, feature_map.bbox.y + feature_map.bbox.h)

    if x1 <= x0 or y1 <= y0:
        raise ValueError("bbox must have a positive area inside the image.")

    grid_x0 = min(grid_size - 1, int((x0 / image_width) * grid_size))
    grid_y0 = min(grid_size - 1, int((y0 / image_height) * grid_size))
    grid_x1 = max(grid_x0 + 1, min(grid_size, int(((x1 / image_width) * grid_size) + 0.999999)))
    grid_y1 = max(grid_y0 + 1, min(grid_size, int(((y1 / image_height) * grid_size) + 0.999999)))

    return grid_x0, grid_y0, grid_x1, grid_y1


def normalize_bbox(feature_map: grid_schema.FeatureMapModel) -> BboxModel:
    if feature_map.size is None:
        raise ValueError("size is required to normalize bbox.")

    image_width = feature_map.size.width
    image_height = feature_map.size.height
    if image_width <= 0 or image_height <= 0:
        raise ValueError("image size must be positive.")

    x = feature_map.bbox.x / image_width
    y = feature_map.bbox.y / image_height
    w = feature_map.bbox.w / image_width
    h = feature_map.bbox.h / image_height

    return BboxModel(x=x, y=y, w=w, h=h)


def fill_bbox_with_annotation_embedding(
    feature_map: grid_schema.FeatureMapModel,
    annotation_embedding: torch.Tensor,
) -> grid_schema.FeatureMapModel:
    if feature_map.feature_map is None:
        raise ValueError("feature_map is not initialized.")

    emb_dim = grid_schema.EMB_DIM
    if emb_dim is None:
        raise ValueError("EMB_DIM is not initialized. Call grid_schema.init_globals(config) first.")

    if annotation_embedding.ndim != 1 or annotation_embedding.shape[0] != emb_dim:
        raise ValueError(f"embedding length must match emb_dim={emb_dim}.")

    grid_x0, grid_y0, grid_x1, grid_y1 = bbox_to_grid_coords(feature_map)
    feature_map.feature_map[grid_y0:grid_y1, grid_x0:grid_x1] = annotation_embedding.to(
        dtype=feature_map.feature_map.dtype,
        device=feature_map.feature_map.device,
    )
    return feature_map


def resize_feature_map_if_needed(
    feature_map: grid_schema.FeatureMapModel,
    target_grid_size: int | None,
) -> grid_schema.FeatureMapModel:
    if target_grid_size is None:
        return feature_map

    if feature_map.feature_map is None:
        raise ValueError("feature_map is not initialized.")

    current_grid_size = feature_map.grid.grid_size
    if current_grid_size == target_grid_size:
        return feature_map

    resized = F.interpolate(
        feature_map.feature_map.permute(2, 0, 1).unsqueeze(0),
        size=(target_grid_size, target_grid_size),
        mode="nearest",
    )
    feature_map.feature_map = resized.squeeze(0).permute(1, 2, 0).contiguous()
    feature_map.grid.grid_size = target_grid_size
    return feature_map


def add_gaussian_noise(
    feature_map: grid_schema.FeatureMapModel,
    noise_scale: float,
) -> grid_schema.FeatureMapModel:
    if noise_scale < 0:
        raise ValueError("noise_scale must be non-negative.")

    if feature_map.feature_map is None:
        raise ValueError("feature_map is not initialized.")

    if noise_scale == 0:
        return feature_map

    noise = torch.randn_like(feature_map.feature_map) * noise_scale
    feature_map.feature_map = feature_map.feature_map + noise
    return feature_map


def preprocess_feature_map(
    feature_map: grid_schema.FeatureMapModel,
    annotation_embedding: torch.Tensor,
    noise_scale: float,
) -> grid_schema.FeatureMapModel:
    feature_map = fill_bbox_with_annotation_embedding(feature_map, annotation_embedding)
    feature_map = resize_feature_map_if_needed(feature_map, grid_schema.MAX_GRID_WIDTH)
    feature_map = add_gaussian_noise(feature_map, noise_scale)
    return feature_map


def build_single_feature_map(
    sample,
    text_embedding: torch.Tensor,
    noise_scale: float,
) -> grid_schema.FeatureMapModel:
    feature_map = grid_schema.FeatureMapModel(
        image_id=sample.image_id,
        image_path=sample.image_path,
        size=sample.size,
        bbox=sample.bbox,
        annotation=sample.annotation,
        grid=grid_schema.Grid(),
    )
    return preprocess_feature_map(
        feature_map=feature_map,
        annotation_embedding=text_embedding,
        noise_scale=noise_scale,
    )


def build_feature_maps(
    refcoco_samples: RefcocoModel_list,
    text_model: CLIPTextModel,
    tokenizer: CLIPTokenizer,
    device: torch.device,
    noise_scale: float,
    num_workers: int,
) -> list[grid_schema.FeatureMapModel]:
    valid_samples = [sample for sample in refcoco_samples if sample.size is not None]
    annotations = [sample.annotation for sample in valid_samples]
    text_embeddings = encode_annotations(annotations, text_model, tokenizer, device)
    worker_inputs = [
        (sample, text_embedding.detach().cpu(), noise_scale)
        for sample, text_embedding in zip(valid_samples, text_embeddings)
    ]

    if num_workers <= 1:
        return [
            build_single_feature_map(sample, text_embedding, current_noise_scale)
            for sample, text_embedding, current_noise_scale in tqdm(
                worker_inputs,
                total=len(worker_inputs),
                desc="Generating feature maps",
            )
        ]

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        feature_maps = list(
            tqdm(
                executor.map(lambda args: build_single_feature_map(*args), worker_inputs),
                total=len(worker_inputs),
                desc="Generating feature maps",
            )
        )

    return feature_maps

class SaveModel(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    image_path : str
    feature_map: torch.Tensor
    size: ImageSizeModel
    bbox: BboxModel
    annotation: str


def to_save_models(
    feature_maps: list[grid_schema.FeatureMapModel],
) -> list[SaveModel]:
    save_models: list[SaveModel] = []

    for feature_map in feature_maps:
        if feature_map.image_path is None:
            continue

        save_models.append(
            SaveModel(
                image_path=feature_map.image_path,
                feature_map=feature_map.feature_map.detach().cpu(),
                size=feature_map.size,
                bbox=normalize_bbox(feature_map),
                annotation=feature_map.annotation,
            )
        )

    return save_models


def save_feature_maps(
    save_models: list[SaveModel],
    dataset: str,
    splitby: str,
    split: str,
) -> Path:
    output_dir = Path("data") / "generated" / dataset / splitby / split
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "feature_maps.pt"
    torch.save(save_models, output_path)
    return output_path



@hydra.main(version_base=None, config_path="../../config", config_name="base")
def main(config : DictConfig)->None:
    grid_schema.init_globals(config)
    device = config.generator.device
    noise_scale = config.generator.noise_scale
    num_workers = max(1, min(8, os.cpu_count() or 1))
    dataset = "refcoco"
    splitby = "unc"
    split = "train"
    text_model = instantiate(config.model.text_model)
    tokenizer = CLIPTokenizer.from_pretrained(config.shared.pretrained_clip)

    text_model.to(device)
    text_model.eval()

    refcoco_samples = load_refcoco(
        dataset=dataset,
        split=split,
        splitby=splitby,
        include_image_metadata=True,
    )

    feature_maps = build_feature_maps(
        refcoco_samples=refcoco_samples,
        text_model=text_model,
        tokenizer=tokenizer,
        device=device,
        noise_scale=noise_scale,
        num_workers=num_workers,
    )
    save_models = to_save_models(feature_maps)
    output_path = save_feature_maps(
        save_models=save_models,
        dataset=dataset,
        splitby=splitby,
        split=split,
    )

    print(f"device: {device}")
    print(f"noise_scale: {noise_scale}")
    print(f"num_workers: {num_workers}")
    print(f"dataset: {dataset}")
    print(f"splitby: {splitby}")
    print(f"split: {split}")
    print(f"loaded refcoco samples: {len(refcoco_samples)}")
    print(f"generated feature maps: {len(feature_maps)}")
    print(f"saved feature maps: {len(save_models)}")
    print(f"save path: {output_path}")

    if feature_maps:
        sample = feature_maps[0]
        print(f"sample annotation: {sample.annotation}")
        print(f"sample grid size: {sample.grid.grid_size}")
        print(f"sample feature_map shape: {tuple(sample.feature_map.shape)}")
        print(f"sample bbox grid coords: {bbox_to_grid_coords(sample)}")


if __name__ == "__main__":
    main()
