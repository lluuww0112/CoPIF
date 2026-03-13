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
from generation.base import schema
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


def bbox_to_grid_coords(feature_map: schema.FeatureMapModel) -> tuple[int, int, int, int]:
    if feature_map.size is None:
        raise ValueError("size is required to map bbox coordinates to the grid.")

    grid_size = feature_map.grid_size
    if grid_size is None:
        raise ValueError("grid_size is not set.")

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


def normalize_bbox(feature_map: schema.FeatureMapModel) -> BboxModel:
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
    feature_map: schema.FeatureMapModel,
    annotation_embedding: torch.Tensor,
) -> schema.FeatureMapModel:
    if feature_map.feature_map is None:
        raise ValueError("feature_map is not initialized.")

    emb_dim = schema.EMB_DIM
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
    feature_map: schema.FeatureMapModel,
    target_grid_size: int | None,
) -> schema.FeatureMapModel:
    if target_grid_size is None:
        return feature_map

    if feature_map.feature_map is None:
        raise ValueError("feature_map is not initialized.")

    current_grid_size = feature_map.grid_size
    if current_grid_size == target_grid_size:
        return feature_map

    resized = F.interpolate(
        feature_map.feature_map.permute(2, 0, 1).unsqueeze(0),
        size=(target_grid_size, target_grid_size),
        mode="nearest",
    )
    feature_map.feature_map = resized.squeeze(0).permute(1, 2, 0).contiguous()
    feature_map.grid_size = target_grid_size
    return feature_map


def preprocess_feature_map(
    feature_map: schema.FeatureMapModel,
    annotation_embedding: torch.Tensor,
) -> schema.FeatureMapModel:
    feature_map = fill_bbox_with_annotation_embedding(feature_map, annotation_embedding)
    feature_map = resize_feature_map_if_needed(feature_map, schema.MAX_GRID_WIDTH)
    return feature_map


def build_single_feature_map(
    sample,
    text_embedding: torch.Tensor,
) -> schema.FeatureMapModel:
    feature_map = schema.FeatureMapModel(
        image_id=sample.image_id,
        image_path=sample.image_path,
        size=sample.size,
        bbox=sample.bbox,
        annotation=sample.annotation,
    )
    return preprocess_feature_map(
        feature_map=feature_map,
        annotation_embedding=text_embedding,
    )


def to_save_model(
    feature_map: schema.FeatureMapModel,
) -> "SaveModel | None":
    if feature_map.image_path is None:
        return None

    return SaveModel(
        image_path=feature_map.image_path,
        feature_map=feature_map.feature_map.detach().cpu(),
        size=feature_map.size,
        bbox=normalize_bbox(feature_map),
        annotation=feature_map.annotation,
    )


def build_save_model_batch(
    samples: list,
    text_embeddings: torch.Tensor,
    num_workers: int,
) -> tuple[list["SaveModel"], schema.FeatureMapModel | None]:
    worker_inputs = [
        (sample, text_embedding.detach().cpu())
        for sample, text_embedding in zip(samples, text_embeddings)
    ]

    save_models = []
    sample_preview: schema.FeatureMapModel | None = None

    if num_workers <= 1:
        for sample, text_embedding in tqdm(
            worker_inputs,
            total=len(worker_inputs),
            desc="Generating feature maps",
            leave=False,
        ):
            feature_map = build_single_feature_map(sample, text_embedding)
            if sample_preview is None:
                sample_preview = feature_map

            save_model = to_save_model(feature_map)
            if save_model is not None:
                save_models.append(save_model)

        return save_models, sample_preview

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        for feature_map in tqdm(
            executor.map(lambda args: build_single_feature_map(*args), worker_inputs),
            total=len(worker_inputs),
            desc="Generating feature maps",
            leave=False,
        ):
            if sample_preview is None:
                sample_preview = feature_map

            save_model = to_save_model(feature_map)
            if save_model is not None:
                save_models.append(save_model)

    return save_models, sample_preview


class SaveModel(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    image_path : str
    feature_map: torch.Tensor
    size: ImageSizeModel
    bbox: BboxModel
    annotation: str


def _feature_map_output_dir(
    dataset: str,
    splitby: str,
    split: str,
) -> Path:
    return Path("data") / "generated" / dataset / splitby / split


def save_feature_map_chunk(
    save_models: list[SaveModel],
    output_dir: Path,
    chunk_index: int,
) -> Path:
    output_path = output_dir / f"feature_maps.chunk_{chunk_index:05d}.pt"
    torch.save(save_models, output_path)
    return output_path


def save_feature_map_manifest(
    output_dir: Path,
    chunk_paths: list[Path],
    total_generated: int,
    total_saved: int,
    annotation_batch_size: int,
) -> Path:
    manifest = {
        "format": "chunked_save_models_v1",
        "total_generated": total_generated,
        "total_saved": total_saved,
        "annotation_batch_size": annotation_batch_size,
        "chunks": [path.name for path in chunk_paths],
    }

    output_path = output_dir / "feature_maps.pt"
    torch.save(manifest, output_path)
    return output_path


def generate_and_save_feature_maps(
    refcoco_samples: RefcocoModel_list,
    text_model: CLIPTextModel,
    tokenizer: CLIPTokenizer,
    device: torch.device,
    num_workers: int,
    dataset: str,
    splitby: str,
    split: str,
    annotation_batch_size: int,
) -> tuple[Path, int, int, schema.FeatureMapModel | None]:
    valid_samples = [sample for sample in refcoco_samples if sample.size is not None]
    output_dir = _feature_map_output_dir(dataset=dataset, splitby=splitby, split=split)
    output_dir.mkdir(parents=True, exist_ok=True)

    chunk_paths: list[Path] = []
    total_saved = 0
    sample_preview: schema.FeatureMapModel | None = None

    total_batches = (len(valid_samples) + annotation_batch_size - 1) // annotation_batch_size

    for chunk_index, start in enumerate(
        tqdm(range(0, len(valid_samples), annotation_batch_size), total=total_batches, desc="Generating feature map chunks")
    ):
        sample_batch = valid_samples[start:start + annotation_batch_size]
        annotations = [sample.annotation for sample in sample_batch]
        text_embeddings = encode_annotations(annotations, text_model, tokenizer, device)
        save_models, chunk_preview = build_save_model_batch(
            samples=sample_batch,
            text_embeddings=text_embeddings,
            num_workers=num_workers,
        )

        if sample_preview is None and chunk_preview is not None:
            sample_preview = chunk_preview

        chunk_path = save_feature_map_chunk(
            save_models=save_models,
            output_dir=output_dir,
            chunk_index=chunk_index,
        )
        chunk_paths.append(chunk_path)
        total_saved += len(save_models)

        del text_embeddings
        del save_models

    manifest_path = save_feature_map_manifest(
        output_dir=output_dir,
        chunk_paths=chunk_paths,
        total_generated=len(valid_samples),
        total_saved=total_saved,
        annotation_batch_size=annotation_batch_size,
    )
    return manifest_path, len(valid_samples), total_saved, sample_preview



@hydra.main(version_base=None, config_path="../../config", config_name="base")
def main(config : DictConfig)->None:
    schema.init_globals(config)
    device = config.generator.device
    num_workers = max(1, min(8, os.cpu_count() or 1))
    annotation_batch_size = int(config.generator.get("annotation_batch_size", 512))
    dataset = config.generator.dataset
    splitby = config.generator.splitby
    split = config.generator.split
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

    output_path, generated_count, saved_count, sample = generate_and_save_feature_maps(
        refcoco_samples=refcoco_samples,
        text_model=text_model,
        tokenizer=tokenizer,
        device=device,
        num_workers=num_workers,
        dataset=dataset,
        splitby=splitby,
        split=split,
        annotation_batch_size=annotation_batch_size,
    )

    print(f"device: {device}")
    print(f"num_workers: {num_workers}")
    print(f"annotation_batch_size: {annotation_batch_size}")
    print(f"dataset: {dataset}")
    print(f"splitby: {splitby}")
    print(f"split: {split}")
    print(f"loaded refcoco samples: {len(refcoco_samples)}")
    print(f"generated feature maps: {generated_count}")
    print(f"saved feature maps: {saved_count}")
    print(f"save path: {output_path}")

    if sample is not None:
        print(f"sample annotation: {sample.annotation}")
        print(f"sample grid size: {sample.grid_size}")
        print(f"sample feature_map shape: {tuple(sample.feature_map.shape)}")
        print(f"sample bbox grid coords: {bbox_to_grid_coords(sample)}")


if __name__ == "__main__":
    main()
