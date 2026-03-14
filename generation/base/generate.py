import math
import random
from pydantic import BaseModel, ConfigDict
import hydra
from omegaconf import DictConfig
from hydra.utils import instantiate
from transformers import CLIPTokenizer
from transformers import CLIPImageProcessor
import torch
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import os
from tqdm import tqdm
from PIL import Image

from APIs.refcocoAPI import RefcocoModel_list, load_refcoco, BboxModel, ImageSizeModel
from generation.base import schema
from model.base.clip_model import CLIPTextModel, CLIPVisionModel

_BICUBIC_RESAMPLE = (
    Image.Resampling.BICUBIC if hasattr(Image, "Resampling") else Image.BICUBIC
)


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


def normalize_sample_bbox(sample) -> BboxModel:
    if sample.size is None:
        raise ValueError("size is required to normalize bbox.")

    image_width = sample.size.width
    image_height = sample.size.height
    if image_width <= 0 or image_height <= 0:
        raise ValueError("image size must be positive.")

    return BboxModel(
        x=sample.bbox.x / image_width,
        y=sample.bbox.y / image_height,
        w=sample.bbox.w / image_width,
        h=sample.bbox.h / image_height,
    )


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
    region_height = grid_y1 - grid_y0
    region_width = grid_x1 - grid_x0
    target_cosines = _build_bbox_similarity_map(
        height=region_height,
        width=region_width,
        min_cosine=float(schema.BBOX_DUMMY_MIN_COSINE or 0.9),
        max_cosine=float(schema.BBOX_DUMMY_MAX_COSINE or 0.98),
        dtype=feature_map.feature_map.dtype,
        device=feature_map.feature_map.device,
    )
    derived_vectors = _derive_vectors_from_target_cosines(
        annotation_embedding=annotation_embedding,
        target_cosines=target_cosines,
        dtype=feature_map.feature_map.dtype,
        device=feature_map.feature_map.device,
    ).reshape(region_height, region_width, emb_dim)
    feature_map.feature_map[grid_y0:grid_y1, grid_x0:grid_x1] = derived_vectors
    return feature_map


def _build_bbox_similarity_map(
    height: int,
    width: int,
    min_cosine: float,
    max_cosine: float,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    if height <= 0 or width <= 0:
        raise ValueError(f"bbox region must be positive, got height={height}, width={width}.")

    min_cosine = max(-1.0, min(1.0, min_cosine))
    max_cosine = max(-1.0, min(1.0, max_cosine))
    if min_cosine > max_cosine:
        min_cosine, max_cosine = max_cosine, min_cosine

    yy, xx = torch.meshgrid(
        torch.linspace(-1.0, 1.0, steps=height, dtype=dtype, device=device),
        torch.linspace(-1.0, 1.0, steps=width, dtype=dtype, device=device),
        indexing="ij",
    )
    radial = torch.sqrt((xx * xx) + (yy * yy)).clamp(max=1.0)
    focus_power = float(schema.BBOX_FOCUS_POWER or 1.75)
    center_weight = torch.pow(1.0 - radial, focus_power)
    similarity_map = min_cosine + ((max_cosine - min_cosine) * center_weight)

    hotspot_count = max(0, int(schema.BBOX_HOTSPOT_COUNT or 0))
    hotspot_boost = max(0.0, float(schema.BBOX_HOTSPOT_BOOST or 0.0))
    if hotspot_count > 0 and hotspot_boost > 0:
        hotspot_map = torch.zeros_like(similarity_map)
        active_hotspot_count = int(
            torch.randint(
                low=0,
                high=hotspot_count + 1,
                size=(1,),
                device=device,
            ).item()
        )
        for _ in range(active_hotspot_count):
            center_x = torch.empty(1, dtype=dtype, device=device).uniform_(-0.6, 0.6).item()
            center_y = torch.empty(1, dtype=dtype, device=device).uniform_(-0.6, 0.6).item()
            sigma_x = torch.empty(1, dtype=dtype, device=device).uniform_(0.18, 0.45).item()
            sigma_y = torch.empty(1, dtype=dtype, device=device).uniform_(0.18, 0.45).item()
            amplitude = torch.empty(1, dtype=dtype, device=device).uniform_(0.35, 1.0).item() * hotspot_boost
            hotspot = torch.exp(
                -(
                    ((xx - center_x) ** 2) / (2 * (sigma_x ** 2))
                    + ((yy - center_y) ** 2) / (2 * (sigma_y ** 2))
                )
            )
            hotspot_map = torch.maximum(hotspot_map, hotspot * amplitude)
        similarity_map = similarity_map + ((max_cosine - similarity_map) * hotspot_map)

    jitter_std = max(0.0, float(schema.BBOX_JITTER_STD or 0.0))
    if jitter_std > 0:
        similarity_map = similarity_map + (torch.randn_like(similarity_map) * jitter_std)

    return similarity_map.clamp(min_cosine, max_cosine)


def _derive_vectors_from_target_cosines(
    annotation_embedding: torch.Tensor,
    target_cosines: torch.Tensor,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    count = int(target_cosines.numel())
    if count <= 0:
        raise ValueError("target_cosines must contain at least one element.")

    base_vector = annotation_embedding.to(device=device, dtype=dtype)
    base_norm = torch.linalg.vector_norm(base_vector).clamp(min=1e-6)
    base_unit = base_vector / base_norm

    noise = torch.randn(count, base_vector.shape[0], device=device, dtype=dtype)
    projection = torch.matmul(noise, base_unit.unsqueeze(-1))
    noise = noise - (projection * base_unit.unsqueeze(0))
    noise_norm = torch.linalg.vector_norm(noise, dim=-1, keepdim=True).clamp(min=1e-6)
    noise_unit = noise / noise_norm

    cosine = target_cosines.reshape(-1, 1).to(device=device, dtype=dtype).clamp(-1.0, 1.0)
    sine = torch.sqrt(torch.clamp(1.0 - cosine.square(), min=0.0))
    derived = (cosine * base_unit.unsqueeze(0)) + (sine * noise_unit)
    return derived * base_norm


def fill_outside_bbox_with_annotation_relative_vectors(
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

    grid_size = feature_map.grid_size
    if grid_size is None:
        raise ValueError("grid_size is not set.")

    target_cosines = _build_outside_similarity_map(
        grid_size=grid_size,
        min_cosine=float(schema.OUTSIDE_BBOX_DUMMY_MIN_COSINE or -0.2),
        max_cosine=float(schema.OUTSIDE_BBOX_DUMMY_MAX_COSINE or 0.1),
        dtype=feature_map.feature_map.dtype,
        device=feature_map.feature_map.device,
    )
    outside_vectors = _derive_vectors_from_target_cosines(
        annotation_embedding=annotation_embedding,
        target_cosines=target_cosines,
        dtype=feature_map.feature_map.dtype,
        device=feature_map.feature_map.device,
    ).reshape(grid_size, grid_size, emb_dim)
    feature_map.feature_map = outside_vectors
    return feature_map


def _build_outside_similarity_map(
    grid_size: int,
    min_cosine: float,
    max_cosine: float,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    if grid_size <= 0:
        raise ValueError(f"grid_size must be positive, got {grid_size}.")

    min_cosine = max(-1.0, min(1.0, min_cosine))
    max_cosine = max(-1.0, min(1.0, max_cosine))
    if min_cosine > max_cosine:
        min_cosine, max_cosine = max_cosine, min_cosine

    similarity_map = torch.empty((grid_size, grid_size), dtype=dtype, device=device).uniform_(
        min_cosine,
        max_cosine,
    )
    jitter_std = max(0.0, float(schema.OUTSIDE_JITTER_STD or 0.0))
    if jitter_std > 0:
        similarity_map = similarity_map + (torch.randn_like(similarity_map) * jitter_std)

    return similarity_map.clamp(min_cosine, max_cosine)


def preprocess_feature_map(
    feature_map: schema.FeatureMapModel,
    annotation_embedding: torch.Tensor,
) -> schema.FeatureMapModel:
    feature_map = fill_outside_bbox_with_annotation_relative_vectors(feature_map, annotation_embedding)
    feature_map = fill_bbox_with_annotation_embedding(feature_map, annotation_embedding)
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
    subset_name: str | None = None,
) -> Path:
    output_dir = Path("data") / "generated" / dataset / splitby / split
    if subset_name:
        output_dir = output_dir / subset_name
    return output_dir


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
    feature_source: str = "synthetic",
    metadata: dict | None = None,
) -> Path:
    manifest = {
        "format": "chunked_save_models_v1",
        "total_generated": total_generated,
        "total_saved": total_saved,
        "annotation_batch_size": annotation_batch_size,
        "feature_source": feature_source,
        "chunks": [path.name for path in chunk_paths],
    }
    if metadata:
        manifest["metadata"] = metadata

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
        feature_source="synthetic",
    )
    return manifest_path, len(valid_samples), total_saved, sample_preview


def select_fewshot_image_samples(
    refcoco_samples,
    num_images: int,
    seed: int,
) -> tuple[list, list[str]]:
    if num_images <= 0:
        raise ValueError(f"num_images must be positive, got {num_images}.")

    image_groups: dict[str, list] = {}
    for sample in refcoco_samples:
        if sample.image_path is None or sample.size is None:
            continue
        if not Path(sample.image_path).is_file():
            continue
        image_groups.setdefault(sample.image_id, []).append(sample)

    if not image_groups:
        raise FileNotFoundError("could not find valid samples with image metadata for few-shot generation.")

    image_ids = sorted(image_groups)
    random.Random(seed).shuffle(image_ids)
    selected_image_ids = image_ids[: min(num_images, len(image_ids))]

    selected_samples = []
    for image_id in selected_image_ids:
        selected_samples.extend(image_groups[image_id])

    return selected_samples, selected_image_ids


def _load_resized_images(
    image_paths: list[str],
    input_res: int,
) -> list[Image.Image]:
    images = []
    for image_path in image_paths:
        with Image.open(image_path) as image:
            images.append(
                image.convert("RGB").resize(
                    (input_res, input_res),
                    resample=_BICUBIC_RESAMPLE,
                )
            )
    return images


def _encode_image_feature_batch(
    image_paths: list[str],
    image_model: CLIPVisionModel,
    image_processor: CLIPImageProcessor,
    device: torch.device,
    input_res: int,
) -> torch.Tensor:
    images = _load_resized_images(image_paths=image_paths, input_res=input_res)
    image_inputs = image_processor(
        images=images,
        return_tensors="pt",
        do_resize=False,
        do_center_crop=False,
    )
    pixel_values = image_inputs["pixel_values"].to(device)

    with torch.no_grad():
        image_features = image_model(pixel_values).detach().cpu()

    return image_features


def _reshape_patch_embeddings_to_grid(image_feature: torch.Tensor) -> torch.Tensor:
    patch_count = int(image_feature.shape[0])
    grid_size = math.isqrt(patch_count)
    if grid_size * grid_size != patch_count:
        raise ValueError(f"patch_count must be a square number, got {patch_count}.")
    return image_feature.reshape(grid_size, grid_size, image_feature.shape[-1])


def generate_and_save_real_image_feature_maps(
    refcoco_samples,
    image_model: CLIPVisionModel,
    image_processor: CLIPImageProcessor,
    device: torch.device,
    image_batch_size: int,
    dataset: str,
    splitby: str,
    split: str,
    subset_name: str,
    output_dir: Path | None = None,
) -> tuple[Path, int, int]:
    grouped_samples: dict[str, list] = {}
    for sample in refcoco_samples:
        if sample.image_path is None or sample.size is None:
            continue
        if not Path(sample.image_path).is_file():
            continue
        grouped_samples.setdefault(sample.image_id, []).append(sample)

    if not grouped_samples:
        raise FileNotFoundError("could not find valid image-backed samples to extract real feature maps.")

    image_groups = list(grouped_samples.values())
    representative_samples = [group[0] for group in image_groups]
    if output_dir is None:
        output_dir = _feature_map_output_dir(
            dataset=dataset,
            splitby=splitby,
            split=split,
            subset_name=subset_name,
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    input_res = int(image_model.vision_model.config.image_size)
    chunk_paths: list[Path] = []
    total_saved = 0

    total_batches = (len(representative_samples) + image_batch_size - 1) // image_batch_size
    for chunk_index, start in enumerate(
        tqdm(
            range(0, len(representative_samples), image_batch_size),
            total=total_batches,
            desc="Extracting real-image feature map chunks",
        )
    ):
        sample_batch = representative_samples[start:start + image_batch_size]
        image_paths = [sample.image_path for sample in sample_batch]
        image_features = _encode_image_feature_batch(
            image_paths=image_paths,
            image_model=image_model,
            image_processor=image_processor,
            device=device,
            input_res=input_res,
        )

        save_models: list[SaveModel] = []
        for sample, image_feature in zip(sample_batch, image_features):
            reshaped_feature_map = _reshape_patch_embeddings_to_grid(image_feature.float())
            for annotation_sample in grouped_samples[sample.image_id]:
                save_models.append(
                    SaveModel(
                        image_path=annotation_sample.image_path,
                        feature_map=reshaped_feature_map.clone(),
                        size=annotation_sample.size,
                        bbox=normalize_sample_bbox(annotation_sample),
                        annotation=annotation_sample.annotation,
                    )
                )

        chunk_path = save_feature_map_chunk(
            save_models=save_models,
            output_dir=output_dir,
            chunk_index=chunk_index,
        )
        chunk_paths.append(chunk_path)
        total_saved += len(save_models)

    manifest_path = save_feature_map_manifest(
        output_dir=output_dir,
        chunk_paths=chunk_paths,
        total_generated=len(representative_samples),
        total_saved=total_saved,
        annotation_batch_size=image_batch_size,
        feature_source="real_image",
        metadata={
            "selected_image_count": len(representative_samples),
            "subset_name": subset_name,
        },
    )
    return manifest_path, len(representative_samples), total_saved



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
