from typing import List
from pydantic import ConfigDict, Field, model_validator
from omegaconf import DictConfig
from omegaconf import OmegaConf
import random

import torch
from torch.utils.data import Dataset

from transformers import CLIPModel

from APIs.refcocoAPI import BboxModel, ImageSizeModel, RefcocoModel

INPUT_RESOLUTION: int | None = None
PATCH_NUM: int | None = None
MAX_GRID_WIDTH: int | None = None
EMB_DIM: int | None = None


def _load_clip_embedding_dims(pretrained_model_name: str) -> tuple[int, int, int]:
    """
        사전학습된 CLIP모델의 입력 해상도, 한 변 패치 수, 임베딩 차원을 불러오는 helper 함수
    """

    clip_model = CLIPModel.from_pretrained(pretrained_model_name)
    input_resolution = clip_model.vision_model.config.image_size
    patch_size = clip_model.vision_model.config.patch_size
    patch_num = input_resolution // patch_size
    emb_dim = clip_model.config.projection_dim

    return input_resolution, patch_num, emb_dim


def init_globals(config : DictConfig) -> None:
    """
        hydra로 정의한 값으로 전역변수 초기화
    """
    global MAX_GRID_WIDTH
    global INPUT_RESOLUTION
    global PATCH_NUM
    global EMB_DIM

    MAX_GRID_WIDTH = config.generator.max_grid_width
    INPUT_RESOLUTION, PATCH_NUM, EMB_DIM = _load_clip_embedding_dims(
        config.shared.pretrained_clip
    )


def _generate_grid_size():
    if MAX_GRID_WIDTH is None:
        if PATCH_NUM is None:
            raise ValueError("PATCH_NUM is not initialized. Call init_globals(config) first.")
        return PATCH_NUM
    return random.randint(1, MAX_GRID_WIDTH)
    



def _generate_random_feature_map(grid_size: int, emb_dim: int) -> torch.Tensor:
    return torch.randn(grid_size, grid_size, emb_dim, dtype=torch.float32)


def _require_embedding_dim() -> int:
    if EMB_DIM is None:
        raise ValueError("EMB_DIM is not initialized. Call init_globals(config) first.")
    return EMB_DIM



class FeatureMapModel(RefcocoModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    grid_size: int | None = Field(
        default_factory=_generate_grid_size,
        description="랜덤으로 생성된 정방형 feature map의 한 변 길이",
    )
    feature_map: torch.Tensor | None = Field(
        default=None,
        description="(grid_size, grid_size, emb_dim) 형태의 feature map",
    )

    @model_validator(mode="after")
    def initialize_feature_map(self) -> "FeatureMapModel":
        grid_size = self.grid_size
        if grid_size is None:
            raise ValueError("grid_size must be set before feature map initialization.")

        emb_dim = _require_embedding_dim()

        if self.feature_map is None:
            self.feature_map = _generate_random_feature_map(grid_size=grid_size, emb_dim=emb_dim)
            return self

        if not isinstance(self.feature_map, torch.Tensor):
            raise ValueError("feature_map must be a torch.Tensor.")

        expected_shape = (grid_size, grid_size, emb_dim)
        if tuple(self.feature_map.shape) != expected_shape:
            raise ValueError(f"feature_map shape must match {expected_shape}.")

        return self

    def bbox_to_grid_coords(self) -> tuple[int, int, int, int]:
        if self.size is None:
            raise ValueError("size is required to map bbox coordinates to the grid.")

        grid_size = self.grid_size
        if grid_size is None:
            raise ValueError("grid_size is not set.")

        image_width = self.size.width
        image_height = self.size.height

        if image_width <= 0 or image_height <= 0:
            raise ValueError("image size must be positive.")

        x0 = max(0.0, self.bbox.x)
        y0 = max(0.0, self.bbox.y)
        x1 = min(image_width, self.bbox.x + self.bbox.w)
        y1 = min(image_height, self.bbox.y + self.bbox.h)

        if x1 <= x0 or y1 <= y0:
            raise ValueError("bbox must have a positive area inside the image.")

        grid_x0 = min(grid_size - 1, int((x0 / image_width) * grid_size))
        grid_y0 = min(grid_size - 1, int((y0 / image_height) * grid_size))
        grid_x1 = max(grid_x0 + 1, min(grid_size, int(((x1 / image_width) * grid_size) + 0.999999)))
        grid_y1 = max(grid_y0 + 1, min(grid_size, int(((y1 / image_height) * grid_size) + 0.999999)))

        return grid_x0, grid_y0, grid_x1, grid_y1

    def apply_text_embedding(self, embedding: torch.Tensor) -> None:
        emb_dim = _require_embedding_dim()
        if embedding.ndim != 1 or embedding.shape[0] != emb_dim:
            raise ValueError(f"embedding length must match emb_dim={emb_dim}.")

        if self.feature_map is None:
            raise ValueError("feature_map is not initialized.")

        grid_x0, grid_y0, grid_x1, grid_y1 = self.bbox_to_grid_coords()
        self.feature_map[grid_y0:grid_y1, grid_x0:grid_x1] = embedding.to(
            dtype=self.feature_map.dtype,
            device=self.feature_map.device,
        )



if __name__ == "__main__":
    example_config = OmegaConf.create({
        "shared": {"pretrained_clip": "openai/clip-vit-base-patch16"},
        "generator": {"max_grid_width": 14},
    })

    try:
        init_globals(example_config)
    except Exception as exc:
        print(f"failed to initialize from CLIP config: {exc}")
        INPUT_RESOLUTION = 224
        PATCH_NUM = 14
        MAX_GRID_WIDTH = example_config.generator.max_grid_width
        EMB_DIM = 512
        print(
            "fallback values are used: "
            f"input_resolution={INPUT_RESOLUTION}, patch_num={PATCH_NUM}, "
            f"max_grid_width={MAX_GRID_WIDTH}, emb_dim={EMB_DIM}"
        )

    sample = FeatureMapModel(
        image_id="example_image",
        image_path=None,
        size=ImageSizeModel(width=640, height=480),
        bbox=BboxModel(x=120, y=80, w=200, h=160),
        annotation="a dog on the grass",
    )

    print(f"input_resolution: {INPUT_RESOLUTION}")
    print(f"patch_num: {PATCH_NUM}")
    print(f"grid_size: {sample.grid_size}")
    print(f"feature_map shape: {tuple(sample.feature_map.shape)}")
    print(f"bbox grid coords: {sample.bbox_to_grid_coords()}")

    text_embedding = torch.randn(EMB_DIM, dtype=torch.float32)
    sample.apply_text_embedding(text_embedding)
    print("text embedding was applied to the bbox region.")
