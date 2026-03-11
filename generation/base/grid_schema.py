from typing import Tuple
from pydantic import BaseModel, Field
from omegaconf import DictConfig
import random

from transformers import CLIPModel

from APIs.refcocoAPI import RefcocoModel


MAX_GRID_WIDTH: int | None = None
IMAGE_EMB_DIM: int | None = None
TEXT_EMB_DIM: int | None = None
EMB_DIM: int | None = None


def _load_clip_embedding_dims(pretrained_model_name: str) -> tuple[int, int, int]:
    """
        사전학습된 CLIP모델의 임베딩 차원을 불러오는 helper 함수
    """

    clip_model = CLIPModel.from_pretrained(pretrained_model_name)
    image_emb_dim = clip_model.vision_model.config.hidden_size
    text_emb_dim = clip_model.text_model.config.hidden_size
    emb_dim = clip_model.config.projection_dim

    return image_emb_dim, text_emb_dim, emb_dim


def init_globals(config : DictConfig) -> None:
    """
        hydra로 정의한 값으로 전역변수 초기화
    """
    global MAX_GRID_WIDTH
    global IMAGE_EMB_DIM
    global TEXT_EMB_DIM
    global EMB_DIM

    MAX_GRID_WIDTH = config.generator.max_grid_width
    IMAGE_EMB_DIM, TEXT_EMB_DIM, EMB_DIM = _load_clip_embedding_dims(
        config.shared.pretrained_clip
    )


def _generate_random_size():
    return random.randint(1, MAX_GRID_WIDTH)


class FeatureMapModel(RefcocoModel):
    grid_size: int | None = Field(default_factory=_generate_random_size)
