from __future__ import annotations

from typing import Any

import torch
from transformers import CLIPTokenizer


def split_reference_expression(annotation: str) -> list[str]:
    """
    Split a reference expression into word-level phrases using whitespace.

    The stack text encoder expects preprocessing to provide one phrase per
    word, so `"guy on cell"` becomes `["guy", "on", "cell"]`.
    """
    phrases = annotation.strip().split()
    if phrases:
        return phrases
    return [annotation.strip()]


def build_phrase_text_inputs(
    annotations: list[str],
    tokenizer: CLIPTokenizer,
    max_phrase_count: int,
) -> dict[str, torch.Tensor]:
    """
    Tokenize pre-split word phrases into stack text-model inputs.

    Returns:
        input_ids: `(batch, phrase_count, context_length)`
        attention_mask: `(batch, phrase_count, context_length)`
        phrase_mask: `(batch, phrase_count)`
    """
    if max_phrase_count <= 0:
        raise ValueError(f"max_phrase_count must be positive, got {max_phrase_count}.")

    batch_size = len(annotations)
    padded_phrases: list[str] = []
    phrase_mask_rows: list[list[int]] = []

    for annotation in annotations:
        phrases = split_reference_expression(annotation)
        phrases = phrases[:max_phrase_count]

        valid_phrase_count = len(phrases)
        pad_phrase_count = max_phrase_count - valid_phrase_count

        padded_phrases.extend(phrases + ([""] * pad_phrase_count))
        phrase_mask_rows.append(([1] * valid_phrase_count) + ([0] * pad_phrase_count))

    tokenized = tokenizer(
        padded_phrases,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )

    context_length = tokenized["input_ids"].shape[-1]
    input_ids = tokenized["input_ids"].reshape(batch_size, max_phrase_count, context_length)
    attention_mask = tokenized["attention_mask"].reshape(batch_size, max_phrase_count, context_length)
    phrase_mask = torch.tensor(phrase_mask_rows, dtype=torch.long)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "phrase_mask": phrase_mask,
    }


class TrainCollator:
    """
    Collate generated feature maps for the stack REC model.

    Compared with the base collator, this version splits each annotation into
    whitespace-delimited word phrases and returns both token-level masks for
    the text encoder and phrase-level masks for decoder cross-attention.
    """

    def __init__(
        self,
        tokenizer: CLIPTokenizer,
        expected_patch_count: int,
        noise_scale: float,
        max_phrase_count: int,
    ):
        self.tokenizer = tokenizer
        self.expected_patch_count = expected_patch_count
        self.noise_scale = noise_scale
        self.max_phrase_count = max_phrase_count

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        annotations = [sample["annotation"] for sample in batch]
        text_inputs = build_phrase_text_inputs(
            annotations=annotations,
            tokenizer=self.tokenizer,
            max_phrase_count=self.max_phrase_count,
        )

        flattened_feature_maps = []
        bbox_targets = []

        for sample in batch:
            feature_map = sample["feature_map"]
            if feature_map.ndim != 3:
                raise ValueError(
                    "feature_map must have shape (grid_size, grid_size, emb_dim), "
                    f"got {tuple(feature_map.shape)}."
                )

            patch_count = feature_map.shape[0] * feature_map.shape[1]
            if patch_count != self.expected_patch_count:
                raise ValueError(
                    f"flattened patch count must match expected_patch_count={self.expected_patch_count}, "
                    f"got {patch_count} from shape {tuple(feature_map.shape)}."
                )

            flattened_feature_maps.append(feature_map.reshape(patch_count, feature_map.shape[-1]))
            bbox_targets.append(sample["bbox"])

        image_features = torch.stack(flattened_feature_maps, dim=0)
        if self.noise_scale > 0:
            image_features = image_features + (torch.randn_like(image_features) * self.noise_scale)

        return {
            "image_features": image_features,
            "input_ids": text_inputs["input_ids"],
            "attention_mask": text_inputs["attention_mask"],
            "phrase_mask": text_inputs["phrase_mask"],
            "bbox_targets": torch.stack(bbox_targets, dim=0),
        }
