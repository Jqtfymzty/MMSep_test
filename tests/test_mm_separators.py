"""成员 B：视觉 token 分隔符筛选测试。"""

from __future__ import annotations

from collections.abc import Callable

import pytest
import torch

from src.mm_separators import graph_rank_separators


def _features(batch_size: int = 1, seq_len: int = 10) -> torch.Tensor:
    values = torch.arange(batch_size * seq_len * 8, dtype=torch.float32)
    return values.reshape(batch_size, seq_len, 8) / 100


def test_visual_ranking_keeps_highest_attention_token_in_original_order(
    mmsep_model_factory: Callable[..., object],
) -> None:
    model = mmsep_model_factory()
    features = _features()
    position_ids = torch.arange(10).unsqueeze(0)
    attention_mask = torch.ones((1, 10), dtype=torch.long)

    new_position_ids, new_mask, result = graph_rank_separators(
        model,
        alpha=0.5,
        theta=0.5,
        layer=0,
        features=features,
        position_ids=position_ids,
        attention_mask=attention_mask,
    )

    kept_positions = [0, 1, 7, 8, 9]
    assert model.visual_sep_pos.tolist() == [7]
    assert model.image_tokens == [1]
    assert new_position_ids.tolist() == [[0, 1, 2, 3, 4]]
    assert new_mask.tolist() == [[1, 1, 1, 1, 1]]
    torch.testing.assert_close(result, features[:, kept_positions, :])


def test_two_selected_visual_tokens_are_restored_to_sequence_order(
    mmsep_model_factory: Callable[..., object],
) -> None:
    model = mmsep_model_factory(image_tokens=[11], image_token_posi=[1])
    features = _features(seq_len=14)

    _, _, result = graph_rank_separators(
        model,
        alpha=0.5,
        theta=0.5,
        layer=0,
        features=features,
        position_ids=torch.arange(14).unsqueeze(0),
        attention_mask=torch.ones((1, 14), dtype=torch.long),
    )

    assert model.visual_sep_pos.tolist() == [10, 11]
    assert model.image_tokens == [2]
    torch.testing.assert_close(result, features[:, [0, 10, 11, 12, 13], :])


def test_small_visual_region_can_be_pruned_to_zero_tokens(
    mmsep_model_factory: Callable[..., object],
) -> None:
    model = mmsep_model_factory(image_tokens=[2], image_token_posi=[2])
    features = _features(seq_len=6)

    new_position_ids, new_mask, result = graph_rank_separators(
        model,
        alpha=0.5,
        theta=0.5,
        layer=0,
        features=features,
        position_ids=torch.arange(6).unsqueeze(0),
        attention_mask=torch.ones((1, 6), dtype=torch.long),
    )

    assert model.visual_sep_pos.numel() == 0
    assert model.image_tokens == [0]
    assert new_position_ids.tolist() == [[0, 1, 2, 3]]
    assert new_mask.tolist() == [[1, 1, 1, 1]]
    torch.testing.assert_close(result, features[:, [0, 1, 4, 5], :])


def test_none_position_ids_and_mask_are_restored_as_none(
    mmsep_model_factory: Callable[..., object],
) -> None:
    model = mmsep_model_factory()

    new_position_ids, new_mask, result = graph_rank_separators(
        model,
        alpha=0.5,
        theta=0.5,
        layer=0,
        features=_features(),
        position_ids=None,
        attention_mask=None,
    )

    assert new_position_ids is None
    assert new_mask is None
    assert result.shape == (1, 5, 8)


def test_left_padding_is_rejected(
    mmsep_model_factory: Callable[..., object],
) -> None:
    model = mmsep_model_factory(padding_side="left")

    with pytest.raises(ValueError, match="tokenizer_padding_side"):
        graph_rank_separators(
            model,
            alpha=0.5,
            theta=0.5,
            layer=0,
            features=_features(),
            position_ids=torch.arange(10).unsqueeze(0),
            attention_mask=torch.ones((1, 10), dtype=torch.long),
        )


def test_non_flash_attention_is_rejected(
    mmsep_model_factory: Callable[..., object],
) -> None:
    model = mmsep_model_factory(attention_type="eager")

    with pytest.raises(NotImplementedError, match="flash_attention_2"):
        graph_rank_separators(
            model,
            alpha=0.5,
            theta=0.5,
            layer=0,
            features=_features(),
            position_ids=torch.arange(10).unsqueeze(0),
            attention_mask=torch.ones((1, 10), dtype=torch.long),
        )


def test_record_without_image_keeps_original_data(
    mmsep_model_factory: Callable[..., object],
) -> None:
    model = mmsep_model_factory(image_tokens=[0], image_token_posi=[-1])
    features = _features(seq_len=5)
    position_ids = torch.arange(5).unsqueeze(0)
    attention_mask = torch.ones((1, 5), dtype=torch.long)

    new_position_ids, new_mask, result = graph_rank_separators(
        model,
        alpha=0.5,
        theta=0.5,
        layer=0,
        features=features,
        position_ids=position_ids,
        attention_mask=attention_mask,
    )

    torch.testing.assert_close(result, features)
    torch.testing.assert_close(new_position_ids, position_ids)
    torch.testing.assert_close(new_mask, attention_mask)


def test_training_mode_can_rank_visual_tokens(
    mmsep_model_factory: Callable[..., object],
) -> None:
    model = mmsep_model_factory(training=True)

    _, _, result = graph_rank_separators(
        model,
        alpha=0.5,
        theta=0.5,
        layer=0,
        features=_features(),
        position_ids=torch.arange(10).unsqueeze(0),
        attention_mask=torch.ones((1, 10), dtype=torch.long),
    )

    assert result.shape == (1, 5, 8)


def test_batch_with_different_visual_lengths_is_padded_consistently(
    mmsep_model_factory: Callable[..., object],
    mmsep_features: torch.Tensor,
    mmsep_position_ids: torch.Tensor,
    mmsep_attention_mask: torch.Tensor,
) -> None:
    model = mmsep_model_factory(
        image_tokens=[6, 3],
        image_token_posi=[2, 2],
    )

    new_position_ids, new_mask, result = graph_rank_separators(
        model,
        alpha=0.5,
        theta=0.5,
        layer=0,
        features=mmsep_features,
        position_ids=mmsep_position_ids,
        attention_mask=mmsep_attention_mask,
    )

    assert result.shape == (2, 7, 8)
    assert new_position_ids.shape == (2, 7)
    assert new_mask.shape == (2, 7)
