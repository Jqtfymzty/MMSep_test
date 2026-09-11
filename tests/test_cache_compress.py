"""成员 B：MMSepCache 历史窗口分隔符提取测试。"""

from __future__ import annotations

from collections.abc import Callable

import pytest
import torch

from src.cache import MMSepCache


def test_no_separator_returns_empty_cache(
    cache_compress_factory: Callable[..., MMSepCache],
    cache_compress_kv_factory: Callable[..., tuple[torch.Tensor, torch.Tensor]],
) -> None:
    cache = cache_compress_factory()
    key, value = cache_compress_kv_factory()
    token_ids = torch.full((2, 8), 9, dtype=torch.long)

    (sep_key, sep_value), sep_ids, min_count, max_count = (
        cache.compress_past_win_2_seps((key, value), token_ids)
    )

    assert min_count.item() == 0
    assert max_count.item() == 0
    assert sep_ids.shape == (2, 0)
    assert sep_key.shape == (2, 2, 0, 4)
    assert sep_value.shape == (2, 2, 0, 4)


def test_single_separator_keeps_matching_kv(
    cache_compress_factory: Callable[..., MMSepCache],
    cache_compress_kv_factory: Callable[..., tuple[torch.Tensor, torch.Tensor]],
) -> None:
    cache = cache_compress_factory()
    key, value = cache_compress_kv_factory(seq_len=5)
    token_ids = torch.tensor(
        [[9, 2, 8, 7, 6], [9, 8, 7, 3, 6]], dtype=torch.long
    )

    (sep_key, sep_value), sep_ids, min_count, max_count = (
        cache.compress_past_win_2_seps((key, value), token_ids)
    )

    assert min_count.item() == 1
    assert max_count.item() == 1
    assert sep_ids.tolist() == [[2], [3]]
    torch.testing.assert_close(sep_key[0, :, 0], key[0, :, 1])
    torch.testing.assert_close(sep_key[1, :, 0], key[1, :, 3])
    torch.testing.assert_close(sep_value[0, :, 0], value[0, :, 1])
    torch.testing.assert_close(sep_value[1, :, 0], value[1, :, 3])


def test_multiple_separators_keep_original_order(
    cache_compress_factory: Callable[..., MMSepCache],
    cache_compress_kv_factory: Callable[..., tuple[torch.Tensor, torch.Tensor]],
) -> None:
    cache = cache_compress_factory()
    key, value = cache_compress_kv_factory(batch_size=1, seq_len=7)
    token_ids = torch.tensor([[2, 9, 3, 8, 2, 7, 6]], dtype=torch.long)

    (sep_key, sep_value), sep_ids, min_count, max_count = (
        cache.compress_past_win_2_seps((key, value), token_ids)
    )

    assert min_count.item() == 3
    assert max_count.item() == 3
    assert sep_ids.tolist() == [[2, 3, 2]]
    torch.testing.assert_close(sep_key, key[:, :, [0, 2, 4], :])
    torch.testing.assert_close(sep_value, value[:, :, [0, 2, 4], :])


def test_batch_truncates_to_min_separator_count(
    cache_compress_factory: Callable[..., MMSepCache],
    cache_compress_kv_factory: Callable[..., tuple[torch.Tensor, torch.Tensor]],
    cache_compress_token_ids: torch.Tensor,
) -> None:
    cache = cache_compress_factory()
    key, value = cache_compress_kv_factory()

    (sep_key, sep_value), sep_ids, min_count, max_count = (
        cache.compress_past_win_2_seps(
            (key, value),
            cache_compress_token_ids,
            SEP_PADDING_IN_BATCH=False,
        )
    )

    assert min_count.item() == 1
    assert max_count.item() == 3
    assert sep_ids.tolist() == [[2], [2]]
    torch.testing.assert_close(sep_key[0, :, 0], key[0, :, 1])
    torch.testing.assert_close(sep_key[1, :, 0], key[1, :, 2])
    torch.testing.assert_close(sep_value[0, :, 0], value[0, :, 1])
    torch.testing.assert_close(sep_value[1, :, 0], value[1, :, 2])


def test_batch_pads_to_max_separator_count_with_window_tail(
    cache_compress_factory: Callable[..., MMSepCache],
    cache_compress_kv_factory: Callable[..., tuple[torch.Tensor, torch.Tensor]],
    cache_compress_token_ids: torch.Tensor,
) -> None:
    cache = cache_compress_factory()
    key, value = cache_compress_kv_factory()

    (sep_key, sep_value), sep_ids, min_count, max_count = (
        cache.compress_past_win_2_seps(
            (key, value),
            cache_compress_token_ids,
            SEP_PADDING_IN_BATCH=True,
        )
    )

    assert min_count.item() == 1
    assert max_count.item() == 3
    assert sep_ids.tolist() == [[2, 3, 2], [2, 4, 1]]
    torch.testing.assert_close(sep_key[0], key[0, :, [1, 3, 5], :])
    torch.testing.assert_close(sep_key[1], key[1, :, [2, 6, 7], :])
    torch.testing.assert_close(sep_value[0], value[0, :, [1, 3, 5], :])
    torch.testing.assert_close(sep_value[1], value[1, :, [2, 6, 7], :])


def test_min_separator_alert_rejects_empty_batch_record(
    cache_compress_factory: Callable[..., MMSepCache],
    cache_compress_kv_factory: Callable[..., tuple[torch.Tensor, torch.Tensor]],
) -> None:
    cache = cache_compress_factory()
    key, value = cache_compress_kv_factory(seq_len=4)
    token_ids = torch.tensor([[9, 2, 8, 7], [9, 8, 7, 6]], dtype=torch.long)

    with pytest.raises(AssertionError, match="at least one"):
        cache.compress_past_win_2_seps(
            (key, value),
            token_ids,
            MIN_SEP_ALERT=True,
            SEP_PADDING_IN_BATCH=False,
        )


def test_noimg_layer_compression_rebuilds_three_cache_sections(
    cache_compress_factory: Callable[..., MMSepCache],
    cache_compress_kv_factory: Callable[..., tuple[torch.Tensor, torch.Tensor]],
) -> None:
    cache = cache_compress_factory()
    key, value = cache_compress_kv_factory(batch_size=1)
    cache.past_tok_ids = [torch.tensor([[9, 2, 8, 3, 7, 6]])]

    (result_key, result_value), result_ids, initial_offset = (
        cache.compress_kv_cache_and_tokids_noimg_layer_wise(
            (key, value),
            layer_idx=0,
            SEP_ACCUMULATION=False,
            USE_MAX_SEP_CACHE=False,
            SEP_PADDING_IN_BATCH=False,
        )
    )

    kept_positions = [0, 1, 3, 5, 6, 7]
    assert initial_offset == 2
    assert result_ids.tolist() == [[2, 3, 7, 6]]
    assert cache.sep_exrange[0] == 4
    torch.testing.assert_close(result_key, key[:, :, kept_positions, :])
    torch.testing.assert_close(result_value, value[:, :, kept_positions, :])


def test_aligned_layer_compression_keeps_initial_token_ids(
    cache_compress_factory: Callable[..., MMSepCache],
    cache_compress_kv_factory: Callable[..., tuple[torch.Tensor, torch.Tensor]],
) -> None:
    cache = cache_compress_factory()
    key, value = cache_compress_kv_factory(batch_size=1)
    cache.key_cache = [key]
    cache.value_cache = [value]
    cache.past_tok_ids = [torch.tensor([[10, 11, 9, 2, 8, 3, 7, 6]])]

    (result_key, result_value), result_ids, initial_offset = (
        cache.compress_kv_cache_and_tokids_layer_wise(
            (key, value),
            layer_idx=0,
            SEP_ACCUMULATION=False,
            USE_MAX_SEP_CACHE=False,
            SEP_PADDING_IN_BATCH=False,
        )
    )

    kept_positions = [0, 1, 3, 5, 6, 7]
    assert initial_offset == 2
    assert result_ids.tolist() == [[10, 11, 2, 3, 7, 6]]
    torch.testing.assert_close(result_key, key[:, :, kept_positions, :])
    torch.testing.assert_close(result_value, value[:, :, kept_positions, :])


def test_disabled_accumulation_discards_previous_separators(
    cache_compress_factory: Callable[..., MMSepCache],
    cache_compress_kv_factory: Callable[..., tuple[torch.Tensor, torch.Tensor]],
) -> None:
    cache = cache_compress_factory()
    key, value = cache_compress_kv_factory(batch_size=1, seq_len=10)
    cache.sep_exrange[0] = 4
    cache.past_tok_ids = [torch.tensor([[2, 3, 9, 2, 8, 3, 7, 6]])]

    (result_key, result_value), result_ids, _ = (
        cache.compress_kv_cache_and_tokids_noimg_layer_wise(
            (key, value),
            layer_idx=0,
            SEP_ACCUMULATION=False,
            USE_MAX_SEP_CACHE=False,
            SEP_PADDING_IN_BATCH=False,
        )
    )

    kept_positions = [0, 1, 5, 7, 8, 9]
    assert result_ids.tolist() == [[2, 3, 7, 6]]
    assert cache.sep_exrange[0] == 4
    torch.testing.assert_close(result_key, key[:, :, kept_positions, :])
    torch.testing.assert_close(result_value, value[:, :, kept_positions, :])


def test_enabled_accumulation_keeps_old_and_new_separators(
    cache_compress_factory: Callable[..., MMSepCache],
    cache_compress_kv_factory: Callable[..., tuple[torch.Tensor, torch.Tensor]],
) -> None:
    cache = cache_compress_factory()
    key, value = cache_compress_kv_factory(batch_size=1, seq_len=10)
    cache.sep_exrange[0] = 4
    cache.past_tok_ids = [torch.tensor([[2, 3, 9, 2, 8, 3, 7, 6]])]

    (result_key, result_value), result_ids, _ = (
        cache.compress_kv_cache_and_tokids_noimg_layer_wise(
            (key, value),
            layer_idx=0,
            SEP_ACCUMULATION=True,
            USE_MAX_SEP_CACHE=False,
            SEP_PADDING_IN_BATCH=False,
        )
    )

    kept_positions = [0, 1, 2, 3, 5, 7, 8, 9]
    assert result_ids.tolist() == [[2, 3, 2, 3, 7, 6]]
    assert cache.sep_exrange[0] == 6
    assert cache.sep_cache_size[0] == 4
    assert cache.cache_size[0] == 9
    torch.testing.assert_close(result_key, key[:, :, kept_positions, :])
    torch.testing.assert_close(result_value, value[:, :, kept_positions, :])


def test_max_separator_cache_keeps_most_recent_separators(
    cache_compress_factory: Callable[..., MMSepCache],
    cache_compress_kv_factory: Callable[..., tuple[torch.Tensor, torch.Tensor]],
) -> None:
    cache = cache_compress_factory()
    key, value = cache_compress_kv_factory(batch_size=1, seq_len=10)
    cache.sep_exrange[0] = 4
    cache.past_tok_ids = [torch.tensor([[2, 3, 9, 2, 8, 3, 7, 6]])]

    (result_key, result_value), result_ids, _ = (
        cache.compress_kv_cache_and_tokids_noimg_layer_wise(
            (key, value),
            layer_idx=0,
            SEP_ACCUMULATION=True,
            USE_MAX_SEP_CACHE=True,
            SEP_PADDING_IN_BATCH=False,
        )
    )

    kept_positions = [0, 1, 3, 5, 7, 8, 9]
    assert result_ids.tolist() == [[3, 2, 3, 7, 6]]
    assert cache.sep_exrange[0] == 5
    assert cache.sep_cache_size[0] == 3
    assert cache.cache_size[0] == 8
    torch.testing.assert_close(result_key, key[:, :, kept_positions, :])
    torch.testing.assert_close(result_value, value[:, :, kept_positions, :])


def test_update_writes_first_prefill_cache(
    cache_compress_factory: Callable[..., MMSepCache],
    cache_compress_kv_factory: Callable[..., tuple[torch.Tensor, torch.Tensor]],
) -> None:
    cache = cache_compress_factory()
    key, value = cache_compress_kv_factory(batch_size=1, seq_len=3)
    input_ids = torch.tensor([[9, 2, 8]], dtype=torch.long)

    result_key, result_value = cache.update(
        key,
        value,
        input_ids,
        layer_idx=0,
        PREFILLING_FLAG=True,
    )

    assert cache.seen_tokens == 3
    assert cache.past_tok_ids[0].tolist() == [[9, 2, 8]]
    torch.testing.assert_close(result_key, key)
    torch.testing.assert_close(result_value, value)


def test_update_appends_one_decode_token_without_compression(
    cache_compress_factory: Callable[..., MMSepCache],
    cache_compress_kv_factory: Callable[..., tuple[torch.Tensor, torch.Tensor]],
) -> None:
    cache = cache_compress_factory()
    key, value = cache_compress_kv_factory(batch_size=1, seq_len=4)

    cache.update(
        key[:, :, :3, :],
        value[:, :, :3, :],
        torch.tensor([[9, 2, 8]]),
        layer_idx=0,
        PREFILLING_FLAG=True,
    )
    result_key, result_value = cache.update(
        key[:, :, 3:, :],
        value[:, :, 3:, :],
        torch.tensor([[7]]),
        layer_idx=0,
        PREFILLING_FLAG=False,
    )

    assert cache.seen_tokens == 4
    assert cache.get_usable_length(0) == 4
    assert cache.past_tok_ids[0].tolist() == [[9, 2, 8, 7]]
    torch.testing.assert_close(result_key, key)
    torch.testing.assert_close(result_value, value)


def test_update_compresses_when_new_length_reaches_capacity(
    cache_compress_factory: Callable[..., MMSepCache],
    cache_compress_kv_factory: Callable[..., tuple[torch.Tensor, torch.Tensor]],
) -> None:
    cache = cache_compress_factory()
    key, value = cache_compress_kv_factory(batch_size=1)
    cache.key_cache = [key[:, :, :7, :].clone()]
    cache.value_cache = [value[:, :, :7, :].clone()]
    cache.past_tok_ids = [torch.tensor([[9, 2, 8, 3, 7]])]

    result_key, result_value = cache.update(
        key[:, :, 7:, :],
        value[:, :, 7:, :],
        torch.tensor([[6]]),
        layer_idx=0,
        PREFILLING_FLAG=False,
    )

    kept_positions = [0, 1, 3, 5, 6, 7]
    assert cache.get_usable_length(0) == 6
    assert cache.past_tok_ids[0].tolist() == [[2, 3, 7, 6]]
    torch.testing.assert_close(result_key, key[:, :, kept_positions, :])
    torch.testing.assert_close(result_value, value[:, :, kept_positions, :])


def test_prefilling_bypasses_compression_at_capacity(
    cache_compress_factory: Callable[..., MMSepCache],
    cache_compress_kv_factory: Callable[..., tuple[torch.Tensor, torch.Tensor]],
) -> None:
    cache = cache_compress_factory()
    key, value = cache_compress_kv_factory(batch_size=1)
    cache.key_cache = [key[:, :, :7, :].clone()]
    cache.value_cache = [value[:, :, :7, :].clone()]
    cache.past_tok_ids = [torch.tensor([[9, 2, 8, 3, 7]])]

    result_key, result_value = cache.update(
        key[:, :, 7:, :],
        value[:, :, 7:, :],
        torch.tensor([[6]]),
        layer_idx=0,
        PREFILLING_FLAG=True,
    )

    assert cache.get_usable_length(0) == 8
    assert cache.past_tok_ids[0].tolist() == [[9, 2, 8, 3, 7, 6]]
    torch.testing.assert_close(result_key, key)
    torch.testing.assert_close(result_value, value)


def test_update_returns_selected_visual_kv_without_overwriting_full_cache(
    cache_compress_factory: Callable[..., MMSepCache],
    cache_compress_kv_factory: Callable[..., tuple[torch.Tensor, torch.Tensor]],
) -> None:
    cache = cache_compress_factory(
        image_token_length=2,
        image_start_pos=[2],
        mmsep_layer=0,
    )
    key, value = cache_compress_kv_factory(batch_size=1, seq_len=6)

    result_key, result_value = cache.update(
        key,
        value,
        torch.tensor([[9, 8, 7, 6, 5, 4]]),
        layer_idx=0,
        PREFILLING_FLAG=False,
        visual_sep_pos=torch.tensor([2]),
    )

    kept_positions = [0, 1, 2, 4, 5]
    assert cache.get_usable_length(0) == 6
    assert result_key.shape[2] == 5
    torch.testing.assert_close(cache.key_cache[0], key)
    torch.testing.assert_close(cache.value_cache[0], value)
    torch.testing.assert_close(result_key, key[:, :, kept_positions, :])
    torch.testing.assert_close(result_value, value[:, :, kept_positions, :])


def test_separator_token_returns_full_kv_instead_of_visual_subset(
    cache_compress_factory: Callable[..., MMSepCache],
    cache_compress_kv_factory: Callable[..., tuple[torch.Tensor, torch.Tensor]],
) -> None:
    cache = cache_compress_factory(
        image_token_length=2,
        image_start_pos=[2],
        mmsep_layer=0,
    )
    key, value = cache_compress_kv_factory(batch_size=1, seq_len=6)

    result_key, result_value = cache.update(
        key,
        value,
        torch.tensor([[9, 8, 7, 6, 5, 2]]),
        layer_idx=0,
        PREFILLING_FLAG=False,
        visual_sep_pos=torch.tensor([2]),
    )

    assert result_key.shape[2] == 6
    torch.testing.assert_close(result_key, key)
    torch.testing.assert_close(result_value, value)
