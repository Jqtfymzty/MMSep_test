"""Basic tests for MMSepCache.

These tests use small CPU tensors and do not load a full language model.
Run with: pytest -q test_cache_basic.py
"""

import pytest
import torch

from cache import MMSepCache


def make_cache(**kwargs):
    """Create a small two-layer cache suitable for unit tests."""
    params = {
        "layer_num": 2,
        "model_type": "llama",
        "init_cache_size": 1,
        "sep_cache_size": 2,
        "local_size": 2,
        "cache_size": 8,
        "image_token_length": 0,
        "image_start_pos": [0, 0],
        "device": "cpu",
    }
    params.update(kwargs)
    return MMSepCache(**params)


def make_kv(seq_len, value_offset=0):
    """Create deterministic [batch, heads, sequence, head_dim] tensors."""
    key = torch.arange(seq_len * 2, dtype=torch.float32).reshape(1, 1, seq_len, 2)
    value = key + value_offset
    return key, value


def test_cache_initializes_layer_parameters():
    cache = make_cache()

    assert cache.layer_num == 2
    assert cache.init_cache_size == [1, 1]
    assert cache.local_size == [2, 2]
    assert len(cache) == 0
    assert cache.get_usable_length(0) == 0


def test_cache_requires_layer_num():
    with pytest.raises(AssertionError):
        MMSepCache(model_type="llama")


def test_cache_rejects_wrong_layer_parameter_length():
    with pytest.raises(AssertionError):
        make_cache(local_size=[2])


def test_prefill_writes_kv_and_token_ids():
    cache = make_cache()
    key, value = make_kv(3, value_offset=100)
    input_ids = torch.tensor([[10, 11, 12]])

    cached_key, cached_value = cache.update(
        key,
        value,
        input_ids=input_ids,
        layer_idx=0,
        PREFILLING_FLAG=True,
    )

    assert torch.equal(cached_key, key)
    assert torch.equal(cached_value, value)
    assert cache.get_usable_length(0) == 3
    assert cache.get_seq_length() == 3
    assert torch.equal(cache.past_tok_ids[0], input_ids)


def test_second_layer_can_store_same_prefill():
    cache = make_cache()
    key, value = make_kv(2)
    input_ids = torch.tensor([[20, 21]])

    cache.update(key, value, input_ids, layer_idx=0, PREFILLING_FLAG=True)
    cache.update(key, value, input_ids, layer_idx=1, PREFILLING_FLAG=True)

    assert len(cache) == 2
    assert cache.get_usable_length(1) == 2
    assert cache.get_batch_size() == 1


def test_decode_appends_one_token_before_cache_limit():
    cache = make_cache()
    key, value = make_kv(3)
    cache.update(key, value, torch.tensor([[1, 2, 3]]), layer_idx=0, PREFILLING_FLAG=True)

    new_key, new_value = make_kv(1, value_offset=50)
    cached_key, cached_value = cache.update(
        new_key,
        new_value,
        input_ids=torch.tensor([[4]]),
        layer_idx=0,
        PREFILLING_FLAG=False,
    )

    assert cached_key.shape[-2] == 4
    assert cached_value.shape[-2] == 4
    assert cache.get_usable_length(0) == 4
    assert torch.equal(cache.past_tok_ids[0], torch.tensor([[1, 2, 3, 4]]))


def test_initial_position_offset_counts_padding_tokens():
    cache = make_cache()
    key, value = make_kv(3)
    token_ids = torch.tensor([[128009, 128009, 10]])
    cache.update(key, value, token_ids, layer_idx=0, PREFILLING_FLAG=True)

    assert cache.get_initial_pos_offset(0) == 2


def test_empty_cache_has_no_batch_size():
    cache = make_cache()

    with pytest.raises(AssertionError):
        cache.get_batch_size()


def test_get_kv_pair_returns_layer_cache():
    cache = make_cache()
    key, value = make_kv(2)
    cache.update(key, value, torch.tensor([[1, 2]]), layer_idx=0, PREFILLING_FLAG=True)

    cached_key, cached_value = cache.get_kv_pair(0)

    assert torch.equal(cached_key, key)
    assert torch.equal(cached_value, value)
