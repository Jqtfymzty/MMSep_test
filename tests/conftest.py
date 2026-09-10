"""Shared pytest fixtures for the MMSep test project.

Member B fixtures use the ``cache_compress_`` and ``mmsep_`` prefixes so that
Member A can add independent fixtures without naming collisions.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest
import torch

from src.cache import MMSepCache


@pytest.fixture
def cache_compress_factory() -> Callable[..., MMSepCache]:
    """Return a factory for a small, CPU-only ``MMSepCache`` instance."""

    def _factory(**overrides: object) -> MMSepCache:
        settings: dict[str, object] = {
            "init_cache_size": 2,
            "sep_cache_size": 3,
            "local_size": 2,
            "cache_size": 8,
            "image_token_length": 4,
            "image_start_pos": [2, 2],
            "mmsep_layer": 1,
            "SEP_ACCUMULATION": True,
            "USE_MAX_SEP_CACHE": False,
            "SEP_PADDING_IN_BATCH": False,
            "separator_token_ids": [2, 3],
            "PADDING_ID": 0,
            "layer_num": 2,
            "device": "cpu",
            "pe_dim": 4,
        }
        settings.update(overrides)
        return MMSepCache(**settings)

    return _factory


@pytest.fixture
def cache_compress_kv_factory() -> Callable[..., tuple[torch.Tensor, torch.Tensor]]:
    """Build traceable Key/Value tensors with sequence length on dimension 2."""

    def _factory(
        *,
        batch_size: int = 2,
        num_heads: int = 2,
        seq_len: int = 8,
        head_dim: int = 4,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        element_count = batch_size * num_heads * seq_len * head_dim
        key = torch.arange(element_count, dtype=torch.float32).reshape(
            batch_size, num_heads, seq_len, head_dim
        )
        value = key + 10_000
        return key, value

    return _factory


@pytest.fixture
def cache_compress_token_ids() -> torch.Tensor:
    """Return two equal-length token sequences with different separator counts."""

    return torch.tensor(
        [
            [9, 2, 8, 3, 7, 2, 6, 5],
            [9, 8, 2, 7, 6, 5, 4, 1],
        ],
        dtype=torch.long,
    )


@pytest.fixture
def mmsep_features() -> torch.Tensor:
    """Return deterministic multimodal features in ``[B, S, hidden]`` form."""

    batch_size, seq_len, hidden_size = 2, 10, 8
    values = torch.arange(
        batch_size * seq_len * hidden_size, dtype=torch.float32
    )
    return values.reshape(batch_size, seq_len, hidden_size) / 100


@pytest.fixture
def mmsep_position_ids() -> torch.Tensor:
    """Return position ids aligned with ``mmsep_features``."""

    return torch.arange(10, dtype=torch.long).unsqueeze(0).repeat(2, 1)


@pytest.fixture
def mmsep_attention_mask() -> torch.Tensor:
    """Return a right-padded attention mask for two records."""

    return torch.tensor(
        [
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
        ],
        dtype=torch.long,
    )
