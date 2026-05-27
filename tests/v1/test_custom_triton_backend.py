# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.v1.attention.backends.custom_triton import (
    CustomTritonAttentionBackend,
    CustomTritonAttentionImpl,
)
from vllm.v1.attention.backends.triton_attn import (
    TritonAttentionImpl,
    TritonAttentionMetadataBuilder,
)


def test_custom_triton_backend_impl():
    assert CustomTritonAttentionBackend.get_impl_cls() is CustomTritonAttentionImpl


def test_custom_triton_backend_builder():
    assert (CustomTritonAttentionBackend.get_builder_cls()
            is TritonAttentionMetadataBuilder)
