"""
Automated unit tests validating numerical precision of OmniSil kernels against baselines.
"""

import numpy as np

from omnisil.kernels.attention import PagedAttentionKernel, flash_attention_tiled
from omnisil.kernels.moe_gating import MoETopKGatingKernel
from omnisil.kernels.quant_gemm import QuantizedGEMMKernel
from omnisil.runtime.engine import ContinuousBatchingEngine, InferenceRequest


def test_flash_attention_tiled_precision():
    np.random.seed(42)
    batch_size, num_heads, seq_len, head_dim = 2, 4, 32, 64
    q = np.random.randn(batch_size, num_heads, seq_len, head_dim)
    k = np.random.randn(batch_size, num_heads, seq_len, head_dim)
    v = np.random.randn(batch_size, num_heads, seq_len, head_dim)

    # Standard exact attention baseline
    scale = 1.0 / np.sqrt(head_dim)
    scores = np.matmul(q, k.transpose(0, 1, 3, 2)) * scale
    scores_max = np.max(scores, axis=-1, keepdims=True)
    exp_scores = np.exp(scores - scores_max)
    attn_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
    expected_out = np.matmul(attn_weights, v)

    # Tiled FlashAttention
    tiled_out = flash_attention_tiled(q, k, v, tile_size_q=16, tile_size_kv=16)

    assert np.allclose(expected_out, tiled_out, atol=1e-5), "Tiled FlashAttention output mismatch!"


def test_paged_attention_forward():
    np.random.seed(42)
    batch_size, num_heads, head_dim = 2, 2, 32
    block_size = 8
    num_blocks = 10

    query = np.random.randn(batch_size, num_heads, head_dim)
    key_cache = np.random.randn(num_blocks, block_size, num_heads, head_dim)
    value_cache = np.random.randn(num_blocks, block_size, num_heads, head_dim)

    block_tables = np.array([[0, 1], [2, 3]])
    context_lens = np.array([12, 16])

    kernel = PagedAttentionKernel(block_size=block_size)
    output = kernel.paged_attention_forward(query, key_cache, value_cache, block_tables, context_lens)

    assert output.shape == (batch_size, num_heads, head_dim), f"Unexpected shape {output.shape}"
    assert not np.isnan(output).any(), "NaN values found in PagedAttention output"


def test_moe_top_k_gating():
    np.random.seed(42)
    num_tokens, num_experts, top_k = 10, 8, 2
    logits = np.random.randn(num_tokens, num_experts)

    kernel = MoETopKGatingKernel(num_experts=num_experts, top_k=top_k)
    indices, weights, metrics = kernel.route_tokens(logits)

    assert indices.shape == (num_tokens, top_k)
    assert weights.shape == (num_tokens, top_k)
    assert np.allclose(np.sum(weights, axis=-1), 1.0, atol=1e-5), "Expert weights do not sum to 1.0"
    assert "load_balance_loss" in metrics and "router_z_loss" in metrics


def test_quantized_gemm_int4():
    np.random.seed(42)
    out_features, in_features = 16, 32
    weights = np.random.randn(out_features, in_features).astype(np.float32)
    activations = np.random.randn(4, in_features).astype(np.float32)

    packed_weights, scales = QuantizedGEMMKernel.quantize_int4_symmetric(weights)
    assert packed_weights.shape == (out_features, in_features // 2)

    output = QuantizedGEMMKernel.gemm_int4(activations, packed_weights, scales)
    expected = np.matmul(activations, weights.T)

    # Relative tolerance correlation check due to quantization compression
    correlation = np.corrcoef(output.flatten(), expected.flatten())[0, 1]
    assert correlation > 0.95, f"INT4 GEMM correlation too low: {correlation}"


def test_continuous_batching_engine():
    engine = ContinuousBatchingEngine(max_batch_size=4)
    req1 = InferenceRequest("req1", prompt_tokens=[1, 2, 3, 4], max_new_tokens=5)
    req2 = InferenceRequest("req2", prompt_tokens=[1, 2, 3, 4, 5, 6], max_new_tokens=5)

    engine.add_request(req1)
    engine.add_request(req2)

    result = engine.run_to_completion()
    assert result["total_steps"] > 0
    assert engine.stats["cached_tokens_saved"] >= 0
    assert req1.status == "COMPLETED" and req2.status == "COMPLETED"
