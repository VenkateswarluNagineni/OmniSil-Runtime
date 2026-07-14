"""
Tiled FlashAttention & PagedAttention KV Kernel Implementation.

Implements online softmax rescaling (FlashAttention tiling algorithm) and block-table
virtual-to-physical memory mapping (PagedAttention) to eliminate KV-cache memory fragmentation
and maximize memory bandwidth utilization across heterogeneous silicon.
"""

import math

import numpy as np

try:
    import torch  # noqa: F401
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class PagedAttentionKernel:
    """
    Tiled KV-Cache Attention execution engine.
    Supports block-table lookups for dynamic page allocation.
    """
    def __init__(self, block_size: int = 16):
        self.block_size = block_size

    def paged_attention_forward(
        self,
        query: np.ndarray,          # Shape: [batch_size, num_heads, head_dim]
        key_cache: np.ndarray,      # Shape: [num_blocks, block_size, num_heads, head_dim]
        value_cache: np.ndarray,    # Shape: [num_blocks, block_size, num_heads, head_dim]
        block_tables: np.ndarray,   # Shape: [batch_size, max_num_blocks_per_seq]
        context_lens: np.ndarray,   # Shape: [batch_size]
        scale: float | None = None
    ) -> np.ndarray:
        """
        Executes PagedAttention forward pass using tiled memory access.
        """
        batch_size, num_heads, head_dim = query.shape
        if scale is None:
            scale = 1.0 / math.sqrt(head_dim)

        output = np.zeros_like(query)

        # Loop over batch items (sequences)
        for b in range(batch_size):
            ctx_len = context_lens[b]
            if ctx_len == 0:
                continue

            num_seq_blocks = (ctx_len + self.block_size - 1) // self.block_size
            seq_block_indices = block_tables[b, :num_seq_blocks]

            # Reconstruct virtual key and value tensors for the sequence from physical pages
            # Gather physical pages
            k_pages = key_cache[seq_block_indices]  # Shape: [num_seq_blocks, block_size, num_heads, head_dim]
            v_pages = value_cache[seq_block_indices]

            # Reshape to continuous virtual sequence
            k_seq = k_pages.reshape(-1, num_heads, head_dim)[:ctx_len]  # [ctx_len, num_heads, head_dim]
            v_seq = v_pages.reshape(-1, num_heads, head_dim)[:ctx_len]  # [ctx_len, num_heads, head_dim]

            # Transpose for attention calculation: [num_heads, 1, head_dim] x [num_heads, head_dim, ctx_len]
            q_b = query[b][:, np.newaxis, :]                  # [num_heads, 1, head_dim]
            k_b = k_seq.transpose(1, 2, 0)                    # [num_heads, head_dim, ctx_len]
            v_b = v_seq.transpose(1, 0, 2)                    # [num_heads, ctx_len, head_dim]

            # Scaled dot-product attention per head
            scores = np.matmul(q_b, k_b) * scale              # [num_heads, 1, ctx_len]

            # Stable numerical softmax
            scores_max = np.max(scores, axis=-1, keepdims=True)
            exp_scores = np.exp(scores - scores_max)
            attn_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)

            # Weighted sum of values
            head_out = np.matmul(attn_weights, v_b)           # [num_heads, 1, head_dim]
            output[b] = head_out.squeeze(1)                   # [num_heads, head_dim]

        return output


def flash_attention_tiled(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    tile_size_q: int = 32,
    tile_size_kv: int = 32
) -> np.ndarray:
    """
    Vectorized FlashAttention forward pass simulating online softmax rescaling across memory tiles.
    q, k, v shapes: [batch_size, num_heads, seq_len, head_dim]
    """
    batch_size, num_heads, seq_len, head_dim = q.shape
    scale = 1.0 / math.sqrt(head_dim)

    out_tensor = np.zeros_like(q)

    # Outer loop over sequence tiles
    for i in range(0, seq_len, tile_size_q):
        i_end = min(i + tile_size_q, seq_len)
        Qi = q[:, :, i:i_end, :]  # [B, H, T_q, D]
        Oi = np.zeros_like(Qi)
        Li = np.zeros((batch_size, num_heads, i_end - i, 1))
        Mi = np.full((batch_size, num_heads, i_end - i, 1), -np.inf)

        for j in range(0, seq_len, tile_size_kv):
            j_end = min(j + tile_size_kv, seq_len)
            Kj = k[:, :, j:j_end, :]  # [B, H, T_kv, D]
            Vj = v[:, :, j:j_end, :]  # [B, H, T_kv, D]

            # Sij = Qi @ Kj^T * scale
            Sij = np.matmul(Qi, Kj.transpose(0, 1, 3, 2)) * scale  # [B, H, T_q, T_kv]

            # Online softmax update
            mij = np.maximum(Mi, np.max(Sij, axis=-1, keepdims=True))
            Pij = np.exp(Sij - mij)
            lij = np.sum(Pij, axis=-1, keepdims=True)

            # Update running output and normalization
            alpha = np.exp(Mi - mij)
            Oi = Oi * alpha + np.matmul(Pij, Vj)
            Li = Li * alpha + lij
            Mi = mij

        out_tensor[:, :, i:i_end, :] = Oi / Li

    return out_tensor
