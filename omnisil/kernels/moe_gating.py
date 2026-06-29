"""
Mixture-of-Experts (MoE) Top-K Gating & Routing Kernel.

Implements token routing dispatch to specialized expert feeds, load-balancing calculation,
and router z-loss regularization to prevent routing collapse in sparse distributed transformers.
"""

from typing import Tuple, Dict, Any
import numpy as np


class MoETopKGatingKernel:
    """
    Executes Top-K routing logic for sparse MoE architectures.
    Calculates dispatch indices, expert gating weights, and auxiliary load-balancing metrics.
    """
    def __init__(self, num_experts: int, top_k: int = 2, z_loss_coeff: float = 1e-4):
        self.num_experts = num_experts
        self.top_k = top_k
        self.z_loss_coeff = z_loss_coeff

    def route_tokens(
        self,
        router_logits: np.ndarray  # Shape: [num_tokens, num_experts]
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
        """
        Routes tokens to top-k experts and computes load balancing stats.
        Returns:
            expert_indices: Shape [num_tokens, top_k]
            expert_weights: Shape [num_tokens, top_k] (normalized softmax probabilities)
            metrics: Dict containing load variance and z-loss penalties
        """
        num_tokens, num_exp = router_logits.shape
        assert num_exp == self.num_experts, f"Expected {self.num_experts} experts, got {num_exp}"

        # Numerical stabilization and softmax over experts
        logits_max = np.max(router_logits, axis=-1, keepdims=True)
        exp_logits = np.exp(router_logits - logits_max)
        probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)  # [num_tokens, num_experts]

        # Top-K selection per token
        # Sort indices along last axis in descending order
        sorted_indices = np.argsort(-probs, axis=-1)
        top_k_indices = sorted_indices[:, :self.top_k]

        # Extract top-k probabilities
        row_indices = np.arange(num_tokens)[:, np.newaxis]
        top_k_probs = probs[row_indices, top_k_indices]

        # Re-normalize weights among selected top-k experts
        weights_sum = np.sum(top_k_probs, axis=-1, keepdims=True)
        expert_weights = top_k_probs / (weights_sum + 1e-9)

        # Auxiliary Load Balancing Calculation (Switch Transformer formulation)
        # Fraction of tokens dispatched to each expert
        expert_counts = np.zeros(self.num_experts)
        for e in range(self.num_experts):
            expert_counts[e] = np.sum(top_k_indices == e)
        fraction_tokens = expert_counts / (num_tokens * self.top_k)

        # Average routing probability per expert
        mean_probs = np.mean(probs, axis=0)

        # Load balancing auxiliary loss: N * sum(f_i * P_i)
        load_balance_loss = float(self.num_experts * np.sum(fraction_tokens * mean_probs))

        # Router Z-loss: log(sum(exp(logits)))^2 penalizing excessively large router logits
        log_z = np.log(np.sum(exp_logits, axis=-1, keepdims=True)) + logits_max
        z_loss = float(self.z_loss_coeff * np.mean(log_z ** 2))

        metrics = {
            "load_balance_loss": load_balance_loss,
            "router_z_loss": z_loss,
            "max_expert_load": float(np.max(fraction_tokens)),
            "min_expert_load": float(np.min(fraction_tokens))
        }

        return top_k_indices, expert_weights, metrics
