"""
Continuous Batching Inference Engine with LMCache Prefix Caching.

Manages dynamic request scheduling, KV-cache page allocation, and prefix tree lookup
to maximize throughput and minimize time-to-first-token (TTFT) across multi-silicon setups.
"""

import hashlib
import time
from typing import Any


class InferenceRequest:
    def __init__(self, request_id: str, prompt_tokens: list[int], max_new_tokens: int = 32):
        self.request_id = request_id
        self.prompt_tokens = prompt_tokens
        self.max_new_tokens = max_new_tokens
        self.generated_tokens: list[int] = []
        self.status = "PENDING"  # PENDING, RUNNING, COMPLETED
        self.arrival_time = time.time()
        self.start_time: float | None = None
        self.end_time: float | None = None


class LMCachePrefixManager:
    """
    Prefix cache manager simulating LMCache token block sharing.
    Computes hash signatures for prompt blocks to recycle precomputed KV cache pages.
    """
    def __init__(self, block_size: int = 16):
        self.block_size = block_size
        self.prefix_pool: dict[str, int] = {}  # Block Hash -> Physical Page ID
        self.page_counter = 0

    def _hash_block(self, tokens: list[int]) -> str:
        return hashlib.sha256(str(tokens).encode('utf-8')).hexdigest()

    def match_prefix(self, prompt_tokens: list[int]) -> tuple[int, list[int]]:
        """
        Matches prompt tokens against cached prefix blocks.
        Returns number of matched tokens and list of recycled physical page IDs.
        """
        matched_tokens = 0
        recycled_pages = []

        num_blocks = len(prompt_tokens) // self.block_size
        for i in range(num_blocks):
            block = prompt_tokens[i * self.block_size : (i + 1) * self.block_size]
            block_hash = self._hash_block(block)

            if block_hash in self.prefix_pool:
                matched_tokens += self.block_size
                recycled_pages.append(self.prefix_pool[block_hash])
            else:
                # Allocate new page and register in pool
                self.page_counter += 1
                self.prefix_pool[block_hash] = self.page_counter
                recycled_pages.append(self.page_counter)

        return matched_tokens, recycled_pages


class ContinuousBatchingEngine:
    """
    Continuous batching request scheduler orchestrating iteration-level scheduling.
    """
    def __init__(self, max_batch_size: int = 8, block_size: int = 16):
        self.max_batch_size = max_batch_size
        self.block_size = block_size
        self.request_queue: list[InferenceRequest] = []
        self.running_batch: list[InferenceRequest] = []
        self.prefix_cache = LMCachePrefixManager(block_size=block_size)
        self.stats = {
            "total_requests": 0,
            "cached_tokens_saved": 0,
            "total_tokens_generated": 0
        }

    def add_request(self, request: InferenceRequest):
        self.request_queue.append(request)
        self.stats["total_requests"] += 1

    def step(self) -> dict[str, Any]:
        """
        Executes one iteration step of continuous batching inference.
        Admits pending requests up to max_batch_size and advances generated tokens.
        """
        # Admit new requests
        while len(self.running_batch) < self.max_batch_size and self.request_queue:
            req = self.request_queue.pop(0)
            req.status = "RUNNING"
            req.start_time = time.time()

            # Check prefix cache
            matched_len, _pages = self.prefix_cache.match_prefix(req.prompt_tokens)
            self.stats["cached_tokens_saved"] += matched_len
            self.running_batch.append(req)

        if not self.running_batch:
            return {"active_requests": 0, "completed_in_step": 0}

        completed_count = 0
        # Simulate token generation iteration for all active requests
        for req in list(self.running_batch):
            # Generate dummy token ID based on sequence position
            next_token = (len(req.prompt_tokens) + len(req.generated_tokens)) % 32000
            req.generated_tokens.append(next_token)
            self.stats["total_tokens_generated"] += 1

            if len(req.generated_tokens) >= req.max_new_tokens:
                req.status = "COMPLETED"
                req.end_time = time.time()
                self.running_batch.remove(req)
                completed_count += 1

        return {
            "active_requests": len(self.running_batch),
            "completed_in_step": completed_count
        }

    def run_to_completion(self) -> dict[str, Any]:
        """
        Runs the scheduler loop until all requests are completed.
        """
        start_time = time.time()
        steps = 0
        while self.request_queue or self.running_batch:
            self.step()
            steps += 1

        total_time = time.time() - start_time
        throughput = self.stats["total_tokens_generated"] / max(total_time, 1e-5)

        return {
            "total_time_sec": total_time,
            "total_steps": steps,
            "throughput_tokens_sec": throughput,
            "stats": self.stats
        }
