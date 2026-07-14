"""
Hardware Dispatcher Engine for OmniSil-Runtime.

Detects available silicon backends (NVIDIA CUDA, AMD ROCm, AWS Trainium Neuron, or CPU)
and routes kernel execution to the optimal hardware-accelerated backend or high-performance
vectorized fallback.
"""

import logging
import os
import platform
from enum import Enum
from typing import Any, Optional

logger = logging.getLogger("OmniSil.Dispatch")


class SiliconBackend(Enum):
    NVIDIA_CUDA = "nvidia_cuda"
    AMD_ROCM = "amd_rocm"
    AWS_TRAINIUM = "aws_trainium"
    VECTORIZED_CPU = "vectorized_cpu"


class HardwareDispatcher:
    """
    Environment sensor and routing engine for unified multi-silicon dispatch.
    """
    _instance: Optional['HardwareDispatcher'] = None

    def __init__(self):
        self.backend: SiliconBackend = self._detect_backend()
        self.device_info: dict[str, Any] = self._get_device_info()

    @classmethod
    def get_instance(cls) -> 'HardwareDispatcher':
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _detect_backend(self) -> SiliconBackend:
        # Check explicit override via environment variable
        override = os.environ.get("OMNISIL_BACKEND", "").lower()
        if override == "cuda":
            return SiliconBackend.NVIDIA_CUDA
        elif override == "rocm":
            return SiliconBackend.AMD_ROCM
        elif override == "trainium":
            return SiliconBackend.AWS_TRAINIUM
        elif override == "cpu":
            return SiliconBackend.VECTORIZED_CPU

        # Auto-detection logic via PyTorch / environment indicators
        try:
            import torch
            if torch.cuda.is_available():
                # Check if ROCm or CUDA
                version_str = getattr(torch.version, "hip", None)
                if version_str is not None:
                    logger.info(f"Detected AMD ROCm/HIP silicon backend (HIP {version_str})")
                    return SiliconBackend.AMD_ROCM
                else:
                    logger.info("Detected NVIDIA CUDA silicon backend")
                    return SiliconBackend.NVIDIA_CUDA
        except ImportError:
            pass

        # Check AWS Trainium / Neuron XLA environment
        if "NEURON_RT_NUM_CORES" in os.environ or os.path.exists("/dev/neuron0"):
            logger.info("Detected AWS Trainium Neuron XLA silicon backend")
            return SiliconBackend.AWS_TRAINIUM

        logger.info("Defaulting to Vectorized CPU silicon backend for high-precision local execution")
        return SiliconBackend.VECTORIZED_CPU

    def _get_device_info(self) -> dict[str, Any]:
        info = {
            "os": platform.system(),
            "architecture": platform.machine(),
            "backend": self.backend.value,
        }
        try:
            import torch
            info["torch_version"] = torch.__version__
            if self.backend in (SiliconBackend.NVIDIA_CUDA, SiliconBackend.AMD_ROCM) and torch.cuda.is_available():
                info["device_name"] = torch.cuda.get_device_name(0)
                info["device_count"] = torch.cuda.device_count()
        except ImportError:
            info["torch_version"] = "Not Installed"
        return info

    def get_backend_name(self) -> str:
        return self.backend.value

    def is_gpu_available(self) -> bool:
        return self.backend in (SiliconBackend.NVIDIA_CUDA, SiliconBackend.AMD_ROCM, SiliconBackend.AWS_TRAINIUM)


def get_dispatcher() -> HardwareDispatcher:
    return HardwareDispatcher.get_instance()
