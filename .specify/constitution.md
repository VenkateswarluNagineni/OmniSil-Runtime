# OmniSil Runtime Constitution (`github/spec-kit` Protocol)

## 1. High-Performance Numerical & Hardware Guardrails
1. **Multi-Silicon Dispatch Isolation**: Hardware dispatch (`omnisil/dispatch.py`) must cleanly separate device selection (`CUDA`, `Triton`, `CPU`) from kernel logic. Custom numerical kernels (`omnisil/kernels/`) must never hardcode device strings without fallback support.
2. **Numerical Accuracy & Tolerance Invariants**: All optimized tensor operations (Attention, MoE Gating, Quantized GEMM) MUST maintain strict numerical equivalence with reference FP32 implementations within `rtol=1e-3, atol=1e-3`. No kernel modification may introduce silent precision degradation or NaNs.
3. **Memory & Quantization Safety**: Quantization routines (`quant_gemm.py`) must explicitly bounds-check tensor shapes and scale factors before casting (`INT4`/`FP16`) to prevent buffer overflows or segmentation faults.
4. **Compiler Backend Determinism**: The TorchDynamo compilation backend (`dynamo_backend.py`) must produce deterministic FX graphs across execution cycles without mutating global PyTorch engine state.
5. **Continuous Benchmark & Kernel Verification**: Every kernel or dispatch update MUST pass `pytest -v tests/test_kernels.py` (`100% pass rate`) and execute `benchmarks/run_benchmarks.py` without performance regression before merge.
