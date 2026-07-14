# Technical Restructuring & Implementation Plan (`plan.md`)

## 1. Numerical & Dispatch Restructuring Objectives
1. **Separation of Compiler vs Execution**: Ensure `omnisil/compiler/dynamo_backend.py` strictly handles graph interception (`torch.compile` backend logic), while `omnisil/runtime/engine.py` orchestrates execution without circular imports.
2. **Device-Agnostic Dispatch Boundaries**: Verify `omnisil/dispatch.py` isolates device detection (`CUDA`, `Triton`, `CPU`) cleanly so custom kernels (`attention.py`, `moe_gating.py`, `quant_gemm.py`) can be tested on CPU fallback environments without hardware crashes.

## 2. Code Restructuring Blueprint
- `omnisil/kernels/attention.py`: Ensure clean tensor shape verification and memory tiling boundaries.
- `omnisil/kernels/moe_gating.py`: Verify router logits are checked for `NaN` and `Inf` before applying `topk`.
- `omnisil/kernels/quant_gemm.py`: Enforce explicit scale verification before `INT4` dequantization.
- `tests/test_kernels.py`: Execute unit tests comparing custom kernels against baseline PyTorch references.

## 3. Continuous Verification Checkpoints
- **SAST**: `ruff check . --select E,F,S,I,UP,B,C90` -> `100% clean`.
- **Numerical Verification**: `pytest -v tests/test_kernels.py` -> `100% pass rate`.
- **Benchmark Run**: `python benchmarks/run_benchmarks.py` -> High throughput verified.
