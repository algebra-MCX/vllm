# DeepEP LL sentinel mapping: single-H200 core validation

Source: [PR #64340](https://github.com/vllm-project/vllm/pull/64340), commit `0e60e1ca696f011fb84579ffd60b31ee8c571030`.
Baseline: `0fddee1f5869465a6235d2076f82f1ea70318064`. Run: 2026-10-28T18:58:51.090732+00:00.

This is a real CUDA tensor / real vLLM adapter check with a recording external transport. It does not exercise native DeepEP communication, native transport CUDA Graph replay, or a model.

| Check | Baseline | Fixed |
| --- | --- | --- |
| Mixed padding, all padding, linear control | 2 failed, 1 passed | 3 passed |
| Adapter CUDA Graph replay: 3 route patterns × mapped/no-map | Not run | 6 configurations passed, both dispatch and combine (12 checks) |
| PyTorch CUDA sync-debug error mode during adapter calls | Not run | Passed |

Graph inputs change between mixed-padding, all-padding and valid-only IDs on the same captured buffers. The replay checks therefore cover changing sentinel values, rather than replaying one constant input.

Hardware/software: 1 × NVIDIA H200, PyTorch 2.13.0+cu129, CUDA 12.9.
PyTorch peak allocated tensor memory: 321024 bytes. This excludes CUDA context and non-PyTorch library memory.

- [Machine-readable results](result.json)
- [Exact runner](run_gpu_core.py)

Runner SHA-256: `88012dd685fb51370281dd34e020d8581982982ed4697b8d534f7677bb7e56c3`.
Results SHA-256: `c5ab9367848956fa43258efe7669088f2d1a65ab32f1a7b29fdce708e3953a4b`.

## Reproduce

Use a clean checkout of the source commit above and the exact runner copied outside that checkout, with an existing PyTorch 2.13.0+cu129 environment and a visible H200:

```bash
PYTHONPATH="$PWD" /path/to/runtime/bin/python /path/to/run_gpu_core.py \
  --source "$PWD" --output /tmp/pr64340-h200-result.json
```

The runner imports the contributed regression and invokes its real fixture/adapter on CUDA tensors. The baseline class is loaded from the parent commit's unmodified source; the fixed implementation is imported from the clean source checkout. No source methods are patched. Only the external transport is replaced.

PyTorch's synchronization debug feature is experimental and does not detect every possible synchronizing operation. This is not a full `VLLM_GPU_SYNC_CHECK=error` model run. No serving correctness or performance claim is made.
