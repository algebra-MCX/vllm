"""Single-GPU adapter regression and graph replay; external transport is stubbed."""

import argparse
import importlib.util
import json
import subprocess
import types
from datetime import datetime, timezone
from pathlib import Path

import pytest
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--source", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
source = args.source.resolve()
head = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
assert head == "0e60e1ca696f011fb84579ffd60b31ee8c571030", head
assert not subprocess.check_output(["git", "-C", str(source), "status", "--porcelain"])
assert torch.cuda.is_available()
assert "H200" in torch.cuda.get_device_name(0)
torch.cuda.set_device(0)
test_path = source / "tests/kernels/moe/test_deepep_ll_prepare_finalize.py"
spec = importlib.util.spec_from_file_location("sentinel_regression", test_path)
regression = importlib.util.module_from_spec(spec)
spec.loader.exec_module(regression)
cases = [
    ("mixed-padding", [[-1, 6], [0, 7]], [[-1, 5], [0, 7]], True),
    ("all-padding", [[-1, -1], [-1, -1]], [[-1, -1], [-1, -1]], True),
    ("linear", [[0, -1], [-1, 6]], [[0, -1], [-1, 6]], False),
]
results = {"base": [], "fixed": [], "graph_replay": []}
with pytest.MonkeyPatch.context() as monkeypatch:
    fixture = regression.deepep_ll_module.__wrapped__(monkeypatch)
    fixed = next(fixture)
    try:
        path = "vllm/model_executor/layers/fused_moe/prepare_finalize/deepep_ll.py"
        baseline = types.ModuleType("sentinel_baseline")
        original = subprocess.check_output(["git", "-C", str(source), "show", f"HEAD^:{path}"])
        exec(compile(original, "baseline_deepep_ll.py", "exec"), baseline.__dict__)
        with torch.device("cuda"):
            for phase, module in [("base", baseline), ("fixed", fixed)]:
                for name, ids, expected, mapped in cases:
                    passed = True
                    try:
                        regression.test_deepep_ll_routing_preserves_padding(module, ids, expected, mapped)
                    except AssertionError:
                        passed = False
                    results[phase].append({"case": name, "passed": passed})
            assert [x["passed"] for x in results["base"]] == [False, False, True]
            assert all(x["passed"] for x in results["fixed"])

            from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
            from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import TopKWeightAndReduceDelegate

            class GPUTransport(regression._RecordingTransport):
                def low_latency_dispatch(self, inputs, ids, *a, **kw):
                    assert inputs.is_cuda and ids.is_cuda
                    return super().low_latency_dispatch(inputs, ids, *a, **kw)

                def low_latency_combine(self, inputs, ids, *a, **kw):
                    assert inputs.is_cuda and ids.is_cuda
                    return super().low_latency_combine(inputs, ids, *a, **kw)

            for mapped in (True, False):
                transport = GPUTransport()
                adapter = fixed.DeepEPLLPrepareAndFinalize(
                    buffer=transport, max_tokens_per_rank=2, num_dispatchers=4,
                    global_to_physical=torch.tensor([0, 2, 4, 6, 1, 3, 5, 7]) if mapped else None,
                )
                routes = torch.tensor([[0, 1], [6, 7]], dtype=torch.int64)
                inputs = torch.zeros((2, 2048), dtype=torch.bfloat16)
                weights = torch.ones((2, 2), dtype=torch.float32)
                output = torch.empty_like(inputs)
                expert_output = torch.empty((2, 8, 2048), dtype=inputs.dtype)
                quant = FusedMoEQuantConfig.make()
                reducer = TopKWeightAndReduceDelegate()

                def forward():
                    previous = torch.cuda.get_sync_debug_mode()
                    torch.cuda.set_sync_debug_mode("error")
                    try:
                        adapter.prepare_async(inputs, weights, routes, 8, None, False, quant)
                        adapter.finalize_async(output, expert_output, weights, routes, False, reducer)
                    finally:
                        torch.cuda.set_sync_debug_mode(previous)

                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    for _ in range(3):
                        forward()
                torch.cuda.current_stream().wait_stream(stream)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    forward()
                replay_cases = cases[:2] + [("valid-only", [[0, 1], [6, 7]], [[0, 2], [5, 7]], True)]
                for name, ids, expected, _ in replay_cases:
                    routes.copy_(torch.tensor(ids, dtype=routes.dtype))
                    graph.replay()
                    reference = torch.tensor(expected if mapped else ids, dtype=routes.dtype)
                    torch.testing.assert_close(transport.dispatch_ids, reference)
                    torch.testing.assert_close(transport.combine_ids, reference)
                    results["graph_replay"].append({"case": name, "mapped": mapped, "dispatch_pass": True, "combine_pass": True})
    finally:
        fixture.close()
torch.cuda.synchronize()
report = {
    "utc": datetime.now(timezone.utc).isoformat(), "source_sha": head,
    "base_sha": subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD^"], text=True).strip(),
    "gpu": torch.cuda.get_device_name(0), "gpu_count": 1,
    "torch": torch.__version__, "cuda": torch.version.cuda,
    "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
    "scope": "Real CUDA tensors and real DeepEP LL adapter; recording external transport. No native transport or model validation.",
    "sync_debug": "PyTorch CUDA sync-debug error mode passed in adapter dispatch/combine calls; not a full VLLM_GPU_SYNC_CHECK run.",
    "results": results, "success": True,
}
args.output.write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2), flush=True)
