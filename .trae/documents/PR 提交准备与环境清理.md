# PR 提交准备计划

## 1. 文件清理
- **保留**: `tests/kernels/attention/test_unified_kv_cache_optimization_real.py`。这是本次 PR 的核心单元测试，证明了新逻辑的正确性，必须随代码一起提交。
- **删除**: `tests/kernels/attention/benchmark_kv_cache_compile.py`。这是一个辅助证明性能的脚本，虽然很有用，但通常不直接包含在核心功能 PR 中（除非放到 `benchmarks/` 目录）。为了保持 PR 简洁，我们将其删除，其结果已记录在 PR 描述中。
- **删除**: `tests/test_real_model_inference.py`。这是本地调试用的临时脚本，包含硬编码路径，严禁提交。

## 2. 最终检查
- 确认 `vllm/forward_context.py` 和 `vllm/attention/layer.py` 的修改已保存且正确。
- 确认保留的单元测试文件内容无误。

## 3. 交付
- 执行文件删除操作。
- 任务结束，您可以直接使用我提供的模板提交 PR。