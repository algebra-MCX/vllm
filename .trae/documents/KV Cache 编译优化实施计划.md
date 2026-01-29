# 完善 KV Cache 优化逻辑并采集性能对比数据

## 1. 源码修改
### `vllm/forward_context.py`
- **目标**: 扩展 `ForwardContext` 类以支持 KV Cache 层的动态名称解析。
- **修改**:
    - 在 `ForwardContext` 类中添加 `remaining_kv_cache_update_layers: list[str] | None` 字段。
    - 在 `create_forward_context` 函数中，遍历 `Attention` 层。
    - 识别 `forward_includes_kv_cache_update` 为 `False` 的层（即需要独立编译 KV update 的层）。
    - 将这些层的名称压入 `remaining_kv_cache_update_layers` 列表（倒序，以便后续 pop）。

### `vllm/attention/layer.py`
- **目标**: 修改 `Attention` 层以使用 `ForwardContext` 中的动态名称。
- **修改**:
    - 在 `Attention.forward` 方法中：
        - 检查 `is_forward_context_available()` 且 `remaining_kv_cache_update_layers` 存在。
        - 如果满足，向 `unified_kv_cache_update` 传入常量字符串 `"from_forward_context"` 而非 `self.layer_name`。
    - 在 `unified_kv_cache_update` 函数中：
        - 检查传入的 `layer_name` 是否为 `"from_forward_context"`。
        - 如果是，从 `get_forward_context().remaining_kv_cache_update_layers` 中 `pop()` 出真实的层名称。
        - 使用该真实名称检索对应的层对象。

## 2. 验证与测试
### 单元测试 (`tests/kernels/attention/test_unified_kv_cache_optimization_real.py`)
- **目标**: 验证优化逻辑的正确性（不 Mock，使用真实 GPU 环境）。
- **逻辑**:
    - 使用 `set_current_vllm_config` 设置真实的 vLLM 配置。
    - 实例化两个 `Attention` 层。
    - 手动构造 `ForwardContext` 并填充 `remaining_kv_cache_update_layers`。
    - 运行 `Attention.forward`。
    - **断言**: 验证 `ForwardContext` 中的栈是否被正确消耗（变为空），证明逻辑生效。

### 性能基准测试 (`tests/kernels/attention/benchmark_kv_cache_compile.py`)
- **目标**: 量化 `torch.compile` 的加速效果。
- **逻辑**:
    - 模拟 32 层 Attention 结构。
    - **Baseline**: 使用唯一的 `layer_name` 进行编译，记录总耗时。
    - **Optimized**: 使用 `"from_forward_context"` 常量名进行编译，记录总耗时。
    - 每次测试前调用 `torch._dynamo.reset()` 清除缓存。
    - 输出对比表格：编译时间、加速比、Graph Capture 次数（推断）。

## 3. 执行计划
1.  **应用代码修改**: 依次更新 `vllm/forward_context.py` 和 `vllm/attention/layer.py`。
2.  **创建测试脚本**: 写入上述两个测试文件。
3.  **运行验证**:
    - 先运行单元测试确保逻辑无误。
    - 后运行 Benchmark 脚本采集最终性能数据。
4.  **提交交付**: 确认所有文件就绪，输出最终结果。