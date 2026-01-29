# 完善 KV Cache 优化逻辑并采集性能对比数据 (最终修正版：严格对齐 PR-33184)

## 1. 源码修改
### `vllm/forward_context.py`
- **目标**: 扩展 `ForwardContext` 类，严格模仿 PR-33184 的 `moe_layer_index` 逻辑。
- **修改**:
    - 添加 `all_kv_cache_update_layers: list[str] | None = None` (对应 PR 中的 `all_moe_layers`)。
    - 添加 `kv_cache_update_index: int = 0` (对应 PR 中的 `moe_layer_index`)。
    - 在 `create_forward_context` 中：
        - 遍历 `Attention` 层，将需要独立编译的层名存入 `all_kv_cache_update_layers`（**正序**，不需要倒序，因为是 index 访问）。
        - 初始化 `kv_cache_update_index = 0`。

### `vllm/attention/layer.py`
- **目标**: 修改 `Attention` 层以使用 `ForwardContext` 中的动态名称。
- **修改**:
    - 在 `Attention.forward` 方法中：
        - 检查 `is_forward_context_available()` 且 `all_kv_cache_update_layers` 存在。
        - 传参改为 `layer_name="from_forward_context"`。
    - 在 `unified_kv_cache_update` 函数中：
        - 检查 `layer_name == "from_forward_context"`。
        - **关键逻辑 (严格对齐 PR-33184)**:
            - 获取 `context.kv_cache_update_index`。
            - 断言 `index < len(context.all_kv_cache_update_layers)`。
            - 从 `context.all_kv_cache_update_layers[index]` 获取真实层名。
            - 执行 `context.kv_cache_update_index += 1` (自增)。
            - 使用真实层名获取 Layer 对象。

## 2. 验证与测试
### 单元测试 (`tests/kernels/attention/test_unified_kv_cache_optimization_real.py`)
- **目标**: 验证 Index 自增逻辑。
- **逻辑**:
    - 构造包含 2 个层的 `ForwardContext`。
    - 模拟两次 Forward 调用。
    - 验证 `index` 从 0 变 1，从 1 变 2。
    - 验证每次取到的层名正确。

### 性能基准测试 (`tests/kernels/attention/benchmark_kv_cache_compile.py`)
- **目标**: 量化加速比。
- **逻辑**: 保持不变（32 层 Baseline vs Optimized），验证编译时间大幅下降。

## 3. 执行计划
1.  **应用代码修改**: 更新 `vllm/forward_context.py` 和 `vllm/attention/layer.py`。
2.  **创建测试脚本**: 编写单元测试和 Benchmark 脚本。
3.  **运行验证**: 运行测试。
4.  **提交交付**: 输出结果。