# Unified Attention 图编译优化实施计划

## 1. 任务核心目标
解决 `unified_attention` 系列操作因传入不同的 `layer_name` 字符串导致 `torch.compile` (Dynamo) 无法复用计算图，从而引发冷启动编译时间过长的问题。

## 2. 技术参考方案 (基于 PR-32805 & PR-33184)
**核心思想**: **“图构建时传占位符，运行时动态查索引”**。
借鉴 PR-33184 的优化思路（List + Index），而非 PR-32805 的早期思路（Stack + Pop）。

*   **静态部分**: 在 `ForwardContext` 中维护一个**只读**的层名列表 `attention_layers`。
*   **动态部分**: 在 `ForwardContext` 中维护一个**可变**的索引计数器 `attention_step`。
*   **图接口**: 向 `torch.ops` 传递固定字符串 `"from_forward_context"`。

## 3. 详细实施步骤

### 阶段一：基础设施改造 (`vllm/forward_context.py`)
1.  **修改 `ForwardContext` 类**:
    *   新增 `attention_layers: list[str] | None` (存储层名序列)。
    *   新增 `attention_step: int = 0` (运行时计数器)。
2.  **修改 `create_forward_context` 函数**:
    *   引入 `Attention` 和 `MLAAttention` 类（注意在函数内 import 避免循环引用）。
    *   遍历 `no_compile_layers`，筛选出所有 Attention 层。
    *   **注意**: 列表保持**正序**（因为我们使用索引 `0 -> N` 递增访问）。
    *   初始化 `ForwardContext` 时传入该列表，并将 `attention_step` 置为 0。

### 阶段二：Attention 层逻辑适配 (`vllm/attention/layer.py`)
1.  **新增辅助函数 `get_layer_name_context(layer_name: str) -> str`**:
    *   **逻辑**:
        *   若 `layer_name == "from_forward_context"`:
            *   获取 `context.attention_layers`。
            *   使用 `context.attention_step` 作为索引取出真实层名。
            *   **关键**: 执行 `context.attention_step += 1`。
            *   返回真实层名。
        *   否则: 直接返回原 `layer_name` (兼容非 Context 模式)。
2.  **修改 Op 实现函数**:
    *   涉及函数:
        *   `unified_attention`
        *   `unified_attention_with_output`
        *   `unified_mla_attention`
        *   `unified_mla_attention_with_output`
        *   `maybe_calc_kv_scales` (建议一并修改以保持一致性)
    *   **动作**: 在函数入口处调用 `get_layer_name_context` 解析层名。
3.  **修改 `Attention.forward` 和 `MLAAttention.forward`**:
    *   新增内部 helper `encode_layer_name()`:
        *   若 Context 可用且 `attention_layers` 存在，返回 `"from_forward_context"`。
        *   否则返回 `self.layer_name`。
    *   **动作**: 将所有调用 `torch.ops.vllm.unified_*` 的地方，将 `self.layer_name` 替换为 `encode_layer_name()`。

### 阶段三：验证与测试
1.  **单元测试验证**:
    *   运行现有的 Attention 相关测试，确保在没有 Context 的情况下（Fallback 逻辑）依然正常工作。
    *   重点关注: `tests/kernels/test_attention.py` (如果有) 或相关集成测试。
2.  **图复用验证 (可选但推荐)**:
    *   通过 `torch.compile` 的日志或简单的脚本，确认多个 Attention 层现在是否生成了相同的 Graph Hash。

## 4. 关键差异点 (对比 MOE PR)
*   **数据结构**: 我们使用 **List + Index** (PR-33184 理念)，而不是 **Stack + Pop** (PR-32805 实现)。
*   **顺序**: 我们的列表是**正序**构建，索引从 0 开始递增；MOE 的 Stack 方案是逆序构建，从尾部 Pop。
