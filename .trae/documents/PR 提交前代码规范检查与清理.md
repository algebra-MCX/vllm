# PR 提交前最终检查与清理计划

## 1. 核心代码规范检查
- **`vllm/forward_context.py`**: 再次确认没有遗留任何未使用的 MoE 字段（`all_moe_layers`, `moe_layer_index`），确保只包含 KV Cache 相关的必要改动。
- **`vllm/attention/layer.py`**: 确认缩进、空格符合 PEP8 规范（特别是 `unified_kv_cache_update` 函数内部的逻辑块）。

## 2. 单元测试文件最终清理 (`test_unified_kv_cache_optimization_real.py`)
为了符合 vLLM 官方的严格测试规范，需要对测试文件进行微调：
- **移除 main 执行块**: vLLM 的测试文件通常不需要 `if __name__ == "__main__":` 块，因为它们是由 CI/CD 中的 `pytest` 自动调用的。保留这个块虽然方便本地调试，但在提交 PR 时最好移除，或者至少用 `sys.exit(pytest.main(...))` 这种标准方式（我们已经改好了）。**建议直接移除**，让文件更纯粹。
- **Docstring**: 确认文件开头的 Docstring 清晰描述了测试目的。
- **Imports**: 确保 `import` 顺序符合规范（标准库 -> 第三方库 -> 本地库）。

## 3. 执行计划
1.  **Read**: 读取上述两个核心源码文件，进行最后的人眼 Review。
2.  **Edit**: 如果发现任何格式问题或多余代码，立即修正。
3.  **Delete/Edit**: 将测试文件末尾的 `if __name__ == "__main__":` 块删除（这是最标准的做法）。

确认无误后，您就可以提交了。