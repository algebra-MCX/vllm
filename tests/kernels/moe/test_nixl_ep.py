# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

pytestmark = pytest.mark.nixl_ep


@pytest.mark.parametrize(
    "tokens,sp,dispatchers,expected", [(128, 1, 2, 128), (64, 1, 8, 64), (5, 2, 4, 3)]
)
def test_runtime_capacity_uses_dp_and_sp_execution(tokens, sp, dispatchers, expected):
    from unittest.mock import Mock

    import nixl_ep

    from vllm.forward_context import (
        DPMetadata,
        ForwardContext,
        override_forward_context,
    )
    from vllm.model_executor.layers.fused_moe.prepare_finalize.nixl_ep import (
        NixlEPPrepareAndFinalize,
    )

    prepare = NixlEPPrepareAndFinalize(Mock(spec=nixl_ep.Buffer), 128, dispatchers, 8)
    metadata = DPMetadata(torch.tensor([tokens, tokens], dtype=torch.int32))
    ctx = ForwardContext({}, {}, {}, dp_metadata=metadata)
    assert prepare._get_runtime_dispatch_capacity() == 128
    with override_forward_context(ctx), metadata.sp_local_sizes(sp, use_ep=True):
        assert prepare._get_runtime_dispatch_capacity() == expected
    assert prepare.max_num_tokens_per_rank() == 128


@pytest.mark.parametrize("k", [2, 3, 4, 8])
@pytest.mark.parametrize("combine", [False, True])
def test_nixl_dbo_communication_runs_on_comm_stream(combine, k):
    from unittest.mock import Mock

    import nixl_ep

    from tests.v1.worker.test_gpu_ubatch_slicing import (
        _make_dbo_config,
        _make_execution_runner,
        _make_model_inputs,
        _make_ubatch_state,
    )
    from vllm.config import ParallelConfig, VllmConfig
    from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
    from vllm.model_executor.layers.fused_moe.prepare_finalize.nixl_ep import (
        NixlEPPrepareAndFinalize,
    )
    from vllm.v1.worker.ubatch_utils import UBatchSlice

    config = VllmConfig(
        model_config=_make_dbo_config().model_config,
        parallel_config=ParallelConfig(ubatch_size=k, all2all_backend="nixl_ep"),
    )
    runner = _make_execution_runner(config)
    buffer = Mock(spec=nixl_ep.Buffer)
    banks = [torch.empty(1, 4, 2048, device="cuda") for _ in range(2)]
    calls = 0

    def dispatch(x, *args, **kwargs):
        assert torch.cuda.current_stream() == runner.comm_stream
        nonlocal calls
        data = banks[calls % 2]
        calls += 1
        data.copy_(x.unsqueeze(0))
        return data, torch.tensor([len(x)], device=x.device), (), None, None

    buffer.dispatch.side_effect = dispatch

    def combine_output(x, *args, out, **kwargs):
        assert torch.cuda.current_stream() == runner.comm_stream
        out.copy_(x.squeeze(0))
        return out, None, None

    buffer.combine.side_effect = combine_output
    prepare = NixlEPPrepareAndFinalize(buffer, 128, 1, 1, num_ubatches=k)

    def model(input_ids, **kwargs):
        x = input_ids[:, None].expand(-1, 2048).float()
        ids = torch.zeros(len(x), 1, dtype=torch.int64, device=x.device)
        receiver = prepare.prepare_async(
            x, torch.ones_like(ids), ids, 1, None, False, FusedMoEQuantConfig.make()
        )
        assert callable(receiver)
        received = receiver()[0]
        if not combine:
            return received.squeeze(0)
        from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
            TopKWeightAndReduceDelegate,
        )

        output = torch.empty_like(x)
        finish = prepare.finalize_async(
            output,
            received,
            torch.ones_like(ids),
            ids,
            False,
            TopKWeightAndReduceDelegate(),
        )
        assert callable(finish)
        finish()
        return output

    state = _make_ubatch_state(
        config,
        [UBatchSlice(slice(i, i + 4), slice(i, i + 4)) for i in range(0, 4 * k, 4)],
    )
    inputs = _make_model_inputs(4 * k, torch.device("cuda", 0))
    output = runner.run(model, inputs, state)
    torch.testing.assert_close(
        output, inputs["input_ids"][:, None].expand_as(output).float()
    )
