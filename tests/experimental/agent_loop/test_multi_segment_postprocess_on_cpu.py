# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""An agent loop that returns a list of outputs (one per segment) becomes several rows, joined to its sample."""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

import verl.experimental.agent_loop.agent_loop as agent_loop_module
from verl import DataProto
from verl.experimental.agent_loop.agent_loop import (
    AgentLoopMetrics,
    AgentLoopOutput,
    AgentLoopWorker,
    _flatten_multi_outputs,
    _globalize_source_index,
    _InternalAgentLoopOutput,
)
from verl.trainer.ppo import multi_segment as ms
from verl.utils.dataset.rl_dataset import RLHFDataset

P, R = 4, 4


def _internal(prompt: list[int], resp: list[int], reward: float, **extra) -> _InternalAgentLoopOutput:
    def pad(x, n, left=False):
        return ([0] * (n - len(x)) + x) if left else (x + [0] * (n - len(x)))

    p, r = pad(prompt, P, left=True), pad(resp, R)
    attn = pad([1] * len(prompt), P, left=True) + pad([1] * len(resp), R)
    t = lambda x: torch.tensor([x], dtype=torch.long)  # noqa: E731
    return _InternalAgentLoopOutput(
        prompt_ids=t(p),
        response_ids=t(r),
        response_mask=t(pad([1] * len(resp), R)),
        attention_mask=t(attn),
        input_ids=t(p + r),
        position_ids=t(list(range(P + R))),
        response_logprobs=torch.tensor([pad([-0.5] * len(resp), R)], dtype=torch.float32),
        reward_score=reward,
        num_turns=2,
        metrics=AgentLoopMetrics(),
        extra_fields=dict(extra),
    )


class _Worker:
    reward_loop_worker_handles = None
    distillation_enabled = False


def test_flatten_records_source_index():
    a0, a1, b, c0, c1, c2 = (object() for _ in range(6))
    flat, src = _flatten_multi_outputs([[a0, a1], b, [c0, c1, c2]])
    assert flat == [a0, a1, b, c0, c1, c2] and src == [0, 0, 1, 2, 2, 2]
    single = [a0, b]
    assert _flatten_multi_outputs(single) == (single, None)
    with pytest.raises(AssertionError):
        _flatten_multi_outputs([[a0], []])


def test_postprocess_joins_input_columns_by_source_index():
    outs = [
        _internal([1, 2], [5], 0.0, segment_index=0, segment_kind="main"),
        _internal([3], [6, 7], 0.0, segment_index=1, segment_kind="summary"),
        _internal([4, 4], [8], 1.0, segment_index=2, segment_kind="main"),
        _internal([9], [9, 9], 0.0, segment_index=0, segment_kind="main"),
    ]
    data = AgentLoopWorker._postprocess(
        _Worker(),
        inputs=outs,
        input_non_tensor_batch={
            "index": np.array([10, 11], dtype=object),
            "agent_name": np.array(["x", "x"], dtype=object),
        },
        source_index=[0, 0, 0, 1],
    )
    assert len(data) == 4
    assert data.non_tensor_batch["index"].tolist() == [10, 10, 10, 11]
    assert data.non_tensor_batch[ms.SOURCE_INDEX_KEY].tolist() == [0, 0, 0, 1]
    assert data.non_tensor_batch["segment_kind"].tolist() == ["main", "summary", "main", "main"]
    # reward sits on the last response token of the row that carries it
    assert data.batch["rm_scores"].sum(-1).tolist() == [0.0, 0.0, 1.0, 0.0]
    assert data.batch["rollout_log_probs"].shape == (4, R)


def test_postprocess_single_outputs_unchanged():
    outs = [_internal([1], [5], 1.0), _internal([2], [6], 0.0)]
    data = AgentLoopWorker._postprocess(
        _Worker(), inputs=outs, input_non_tensor_batch={"index": np.array([0, 1], dtype=object)}
    )
    assert ms.SOURCE_INDEX_KEY not in data.non_tensor_batch and data.non_tensor_batch["index"].tolist() == [0, 1]


def test_globalize_source_index_offsets_worker_chunks():
    def chunk(n_rows, src=None):
        nt = {"x": np.zeros(n_rows, dtype=object)}
        if src is not None:
            nt[ms.SOURCE_INDEX_KEY] = np.array(src, dtype=np.int64)
        return DataProto.from_dict(tensors={"t": torch.zeros(n_rows, 1)}, non_tensors=nt)

    outs = [chunk(3, [0, 0, 1]), chunk(2), chunk(4, [0, 1, 1, 1])]  # chunks of 2, 2, 2 input rows
    _globalize_source_index([2, 2, 2], outs)
    merged = DataProto.concat(outs)
    assert merged.non_tensor_batch[ms.SOURCE_INDEX_KEY].tolist() == [0, 0, 1, 2, 3, 4, 5, 5, 5]
    plain = [chunk(2), chunk(2)]
    _globalize_source_index([2, 2], plain)
    assert all(ms.SOURCE_INDEX_KEY not in o.non_tensor_batch for o in plain)


@pytest.mark.asyncio
@pytest.mark.parametrize("validate", [False, True])
async def test_run_agent_loop_list_output(monkeypatch, validate):
    segs = [
        AgentLoopOutput(
            prompt_ids=[1],
            response_ids=[2],
            response_mask=[1],
            metrics=AgentLoopMetrics(),
            extra_fields={"segment_index": i},
        )
        for i in range(3)
    ]

    class _ListLoop:
        async def run(self, sampling_params: dict[str, Any], **kwargs):
            return list(segs)

    monkeypatch.setattr(agent_loop_module.hydra.utils, "instantiate", lambda *, config, **kw: _ListLoop())
    monkeypatch.setattr(agent_loop_module, "rollout_trace_attr", lambda **kwargs: nullcontext())
    monkeypatch.setitem(agent_loop_module._agent_loop_registry, "list_loop_test", {"_target_": "unused"})
    worker = object.__new__(AgentLoopWorker)
    worker.config = OmegaConf.create({"data": {}})
    worker.llm_client = object()
    worker.tokenizer = None
    worker.processor = None
    worker.hf_model_type = None
    worker.dataset_cls = RLHFDataset
    worker.tools = []
    seen = []

    async def record(output, validate, **kwargs):
        seen.append((output.extra_fields["segment_index"], validate))
        return output

    worker._agent_loop_postprocess = record
    result = await worker._run_agent_loop(
        {}, {"step": 0, "sample_index": 0, "rollout_n": 0, "validate": validate}, agent_name="list_loop_test"
    )
    if validate:  # validation stays 1:1 with its inputs: the final segment only
        assert result is segs[-1] and seen == [(2, True)]
    else:
        assert result == segs and seen == [(0, False), (1, False), (2, False)]
