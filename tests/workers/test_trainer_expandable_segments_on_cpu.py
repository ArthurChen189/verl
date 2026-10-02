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

import pytest

from verl.workers.engine_workers import _trainer_wants_expandable_segments


@pytest.mark.parametrize("device_name", ["cuda", "npu"])
def test_trainer_only_worker_enables_expandable_segments(monkeypatch, device_name):
    monkeypatch.delenv("VERL_TRAINER_EXPANDABLE_SEGMENTS", raising=False)
    assert _trainer_wants_expandable_segments(is_rollout=False, device_name=device_name)


def test_worker_hosting_a_rollout_engine_is_left_alone(monkeypatch):
    monkeypatch.delenv("VERL_TRAINER_EXPANDABLE_SEGMENTS", raising=False)
    assert not _trainer_wants_expandable_segments(is_rollout=True, device_name="cuda")


def test_cpu_is_left_alone(monkeypatch):
    monkeypatch.delenv("VERL_TRAINER_EXPANDABLE_SEGMENTS", raising=False)
    assert not _trainer_wants_expandable_segments(is_rollout=False, device_name="cpu")


def test_opt_out(monkeypatch):
    monkeypatch.setenv("VERL_TRAINER_EXPANDABLE_SEGMENTS", "0")
    assert not _trainer_wants_expandable_segments(is_rollout=False, device_name="cuda")
