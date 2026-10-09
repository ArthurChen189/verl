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
"""Multi-segment episodes (verl/trainer/ppo/multi_segment.py) on a synthetic 4 tasks x 8 episodes batch."""

from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from verl import DataProto
from verl.trainer.config.algorithm import RolloutCorrectionConfig
from verl.trainer.ppo import multi_segment as ms
from verl.trainer.ppo.core_algos import (
    AdvantageEstimator,
    agg_loss,
    compute_policy_loss_bypass_mode,
    compute_policy_loss_vanilla,
)
from verl.trainer.ppo.ray_trainer import RayPPOTrainer, compute_advantage
from verl.trainer.ppo.rollout_corr_helper import (
    apply_bypass_mode,
    compute_offpolicy_metrics,
    compute_rollout_correction_and_rejection_mask,
)
from verl.workers.config import ActorConfig, PolicyLossConfig

P, R = 6, 8  # padded prompt / response length
EOS, PAD = 2, 0
N = 8  # rollout.n
# segments per episode for 4 tasks x 8 episodes; reward per episode (on its final segment)
SEGMENTS = [
    [1, 1, 2, 1, 3, 1, 1, 1],
    [2, 2, 2, 2, 2, 2, 2, 2],
    [4, 1, 1, 1, 1, 1, 1, 1],
    [1, 3, 1, 2, 1, 1, 4, 1],
]
REWARDS = [
    [0, 1, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],  # all-zero group: no gradient
    [1, 0, 0, 0, 0, 0, 0, 0],  # the 4-segment episode is the only success
    [1, 0, 1, 1, 0, 1, 0, 1],
]


def _row(n_prompt: int, n_resp: int, mask_pattern: list[int] | None, reward: float, seed: int) -> dict:
    g = torch.Generator().manual_seed(seed)
    prompt = torch.full((P,), PAD, dtype=torch.long)
    prompt[P - n_prompt :] = torch.randint(10, 100, (n_prompt,), generator=g)
    resp = torch.full((R,), PAD, dtype=torch.long)
    resp[:n_resp] = torch.randint(10, 100, (n_resp,), generator=g)
    attn = torch.zeros(P + R, dtype=torch.long)
    attn[P - n_prompt : P + n_resp] = 1
    rmask = torch.zeros(R, dtype=torch.long)
    rmask[:n_resp] = torch.tensor(mask_pattern if mask_pattern is not None else [1] * n_resp)
    scores = torch.zeros(R)
    scores[n_resp - 1] = reward
    pos = torch.clamp(torch.cumsum(attn, 0) - 1, min=0)
    return {
        "prompts": prompt,
        "responses": resp,
        "input_ids": torch.cat([prompt, resp]),
        "attention_mask": attn,
        "position_ids": pos,
        "response_mask": rmask,
        "rm_scores": scores,
        "token_level_scores": scores.clone(),
        "token_level_rewards": scores.clone(),
        "rollout_log_probs": -torch.rand(R, generator=g) * rmask,
    }


def _make_batch(segments=SEGMENTS, rewards=REWARDS, flags=None, kinds=True):
    """Repeated input batch (one row per episode) + generation output (one row per segment), as the trainer sees them.

    ``flags``: {(task, episode): {"exclude_from_loss": .., "exclude_from_baseline": .., "episode_overlong": ..}}
    """
    flags = flags or {}
    n_tasks = len(segments)
    uids = [f"task{t}" for t in range(n_tasks)]
    batch_in = DataProto.from_dict(
        tensors={"dummy": torch.zeros(n_tasks * N, 1)},
        non_tensors={
            "uid": np.array([u for u in uids for _ in range(N)], dtype=object),
            "extra_info": np.array([{"task": t, "episode": e} for t in range(n_tasks) for e in range(N)], dtype=object),
        },
    )
    rows, src, nt = (
        [],
        [],
        {
            k: []
            for k in (
                "segment_index",
                "segment_kind",
                "is_final_segment",
                "episode_overlong",
                "exclude_from_loss",
                "exclude_from_baseline",
            )
        },
    )
    for t in range(n_tasks):
        for e in range(N):
            n_seg = segments[t][e]
            f = flags.get((t, e), {})
            for s in range(n_seg):
                final = s == n_seg - 1
                n_resp = 2 + (s + e + t) % (R - 2)
                rows.append(
                    _row(2 + s % 3, n_resp, None, float(rewards[t][e]) if final else 0.0, seed=t * 100 + e * 10 + s)
                )
                src.append(t * N + e)
                nt["segment_index"].append(s)
                nt["segment_kind"].append(("summary" if s % 2 else "main") if kinds and not final else "main")
                nt["is_final_segment"].append(final)
                nt["episode_overlong"].append(f.get("episode_overlong", False))
                nt["exclude_from_loss"].append(f.get("exclude_from_loss", False))
                nt["exclude_from_baseline"].append(f.get("exclude_from_baseline", False))
    gen = DataProto.from_dict(
        tensors={k: torch.stack([r[k] for r in rows]) for k in rows[0]},
        non_tensors={
            **{k: np.array(v, dtype=object) for k, v in nt.items()},
            ms.SOURCE_INDEX_KEY: np.array(src, dtype=np.int64),
        },
    )
    return batch_in, gen


def _aligned(**kw) -> DataProto:
    batch_in, gen = _make_batch(**kw)
    return ms.align_to_outputs(batch_in, gen)


def _episode_rows(batch: DataProto) -> dict[str, list[int]]:
    out: dict[str, list[int]] = {}
    for i, ep in enumerate(batch.non_tensor_batch[ms.EPISODE_KEY]):
        out.setdefault(ep, []).append(i)
    return out


# ---------------------------------------------------------------- 1. every segment is its own row, joined to its sample
def test_rows_equal_segments_and_join_to_episode():
    batch = _aligned()
    assert len(batch) == sum(map(sum, SEGMENTS))
    eps = _episode_rows(batch)
    assert len(eps) == 4 * N
    for rows in eps.values():
        uid = {batch.non_tensor_batch["uid"][i] for i in rows}
        info = {json.dumps(batch.non_tensor_batch["extra_info"][i], sort_keys=True) for i in rows}
        assert len(uid) == 1 and len(info) == 1  # input columns copied onto every segment of the episode
        t, e = batch.non_tensor_batch["extra_info"][rows[0]].values()
        assert len(rows) == SEGMENTS[t][e]
        assert [batch.non_tensor_batch["segment_index"][i] for i in rows] == list(range(len(rows)))
    # segments do NOT get their own uid: a group is still the N episodes of a task
    assert len(set(batch.non_tensor_batch["uid"])) == 4


def test_align_without_source_index_is_plain_union():
    batch_in, gen = _make_batch(segments=[[1] * N] * 4)
    gen.non_tensor_batch.pop(ms.SOURCE_INDEX_KEY)
    out = ms.align_to_outputs(batch_in, gen)
    assert len(out) == 4 * N and not ms.is_multi_segment(out)


def test_align_rejects_episode_without_rows():
    batch_in, gen = _make_batch()
    keep = np.flatnonzero(gen.non_tensor_batch[ms.SOURCE_INDEX_KEY] != 5)
    with pytest.raises(AssertionError, match="without any output row"):
        ms.align_to_outputs(batch_in, gen.select_idxs(keep))


# ---------------------------------------------------------------- 2. one advantage per episode, broadcast to all rows
def test_episode_advantage_is_shared_by_all_rows_and_tokens():
    batch = ms.compute_episode_grpo_advantage(_aligned(), norm_adv_by_std_in_grpo=False)
    adv, mask = batch.batch["advantages"], batch.batch["response_mask"].bool()
    for rows in _episode_rows(batch).values():
        vals = torch.cat([adv[i][mask[i]] for i in rows])
        assert torch.allclose(vals, vals[0].expand_as(vals))
        assert (adv[rows][~mask[rows]] == 0).all()
    torch.testing.assert_close(batch.batch["returns"], adv)


# ---------------------------------------------------------------- 3. the baseline is a mean over EPISODES, not rows
def test_baseline_is_over_episodes_not_rows():
    batch = ms.compute_episode_grpo_advantage(_aligned(), norm_adv_by_std_in_grpo=False)
    adv = {ep: float(batch.non_tensor_batch[ms.EPISODE_ADV_KEY][rows[0]]) for ep, rows in _episode_rows(batch).items()}
    # task 2: the only success has 4 segments. Episode mean = 1/8; a mean over its 11 rows would be 1/11.
    assert math.isclose(adv["task2#16"], 1 - 1 / 8, rel_tol=1e-6)
    assert math.isclose(adv["task2#17"], -1 / 8, rel_tol=1e-6)
    # the row-level estimator (what grouping segments by uid would do) gives a different, wrong baseline
    from verl.trainer.ppo.core_algos import compute_grpo_outcome_advantage

    row_adv, _ = compute_grpo_outcome_advantage(
        batch.batch["token_level_rewards"], batch.batch["response_mask"], batch.non_tensor_batch["uid"], False
    )
    first_row = _episode_rows(batch)["task2#17"][0]
    assert not math.isclose(float(row_adv[first_row][0]), -1 / 8, rel_tol=1e-3)


def test_episode_score_is_summed_over_rows_wherever_the_reward_sits():
    batch_last = ms.compute_episode_grpo_advantage(_aligned(), norm_adv_by_std_in_grpo=False)
    moved = _aligned()
    for rows in _episode_rows(moved).values():  # move each episode's reward from its final row to its first row
        r = moved.batch["token_level_rewards"]
        total = r[rows].sum()
        r[rows] = 0
        r[rows[0], 0] = total
    batch_first = ms.compute_episode_grpo_advantage(moved, norm_adv_by_std_in_grpo=False)
    np.testing.assert_allclose(
        batch_last.non_tensor_batch[ms.EPISODE_ADV_KEY].astype(float),
        batch_first.non_tensor_batch[ms.EPISODE_ADV_KEY].astype(float),
    )


# ---------------------------------------------------------------- 4. Dr.GRPO (no std) vs std-normalized GRPO
@pytest.mark.parametrize("norm", [False, True])
def test_dr_grpo_and_std_normalization(norm):
    batch = ms.compute_episode_grpo_advantage(_aligned(), norm_adv_by_std_in_grpo=norm)
    adv = {ep: float(batch.non_tensor_batch[ms.EPISODE_ADV_KEY][rows[0]]) for ep, rows in _episode_rows(batch).items()}
    for t in range(4):
        r = torch.tensor(REWARDS[t], dtype=torch.float64)
        expect = (r - r.mean()) / (r.std() + 1e-6) if norm else r - r.mean()
        got = torch.tensor([adv[f"task{t}#{t * N + e}"] for e in range(N)], dtype=torch.float64)
        torch.testing.assert_close(got, expect, rtol=1e-5, atol=1e-6)
    # the all-zero group gives zero advantage, so no gradient
    assert all(adv[f"task1#{N + e}"] == 0 for e in range(N))


def test_single_episode_group_follows_grpo_convention():
    batch = ms.compute_episode_grpo_advantage(
        _aligned(
            segments=[[2] * N],
            rewards=[[1] + [0] * (N - 1)],
            flags={(0, e): {"exclude_from_baseline": True} for e in range(1, N)},
        ),
        norm_adv_by_std_in_grpo=True,
    )
    # only episode 0 is in the baseline -> a group of one: mean 0, std 1 (as compute_grpo_outcome_advantage)
    adv = batch.non_tensor_batch[ms.EPISODE_ADV_KEY].astype(float)
    assert math.isclose(adv[0], 1 / (1 + 1e-6), rel_tol=1e-6)


# ---------------------------------------------------------------- 6. overlong: in the baseline, out of the loss
def test_overlong_episode_counts_in_baseline_and_is_masked_after_advantage():
    flags = {(3, 4): {"episode_overlong": True, "exclude_from_loss": True}}  # task 3 episode 4: reward 0, capped
    batch = ms.compute_episode_grpo_advantage(_aligned(flags=flags), norm_adv_by_std_in_grpo=False)
    eps = _episode_rows(batch)
    mean_with = np.mean(REWARDS[3])  # 5/8: the overlong episode's 0 is in the baseline
    assert math.isclose(
        float(batch.non_tensor_batch[ms.EPISODE_ADV_KEY][eps["task3#24"][0]]), 1 - mean_with, rel_tol=1e-6
    )
    for i in eps["task3#28"]:
        assert batch.batch["response_mask"][i].sum() == 0 and (batch.batch["advantages"][i] == 0).all()
    # the advantage itself was computed (stored per row) before the mask was applied
    assert math.isclose(float(batch.non_tensor_batch[ms.EPISODE_ADV_KEY][eps["task3#28"][0]]), -mean_with, rel_tol=1e-6)


def test_exclude_from_baseline_removes_episode_from_the_mean():
    flags = {(3, 4): {"exclude_from_baseline": True, "exclude_from_loss": True}}
    batch = ms.compute_episode_grpo_advantage(_aligned(flags=flags), norm_adv_by_std_in_grpo=False)
    mean_without = np.mean([r for e, r in enumerate(REWARDS[3]) if e != 4])  # 5/7
    first = _episode_rows(batch)["task3#24"][0]
    assert math.isclose(float(batch.non_tensor_batch[ms.EPISODE_ADV_KEY][first]), 1 - mean_without, rel_tol=1e-6)


# ---------------------------------------------------------------- 8. token-mean loss, padding / dummy rows are neutral
def _token_mean(batch: DataProto) -> torch.Tensor:
    loss_mat = -batch.batch["advantages"] * batch.batch["rollout_log_probs"]  # any per-token loss
    mask = batch.batch["response_mask"]
    return agg_loss(loss_mat, mask, "token-mean", dp_size=1, batch_num_tokens=int(mask.sum()))


def test_token_mean_loss_weights_tokens_and_ignores_padding_rows():
    batch = ms.compute_episode_grpo_advantage(_aligned(), norm_adv_by_std_in_grpo=False)
    loss = _token_mean(batch)
    mask = batch.batch["response_mask"].float()
    expect = (-batch.batch["advantages"] * batch.batch["rollout_log_probs"] * mask).sum() / mask.sum()
    torch.testing.assert_close(loss, expect)  # every trained token weighs the same, whatever its row
    padded, n_pad = ms.pad_rows(_aligned(), divisor=64, pad_token_id=PAD, eos_token_id=EOS)
    assert n_pad > 0
    padded = ms.compute_episode_grpo_advantage(padded, norm_adv_by_std_in_grpo=False)
    torch.testing.assert_close(_token_mean(padded), loss)


def test_fully_masked_mini_batch_gives_zero_loss_not_nan():
    loss_mat = torch.randn(3, 4)
    mask = torch.zeros(3, 4)
    assert agg_loss(loss_mat, mask, "token-mean", dp_size=2, batch_num_tokens=0).item() == 0.0
    assert agg_loss(loss_mat, mask, "token-mean").item() == 0.0  # local count path


# ---------------------------------------------------------------- padding rows
@pytest.mark.parametrize("pos_3d", [False, True])
def test_pad_rows_shape_and_neutrality(pos_3d):
    batch = _aligned()
    batch.meta_info["metrics"] = [{"x": 1}]
    if pos_3d:
        batch.batch["position_ids"] = batch.batch["position_ids"].unsqueeze(1).expand(-1, 4, -1).clone()
    padded, n_pad = ms.pad_rows(batch, divisor=16, pad_token_id=PAD, eos_token_id=EOS)
    assert len(padded) % 16 == 0 and n_pad == len(padded) - len(batch) and n_pad > 0
    assert padded.meta_info == batch.meta_info
    pad = slice(len(batch), None)
    assert (padded.batch["response_mask"][pad] == 0).all()
    assert (padded.batch["token_level_rewards"][pad] == 0).all()
    assert (padded.batch["attention_mask"][pad].sum(-1) == 2).all()  # 1 prompt + 1 response token
    assert (padded.batch["responses"][pad][:, 0] == EOS).all() and (padded.batch["prompts"][pad][:, -1] == EOS).all()
    for key in (ms.PADDING_KEY, ms.EXCLUDE_LOSS_KEY, ms.EXCLUDE_BASELINE_KEY):
        assert padded.non_tensor_batch[key][pad].tolist() == [True] * n_pad
        assert not any(padded.non_tensor_batch[key][: len(batch)])
    assert not set(padded.non_tensor_batch["uid"][pad]) & set(batch.non_tensor_batch["uid"])
    assert len(set(padded.non_tensor_batch[ms.EPISODE_KEY][pad])) == n_pad
    # the real rows' advantages do not change
    a = ms.compute_episode_grpo_advantage(_aligned(), norm_adv_by_std_in_grpo=False).batch["advantages"]
    b = ms.compute_episode_grpo_advantage(padded, norm_adv_by_std_in_grpo=False).batch["advantages"]
    torch.testing.assert_close(b[: len(batch)], a)
    assert (b[pad] == 0).all()


def test_pad_rows_noop_when_divisible():
    batch = _aligned()
    out, n_pad = ms.pad_rows(batch, divisor=len(batch), pad_token_id=PAD, eos_token_id=EOS)
    assert n_pad == 0 and out is batch


def test_pad_divisor():
    assert ms.pad_divisor(dp_size=4, ppo_mini_batch_size=8, rollout_n=8, train_batch_size=8, update_count="fixed") == 4
    assert ms.pad_divisor(4, 4, 8, 8, "fixed") == 8  # 2 updates per step
    assert ms.pad_divisor(4, 8, 8, 8, "rows") == 64
    with pytest.raises(ValueError):
        ms.pad_divisor(4, 8, 8, 8, "bogus")


# ---------------------------------------------------------------- trainer integration
def test_compute_advantage_dispatches_multi_segment_batches():
    batch = _aligned()
    out = compute_advantage(batch, AdvantageEstimator.GRPO, norm_adv_by_std_in_grpo=False)
    assert ms.EPISODE_ADV_KEY in out.non_tensor_batch
    with pytest.raises(NotImplementedError):
        compute_advantage(_aligned(), AdvantageEstimator.RLOO)


def test_compute_advantage_single_row_path_unchanged():
    batch_in, gen = _make_batch(segments=[[1] * N] * 4)
    gen.non_tensor_batch.pop(ms.SOURCE_INDEX_KEY)
    for key in (
        "segment_index",
        "segment_kind",
        "is_final_segment",
        "episode_overlong",
        "exclude_from_loss",
        "exclude_from_baseline",
    ):
        gen.non_tensor_batch.pop(key)
    plain = batch_in.union(gen)
    out = compute_advantage(plain, AdvantageEstimator.GRPO, norm_adv_by_std_in_grpo=False)
    assert ms.EPISODE_ADV_KEY not in out.non_tensor_batch
    # one segment per episode: the episode estimator equals the row estimator
    multi = ms.compute_episode_grpo_advantage(_aligned(segments=[[1] * N] * 4), norm_adv_by_std_in_grpo=False)
    torch.testing.assert_close(multi.batch["advantages"], out.batch["advantages"])


def _stub_trainer(update_count="fixed", ppo_mini=8, train_bs=8, dp=4, micro=1, dynamic=False):
    cfg = OmegaConf.create(
        {
            "algorithm": {"multi_segment": {"update_count": update_count, "dump_dir": None}},
            "data": {"train_batch_size": train_bs},
            "actor_rollout_ref": {
                "actor": {
                    "ppo_mini_batch_size": ppo_mini,
                    "ppo_micro_batch_size_per_gpu": micro,
                    "use_dynamic_bsz": dynamic,
                    "ppo_epochs": 1,
                    "data_loader_seed": 1,
                    "shuffle": False,
                    "calculate_entropy": False,
                    "entropy_coeff": 0.0,
                },
                "rollout": {"n": N, "temperature": 1.0, "multi_turn": {"enable": True}},
            },
        }
    )
    captured = {}

    class _WG:
        def update_actor(self, td):
            from verl.utils import tensordict_utils as tu

            captured.update(
                {
                    k: tu.get_non_tensor_data(td, k, None)
                    for k in ("mini_batch_size", "num_mini_batch", "global_batch_size")
                }
            )
            return tu.get_tensordict(tensor_dict={}, non_tensor_dict={"metrics": {"mfu": 0.0}})

    stub = SimpleNamespace(
        config=cfg,
        actor_rollout_wg=_WG(),
        tokenizer=SimpleNamespace(eos_token_id=EOS, pad_token_id=PAD),
        _get_dp_size=lambda wg, role: dp,
    )
    stub._multi_segment_update_count = lambda: RayPPOTrainer._multi_segment_update_count(stub)
    return stub, captured


def _trainable(batch: DataProto) -> DataProto:
    return DataProto(
        batch=batch.batch.select(
            "input_ids",
            "attention_mask",
            "position_ids",
            "response_mask",
            "responses",
            "prompts",
            "advantages",
            "rollout_log_probs",
        ),
        non_tensor_batch=batch.non_tensor_batch,
        meta_info={},
    )


@pytest.mark.parametrize("ppo_mini,k", [(8, 1), (4, 2)])
def test_update_actor_keeps_update_count_fixed(ppo_mini, k):
    stub, captured = _stub_trainer(ppo_mini=ppo_mini)
    metrics = {}
    batch = RayPPOTrainer._pad_multi_segment_batch(stub, _aligned(), metrics)
    assert len(batch) % (4 * k) == 0 and metrics["multi_segment/dummy_rows"] == len(batch) - sum(map(sum, SEGMENTS))
    batch = ms.compute_episode_grpo_advantage(batch, norm_adv_by_std_in_grpo=False)
    RayPPOTrainer._update_actor(stub, _trainable(batch))
    assert captured == {"mini_batch_size": None, "num_mini_batch": k, "global_batch_size": len(batch) // k}


def test_update_actor_rows_mode_and_single_row_path_use_row_mini_batches():
    stub, captured = _stub_trainer(update_count="rows")
    batch = RayPPOTrainer._pad_multi_segment_batch(stub, _aligned(), {})
    assert len(batch) % 64 == 0
    RayPPOTrainer._update_actor(stub, _trainable(ms.compute_episode_grpo_advantage(batch, False)))
    assert captured == {"mini_batch_size": 64, "num_mini_batch": None, "global_batch_size": 64}
    stub, captured = _stub_trainer()
    batch_in, gen = _make_batch(segments=[[1] * N] * 4)
    gen.non_tensor_batch.pop(ms.SOURCE_INDEX_KEY)
    plain = compute_advantage(batch_in.union(gen), AdvantageEstimator.GRPO)
    RayPPOTrainer._update_actor(stub, _trainable(plain))
    assert captured == {"mini_batch_size": 64, "num_mini_batch": None, "global_batch_size": 64}


def test_pad_multiplies_micro_batch_without_dynamic_bsz():
    stub, _ = _stub_trainer(micro=3)
    assert len(RayPPOTrainer._pad_multi_segment_batch(stub, _aligned(), {})) % 12 == 0
    stub, _ = _stub_trainer(micro=3, dynamic=True)
    assert len(RayPPOTrainer._pad_multi_segment_batch(stub, _aligned(), {})) % 4 == 0


# ---------------------------------------------------------------- guards for fully masked micro-batches (bypass mode)
def test_one_step_off_bypass_uses_rollout_log_probs_as_old():
    # one-step-off sets bypass_mode: the driver uses rollout_log_probs as old_log_probs. Its loss_mode flip only edits
    # the driver's copy of the config; engine workers keep the loss they were built with (vanilla PPO-clip, against
    # those same old log-probs; the 2026-10-05 GPU smoke logged no rollout_corr/* keys). Both losses are tested below.
    cfg = OmegaConf.load(
        Path(__file__).parents[3] / "verl/experimental/one_step_off_policy/config/one_step_off_ppo_trainer.yaml"
    )
    assert cfg.algorithm.rollout_correction.bypass_mode is True
    policy_loss = OmegaConf.create({"loss_mode": "vanilla"})
    batch = _aligned()
    apply_bypass_mode(batch, OmegaConf.create({"bypass_mode": True, "loss_type": "ppo_clip"}), policy_loss)
    assert policy_loss.loss_mode == "bypass_mode"
    assert torch.equal(batch.batch["old_log_probs"], batch.batch["rollout_log_probs"])


def test_empty_mask_guards():
    lp = torch.randn(2, 4)
    empty = torch.zeros(2, 4, dtype=torch.bool)
    assert compute_offpolicy_metrics(lp, lp, empty) == {}
    weights, mask, metrics = compute_rollout_correction_and_rejection_mask(lp, lp, empty)
    assert weights is None and metrics == {} and not mask.any()


def test_vanilla_ppo_clip_loss_on_fully_masked_micro_batch():
    config = ActorConfig(strategy="fsdp", rollout_n=1, ppo_micro_batch_size_per_gpu=1, loss_agg_mode="token-mean")
    config.global_batch_info.update(dp_size=2, batch_num_tokens=7, global_batch_size=None, loss_scale_factor=None)
    log_prob = torch.randn(1, 5, requires_grad=True)
    loss, metrics = compute_policy_loss_vanilla(
        log_prob.detach() - 0.1, log_prob, torch.ones(1, 5), torch.zeros(1, 5, dtype=torch.bool), "token-mean", config
    )
    loss.backward()
    assert loss.item() == 0.0 and (log_prob.grad == 0).all()
    assert all(math.isfinite(float(v)) for v in metrics.values())


def test_bypass_ppo_clip_loss_on_fully_masked_micro_batch():
    config = ActorConfig(
        strategy="fsdp",
        rollout_n=1,
        ppo_micro_batch_size_per_gpu=1,
        policy_loss=PolicyLossConfig(
            loss_mode="bypass_mode", rollout_correction=RolloutCorrectionConfig.bypass_ppo_clip()
        ),
        loss_agg_mode="token-mean",
    )
    config.global_batch_info.update(dp_size=2, batch_num_tokens=7, global_batch_size=None, loss_scale_factor=None)
    log_prob = torch.randn(1, 5, requires_grad=True)
    loss, _ = compute_policy_loss_bypass_mode(
        log_prob.detach() - 0.1, log_prob, torch.ones(1, 5), torch.zeros(1, 5, dtype=torch.bool), "token-mean", config
    )
    loss.backward()
    assert loss.item() == 0.0 and torch.isfinite(log_prob.grad).all() and (log_prob.grad == 0).all()


# ---------------------------------------------------------------- metrics and row dump
def test_multi_segment_metrics():
    flags = {(3, 4): {"episode_overlong": True, "exclude_from_loss": True}}
    batch, n_pad = ms.pad_rows(_aligned(flags=flags), divisor=64, pad_token_id=PAD, eos_token_id=EOS)
    batch = ms.compute_episode_grpo_advantage(batch, norm_adv_by_std_in_grpo=False)
    m = ms.compute_multi_segment_metrics(batch)
    seg = [s for task in SEGMENTS for s in task]
    assert m["multi_segment/rows"] == sum(seg) and m["multi_segment/dummy_rows"] == n_pad
    assert m["multi_segment/episodes"] == 32
    assert math.isclose(m["multi_segment/segments_per_episode/mean"], np.mean(seg))
    assert m["multi_segment/segments_per_episode/max"] == 4
    assert math.isclose(m["multi_segment/overlong_episode_rate"], 1 / 32)
    assert math.isclose(m["multi_segment/loss_excluded_episode_rate"], 1 / 32)
    assert math.isclose(m["multi_segment/compaction_rate"], sum(s >= 3 for s in seg) / 32)  # kinds: index 1 = summary
    assert math.isclose(m["multi_segment/zero_adv_group_frac"], 1 / 4)
    assert math.isclose(m["multi_segment/success_rate"], np.mean(REWARDS))
    assert m["multi_segment/trained_tokens"] == float(batch.batch["response_mask"].sum())
    assert 0 < m["multi_segment/summary_token_frac"] < 1


def test_dump_rows(tmp_path):
    batch, _ = ms.pad_rows(_aligned(), divisor=64, pad_token_id=PAD, eos_token_id=EOS)
    batch = ms.compute_episode_grpo_advantage(batch, norm_adv_by_std_in_grpo=False)
    path = ms.dump_rows(batch, str(tmp_path), step=3)
    rows = [json.loads(line) for line in Path(path).read_text().splitlines()]
    assert path.endswith("step_3.jsonl") and len(rows) == sum(map(sum, SEGMENTS))
    first = rows[0]
    assert first[ms.EPISODE_KEY] == "task0#0" and len(first["response_ids"]) == len(first["response_mask"])
    assert first["segment_kind"] == "main" and isinstance(first[ms.EPISODE_ADV_KEY], float)


# ---------------------------------------------------------------- zero-advantage rows leave the loss, the KL and N_tok
def test_mask_zero_adv_drops_uniform_groups_only():
    plain = ms.compute_episode_grpo_advantage(_aligned(), norm_adv_by_std_in_grpo=False)
    masked = ms.compute_episode_grpo_advantage(_aligned(), norm_adv_by_std_in_grpo=False, mask_zero_adv=True)
    uid = masked.non_tensor_batch["uid"]
    in_uniform = np.array([u == "task1" for u in uid])  # task 1: all 8 episodes scored 0
    assert masked.non_tensor_batch[ms.ZERO_ADV_KEY].tolist() == in_uniform.tolist()
    assert (masked.batch["response_mask"][torch.from_numpy(in_uniform)] == 0).all()
    keep = torch.from_numpy(~in_uniform)
    torch.testing.assert_close(masked.batch["response_mask"][keep], plain.batch["response_mask"][keep])
    torch.testing.assert_close(masked.batch["advantages"], plain.batch["advantages"])  # A was 0 there anyway
    # token-mean now averages over the nonzero-advantage tokens only: a larger step, same direction
    assert _token_mean(masked).item() == pytest.approx(
        _token_mean(plain).item()
        * plain.batch["response_mask"].sum().item()
        / masked.batch["response_mask"].sum().item()
    )
    # a KL-like term over the mask no longer includes the uniform group's tokens
    kld = torch.rand_like(masked.batch["advantages"])
    m = masked.batch["response_mask"]
    kl = agg_loss(kld, m, "token-mean", dp_size=1, batch_num_tokens=int(m.sum()))
    torch.testing.assert_close(kl, (kld * m).sum() / m.sum())
    met = ms.compute_multi_segment_metrics(masked)
    assert met["multi_segment/zero_adv_masked_rows"] == float(in_uniform.sum())
    pre = sum(int(x) for x in masked.non_tensor_batch[ms.PRE_ZERO_TOKENS_KEY])
    assert met["multi_segment/nonzero_adv_token_frac"] == pytest.approx(float(m.sum()) / pre)
    assert ms.compute_multi_segment_metrics(plain)["multi_segment/nonzero_adv_token_frac"] == 1.0
    assert ms.compute_multi_segment_metrics(plain)["multi_segment/zero_adv_masked_rows"] == 0.0


def test_mask_zero_adv_keeps_single_episode_groups_and_exclusions():
    flags = {(0, e): {"exclude_from_baseline": True} for e in range(1, N)}
    flags[(3, 4)] = {"episode_overlong": True, "exclude_from_loss": True}
    batch = ms.compute_episode_grpo_advantage(_aligned(flags=flags), norm_adv_by_std_in_grpo=True, mask_zero_adv=True)
    zero = batch.non_tensor_batch[ms.ZERO_ADV_KEY]
    eps = _episode_rows(batch)
    # task 0 episode 0 is a group of one in the baseline: A = s (GRPO convention) -> kept; s = 0 here, so it IS zero
    assert all(zero[i] == (abs(batch.non_tensor_batch[ms.EPISODE_ADV_KEY][i]) < 1e-12) for i in eps["task0#0"])
    # an episode already excluded from the loss is not counted as zero-advantage-masked
    assert not any(zero[i] for i in eps["task3#28"])


def test_compute_advantage_reads_mask_zero_adv_from_config():
    on = compute_advantage(_aligned(), AdvantageEstimator.GRPO, norm_adv_by_std_in_grpo=False)
    off = compute_advantage(
        _aligned(),
        AdvantageEstimator.GRPO,
        norm_adv_by_std_in_grpo=False,
        config=OmegaConf.create({"multi_segment": {"mask_zero_adv": False}}),
    )
    assert any(on.non_tensor_batch[ms.ZERO_ADV_KEY]) and not any(off.non_tensor_batch[ms.ZERO_ADV_KEY])


def test_all_masked_batch_skips_the_actor_update():
    from verl.experimental.separation.ray_trainer import SeparateRayPPOTrainer

    calls = []
    stub = SimpleNamespace(
        metrics={},
        timing_raw={},
        global_steps=1,
        config=OmegaConf.create({"trainer": {"critic_warmup": 0}}),
        _update_actor=lambda b: calls.append(b) or DataProto(meta_info={"metrics": {}}),
    )
    batch = ms.compute_episode_grpo_advantage(_aligned(), norm_adv_by_std_in_grpo=False)
    batch.batch["response_mask"].zero_()
    SeparateRayPPOTrainer._fit_update_actor(stub, batch)
    assert calls == [] and stub.metrics["multi_segment/skipped_update"] == 1.0
    SeparateRayPPOTrainer._fit_update_actor(stub, ms.compute_episode_grpo_advantage(_aligned(), False))
    assert len(calls) == 1 and stub.metrics["multi_segment/skipped_update"] == 0.0


# ---------------------------------------------------------------- bypass mode reaches the workers, and is checked
def _trainer_cfg(bypass=True):
    cfg = OmegaConf.load(Path(__file__).parents[3] / "verl/trainer/config/_generated_ppo_trainer.yaml")
    cfg.algorithm.rollout_correction = OmegaConf.create(
        {"bypass_mode": bypass, "loss_type": "ppo_clip", "rollout_is": None, "rollout_rs": None}
    )
    cfg.actor_rollout_ref.actor.clip_ratio_high = 0.28
    cfg.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu = 1
    return cfg


def test_bypass_mode_is_applied_before_the_actor_workers_are_built():
    from verl.experimental.separation.ray_trainer import SeparateRayPPOTrainer

    seen = {}
    stub = SimpleNamespace(config=_trainer_cfg())
    stub._apply_bypass_mode_to_actor_config = lambda: RayPPOTrainer._apply_bypass_mode_to_actor_config(stub)
    stub._create_actor_rollout_classes = lambda: seen.update(
        loss_mode=stub.config.actor_rollout_ref.actor.policy_loss.loss_mode,
        bypass=stub.config.actor_rollout_ref.actor.policy_loss.rollout_correction.bypass_mode,
    )
    stub._create_critic_class = stub._create_reference_policy_class = stub._create_reward_model_class = lambda: None
    SeparateRayPPOTrainer._create_worker_classes(stub)
    assert seen == {"loss_mode": "bypass_mode", "bypass": True}
    off = SimpleNamespace(config=_trainer_cfg(bypass=False))
    RayPPOTrainer._apply_bypass_mode_to_actor_config(off)
    assert off.config.actor_rollout_ref.actor.policy_loss.loss_mode == "vanilla"


def _worker_info(cfg):
    """What get_policy_loss_info returns on a worker built from this actor config (the worker's own conversion)."""
    from functools import partial

    from verl.utils.config import omega_conf_to_dataclass
    from verl.workers.engine_workers import ActorRolloutRefWorker
    from verl.workers.utils.losses import ppo_loss

    actor_config = omega_conf_to_dataclass(cfg.actor_rollout_ref.actor)
    worker = SimpleNamespace(actor=SimpleNamespace(loss_fn=partial(ppo_loss, config=actor_config)))
    return ActorRolloutRefWorker.get_policy_loss_info(worker)


def test_worker_policy_loss_info_and_startup_check():
    cfg = _trainer_cfg()
    stub = SimpleNamespace(config=cfg)
    RayPPOTrainer._apply_bypass_mode_to_actor_config(stub)
    info = _worker_info(cfg)
    assert info["loss_fn"] == "ppo_loss" and info["loss_mode"] == "bypass_mode"
    assert info["rollout_correction.bypass_mode"] is True and info["clip_ratio_high"] == 0.28
    stub.actor_rollout_wg = SimpleNamespace(get_policy_loss_info=lambda: [info, info])
    RayPPOTrainer._check_worker_policy_loss(stub)  # matches: no error
    # the old failure: workers built before the driver switched to bypass mode
    stale = _worker_info(_trainer_cfg())
    assert stale["loss_mode"] == "vanilla"
    stub.actor_rollout_wg = SimpleNamespace(get_policy_loss_info=lambda: [info, stale])
    with pytest.raises(RuntimeError, match="different policy loss"):
        RayPPOTrainer._check_worker_policy_loss(stub)


# ---------------------------------------------------------------- cost penalty (algorithm.multi_segment.cost_penalty)
def _ep_cost(t: int, e: int) -> float:
    return 0.01 * (t * N + e + 1)  # a distinct cost per episode


def _with_cost(batch: DataProto, alias: bool = True) -> DataProto:
    """Cost columns on every row (identical within an episode), as the agent loop emits them. ``alias``: the trainer's
    ``token_level_rewards`` IS ``token_level_scores`` (separation/ray_trainer.py assigns it without a copy)."""
    eps = batch.non_tensor_batch[ms.EPISODE_KEY]
    te = [(int(ep.split("#")[0].removeprefix("task")), int(ep.split("#")[1]) % N) for ep in eps]
    batch.non_tensor_batch["episode_cost_usd"] = np.array([_ep_cost(t, e) for t, e in te], dtype=object)
    batch.non_tensor_batch["episode_output_tokens"] = np.array([1000 * (e + 1) for _, e in te], dtype=object)
    batch.non_tensor_batch["episode_capped"] = np.array([e == 7 for _, e in te], dtype=object)
    if alias:
        batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]
    return batch


CP = {"enable": True, "signal": "cost_usd", "budget": 0.2, "coef": 0.25, "apply_to": "success", "success_threshold": 0.5}


def _ep_sum(batch: DataProto, key: str) -> dict[str, float]:
    v = batch.batch[key].sum(-1).tolist()
    return {ep: sum(v[i] for i in rows) for ep, rows in _episode_rows(batch).items()}


def test_cost_penalty_charges_solves_only_and_keeps_raw_scores():
    batch = _with_cost(_aligned())
    raw = batch.batch["token_level_scores"].clone()
    out = ms.apply_episode_cost_penalty(batch, CP, norm_adv_by_std_in_grpo=False)
    torch.testing.assert_close(out.batch["token_level_scores"], raw)  # the aliased raw outcome is untouched
    shaped, rows = _ep_sum(out, "token_level_rewards"), _episode_rows(out)
    for ep, idx in rows.items():
        t, e = int(ep.split("#")[0][4:]), int(ep.split("#")[1]) % N
        want = CP["coef"] * min(_ep_cost(t, e) / CP["budget"], 1.0) if REWARDS[t][e] else 0.0
        assert shaped[ep] == pytest.approx(REWARDS[t][e] - want)
        assert all(out.non_tensor_batch[ms.COST_PENALTY_KEY][i] == pytest.approx(want) for i in idx)
    # the penalty sits on the final (reward-carrying) row only
    final = [i for i, f in enumerate(out.non_tensor_batch["is_final_segment"]) if f]
    diff = (raw - out.batch["token_level_rewards"]).sum(-1)
    assert diff[[i for i in range(len(out)) if i not in final]].abs().max() == 0


def test_cost_penalty_orders_solves_by_cost_and_trains_all_solved_groups():
    rewards = [[1] * N, [0, 1, 0, 1, 0, 0, 0, 0]]
    segs = [[1] * N, [2] * N]
    base = ms.compute_episode_grpo_advantage(
        _with_cost(_aligned(segments=segs, rewards=rewards)), norm_adv_by_std_in_grpo=False, mask_zero_adv=True
    )
    assert all(base.non_tensor_batch[ms.ZERO_ADV_KEY][i] for i in _episode_rows(base)["task0#0"])  # uniform: masked
    pen = ms.apply_episode_cost_penalty(
        _with_cost(_aligned(segments=segs, rewards=rewards)), CP, norm_adv_by_std_in_grpo=False
    )
    pen = ms.compute_episode_grpo_advantage(pen, norm_adv_by_std_in_grpo=False, mask_zero_adv=True)
    adv = {ep: pen.non_tensor_batch[ms.EPISODE_ADV_KEY][rows[0]] for ep, rows in _episode_rows(pen).items()}
    task0 = [adv[f"task0#{e}"] for e in range(N)]
    assert task0 == sorted(task0, reverse=True) and task0[0] > 0 > task0[-1]  # cheaper solve -> larger advantage
    assert not any(pen.non_tensor_batch[ms.ZERO_ADV_KEY][i] for i in _episode_rows(pen)["task0#0"])
    # a mixed group: every solve still beats every failure
    assert min(adv[f"task1#{e}"] for e in (9, 11)) > max(adv[f"task1#{e}"] for e in (8, 10, 12, 13, 14, 15))


def test_cost_penalty_apply_to_all_saturates_and_noop_when_disabled():
    cfg = {**CP, "apply_to": "all", "budget": 0.05}
    out = ms.apply_episode_cost_penalty(_with_cost(_aligned()), cfg, norm_adv_by_std_in_grpo=False)
    pen = out.non_tensor_batch[ms.COST_PENALTY_KEY]
    rows = _episode_rows(out)
    assert REWARDS[0][0] == 0 and pen[rows["task0#0"][0]] == pytest.approx(0.25 * 0.01 / 0.05)  # a failure pays too
    assert pen[rows["task3#31"][0]] == pytest.approx(0.25)  # cost 0.32 > budget: saturates at coef
    plain = _with_cost(_aligned())
    before = plain.batch["token_level_rewards"]
    for off in (None, {}, {**CP, "enable": False}):
        same = ms.apply_episode_cost_penalty(plain, off, norm_adv_by_std_in_grpo=True)
        assert same.batch["token_level_rewards"] is before and ms.COST_PENALTY_KEY not in same.non_tensor_batch


def test_cost_penalty_guards():
    b = _with_cost(_aligned())
    with pytest.raises(ValueError, match="norm_adv_by_std_in_grpo"):
        ms.apply_episode_cost_penalty(b, CP, norm_adv_by_std_in_grpo=True)
    for bad in ({"coef": 1.0}, {"coef": -0.1}, {"budget": 0}, {"apply_to": "failed"}):
        with pytest.raises(ValueError):
            ms.apply_episode_cost_penalty(_with_cost(_aligned()), {**CP, **bad}, norm_adv_by_std_in_grpo=False)
    with pytest.raises(KeyError, match="episode_think_tokens"):
        ms.apply_episode_cost_penalty(_with_cost(_aligned()), {**CP, "signal": "think_tokens"}, False)


def test_cost_penalty_skips_padding_and_reaches_metrics_and_dump(tmp_path):
    batch, n_pad = ms.pad_rows(_with_cost(_aligned()), divisor=64, pad_token_id=PAD, eos_token_id=EOS)
    assert n_pad > 0
    batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]
    cfg = OmegaConf.create({"multi_segment": {"cost_penalty": CP, "mask_zero_adv": True}})
    batch = compute_advantage(
        batch, AdvantageEstimator.GRPO, num_repeat=N, norm_adv_by_std_in_grpo=False, config=cfg
    )
    pen = batch.non_tensor_batch[ms.COST_PENALTY_KEY]
    pad = batch.non_tensor_batch[ms.PADDING_KEY]
    assert all(pen[i] == 0 for i in range(len(batch)) if pad[i])
    m = ms.compute_multi_segment_metrics(batch)
    solved = [(t, e) for t in range(4) for e in range(N) if REWARDS[t][e]]
    want = [CP["coef"] * min(_ep_cost(t, e) / CP["budget"], 1.0) for t, e in solved]
    assert m["multi_segment/cost_penalty/mean"] == pytest.approx(sum(want) / 32)
    assert m["multi_segment/cost_penalty/applied_frac"] == pytest.approx(len(solved) / 32)
    assert m["multi_segment/cost_usd/mean"] == pytest.approx(np.mean([_ep_cost(t, e) for t in range(4) for e in range(N)]))
    assert m["multi_segment/cost_usd/solved_mean"] == pytest.approx(np.mean([_ep_cost(t, e) for t, e in solved]))
    assert m["multi_segment/output_tokens/mean"] == pytest.approx(1000 * np.mean(range(1, N + 1)))
    assert m["multi_segment/capped_episode_rate"] == pytest.approx(1 / N)
    assert m["multi_segment/success_rate"] == pytest.approx(np.mean(REWARDS))  # raw, not shaped
    assert m["multi_segment/shaped_reward_mean"] == pytest.approx(np.mean(REWARDS) - sum(want) / 32)
    row = json.loads(Path(ms.dump_rows(batch, str(tmp_path), step=1)).read_text().splitlines()[0])
    assert row["episode_cost_usd"] == pytest.approx(_ep_cost(0, 0)) and ms.COST_PENALTY_KEY in row
