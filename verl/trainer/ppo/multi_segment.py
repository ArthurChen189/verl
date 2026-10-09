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
"""Multi-segment episodes on the DataProto (v0) trainer path.

An agent loop may return several ``AgentLoopOutput`` for one sample, e.g. one per context-compaction segment of an
agentic episode (SUPO, arXiv 2510.06727; MemAgent, arXiv 2507.02259). Each segment becomes its own training row:
its own prompt (the context the segment started from) and response (everything generated after it). Because the
segments' contexts are built deterministically, log P(episode) is the sum of the segments' log-probs, so training
the rows independently with ONE shared episode advantage is the exact policy gradient of the episode.

Row layout produced by ``AgentLoopWorker`` when ``run()`` returns a list:
  ``__source_index__``      index of the generation input row (one per episode = prompt x rollout.n) the row came from

Columns this module adds or reads (non_tensor_batch, one value per row):
  ``episode_id``            ``f"{uid}#{source_index}"``: identical for all rows of one episode
  ``exclude_from_loss``     True -> the row's response_mask is zeroed after the advantage is computed
  ``exclude_from_baseline`` True -> the row's episode is left out of its group's baseline (mean / std)
  ``is_padding``            True for synthetic rows added by :func:`pad_rows`
  ``zero_adv_masked``       True -> the row's episode advantage is 0 and ``mask_zero_adv`` zeroed its mask

Advantage (:func:`compute_episode_grpo_advantage`): the episode score is the sum of ``token_level_rewards`` over ALL
rows of the episode (so it does not matter which row carries the reward), grouped by ``uid`` OVER EPISODES (never
over rows, which would weight episodes by their number of segments), baseline = mean over the group's episodes,
optional std normalisation (``norm_adv_by_std_in_grpo``; Dr.GRPO = off), broadcast to every response token of every
row of the episode.

Zero-advantage rows (``mask_zero_adv``, algorithm.multi_segment.mask_zero_adv): under token-mean a row with advantage 0
adds nothing to the policy-gradient numerator but still counts in the token denominator and in the KL term, so a step
in which half the groups are uniform (all episodes scored the same) takes a ~2x smaller policy step with a relatively
stronger KL pull than one with none. With the flag, rows whose episode advantage is 0 get their mask zeroed too: they
leave the loss, the KL and the token count. For binary rewards that is exactly the uniform groups of >= 2 episodes (the
mean of identical values is exact, so their advantage is exactly 0, with or without std normalisation).

Cost penalty (:func:`apply_episode_cost_penalty`, algorithm.multi_segment.cost_penalty, off by default): before the
advantage, a solved episode's reward is reduced by ``lambda * C(x)`` (R' = R (1 - lambda C(x)) for R in {0, 1}), the
calibrated concave penalty of docs/arthur/episode-cost-patcheval-cwe in cyber-train:
  x     = sum_i w_i * c_i / m_i over per-episode cost columns ``episode_<i>`` the agent loop emits (cyber-train's
          training/rollout/segments.py ``episode_cost``), so x = 1 is a typical solved episode; with no weights,
          x = ``episode_<signal>`` / scale
  C(x)  = [(1 + k x)^(1-q) - 1] / (k (1-q)), log(1 + k x) / k at q = 1 (concave: the penalty sees relative cost, the
          same on easy and hard tasks), x at q = 0
  lambda = lam1 / C(1): lam1 is what a typical solved episode pays; one episode pays at most max_penalty (< 1).
With ``apply_to: success`` (the default) only episodes whose raw score is above ``success_threshold`` pay, so the
policy is never rewarded for failing cheaply (giving up early), a solve always beats a failure, and among a group's
solves the cheaper ones get the larger advantage (all-solved groups, which carry no gradient otherwise, now train on
cost). The penalty goes into a copy of ``token_level_rewards`` only: ``token_level_scores`` (and so
``multi_segment/success_rate``, ``critic/score/*`` and validation) stay the raw outcome. Requires Dr.GRPO
(``norm_adv_by_std_in_grpo: false``): with std normalisation the small cost differences inside an all-solved group
would be blown up to unit-scale advantages.
"""

from __future__ import annotations

import json
import math
import os
import uuid
from collections import defaultdict
from typing import Any

import numpy as np
import torch

from verl import DataProto
from verl.utils.model import compute_position_id_with_mask

SOURCE_INDEX_KEY = "__source_index__"
EPISODE_KEY = "episode_id"
EXCLUDE_LOSS_KEY = "exclude_from_loss"
EXCLUDE_BASELINE_KEY = "exclude_from_baseline"
PADDING_KEY = "is_padding"
EPISODE_ADV_KEY = "episode_advantage"
ZERO_ADV_KEY = "zero_adv_masked"
PRE_ZERO_TOKENS_KEY = "loss_tokens_pre_zero_mask"
COST_PENALTY_KEY = "episode_cost_penalty"
COST_X_KEY = "episode_cost_x"
# per-episode cost columns emitted by the agent loop (identical on every row of an episode); the first six are the
# components of the cost penalty's x
COST_COLUMNS = (
    "episode_thinking",
    "episode_tool_call",
    "episode_message",
    "episode_tool_output",
    "episode_tool_calls",
    "episode_turns",
    "episode_cost_usd",
    "episode_input_tokens",
    "episode_output_tokens",
    "episode_title_tokens",
    "episode_model_calls",
    "episode_n_traces",
    "episode_summaries",
    "episode_run_s",
)
ZERO_ADV_EPS = 1e-12
_FLAG_KEYS = (EXCLUDE_LOSS_KEY, EXCLUDE_BASELINE_KEY, PADDING_KEY)


def is_multi_segment(data: DataProto) -> bool:
    return EPISODE_KEY in data.non_tensor_batch


def _flag(data: DataProto, key: str) -> np.ndarray:
    """Per-row boolean column; missing column or None entries count as False."""
    col = data.non_tensor_batch.get(key)
    if col is None:
        return np.zeros(len(data), dtype=bool)
    return np.array([bool(x) if x is not None else False for x in col], dtype=bool)


def align_to_outputs(batch_repeated: DataProto, gen_output: DataProto) -> DataProto:
    """Union the (prompt x n)-repeated input batch with the generation output.

    Without ``__source_index__`` this is the usual 1:1 ``union``. With it, every output row is joined to the input row
    it came from (so the episode's ``uid``, ``extra_info``, ... are copied onto each of its segments) and an
    ``episode_id`` column is added.
    """
    if SOURCE_INDEX_KEY not in gen_output.non_tensor_batch:
        return batch_repeated.union(gen_output)
    src = np.asarray(gen_output.non_tensor_batch[SOURCE_INDEX_KEY]).astype(np.int64)
    assert src.min(initial=0) >= 0 and src.max(initial=-1) < len(batch_repeated), (
        f"{SOURCE_INDEX_KEY} out of range for a batch of {len(batch_repeated)} episodes"
    )
    missing = sorted(set(range(len(batch_repeated))) - set(src.tolist()))
    assert not missing, f"episodes without any output row: {missing[:8]}"
    batch = batch_repeated.select_idxs(src).union(gen_output)
    uid = batch.non_tensor_batch["uid"]
    batch.non_tensor_batch[EPISODE_KEY] = np.array([f"{u}#{s}" for u, s in zip(uid, src, strict=True)], dtype=object)
    for key in _FLAG_KEYS:
        batch.non_tensor_batch[key] = _flag(batch, key).astype(object)
    return batch


def pad_divisor(
    dp_size: int, ppo_mini_batch_size: int, rollout_n: int, train_batch_size: int, update_count: str
) -> int:
    """Row-count multiple the update needs.

    ``fixed``: the number of optimizer updates per step stays ``train_batch_size // ppo_mini_batch_size`` (k); the
    update splits each dp shard into k mini-batches -> rows must divide by ``dp * k``.
    ``rows``: mini-batches of ``ppo_mini_batch_size * n`` rows (the update count then grows with the number of
    segments) -> rows must divide by ``lcm(dp, ppo_mini_batch_size * n)``.
    """
    if update_count == "fixed":
        return dp_size * num_fixed_updates(ppo_mini_batch_size, train_batch_size)
    if update_count == "rows":
        return math.lcm(dp_size, ppo_mini_batch_size * rollout_n)
    raise ValueError(f"algorithm.multi_segment.update_count must be 'fixed' or 'rows', got {update_count!r}")


def num_fixed_updates(ppo_mini_batch_size: int, train_batch_size: int) -> int:
    assert train_batch_size % ppo_mini_batch_size == 0, (
        f"train_batch_size {train_batch_size} must be a multiple of ppo_mini_batch_size {ppo_mini_batch_size}"
    )
    return train_batch_size // ppo_mini_batch_size


def pad_rows(batch: DataProto, divisor: int, pad_token_id: int, eos_token_id: int) -> tuple[DataProto, int]:
    """Append neutral rows until ``len(batch) % divisor == 0``; returns the batch and the number of rows added.

    A padding row is a 1-token prompt (EOS) + 1-token response (EOS) with ``response_mask`` 0, zero reward and zero
    rollout log-probs, its own ``uid`` / ``episode_id`` (a group of its own), and ``exclude_from_loss`` /
    ``exclude_from_baseline`` / ``is_padding`` set, so it contributes nothing to the advantage, the loss or the
    token-mean denominator.
    """
    n_pad = (-len(batch)) % divisor
    if n_pad == 0:
        return batch, 0
    td = batch.batch
    prompt_len = td["prompts"].shape[-1]
    resp_len = td["responses"].shape[-1]
    prompts = torch.full((n_pad, prompt_len), pad_token_id, dtype=td["prompts"].dtype)
    prompts[:, -1] = eos_token_id
    responses = torch.full((n_pad, resp_len), pad_token_id, dtype=td["responses"].dtype)
    responses[:, 0] = eos_token_id
    attention_mask = torch.zeros((n_pad, prompt_len + resp_len), dtype=td["attention_mask"].dtype)
    attention_mask[:, prompt_len - 1 : prompt_len + 1] = 1
    special = {
        "prompts": prompts,
        "responses": responses,
        "input_ids": torch.cat([prompts, responses], dim=-1),
        "attention_mask": attention_mask,
    }
    pos = compute_position_id_with_mask(attention_mask)
    src_pos = td["position_ids"]
    if src_pos.dim() == 3:  # (bsz, k, seq) e.g. multi-rope: same 1-D positions on every channel
        pos = pos.unsqueeze(1).expand(-1, src_pos.shape[1], -1).clone()
    special["position_ids"] = pos.to(src_pos.dtype)
    pad_td = {}
    for key, value in td.items():
        pad_td[key] = special[key] if key in special else torch.zeros((n_pad, *value.shape[1:]), dtype=value.dtype)
    pad_uid = f"pad-{uuid.uuid4().hex}"
    non_tensor = {}
    for key, col in batch.non_tensor_batch.items():
        arr = np.empty(n_pad, dtype=object)
        arr[:] = [col[0]] * n_pad
        non_tensor[key] = arr.astype(col.dtype) if col.dtype != object else arr
    non_tensor["uid"] = np.array([pad_uid] * n_pad, dtype=object)
    non_tensor[EPISODE_KEY] = np.array([f"{pad_uid}#{i}" for i in range(n_pad)], dtype=object)
    for key in _FLAG_KEYS:
        non_tensor[key] = np.array([True] * n_pad, dtype=object)
    if "__num_turns__" in non_tensor:
        non_tensor["__num_turns__"] = np.zeros(n_pad, dtype=batch.non_tensor_batch["__num_turns__"].dtype)
    pad = DataProto.from_dict(tensors=pad_td, non_tensors=non_tensor)
    # concat with empty meta_info: DataProto.concat merges/aggregates meta (e.g. "metrics"); keep the batch's as-is
    out = DataProto.concat([DataProto(batch=batch.batch, non_tensor_batch=batch.non_tensor_batch), pad])
    out.meta_info = batch.meta_info
    return out, n_pad


def _episode_rows(data: DataProto) -> dict[str, list[int]]:
    """episode_id -> its real (non-padding) row indices, in batch order."""
    pad = _flag(data, PADDING_KEY)
    rows: dict[str, list[int]] = defaultdict(list)
    for i, ep in enumerate(data.non_tensor_batch[EPISODE_KEY]):
        if not pad[i]:
            rows[ep].append(i)
    return rows


def cost_curve(x: float, k: float, q: float) -> float:
    """C(x) = [(1 + k x)^(1-q) - 1] / (k (1-q)); log(1 + k x) / k at q = 1. C(0) = 0, C'(0) = 1."""
    if abs(q - 1.0) < 1e-9:
        return math.log1p(k * x) / k
    return ((1.0 + k * x) ** (1.0 - q) - 1.0) / (k * (1.0 - q))


def _cost_x(nt: dict, row: int, weights: dict, norms: dict, signal: str, scale: float) -> float | None:
    """The episode's normalised cost x (None if a column is None on this row)."""
    if not weights:
        v = nt[f"episode_{signal}"][row]
        return None if v is None else max(float(v), 0.0) / scale
    x = 0.0
    for name, w in weights.items():
        v = nt[f"episode_{name}"][row]
        if v is None:
            return None
        x += float(w) * max(float(v), 0.0) / float(norms[name])
    return x


def apply_episode_cost_penalty(data: DataProto, cfg: dict | None, norm_adv_by_std_in_grpo: bool) -> DataProto:
    """Subtract each episode's cost penalty from (a copy of) ``token_level_rewards``; see the module docstring.

    cfg (algorithm.multi_segment.cost_penalty): enable; weights {component: w} and norms {component: m} (x = sum w c / m)
    or, with no weights, signal + scale (x = episode_<signal> / scale); k, q (the shape of C); lam1 (the penalty at
    x = 1); max_penalty (< 1); apply_to (success | all); success_threshold. Writes ``episode_cost_x`` and
    ``episode_cost_penalty`` on every real row (0 on padding). No-op unless enabled."""
    if not cfg or not cfg.get("enable", False):
        return data
    if norm_adv_by_std_in_grpo:
        raise ValueError(
            "algorithm.multi_segment.cost_penalty needs norm_adv_by_std_in_grpo=false (Dr.GRPO): std normalisation "
            "would scale the cost differences inside an all-solved group up to unit-size advantages"
        )
    weights = {str(k): float(v) for k, v in (cfg.get("weights") or {}).items() if float(v) != 0.0}
    norms = {str(k): float(v) for k, v in (cfg.get("norms") or {}).items()}
    signal, scale = str(cfg.get("signal", "cost_usd")), float(cfg.get("scale", 1.0))
    k, q = float(cfg.get("k", 2.0)), float(cfg.get("q", 1.0))
    lam1, cap = float(cfg.get("lam1", 0.05)), float(cfg.get("max_penalty", 0.5))
    apply_to, threshold = str(cfg.get("apply_to", "success")), float(cfg.get("success_threshold", 0.5))
    if k <= 0 or q < 0:
        raise ValueError(f"cost_penalty needs k > 0 and q >= 0, got k={k} q={q}")
    if lam1 < 0 or not 0.0 <= cap < 1.0:
        raise ValueError(f"cost_penalty needs lam1 >= 0 and max_penalty in [0, 1) (a solve beats a failure), got {lam1}, {cap}")
    if apply_to not in ("success", "all"):
        raise ValueError(f"cost_penalty.apply_to must be 'success' or 'all', got {apply_to!r}")
    if weights:
        bad = [n for n in weights if norms.get(n, 0.0) <= 0]
        if bad:
            raise ValueError(f"cost_penalty.norms needs a positive normaliser for every weighted component; missing {bad}")
        need = [f"episode_{n}" for n in weights]
    else:
        if scale <= 0:
            raise ValueError(f"cost_penalty.scale must be > 0, got {scale}")
        need = [f"episode_{signal}"]
    missing = [c for c in need if c not in data.non_tensor_batch]
    if missing:
        raise KeyError(f"cost_penalty needs the per-row columns {missing}; the agent loop did not emit them")

    nt = data.non_tensor_batch
    lam = lam1 / cost_curve(1.0, k, q)
    rewards = data.batch["token_level_rewards"].clone()  # it aliases token_level_scores: keep the raw outcome intact
    row_score = rewards.sum(dim=-1).to(torch.float64)
    prompt_len = data.batch["prompts"].shape[-1]
    resp_attn = data.batch["attention_mask"][:, prompt_len:]
    final = _flag(data, "is_final_segment")
    penalty = np.zeros(len(data), dtype=np.float64)
    xs = np.array([None] * len(data), dtype=object)
    for rows in _episode_rows(data).values():
        x = _cost_x(nt, rows[0], weights, norms, signal, scale)
        xs[rows] = x
        if x is None or (apply_to == "success" and float(row_score[rows].sum()) <= threshold):
            continue
        p = min(lam * cost_curve(x, k, q), cap)
        if p <= 0.0:
            continue
        i = next((r for r in reversed(rows) if final[r]), rows[-1])  # the row carrying the reward
        valid = torch.nonzero(resp_attn[i]).flatten()
        pos = int(valid[-1]) if len(valid) else rewards.shape[-1] - 1
        rewards[i, pos] -= p
        penalty[rows] = p
    data.batch["token_level_rewards"] = rewards
    data.non_tensor_batch[COST_PENALTY_KEY] = penalty.astype(object)
    data.non_tensor_batch[COST_X_KEY] = xs
    return data


def compute_episode_grpo_advantage(
    data: DataProto, norm_adv_by_std_in_grpo: bool = True, epsilon: float = 1e-6, mask_zero_adv: bool = False
) -> DataProto:
    """Episode-level GRPO for multi-segment batches; writes ``advantages`` / ``returns`` and applies exclusions.

    ``mask_zero_adv``: also zero the mask of rows whose episode advantage is 0 (see the module docstring)."""
    rewards = data.batch["token_level_rewards"]
    response_mask = data.batch["response_mask"]
    row_score = rewards.sum(dim=-1).to(torch.float64)
    episodes = data.non_tensor_batch[EPISODE_KEY]
    uids = data.non_tensor_batch["uid"]
    excl_base = _flag(data, EXCLUDE_BASELINE_KEY) | _flag(data, PADDING_KEY)
    excl_loss = _flag(data, EXCLUDE_LOSS_KEY) | _flag(data, PADDING_KEY)

    ep_score: dict[str, float] = defaultdict(float)
    ep_uid: dict[str, Any] = {}
    ep_in_base: dict[str, bool] = {}
    for i, ep in enumerate(episodes):
        ep_score[ep] += float(row_score[i])
        if ep in ep_uid:
            assert ep_uid[ep] == uids[i], f"episode {ep} spans two uids"
        ep_uid[ep] = uids[i]
        ep_in_base[ep] = ep_in_base.get(ep, True) and not bool(excl_base[i])

    group: dict[Any, list[float]] = defaultdict(list)
    for ep, score in ep_score.items():
        if ep_in_base[ep]:
            group[ep_uid[ep]].append(score)
    stats = {}
    for uid, scores in group.items():
        t = torch.tensor(scores, dtype=torch.float64)
        # same convention as compute_grpo_outcome_advantage: a group of one has mean 0, std 1
        stats[uid] = (t.mean().item(), t.std().item()) if len(scores) > 1 else (0.0, 1.0)
    ep_adv = {}
    for ep, score in ep_score.items():
        mean, std = stats.get(ep_uid[ep], (0.0, 1.0))
        ep_adv[ep] = (score - mean) / (std + epsilon) if norm_adv_by_std_in_grpo else score - mean

    row_adv = torch.tensor([ep_adv[ep] for ep in episodes], dtype=torch.float32)
    new_mask = response_mask.clone()
    if excl_loss.any():
        new_mask[torch.from_numpy(excl_loss)] = 0
    data.non_tensor_batch[PRE_ZERO_TOKENS_KEY] = new_mask.sum(-1).numpy().astype(object)
    zero_rows = np.array([abs(ep_adv[ep]) < ZERO_ADV_EPS for ep in episodes], dtype=bool) & ~excl_loss
    zero_rows &= (new_mask.sum(-1) > 0).numpy()
    if mask_zero_adv and zero_rows.any():
        new_mask[torch.from_numpy(zero_rows)] = 0
    else:
        zero_rows[:] = False
    data.non_tensor_batch[ZERO_ADV_KEY] = zero_rows.astype(object)
    advantages = row_adv.unsqueeze(-1) * new_mask.to(torch.float32)
    data.batch["response_mask"] = new_mask
    data.batch["advantages"] = advantages
    data.batch["returns"] = advantages
    data.non_tensor_batch[EPISODE_ADV_KEY] = row_adv.numpy().astype(object)
    return data


def compute_multi_segment_metrics(batch: DataProto) -> dict[str, float]:
    """Per-step ``multi_segment/*`` metrics (padding rows excluded from everything but ``dummy_rows``)."""
    nt = batch.non_tensor_batch
    pad = _flag(batch, PADDING_KEY)
    real = ~pad
    episodes = nt[EPISODE_KEY]
    kinds = nt.get("segment_kind", np.array([None] * len(batch), dtype=object))
    scores = batch.batch["token_level_scores"].sum(-1).tolist() if "token_level_scores" in batch.batch else None
    prompt_len = batch.batch["prompts"].shape[-1]
    resp_tokens = batch.batch["attention_mask"][:, prompt_len:].sum(-1)
    resp_cap = batch.batch["responses"].shape[-1]
    trained = batch.batch["response_mask"].sum(-1)

    ep_rows: dict[str, list[int]] = defaultdict(list)
    for i in np.flatnonzero(real):
        ep_rows[episodes[i]].append(int(i))
    n_ep = max(1, len(ep_rows))
    seg_counts = [len(v) for v in ep_rows.values()] or [0]

    def ep_any(key: str) -> float:
        col = _flag(batch, key)
        return sum(bool(col[rows].any()) for rows in ep_rows.values()) / n_ep

    ep_reward = {ep: sum(scores[i] for i in rows) for ep, rows in ep_rows.items()} if scores is not None else {}
    compacted = {ep: any(kinds[i] == "summary" for i in rows) for ep, rows in ep_rows.items()}
    n_comp = sum(compacted.values())
    by_uid: dict[Any, list[float]] = defaultdict(list)
    for ep, rows in ep_rows.items():
        if ep in ep_reward:
            by_uid[nt["uid"][rows[0]]].append(ep_reward[ep])
    trained_total = float(trained[torch.from_numpy(real)].sum())
    zero_masked = _flag(batch, ZERO_ADV_KEY) & real
    pre_zero = nt.get(PRE_ZERO_TOKENS_KEY)
    loss_tokens_pre = (
        float(sum(int(pre_zero[i]) for i in np.flatnonzero(real))) if pre_zero is not None else trained_total
    )
    summary_rows = torch.from_numpy(np.array([k == "summary" for k in kinds], dtype=bool) & real)
    real_t = torch.from_numpy(real)
    m = {
        "multi_segment/rows": float(real.sum()),
        "multi_segment/dummy_rows": float(pad.sum()),
        "multi_segment/episodes": float(len(ep_rows)),
        "multi_segment/segments_per_episode/mean": float(np.mean(seg_counts)),
        "multi_segment/segments_per_episode/max": float(np.max(seg_counts)),
        "multi_segment/response_tokens_per_row/mean": float(resp_tokens[real_t].float().mean()) if real.any() else 0.0,
        "multi_segment/response_tokens_per_row/max": float(resp_tokens[real_t].max()) if real.any() else 0.0,
        "multi_segment/rows_at_response_length_frac": float((resp_tokens[real_t] >= resp_cap).float().mean())
        if real.any()
        else 0.0,
        "multi_segment/overlong_episode_rate": ep_any("episode_overlong"),
        "multi_segment/truncation_masked_episode_rate": ep_any("segment_truncated"),
        "multi_segment/loss_excluded_episode_rate": ep_any(EXCLUDE_LOSS_KEY),
        "multi_segment/compaction_rate": n_comp / n_ep,
        "multi_segment/zero_adv_group_frac": (sum(len(set(v)) <= 1 for v in by_uid.values()) / len(by_uid))
        if by_uid
        else 0.0,
        "multi_segment/summary_token_frac": (float(trained[summary_rows].sum()) / trained_total)
        if trained_total
        else 0.0,
        "multi_segment/trained_tokens": trained_total,
        # share of the loss tokens (after the exclusions) with a nonzero advantage, i.e. that survive mask_zero_adv
        "multi_segment/nonzero_adv_token_frac": (trained_total / loss_tokens_pre) if loss_tokens_pre else 0.0,
        "multi_segment/zero_adv_masked_rows": float(zero_masked.sum()),
    }
    # per-episode cost columns (from the agent loop) and the cost penalty, averaged over episodes, not rows
    for key in (*COST_COLUMNS, COST_X_KEY, COST_PENALTY_KEY):
        col = nt.get(key)
        if col is None:
            continue
        vals = {ep: col[rows[0]] for ep, rows in ep_rows.items() if col[rows[0]] is not None}
        if not vals:
            continue
        name = key.removeprefix("episode_")
        m[f"multi_segment/{name}/mean"] = float(np.mean([float(v) for v in vals.values()]))
        if key == COST_PENALTY_KEY:
            m["multi_segment/cost_penalty/applied_frac"] = float(np.mean([float(v) > 0 for v in vals.values()]))
        elif key in ("episode_cost_usd", "episode_output_tokens", "episode_thinking", COST_X_KEY) and ep_reward:
            for label, keep in (("solved", lambda r: r > 0.5), ("failed", lambda r: r <= 0.5)):
                sel = [float(v) for ep, v in vals.items() if ep in ep_reward and keep(ep_reward[ep])]
                if sel:
                    m[f"multi_segment/{name}/{label}_mean"] = float(np.mean(sel))
    capped = nt.get("episode_capped")
    if capped is not None:
        m["multi_segment/capped_episode_rate"] = sum(bool(capped[rows[0]]) for rows in ep_rows.values()) / n_ep
    if "token_level_rewards" in batch.batch:
        shaped = batch.batch["token_level_rewards"].sum(-1).tolist()
        m["multi_segment/shaped_reward_mean"] = float(np.mean([sum(shaped[i] for i in rows) for rows in ep_rows.values()]))
    if ep_reward:
        comp_rewards = [ep_reward[ep] for ep, c in compacted.items() if c]
        m["multi_segment/success_rate_given_compaction"] = float(np.mean(comp_rewards)) if comp_rewards else 0.0
        m["multi_segment/success_rate"] = float(np.mean(list(ep_reward.values())))
    return m


def dump_rows(batch: DataProto, dump_dir: str, step: int) -> str:
    """Write every real row (token ids, mask, advantage, flags) to ``<dump_dir>/step_<step>.jsonl`` for inspection."""
    os.makedirs(dump_dir, exist_ok=True)
    path = os.path.join(dump_dir, f"step_{step}.jsonl")
    nt = batch.non_tensor_batch
    pad = _flag(batch, PADDING_KEY)
    prompt_len = batch.batch["prompts"].shape[-1]
    attn = batch.batch["attention_mask"]
    keys = [
        k
        for k in (
            "uid",
            EPISODE_KEY,
            "segment_index",
            "segment_kind",
            "num_segments",
            "is_final_segment",
            "trace_index",
            "episode_overlong",
            "segment_truncated",
            EXCLUDE_LOSS_KEY,
            EXCLUDE_BASELINE_KEY,
            ZERO_ADV_KEY,
            PRE_ZERO_TOKENS_KEY,
            EPISODE_ADV_KEY,
            COST_PENALTY_KEY,
            COST_X_KEY,
            *COST_COLUMNS,
            "episode_capped",
            "polar_session_id",
            "cve",
            "slug",
        )
        if k in nt
    ]
    with open(path, "w") as fh:
        for i in np.flatnonzero(~pad):
            p_mask = attn[i, :prompt_len].bool()
            r_mask = attn[i, prompt_len:].bool()
            row = {k: (nt[k][i].item() if hasattr(nt[k][i], "item") else nt[k][i]) for k in keys}
            row.update(
                prompt_ids=batch.batch["prompts"][i][p_mask].tolist(),
                response_ids=batch.batch["responses"][i][r_mask].tolist(),
                response_mask=batch.batch["response_mask"][i][r_mask].tolist(),
                score=float(batch.batch["token_level_scores"][i].sum())
                if "token_level_scores" in batch.batch
                else None,
            )
            fh.write(json.dumps(row, default=str) + "\n")
    return path
