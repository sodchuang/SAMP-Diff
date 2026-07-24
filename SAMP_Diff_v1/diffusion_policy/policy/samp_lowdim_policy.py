"""SampLowdimPolicy — SAMP (Spectral-Adaptive Modulated Prior) lowdim policy.

Wraps SampNet + TorchFlowMatcher into the standard BaseLowdimPolicy interface.

Key differences from FreqpolicyLowdimPolicy
--------------------------------------------
  • No DDPMScheduler / DiffLoss — replaced by Flow Matching.
  • A2A warm-start: last predicted action stored as `self._prev_action` and fed
    as x_0 at the next call to predict_action().  First call uses x_0 ~ N(0,I).
  • predict_action() triggers Euler ODE in DCT space, then iDCT → time domain.
  • compute_loss() derives prev_actions by shifting the batch's action sequence
    one step back (first step padded with zeros).
"""
from typing import Dict, Optional
from functools import partial

import torch
import torch.nn as nn

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_lowdim_policy import BaseLowdimPolicy
from diffusion_policy.model.SAMP.samp_net import SampNet
from diffusion_policy.utils.flow.flow_matchers import (
    ConditionalFlowMatcher,
    TorchFlowMatcher,
)


class SampLowdimPolicy(BaseLowdimPolicy):
    """Low-dimensional policy using Flow Matching over DCT action space.

    Parameters
    ----------
    horizon          : int   Full prediction horizon length.
    obs_dim          : int   Observation vector dimension.
    action_dim       : int   Action vector dimension (Da).
    n_action_steps   : int   Steps to execute per predict_action call.
    n_obs_steps      : int   Number of observation steps used as condition (To).
    num_inference_steps : int  Euler ODE steps during inference.
    sigma            : float Warm-start noise level (std of perturbation on x_0).
    fm_sigma         : float Sigma for ConditionalFlowMatcher (default 0.0).
    obs_as_global_cond : bool  Use obs as global condition (recommended).
    pred_action_steps_only : bool  If True, only predict n_action_steps frames.
    oa_step_convention : bool  Offset obs/action boundary by 1 (standard DP conv).
    """

    def __init__(
        self,
        horizon: int,
        obs_dim: int,
        action_dim: int,
        n_action_steps: int,
        n_obs_steps: int,
        # SampNet hyper-parameters
        encoder_embed_dim: int = 512,
        decoder_embed_dim: int = 512,
        encoder_depth: int = 4,
        decoder_depth: int = 4,
        encoder_num_heads: int = 8,
        decoder_num_heads: int = 8,
        mask: bool = True,
        num_iter: int = 4,
        # Flow matching
        num_inference_steps: int = 6,
        sigma: float = 0.1,
        fm_sigma: float = 0.0,
        cold_start_prob: float = 0.2,
        freq_split_low: int = 0,
        freq_split_high: int = 4,
        sigma_high: float = -1.0,
        action_group_spectral_params=None,
        action_group_loss_weights=None,
        action_group_history_params=None,
        action_group_timing_params=None,
        action_phase_loss_params=None,
        history_training_mode: str = "legacy_shift",
        # Policy convention flags
        obs_as_global_cond: bool = True,
        obs_as_local_cond: bool = False,
        pred_action_steps_only: bool = False,
        oa_step_convention: bool = False,
        **kwargs,
    ):
        super().__init__()
        assert not (obs_as_local_cond and obs_as_global_cond)
        if pred_action_steps_only:
            assert obs_as_global_cond

        # ---- configuration ----
        self.horizon = horizon
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_global_cond = obs_as_global_cond
        self.obs_as_local_cond = obs_as_local_cond
        self.pred_action_steps_only = pred_action_steps_only
        self.oa_step_convention = oa_step_convention
        self.num_inference_steps = num_inference_steps
        self.sigma = sigma
        self.cold_start_prob = cold_start_prob
        self.action_group_history_params = action_group_history_params
        self.action_group_timing_params = action_group_timing_params
        self.history_training_mode = str(history_training_mode).lower()
        if self.history_training_mode not in ("legacy_shift", "aligned"):
            raise ValueError(
                "history_training_mode must be 'legacy_shift' or 'aligned', "
                f"got {history_training_mode!r}"
            )

        # condition dimension = To * obs_dim (flattened obs window)
        condition_dim = n_obs_steps * obs_dim

        # ---- core model ----
        self.samp_net = SampNet(
            trajectory_dim=action_dim,
            horizon=horizon,
            n_obs_steps=n_obs_steps,
            condition_dim=condition_dim,
            encoder_embed_dim=encoder_embed_dim,
            decoder_embed_dim=decoder_embed_dim,
            encoder_depth=encoder_depth,
            decoder_depth=decoder_depth,
            encoder_num_heads=encoder_num_heads,
            decoder_num_heads=decoder_num_heads,
            mask=mask,
            num_iter=num_iter,
            sigma=sigma,
            freq_split_low=freq_split_low,
            freq_split_high=freq_split_high,
            sigma_high=sigma_high,
            action_group_spectral_params=action_group_spectral_params,
            action_group_loss_weights=action_group_loss_weights,
            action_phase_loss_params=action_phase_loss_params,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
        )

        # ---- flow matcher ----
        self.flow_matcher: TorchFlowMatcher = ConditionalFlowMatcher(
            num_sampling_steps=num_inference_steps,
            sigma=fm_sigma,
        )

        # ---- normalizer ----
        self.normalizer = LinearNormalizer()

        # ---- warm-start state ----
        # Stores the last predicted action trajectory (normalised) per batch item.
        self._prev_action: Optional[torch.Tensor] = None
        self._history_buffers = dict()
        self._gripper_latch_remaining: Optional[torch.Tensor] = None
        self._gripper_latch_chunk_budget: Optional[torch.Tensor] = None
        self._gripper_latch_has_started: Optional[torch.Tensor] = None

        n_params = sum(p.numel() for p in self.samp_net.parameters() if p.requires_grad)
        print(f"[SampLowdimPolicy] SampNet trainable params: {n_params / 1e6:.1f}M")

    def _history_groups(self):
        cfg = self.action_group_history_params
        if not cfg or not bool(cfg.get('enabled', False)):
            return None
        return cfg.get('groups', None)

    def _fresh_history_indices(self):
        groups = self._history_groups()
        if not groups:
            return []
        return [
            int(index)
            for raw_cfg in groups.values()
            if not bool(raw_cfg.get('use_history', True))
            for index in raw_cfg['indices']
        ]

    def _build_training_history_prior(
            self, trajectory: torch.Tensor) -> torch.Tensor:
        """Build the teacher-forced prior used during flow-matching training.

        At rollout time a predicted horizon is advanced by ``history_shift``
        before it is reused.  The surviving part of that advanced horizon is
        aligned with the new target horizon.  ``aligned`` therefore uses the
        current ground-truth trajectory as the teacher-forced estimate of that
        prior; SampNet still corrupts / filters it and cold-starts a configured
        fraction of the batch.

        ``legacy_shift`` preserves the original one-step, right-shifted prior
        for old checkpoints and non-HG experiments.
        """
        if self.history_training_mode == "aligned":
            prior = trajectory.detach().clone()
            groups = self._history_groups()
            default_shift = self.n_action_steps
            if self.action_group_history_params:
                default_shift = int(
                    self.action_group_history_params.get(
                        'shift', self.n_action_steps))

            def hold_unknown_tail(indices, shift):
                shift = min(max(int(shift), 0), trajectory.shape[1])
                if shift <= 0 or not indices:
                    return
                tail_start = trajectory.shape[1] - shift
                source_index = max(0, tail_start - 1)
                source = trajectory[
                    :, source_index:source_index + 1, indices].detach()
                prior[:, tail_start:, indices] = source.expand(
                    -1, shift, -1)

            if groups:
                assigned = set()
                for raw_cfg in groups.values():
                    cfg = dict(raw_cfg)
                    indices = [int(index) for index in cfg['indices']]
                    assigned.update(indices)
                    if bool(cfg.get('use_history', True)):
                        hold_unknown_tail(
                            indices, cfg.get('shift', default_shift))
                remaining = sorted(
                    set(range(self.action_dim)).difference(assigned))
                hold_unknown_tail(remaining, default_shift)
            else:
                hold_unknown_tail(
                    list(range(self.action_dim)), default_shift)
            return prior

        prior = torch.zeros_like(trajectory)
        prior[:, 1:] = trajectory[:, :-1].detach()
        return prior

    @staticmethod
    def _shift_history(buffer: torch.Tensor, shift: int) -> torch.Tensor:
        """Advance a predicted horizon and hold its final state at the tail."""
        horizon = buffer.shape[1]
        shift = min(max(int(shift), 0), horizon)
        if shift == 0:
            return buffer.clone()
        if shift == horizon:
            return buffer[:, -1:].expand_as(buffer).clone()

        shifted = torch.empty_like(buffer)
        shifted[:, :-shift] = buffer[:, shift:]
        shifted[:, -shift:] = buffer[:, -1:].expand(-1, shift, -1)
        return shifted

    def _build_history_prior(self, batch_size, device, dtype):
        groups = self._history_groups()
        if not groups:
            if self._prev_action is None or self._prev_action.shape[0] != batch_size:
                return None, None, []
            return self._prev_action.to(device=device, dtype=dtype), None, []

        if not self._history_buffers:
            return None, None, self._fresh_history_indices()

        # Fresh groups are replaced with Gaussian coefficients in SampNet.
        prior = torch.zeros(
            batch_size,
            self.horizon,
            self.action_dim,
            device=device,
            dtype=dtype,
        )
        assigned = set()
        aligned = dict()
        default_shift = int(
            self.action_group_history_params.get('shift', self.n_action_steps)
        )

        for name, raw_cfg in groups.items():
            cfg = dict(raw_cfg)
            indices = [int(index) for index in cfg['indices']]
            invalid = [
                index for index in indices
                if index < 0 or index >= self.action_dim
            ]
            if invalid:
                raise ValueError(
                    f"history group {name!r} has invalid action indices: {invalid}"
                )
            overlap = assigned.intersection(indices)
            if overlap:
                raise ValueError(
                    f"history group {name!r} overlaps action indices: "
                    f"{sorted(overlap)}"
                )
            assigned.update(indices)

            if not bool(cfg.get('use_history', True)):
                continue

            buffer = self._history_buffers.get(name)
            if buffer is None or buffer.shape[0] != batch_size:
                return None, None, self._fresh_history_indices()
            buffer = buffer.to(device=device, dtype=dtype)
            shifted = self._shift_history(
                buffer,
                cfg.get('shift', default_shift),
            )
            aligned[name] = shifted
            prior[:, :, indices] = shifted

        remaining = sorted(set(range(self.action_dim)).difference(assigned))
        if remaining:
            if self._prev_action is None or self._prev_action.shape[0] != batch_size:
                return None, None, self._fresh_history_indices()
            fallback = self._shift_history(
                self._prev_action.to(device=device, dtype=dtype),
                default_shift,
            )
            prior[:, :, remaining] = fallback[:, :, remaining]
        return prior, aligned, self._fresh_history_indices()

    def _update_history_buffers(self, prediction, aligned):
        groups = self._history_groups()
        if not groups:
            return

        aligned = aligned or dict()
        for name, raw_cfg in groups.items():
            cfg = dict(raw_cfg)
            if not bool(cfg.get('use_history', True)):
                continue
            indices = [int(index) for index in cfg['indices']]
            update_rate = float(cfg.get('update_rate', 1.0))
            if not 0.0 <= update_rate <= 1.0:
                raise ValueError(
                    f"history group {name!r} update_rate must be in [0, 1], "
                    f"got {update_rate}"
                )

            new_value = prediction[:, :, indices].detach().clone()
            old_value = aligned.get(name)
            if old_value is not None and update_rate < 1.0:
                new_value = (
                    update_rate * new_value
                    + (1.0 - update_rate) * old_value
                )
            self._history_buffers[name] = new_value

    def _gripper_timing_cfg(self):
        cfg = self.action_group_timing_params
        if not cfg:
            return None
        gripper_cfg = cfg.get('gripper', None)
        if not gripper_cfg or not bool(gripper_cfg.get('enabled', False)):
            return None
        return gripper_cfg

    def _apply_gripper_timing(self, action: torch.Tensor) -> torch.Tensor:
        """Keep gripper close commands stable during the executed action chunk."""
        cfg = self._gripper_timing_cfg()
        if cfg is None:
            return action

        indices = [int(index) for index in cfg.get('indices', [self.action_dim - 1])]
        invalid = [
            index for index in indices
            if index < 0 or index >= self.action_dim
        ]
        if invalid:
            raise ValueError(f"gripper timing has invalid action indices: {invalid}")

        min_close_steps = int(cfg.get('min_close_steps', self.n_action_steps))
        if min_close_steps <= 1:
            return action

        close_threshold = float(cfg.get('close_threshold', 0.0))
        close_is_greater = bool(cfg.get('close_is_greater', True))
        close_value = cfg.get('close_value', None)
        max_latch_chunks = cfg.get('max_latch_chunks_after_close', None)
        phase_gate_enabled = max_latch_chunks is not None
        if phase_gate_enabled:
            max_latch_chunks = int(max_latch_chunks)
            if max_latch_chunks <= 0:
                return action

        B, T, _ = action.shape
        device = action.device
        if (
            self._gripper_latch_remaining is None
            or self._gripper_latch_remaining.shape[0] != B
        ):
            self._gripper_latch_remaining = torch.zeros(
                B,
                device=device,
                dtype=torch.long,
            )
        else:
            self._gripper_latch_remaining = self._gripper_latch_remaining.to(device=device)

        if phase_gate_enabled:
            if (
                self._gripper_latch_chunk_budget is None
                or self._gripper_latch_chunk_budget.shape[0] != B
            ):
                self._gripper_latch_chunk_budget = torch.zeros(
                    B,
                    device=device,
                    dtype=torch.long,
                )
                self._gripper_latch_has_started = torch.zeros(
                    B,
                    device=device,
                    dtype=torch.bool,
                )
            else:
                self._gripper_latch_chunk_budget = (
                    self._gripper_latch_chunk_budget.to(device=device)
                )
                self._gripper_latch_has_started = (
                    self._gripper_latch_has_started.to(device=device)
                )

        output = action.clone()
        threshold = torch.as_tensor(
            close_threshold,
            device=device,
            dtype=output.dtype,
        )
        forced_value = None
        if close_value is not None:
            forced_value = torch.as_tensor(
                float(close_value),
                device=device,
                dtype=output.dtype,
            )

        remaining = self._gripper_latch_remaining
        if phase_gate_enabled:
            chunk_budget = self._gripper_latch_chunk_budget
            has_started = self._gripper_latch_has_started
        else:
            chunk_budget = None
            has_started = None

        for t in range(T):
            gripper_action = output[:, t, indices]
            signal = gripper_action.mean(dim=-1)
            if close_is_greater:
                close_now = signal >= close_threshold
            else:
                close_now = signal <= close_threshold

            if phase_gate_enabled:
                new_start = close_now & (~has_started)
                if new_start.any():
                    chunk_budget = torch.where(
                        new_start,
                        torch.full_like(chunk_budget, max_latch_chunks),
                        chunk_budget,
                    )
                    has_started = has_started | new_start
                gate_active = chunk_budget > 0
            else:
                gate_active = torch.ones(B, device=device, dtype=torch.bool)

            force_close = (close_now | (remaining > 0)) & gate_active
            if force_close.any():
                forced = gripper_action[force_close]
                if forced_value is not None:
                    forced = torch.ones_like(forced) * forced_value
                elif close_is_greater:
                    forced = torch.maximum(forced, threshold)
                else:
                    forced = torch.minimum(forced, threshold)
                gripper_action = gripper_action.clone()
                gripper_action[force_close] = forced
                output[:, t, indices] = gripper_action

            remaining = torch.where(
                gate_active & close_now,
                torch.full_like(remaining, min_close_steps - 1),
                torch.where(
                    gate_active,
                    torch.clamp(remaining - 1, min=0),
                    torch.zeros_like(remaining),
                ),
            )

        self._gripper_latch_remaining = remaining.detach()
        if phase_gate_enabled:
            self._gripper_latch_chunk_budget = torch.clamp(
                chunk_budget - 1,
                min=0,
            ).detach()
            self._gripper_latch_has_started = has_started.detach()
        return output

    # ------------------------------------------------------------------
    # Normalizer
    # ------------------------------------------------------------------

    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Generate action prediction from observation.

        obs_dict must contain:
            'obs': (B, To, obs_dim)

        Returns dict with:
            'action'     : (B, n_action_steps, action_dim)
            'action_pred': (B, horizon, action_dim)
        """
        assert 'obs' in obs_dict

        nobs = self.normalizer['obs'].normalize(obs_dict['obs'])  # (B, To, Do)
        B, _, Do = nobs.shape
        To = self.n_obs_steps
        assert Do == self.obs_dim
        device = nobs.device
        dtype = nobs.dtype

        # ---- build global condition ----
        global_cond = nobs[:, :To].reshape(B, -1)  # (B, To*Do)

        # ---- warm-start ----
        prev_actions, aligned_history, fresh_action_indices = self._build_history_prior(
            batch_size=B,
            device=device,
            dtype=dtype,
        )

        # ---- sample ----
        self.samp_net.eval()
        nsample = self.samp_net.sample(
            flow_matcher=self.flow_matcher,
            prev_actions=prev_actions,
            global_cond=global_cond,
            num_steps=self.num_inference_steps,
            fresh_action_indices=fresh_action_indices,
        )  # (B, H, Da) normalised

        # store for next call
        self._prev_action = nsample.detach().clone()
        self._update_history_buffers(nsample, aligned_history)

        # ---- unnormalize ----
        action_pred = self.normalizer['action'].unnormalize(nsample)  # (B, H, Da)

        # ---- slice to execution window ----
        if self.pred_action_steps_only:
            action = action_pred
        else:
            start = To
            if self.oa_step_convention:
                start = To - 1
            end = start + self.n_action_steps
            action = action_pred[:, start:end]

        action = self._apply_gripper_timing(action)
        if self._gripper_timing_cfg() is not None:
            action_pred = action_pred.clone()
            if self.pred_action_steps_only:
                action_pred = action
            else:
                action_pred[:, start:end] = action

        return {
            'action': action,
            'action_pred': action_pred,
        }

    def reset(self):
        """Clear warm-start buffer (call between episodes)."""
        self._prev_action = None
        self._history_buffers.clear()
        self._gripper_latch_remaining = None
        self._gripper_latch_chunk_budget = None
        self._gripper_latch_has_started = None

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute flow-matching training loss.

        Expects batch with:
            'obs'   : (B, To, obs_dim)
            'action': (B, horizon, action_dim)

        Returns:
            loss : scalar tensor
        """
        assert 'valid_mask' not in batch, "valid_mask not supported"

        nbatch = self.normalizer.normalize(batch)
        obs = nbatch['obs']      # (B, To, Do)  normalised
        action = nbatch['action']  # (B, H, Da) normalised
        B = action.shape[0]

        # ---- global condition ----
        global_cond = obs[:, :self.n_obs_steps].reshape(B, -1)  # (B, To*Do)

        # ---- trajectory for loss ----
        if self.pred_action_steps_only:
            To = self.n_obs_steps
            start = To - 1 if self.oa_step_convention else To
            end = start + self.n_action_steps
            trajectory = action[:, start:end]
        else:
            trajectory = action  # (B, H, Da)

        # ---- warm-start x_0 ----
        # HG rollouts advance the previous prediction before reusing it, so HG
        # training must use an aligned teacher-forced prior. The legacy mode is
        # retained for checkpoint compatibility and controlled ablations.
        prev_actions = self._build_training_history_prior(trajectory)

        # ---- loss ----
        loss = self.samp_net(
            flow_matcher=self.flow_matcher,
            actions_gt=trajectory,
            prev_actions=prev_actions,
            global_cond=global_cond,
            cold_start_prob=self.cold_start_prob,
            fresh_action_indices=self._fresh_history_indices(),
        )
        return loss
