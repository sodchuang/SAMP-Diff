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

        n_params = sum(p.numel() for p in self.samp_net.parameters() if p.requires_grad)
        print(f"[SampLowdimPolicy] SampNet trainable params: {n_params / 1e6:.1f}M")

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
        if self._prev_action is not None and self._prev_action.shape[0] == B:
            prev_actions = self._prev_action.to(device=device, dtype=dtype)
        else:
            prev_actions = None  # first frame → SampNet will use N(0,I)

        # ---- sample ----
        self.samp_net.eval()
        nsample = self.samp_net.sample(
            flow_matcher=self.flow_matcher,
            prev_actions=prev_actions,
            global_cond=global_cond,
            num_steps=self.num_inference_steps,
        )  # (B, H, Da) normalised

        # store for next call
        self._prev_action = nsample.detach().clone()

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

        return {
            'action': action,
            'action_pred': action_pred,
        }

    def reset(self):
        """Clear warm-start buffer (call between episodes)."""
        self._prev_action = None

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

        # ---- warm-start x_0: prev_actions derived by shifting trajectory ----
        # Shift by 1 step along the horizon axis; pad the first position with zeros.
        prev_actions = torch.zeros_like(trajectory)
        prev_actions[:, 1:] = trajectory[:, :-1].detach()

        # ---- loss ----
        loss = self.samp_net(
            flow_matcher=self.flow_matcher,
            actions_gt=trajectory,
            prev_actions=prev_actions,
            global_cond=global_cond,
            cold_start_prob=self.cold_start_prob,
        )
        return loss
