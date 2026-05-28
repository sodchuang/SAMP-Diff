"""SampImagePolicy — SAMP policy with image-based observations (v3).

Wraps MultiImageObsEncoder (ResNet18 backbone) + SampNet (MAE Transformer,
Flow Matching) into the BaseImagePolicy interface.

Supports:
  * A2A warm-start: x_0 initialised from previous action's DCT
  * v2 freq-split prior: sigma / sigma_high per frequency band
  * image + low-dim (agent_pos) observations via MultiImageObsEncoder
"""

from typing import Dict, Optional, Tuple
from functools import partial

import torch
import torch.nn as nn

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.model.SAMP.samp_net import SampNet
from diffusion_policy.model.vision.multi_image_obs_encoder import MultiImageObsEncoder
from diffusion_policy.model.vision.model_getter import get_resnet
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.utils.flow.flow_matchers import (
    ConditionalFlowMatcher,
    TorchFlowMatcher,
)


class SampImagePolicy(BaseImagePolicy):
    """SAMP image policy: ResNet18 encoder + SampNet transformer.

    Parameters
    ----------
    shape_meta       : dict  Shape metadata (see pusht.yaml shape_meta section).
                             Must have 'obs' (per-key shapes/types) and 'action'.
    horizon          : int   Full prediction horizon H.
    n_obs_steps      : int   Number of observation timesteps used as condition (To).
    n_action_steps   : int   Steps to execute per predict_action call.
    crop_shape       : tuple Image crop (h, w) for training augmentation. None = no crop.
    use_group_norm   : bool  Replace BatchNorm with GroupNorm in ResNet.
    imagenet_norm    : bool  Apply ImageNet mean/std normalisation inside encoder.
    encoder_embed_dim: int   Transformer encoder embedding dim.
    decoder_embed_dim: int   Transformer decoder embedding dim.
    encoder_depth    : int   Number of transformer encoder layers.
    decoder_depth    : int   Number of transformer decoder layers.
    encoder_num_heads: int   Transformer encoder attention heads.
    decoder_num_heads: int   Transformer decoder attention heads.
    mask             : bool  Use MAE random masking during training.
    num_iter         : int   MAE decoder iterations.
    num_inference_steps: int Euler ODE steps at inference.
    sigma            : float Noise std added to warm-start (in-band) DCT coefficients.
    fm_sigma         : float ConditionalFlowMatcher sigma (0 = straight paths).
    cold_start_prob  : float Fraction of training batches that use N(0,I) as x_0.
    freq_split_low   : int   Warm-start band start index (inclusive).
    freq_split_high  : int   Warm-start band end index (exclusive).
    sigma_high       : float Noise std for out-of-band coefficients; <0 = pure N(0,I).
    pred_action_steps_only : bool  If True only predict n_action_steps frames.
    oa_step_convention     : bool  Offset obs/action boundary by 1 (standard DP conv).
    """

    def __init__(
        self,
        shape_meta: dict,
        horizon: int,
        n_obs_steps: int,
        n_action_steps: int,
        # ---- image encoder ----
        crop_shape: Optional[Tuple[int, int]] = (76, 76),
        use_group_norm: bool = True,
        imagenet_norm: bool = True,
        # ---- SampNet ----
        encoder_embed_dim: int = 512,
        decoder_embed_dim: int = 512,
        encoder_depth: int = 6,
        decoder_depth: int = 6,
        encoder_num_heads: int = 8,
        decoder_num_heads: int = 8,
        mask: bool = False,
        num_iter: int = 4,
        # ---- flow matching ----
        num_inference_steps: int = 6,
        sigma: float = 0.3,
        fm_sigma: float = 0.0,
        cold_start_prob: float = 0.3,
        freq_split_low: int = 0,
        freq_split_high: int = 8,
        sigma_high: float = 0.2,
        # ---- policy flags ----
        pred_action_steps_only: bool = False,
        oa_step_convention: bool = True,
        **kwargs,
    ):
        super().__init__()

        # ---- action dim from shape_meta ----
        action_shape = tuple(shape_meta['action']['shape'])
        assert len(action_shape) == 1, "action must be 1-D vector"
        action_dim = action_shape[0]

        # ---- image encoder (ResNet18) ----
        if isinstance(crop_shape, (list,)):
            crop_shape = tuple(crop_shape)
        rgb_model = get_resnet('resnet18')
        self.obs_encoder = MultiImageObsEncoder(
            shape_meta=shape_meta,
            rgb_model=rgb_model,
            crop_shape=crop_shape,
            random_crop=True,
            use_group_norm=use_group_norm,
            share_rgb_model=False,
            imagenet_norm=imagenet_norm,
        )
        obs_feature_dim = self.obs_encoder.output_shape()[0]   # per timestep
        condition_dim   = n_obs_steps * obs_feature_dim

        # ---- SampNet (MAE Transformer + Flow Matching) ----
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

        # ---- normalizer (set via set_normalizer) ----
        self.normalizer = LinearNormalizer()

        # ---- policy config ----
        self.horizon                = horizon
        self.action_dim             = action_dim
        self.n_obs_steps            = n_obs_steps
        self.n_action_steps         = n_action_steps
        self.pred_action_steps_only = pred_action_steps_only
        self.oa_step_convention     = oa_step_convention
        self.num_inference_steps    = num_inference_steps
        self.cold_start_prob        = cold_start_prob

        # ---- A2A warm-start state ----
        self._prev_action: Optional[torch.Tensor] = None

        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[SampImagePolicy] trainable params: {n_params / 1e6:.1f}M")

    # ------------------------------------------------------------------
    # Normalizer
    # ------------------------------------------------------------------

    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _normalize_obs(
        self, obs_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Normalize each obs key.

        RGB image keys are passed through unchanged — the MultiImageObsEncoder
        applies its own crop/imagenet normalisation internally.
        Low-dim keys (e.g. agent_pos) are normalised via the LinearNormalizer.
        """
        result = {}
        for key, val in obs_dict.items():
            if key in self.obs_encoder.rgb_keys:
                result[key] = val          # encoder handles normalisation
            else:
                result[key] = self.normalizer[key].normalize(val)
        return result

    def _encode_obs(
        self, nobs_dict: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Encode multi-step obs dict to a flat global condition vector.

        Each value has shape (B, To, *).
        Returns (B, To * encoder_feat_dim).
        """
        B  = next(iter(nobs_dict.values())).shape[0]
        To = self.n_obs_steps
        # flatten time into batch axis: (B*To, *)
        flat      = dict_apply(nobs_dict, lambda x: x.reshape(B * To, *x.shape[2:]))
        # encode: (B*To, feat_dim)
        flat_feat = self.obs_encoder(flat)
        # restore: (B, To * feat_dim)
        return flat_feat.reshape(B, To * flat_feat.shape[-1])

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict_action(
        self, obs_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Generate action prediction from image observations.

        Args:
            obs_dict : dict with at least
                'image'     : (B, To, 3, H, W)  float32 in [0, 1]
                'agent_pos' : (B, To, 2)

        Returns:
            'action'     : (B, n_action_steps, action_dim)
            'action_pred': (B, horizon,        action_dim)
        """
        B      = next(iter(obs_dict.values())).shape[0]
        device = next(iter(obs_dict.values())).device
        dtype  = next(iter(obs_dict.values())).dtype

        nobs        = self._normalize_obs(obs_dict)
        global_cond = self._encode_obs(nobs)       # (B, To * feat_dim)

        # ---- warm-start ----
        if self._prev_action is not None and self._prev_action.shape[0] == B:
            prev_actions = self._prev_action.to(device=device, dtype=dtype)
        else:
            prev_actions = None   # first frame → SampNet uses N(0,I)

        # ---- sample ----
        self.samp_net.eval()
        nsample = self.samp_net.sample(
            flow_matcher=self.flow_matcher,
            prev_actions=prev_actions,
            global_cond=global_cond,
            num_steps=self.num_inference_steps,
        )  # (B, H, Da) normalised

        self._prev_action = nsample.detach().clone()

        # ---- unnormalize ----
        action_pred = self.normalizer['action'].unnormalize(nsample)  # (B, H, Da)

        # ---- slice to execution window ----
        if self.pred_action_steps_only:
            action = action_pred
        else:
            start = self.n_obs_steps - 1 if self.oa_step_convention else self.n_obs_steps
            end   = start + self.n_action_steps
            action = action_pred[:, start:end]

        return {
            'action':      action,
            'action_pred': action_pred,
        }

    def reset(self):
        """Clear warm-start buffer (call between episodes)."""
        self._prev_action = None

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def compute_loss(self, batch: Dict) -> torch.Tensor:
        """Compute flow-matching training loss.

        Args:
            batch : dict with
                'obs'    : dict('image': (B,T,3,H,W), 'agent_pos': (B,T,2))
                'action' : (B, H, Da)

        Returns:
            loss : scalar tensor
        """
        obs    = batch['obs']     # {'image': ..., 'agent_pos': ...}
        action = batch['action']  # (B, H, Da)

        naction = self.normalizer['action'].normalize(action)
        nobs    = self._normalize_obs(obs)
        B       = naction.shape[0]

        global_cond = self._encode_obs(nobs)   # (B, To * feat_dim)

        # trajectory (support pred_action_steps_only)
        if self.pred_action_steps_only:
            start      = self.n_obs_steps - 1 if self.oa_step_convention else self.n_obs_steps
            end        = start + self.n_action_steps
            trajectory = naction[:, start:end]
        else:
            trajectory = naction  # (B, H, Da)

        # warm-start x_0: shift trajectory by 1, pad first step with zeros
        prev_actions = torch.zeros_like(trajectory)
        prev_actions[:, 1:] = trajectory[:, :-1].detach()

        loss = self.samp_net(
            flow_matcher=self.flow_matcher,
            actions_gt=trajectory,
            prev_actions=prev_actions,
            global_cond=global_cond,
            cold_start_prob=self.cold_start_prob,
        )
        return loss
