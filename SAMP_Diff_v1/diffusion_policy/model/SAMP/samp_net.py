"""SampNet — Spectral-Adaptive Modulated Prior (SAMP) core model.

Architecture
------------
  • Freqpolicy MAE Transformer backbone (encoder + decoder) for sequence modelling.
  • DCT pre-processing on action trajectories (borrowed from Freqpolicy).
  • Flow Matching velocity head replacing DiffLoss.
  • A2A warm-start: previous action's DCT coefficients serve as x_0 for the
    Flow Matching ODE, enabling few-step inference.

High-level forward pass (training)
-----------------------------------
  actions_gt  (B, H, Da)
      → DCT → x_1                        target in frequency space
  prev_actions (B, H, Da)
      → DCT → x_0_base
      → x_0 = x_0_base + sigma * eps     warm-start with small perturbation

  TorchFlowMatcher interpolates:
      x_t = (1-t)*x_0 + t*x_1 + noise
  MAE Transformer + flow head predicts velocity:
      v_pred = flow_head(decoder_out(x_t, t, obs_cond))
  Loss:
      L = || v_pred - (x_1 - x_0) ||^2

Inference
----------
  x_0 = DCT(prev_action) + sigma * eps   (first frame: x_0 ~ N(0, I))
  Euler ODE  num_inference_steps steps
  x_1 → iDCT → actions
"""
import math
import numpy as np
import torch
import torch.nn as nn
import torch_dct
import scipy.stats as stats

from functools import partial
from diffusers.models.attention import BasicTransformerBlock


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def mask_by_order(mask_len, order, bsz, seq_len):
    masking = torch.zeros(bsz, seq_len, device=order.device)
    masking = torch.scatter(
        masking, dim=-1,
        index=order[:, :mask_len.long()],
        src=torch.ones(bsz, seq_len, device=order.device),
    )
    return masking


def sinusoidal_time_embed(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Map continuous time t ∈ [0, 1] to a sinusoidal embedding of size dim.

    Args:
        t  : (B,) float tensor in [0, 1].
        dim: embedding dimensionality (must be even).

    Returns:
        (B, dim) float tensor.
    """
    assert dim % 2 == 0, "dim must be even"
    device = t.device
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000) * torch.arange(half, device=device).float() / half
    )                                       # (half,)
    args = t[:, None] * freqs[None, :]     # (B, half)
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # (B, dim)


# ---------------------------------------------------------------------------
# SampNet
# ---------------------------------------------------------------------------

class SampNet(nn.Module):
    """Flow-Matching variant of Freqpolicy for action-trajectory generation.

    Parameters
    ----------
    trajectory_dim  : int   Action dimension (Da).
    horizon         : int   Prediction horizon (H).
    n_obs_steps     : int   Number of observation steps used as condition.
    condition_dim   : int   Raw condition vector dimension (before projection).
    encoder_embed_dim / decoder_embed_dim : int  Transformer hidden sizes.
    encoder_depth / decoder_depth         : int  Number of Transformer blocks.
    encoder_num_heads / decoder_num_heads : int  Attention heads.
    mask            : bool  Whether to use random MAE masking during training.
    num_iter        : int   Number of progressive sampling iterations
                            (used in training DCT processing, kept for compat.).
    sigma           : float Warm-start noise level added to DCT(prev_action).
    patch_size      : int   Temporal patch size (default 1 = per-step tokens).
    """

    def __init__(
        self,
        trajectory_dim: int = 26,
        horizon: int = 16,
        n_obs_steps: int = 2,
        condition_dim: int = 128,
        encoder_embed_dim: int = 256,
        encoder_depth: int = 4,
        encoder_num_heads: int = 8,
        decoder_embed_dim: int = 256,
        decoder_depth: int = 4,
        decoder_num_heads: int = 8,
        norm_layer=nn.LayerNorm,
        mask: bool = True,
        num_iter: int = 4,
        sigma: float = 0.1,
        freq_split_low: int = 0,
        freq_split_high: int = 4,
        sigma_high: float = -1.0,   # noise for out-of-band; <0 → pure N(0,I)
        action_group_spectral_params=None,
        action_group_loss_weights=None,
        action_phase_loss_params=None,
        patch_size: int = 1,
        **kwargs,
    ):
        super().__init__()

        # ---- sequence geometry ----
        self.trajectory_dim = trajectory_dim
        self.horizon = horizon
        self.n_obs_steps = n_obs_steps
        self.patch_size = patch_size
        self.seq_len = horizon // patch_size
        self.token_embed_dim = trajectory_dim
        self.encoder_embed_dim = encoder_embed_dim
        self.decoder_embed_dim = decoder_embed_dim
        self.condition_embed_dim = encoder_embed_dim
        self.buffer_size = n_obs_steps   # one buffer slot per observation step
        self.mask = mask
        self.sigma = sigma
        # v2: warm-start band [freq_split_low : freq_split_high] with sigma noise
        #     out-of-band: DCT(prev) + sigma_high * noise  (sigma_high<0 → pure N(0,I))
        self.freq_split_low  = max(0, freq_split_low)
        self.freq_split_high = max(self.freq_split_low, freq_split_high)
        self.sigma_high = sigma_high   # <0 means pure random (original v2 behaviour)
        self.action_group_spectral_params = action_group_spectral_params
        self.action_group_loss_weights = action_group_loss_weights
        self.action_phase_loss_params = action_phase_loss_params

        # ---- progressive masking schedule (kept from Freqpolicy) ----
        core_2 = 5
        if num_iter == 1:
            self.core = [0]
        else:
            self.core = [int(i * self.seq_len / (num_iter - 1)) for i in range(num_iter)]
            if self.core[1] < core_2:
                remain = num_iter - 2
                if remain > 0:
                    interval = (self.seq_len - core_2) / remain
                    self.core = [0, core_2] + [
                        int(core_2 + interval * i) for i in range(1, remain + 1)
                    ]
                else:
                    self.core = [0, self.seq_len]

        self.loss_weight = [
            2 - np.sin(math.pi / 2.0 * (b + 1) / self.seq_len)
            for b in range(self.seq_len)
        ]

        # ---- condition projection ----
        # Projects a single observation step (obs_dim) to encoder embedding dim.
        # global_cond is split into n_obs_steps tokens, each of size obs_dim.
        self.condition_dim = condition_dim
        self.obs_dim_per_step = condition_dim // n_obs_steps
        self.condition_proj = nn.Linear(self.obs_dim_per_step, self.condition_embed_dim)
        self.embedding_index = nn.Linear(1, self.condition_embed_dim)

        # ---- time embedding for flow matching t ∈ [0, 1] ----
        self.time_embed_dim = encoder_embed_dim
        self.time_proj = nn.Sequential(
            nn.Linear(encoder_embed_dim, encoder_embed_dim),
            nn.SiLU(),
            nn.Linear(encoder_embed_dim, encoder_embed_dim),
        )

        # ---- MAE masking ----
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.decoder_embed_dim))

        # ---- encoder ----
        self.z_proj = nn.Linear(self.token_embed_dim, encoder_embed_dim)
        self.z_proj_ln = nn.LayerNorm(encoder_embed_dim, eps=1e-6)
        self.encoder_pos_embed_learned = nn.Parameter(
            torch.zeros(1, self.seq_len + self.buffer_size, encoder_embed_dim)
        )
        self.encoder_blocks = nn.ModuleList([
            BasicTransformerBlock(
                encoder_embed_dim,
                encoder_num_heads,
                64,
                dropout=0.0,
                cross_attention_dim=encoder_embed_dim,
                activation_fn="geglu",
                attention_bias=True,
                upcast_attention=False,
            )
            for _ in range(encoder_depth)
        ])
        self.encoder_norm = norm_layer(encoder_embed_dim)

        # ---- decoder ----
        self.decoder_embed = nn.Linear(encoder_embed_dim, decoder_embed_dim)
        self.decoder_pos_embed_learned = nn.Parameter(
            torch.zeros(1, self.seq_len + self.buffer_size, decoder_embed_dim)
        )
        self.decoder_blocks = nn.ModuleList([
            BasicTransformerBlock(
                decoder_embed_dim,
                decoder_num_heads,
                64,
                dropout=0.0,
                cross_attention_dim=decoder_embed_dim,
                activation_fn="geglu",
                attention_bias=True,
                upcast_attention=False,
            )
            for _ in range(decoder_depth)
        ])
        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.diffusion_pos_embed_learned = nn.Parameter(
            torch.zeros(1, self.seq_len, decoder_embed_dim)
        )

        # ---- flow velocity head (replaces DiffLoss) ----
        # Projects decoder output → velocity in trajectory token space
        self.flow_head = nn.Sequential(
            nn.Linear(decoder_embed_dim, decoder_embed_dim),
            nn.SiLU(),
            nn.Linear(decoder_embed_dim, self.token_embed_dim),
        )

        self._init_weights()

    # ------------------------------------------------------------------
    # Weight initialisation
    # ------------------------------------------------------------------

    def _init_weights(self):
        if self.mask:
            nn.init.normal_(self.mask_token, std=0.02)
        nn.init.normal_(self.encoder_pos_embed_learned, std=0.02)
        nn.init.normal_(self.decoder_pos_embed_learned, std=0.02)
        nn.init.normal_(self.diffusion_pos_embed_learned, std=0.02)
        self.apply(self._init_module_weights)

    @staticmethod
    def _init_module_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)

    # ------------------------------------------------------------------
    # Random masking helpers (carried over from Freqpolicy)
    # ------------------------------------------------------------------

    def sample_orders(self, bsz: int) -> torch.Tensor:
        orders = []
        for _ in range(bsz):
            order = np.array(list(range(self.seq_len)))
            np.random.shuffle(order)
            orders.append(order)
        return torch.tensor(np.array(orders), dtype=torch.long).cuda()

    def random_masking(self, x: torch.Tensor, orders: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        mask_ratio_min = 0.7
        mask_rate = stats.truncnorm(
            (mask_ratio_min - 1.0) / 0.25, 0, loc=1.0, scale=0.25
        ).rvs(1)[0]
        num_masked = int(np.ceil(seq_len * mask_rate))
        mask = torch.zeros(bsz, seq_len, device=x.device)
        mask = torch.scatter(
            mask, dim=-1,
            index=orders[:, :num_masked],
            src=torch.ones(bsz, seq_len, device=x.device),
        )
        return mask

    # ------------------------------------------------------------------
    # DCT helpers
    # ------------------------------------------------------------------

    def dct_transform(self, trajectory: torch.Tensor):
        """Apply DCT and return smoothed trajectory + sampled frequency index.

        Args:
            trajectory: (B, H, Da)

        Returns:
            out       : (B, H, Da)  DCT-filtered trajectory (idct of masked coeffs)
            core_index: (B,)        number of retained frequency components
        """
        B, H, D = trajectory.shape
        chosen_cores = torch.randint(0, H + 1, (B,))
        core_index = chosen_cores.to(trajectory.device, dtype=torch.float32)

        traj_t = trajectory.transpose(1, 2).to(torch.float64)          # (B, D, H)
        dct_coeffs = torch_dct.dct(traj_t, norm='ortho')               # (B, D, H)

        freq_idx = torch.arange(H, device=trajectory.device).view(1, 1, H)
        thresh = core_index.view(B, 1, 1)
        dct_mask = (freq_idx < thresh).float().expand(B, D, H)

        masked = dct_coeffs * dct_mask
        out = torch_dct.idct(masked, norm='ortho').to(trajectory.dtype) # (B, D, H)
        return out.transpose(1, 2), core_index                          # (B, H, D)

    def full_dct(self, trajectory: torch.Tensor) -> torch.Tensor:
        """Return full DCT coefficients (no masking), shape (B, H, Da)."""
        traj_t = trajectory.transpose(1, 2).to(torch.float64)
        coeffs = torch_dct.dct(traj_t, norm='ortho').to(trajectory.dtype)
        return coeffs.transpose(1, 2)

    def full_idct(self, coeffs: torch.Tensor) -> torch.Tensor:
        """Inverse DCT, shape (B, H, Da) → (B, H, Da)."""
        c_t = coeffs.transpose(1, 2).to(torch.float64)
        out = torch_dct.idct(c_t, norm='ortho').to(coeffs.dtype)
        return out.transpose(1, 2)

    def _build_action_phase_loss_weight(self, actions_gt: torch.Tensor):
        cfg = self.action_phase_loss_params
        if not cfg or not bool(cfg.get('enabled', False)):
            return None

        action_dim = actions_gt.shape[-1]
        gripper_indices = [
            int(index) for index in cfg.get('gripper_indices', [action_dim - 1])
        ]
        invalid = [
            index for index in gripper_indices
            if index < 0 or index >= action_dim
        ]
        if invalid:
            raise ValueError(
                f"action_phase_loss gripper_indices invalid: {invalid}"
            )

        close_threshold = float(cfg.get('close_threshold', 0.0))
        close_is_greater = bool(cfg.get('close_is_greater', True))
        transition_radius = max(0, int(cfg.get('transition_radius', 2)))
        device = actions_gt.device

        gripper = actions_gt.index_select(
            -1,
            torch.as_tensor(gripper_indices, device=device),
        ).mean(dim=-1)
        if close_is_greater:
            close = gripper >= close_threshold
        else:
            close = gripper <= close_threshold

        prev_close = close.clone()
        prev_close[:, 1:] = close[:, :-1]

        def expand_transition(transition, radius):
            radius = max(0, int(radius))
            if radius <= 0:
                return transition
            expanded = transition.clone()
            for offset in range(1, radius + 1):
                expanded[:, offset:] = expanded[:, offset:] | transition[:, :-offset]
                expanded[:, :-offset] = expanded[:, :-offset] | transition[:, offset:]
            return expanded

        close_transition = close & (~prev_close)
        close_transition = expand_transition(close_transition, transition_radius)

        release_enabled = bool(cfg.get('release_enabled', False))
        release_transition = None
        if release_enabled:
            release_radius = max(
                0,
                int(cfg.get('release_transition_radius', transition_radius)),
            )
            release_transition = (~close) & prev_close
            release_transition = expand_transition(release_transition, release_radius)

        weight = torch.zeros_like(actions_gt)

        def add_group(transition, indices, group_weight):
            if transition is None or not transition.any():
                return
            indices = [int(index) for index in indices]
            invalid_group = [
                index for index in indices
                if index < 0 or index >= action_dim
            ]
            if invalid_group:
                raise ValueError(
                    f"action_phase_loss group indices invalid: {invalid_group}"
                )
            if not indices or float(group_weight) <= 0:
                return
            index_tensor = torch.as_tensor(indices, device=device)
            group_mask = transition.to(dtype=actions_gt.dtype).unsqueeze(-1)
            update = group_mask.expand(-1, -1, len(indices)) * float(group_weight)
            current = weight.index_select(-1, index_tensor)
            weight.index_copy_(
                -1,
                index_tensor,
                current + update,
            )

        if close_transition.any():
            add_group(
                close_transition,
                cfg.get('translation_indices', [0, 1, 2]),
                cfg.get('translation_weight', 1.0),
            )
            add_group(
                close_transition,
                cfg.get('rotation_indices', [3, 4, 5, 6, 7, 8]),
                cfg.get('rotation_weight', 1.0),
            )
            add_group(close_transition, gripper_indices, cfg.get('gripper_weight', 1.0))

        if release_enabled and release_transition is not None and release_transition.any():
            add_group(
                release_transition,
                cfg.get(
                    'release_translation_indices',
                    cfg.get('translation_indices', [0, 1, 2]),
                ),
                cfg.get(
                    'release_translation_weight',
                    cfg.get('translation_weight', 1.0),
                ),
            )
            add_group(
                release_transition,
                cfg.get(
                    'release_rotation_indices',
                    cfg.get('rotation_indices', [3, 4, 5, 6, 7, 8]),
                ),
                cfg.get(
                    'release_rotation_weight',
                    cfg.get('rotation_weight', 1.0),
                ),
            )
            add_group(
                release_transition,
                gripper_indices,
                cfg.get('release_gripper_weight', cfg.get('gripper_weight', 1.0)),
            )

        extra_windows = cfg.get('extra_windows', None)
        if extra_windows:
            if hasattr(extra_windows, 'values'):
                window_iter = extra_windows.values()
            else:
                window_iter = extra_windows
            horizon = actions_gt.shape[1]

            def select_occurrence(events, occurrence):
                if occurrence is None:
                    return events
                if isinstance(occurrence, str):
                    occurrence = occurrence.lower()
                    if occurrence in ('all', 'any'):
                        return events
                    if occurrence == 'first':
                        occurrence = 1
                    elif occurrence == 'second':
                        occurrence = 2
                    elif occurrence == 'third':
                        occurrence = 3
                    elif occurrence == 'last':
                        counts = events.to(torch.int64).cumsum(dim=1)
                        totals = events.to(torch.int64).sum(dim=1, keepdim=True)
                        return events & (counts == totals) & (totals > 0)
                    else:
                        occurrence = int(occurrence)
                occurrence = int(occurrence)
                if occurrence <= 0:
                    raise ValueError(
                        f"action_phase_loss occurrence must be positive, got {occurrence}"
                    )
                counts = events.to(torch.int64).cumsum(dim=1)
                return events & (counts == occurrence)

            for window_cfg in window_iter:
                if not window_cfg or not bool(window_cfg.get('enabled', True)):
                    continue
                # Prefer event-anchored windows for multi-stage tasks. A fixed
                # fraction of a randomly sampled training chunk is not a task
                # phase and can otherwise teach the hang motion before grasp.
                anchor = str(window_cfg.get('anchor', 'fixed')).lower()
                if anchor in ('close', 'gripper_close'):
                    anchor_event = close & (~prev_close)
                elif anchor in ('release', 'gripper_release'):
                    anchor_event = (~close) & prev_close
                elif anchor in ('closed', 'gripper_closed', 'hold'):
                    anchor_event = close
                elif anchor == 'fixed':
                    anchor_event = None
                else:
                    raise ValueError(
                        f"action_phase_loss window anchor invalid: {anchor!r}"
                    )

                if anchor_event is not None:
                    anchor_event = select_occurrence(
                        anchor_event,
                        window_cfg.get('occurrence', None),
                    )
                    before = max(0, int(window_cfg.get('before', 0)))
                    after = max(0, int(window_cfg.get('after', 0)))
                    window = anchor_event.clone()
                    for offset in range(1, before + 1):
                        window[:, :-offset] |= anchor_event[:, offset:]
                    for offset in range(1, after + 1):
                        window[:, offset:] |= anchor_event[:, :-offset]
                    # When the sampled chunk contains no matching event, this
                    # layer deliberately contributes no auxiliary loss.
                    if not window.any():
                        continue
                else:
                    start = window_cfg.get('start', window_cfg.get('start_frac', 0.0))
                    end = window_cfg.get('end', window_cfg.get('end_frac', 1.0))
                    if isinstance(start, float) and 0.0 <= start <= 1.0:
                        start_idx = int(round(start * horizon))
                    else:
                        start_idx = int(start)
                    if isinstance(end, float) and 0.0 <= end <= 1.0:
                        end_idx = int(round(end * horizon))
                    else:
                        end_idx = int(end)
                    start_idx = max(0, min(horizon, start_idx))
                    end_idx = max(start_idx, min(horizon, end_idx))
                    if end_idx <= start_idx:
                        continue
                    window = torch.zeros(
                        actions_gt.shape[:2],
                        dtype=torch.bool,
                        device=device,
                    )
                    window[:, start_idx:end_idx] = True
                add_group(
                    window,
                    window_cfg.get(
                        'translation_indices',
                        cfg.get('translation_indices', [0, 1, 2]),
                    ),
                    window_cfg.get('translation_weight', 0.0),
                )
                add_group(
                    window,
                    window_cfg.get(
                        'rotation_indices',
                        cfg.get('rotation_indices', [3, 4, 5, 6, 7, 8]),
                    ),
                    window_cfg.get('rotation_weight', 0.0),
                )
                add_group(
                    window,
                    window_cfg.get('gripper_indices', gripper_indices),
                    window_cfg.get('gripper_weight', 0.0),
                )

        if not (weight > 0).any():
            return None

        return weight

    def _frequency_prior(
        self,
        coeffs: torch.Tensor,
        freq_split_low: int,
        freq_split_high: int,
        sigma: float,
        sigma_high: float,
    ) -> torch.Tensor:
        """Apply one frequency-band warm-start schedule."""
        horizon = coeffs.shape[1]
        fl = min(max(0, int(freq_split_low)), horizon)
        fh = min(max(fl, int(freq_split_high)), horizon)
        mid = coeffs[:, fl:fh] + sigma * torch.randn_like(coeffs[:, fl:fh])
        if sigma_high < 0:
            pre = torch.randn_like(coeffs[:, :fl])
            post = torch.randn_like(coeffs[:, fh:])
        else:
            pre = coeffs[:, :fl] + sigma_high * torch.randn_like(coeffs[:, :fl])
            post = coeffs[:, fh:] + sigma_high * torch.randn_like(coeffs[:, fh:])
        return torch.cat([pre, mid, post], dim=1)

    def spectral_prior(self, prev_actions: torch.Tensor) -> torch.Tensor:
        """Build a full-DCT prior with optional action-group parameters."""
        coeffs = self.full_dct(prev_actions)
        groups = self.action_group_spectral_params
        if not groups:
            return self._frequency_prior(
                coeffs,
                self.freq_split_low,
                self.freq_split_high,
                self.sigma,
                self.sigma_high,
            )

        prior = torch.empty_like(coeffs)
        assigned = set()
        for name, raw_cfg in groups.items():
            cfg = dict(raw_cfg)
            indices = [int(i) for i in cfg["indices"]]
            bad = [i for i in indices if i < 0 or i >= self.trajectory_dim]
            if bad:
                raise ValueError(f"group {name!r} has invalid action indices: {bad}")
            overlap = assigned.intersection(indices)
            if overlap:
                raise ValueError(
                    f"group {name!r} overlaps action indices: {sorted(overlap)}"
                )
            assigned.update(indices)
            prior[:, :, indices] = self._frequency_prior(
                coeffs[:, :, indices],
                cfg.get("freq_split_low", self.freq_split_low),
                cfg.get("freq_split_high", self.freq_split_high),
                cfg.get("sigma", self.sigma),
                cfg.get("sigma_high", self.sigma_high),
            )

        remaining = sorted(set(range(self.trajectory_dim)).difference(assigned))
        if remaining:
            prior[:, :, remaining] = self._frequency_prior(
                coeffs[:, :, remaining],
                self.freq_split_low,
                self.freq_split_high,
                self.sigma,
                self.sigma_high,
            )
        return prior

    # ------------------------------------------------------------------
    # MAE encoder / decoder (unchanged API from Freqpolicy)
    # ------------------------------------------------------------------

    def forward_mae_encoder(
        self,
        x: torch.Tensor,
        condition_embedding: torch.Tensor,
        mask=None,
        index=None,
        time_emb=None,
    ) -> torch.Tensor:
        """Encode a (possibly masked) token sequence.

        Args:
            x                 : (B, seq_len, token_embed_dim)
            condition_embedding: (B, n_obs_steps [+1], encoder_embed_dim)
            mask              : (B, seq_len) float mask (1 = masked, 0 = visible)
            index             : (B,) DCT frequency index
            time_emb          : (B, encoder_embed_dim) flow-time embedding

        Returns:
            (B, n_visible, encoder_embed_dim)
        """
        x = self.z_proj(x)                                  # (B, L, E)
        bsz, seq_len, embed_dim = x.shape

        # prepend buffer slots for condition tokens
        x = torch.cat([torch.zeros(bsz, self.buffer_size, embed_dim, device=x.device), x], dim=1)
        if mask is not None:
            mask_with_buffer = torch.cat(
                [torch.zeros(bsz, self.buffer_size, device=x.device), mask], dim=1
            )
        else:
            mask_with_buffer = None

        # inject DCT frequency index as extra condition token
        if index is not None:
            idx_emb = self.embedding_index(index.unsqueeze(-1).unsqueeze(-1))  # (B, 1, E)
            condition_embedding = torch.cat([condition_embedding, idx_emb], dim=-2)

        # inject flow-matching time embedding into buffer position 0
        if time_emb is not None:
            x[:, 0] = x[:, 0] + time_emb

        x[:, :self.buffer_size] = condition_embedding
        x = x + self.encoder_pos_embed_learned
        x = self.z_proj_ln(x)

        if self.mask and mask_with_buffer is not None:
            x = x[(1 - mask_with_buffer).nonzero(as_tuple=True)].reshape(bsz, -1, embed_dim)

        for blk in self.encoder_blocks:
            x = blk(
                x,
                attention_mask=None,
                encoder_hidden_states=condition_embedding,
            )
        return self.encoder_norm(x)

    def forward_mae_decoder(
        self,
        x: torch.Tensor,
        condition_embedding: torch.Tensor,
        mask=None,
        index=None,
    ) -> torch.Tensor:
        """Decode encoder output back to full sequence length.

        Returns:
            (B, seq_len, decoder_embed_dim)
        """
        x = self.decoder_embed(x)
        bsz = x.size(0)

        if mask is not None:
            mask_with_buffer = torch.cat(
                [torch.zeros(bsz, self.buffer_size, device=x.device), mask], dim=1
            )
        else:
            mask_with_buffer = None

        if index is not None:
            idx_emb = self.embedding_index(index.unsqueeze(-1).unsqueeze(-1))
            condition_embedding = torch.cat([condition_embedding, idx_emb], dim=-2)

        if self.mask and mask_with_buffer is not None:
            mask_tokens = self.mask_token.expand(
                mask_with_buffer.shape[0], mask_with_buffer.shape[1], -1
            ).to(x.dtype)
            x_padded = mask_tokens.clone()
            x_padded[(1 - mask_with_buffer).nonzero(as_tuple=True)] = x.reshape(
                x.shape[0] * x.shape[1], x.shape[2]
            )
            x = x_padded + self.decoder_pos_embed_learned
        else:
            x = x + self.decoder_pos_embed_learned

        for blk in self.decoder_blocks:
            x = blk(
                x,
                attention_mask=None,
                encoder_hidden_states=condition_embedding,
            )
        x = self.decoder_norm(x)
        x = x[:, self.buffer_size:]                 # strip buffer
        x = x + self.diffusion_pos_embed_learned
        return x                                    # (B, seq_len, D_dec)

    # ------------------------------------------------------------------
    # Velocity network (called by TorchFlowMatcher)
    # ------------------------------------------------------------------

    def velocity_fn(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        global_cond: torch.Tensor,
        mask=None,
        index=None,
    ) -> torch.Tensor:
        """Predict velocity v_θ(x_t, t | obs_cond).

        Args:
            x_t       : (B, H, Da)  noisy trajectory in DCT space
            t         : (B,)        flow time ∈ [0, 1]
            global_cond: (B, obs_steps * obs_dim) flattened observation condition
            mask      : (B, seq_len) optional MAE mask
            index     : (B,) DCT frequency index

        Returns:
            v_pred : (B, H, Da)  predicted velocity in DCT space
        """
        B = x_t.shape[0]

        # --- condition embedding ---
        # Reshape global_cond from (B, To*obs_dim) → (B, To, obs_dim) so each
        # observation step becomes an independent cross-attention key/value token.
        # This makes full use of the buffer_size slots in the encoder/decoder.
        To = self.n_obs_steps
        obs_dim = global_cond.shape[-1] // To
        cond = global_cond.view(B, To, obs_dim)   # (B, To, obs_dim)
        cond_emb = self.condition_proj(cond)       # (B, To, E)

        # --- time embedding ---
        t_emb = sinusoidal_time_embed(t, self.time_embed_dim)  # (B, E)
        t_emb = self.time_proj(t_emb)                          # (B, E)

        # --- encode ---
        enc_out = self.forward_mae_encoder(
            x_t, cond_emb, mask=mask, index=index, time_emb=t_emb
        )

        # --- decode ---
        dec_out = self.forward_mae_decoder(enc_out, cond_emb, mask=mask, index=index)
        # dec_out: (B, seq_len, D_dec)

        # --- velocity head ---
        v_pred = self.flow_head(dec_out)   # (B, H, Da)
        return v_pred

    # ------------------------------------------------------------------
    # Training forward
    # ------------------------------------------------------------------

    def forward(
        self,
        flow_matcher,
        actions_gt: torch.Tensor,
        prev_actions: torch.Tensor,
        global_cond: torch.Tensor,
        cold_start_prob: float = 0.2,
        fresh_action_indices=None,
    ) -> torch.Tensor:
        """Compute flow-matching loss.

        Args:
            flow_matcher    : TorchFlowMatcher instance.
            actions_gt      : (B, H, Da) ground-truth normalised actions.
            prev_actions    : (B, H, Da) previous-frame normalised actions (warm-start).
                              Pass zeros for the first batch item when unavailable.
            global_cond     : (B, To * obs_dim) flattened observation condition.
            cold_start_prob : Probability of replacing warm-start with N(0,I) to
                              match the first-step inference distribution.

        Returns:
            loss : scalar tensor.
        """
        B = actions_gt.shape[0]

        # x_1 — target in full DCT space
        x_1 = self.full_dct(actions_gt)

        # Every action dimension stays in DCT space. Only the prior schedule
        # differs by action group when action_group_spectral_params is set.
        x_0_warm = self.spectral_prior(prev_actions)
        if fresh_action_indices:
            x_0_warm[:, :, fresh_action_indices] = torch.randn_like(
                x_0_warm[:, :, fresh_action_indices]
            )
        x_0_cold = torch.randn_like(x_1)
        cold_mask = (torch.rand(B, device=x_1.device) < cold_start_prob).view(B, 1, 1)
        x_0 = torch.where(cold_mask, x_0_cold, x_0_warm)

        # optional MAE mask
        mask = None
        if self.mask:
            orders = self.sample_orders(actions_gt.shape[0])
            mask = self.random_masking(x_1, orders)

        def _vel_fn(x_t, t, **kw):
            return self.velocity_fn(x_t, t, global_cond=global_cond, mask=mask)

        aux_loss_weight = self._build_action_phase_loss_weight(actions_gt)
        aux_loss_scale = 0.0
        if aux_loss_weight is not None:
            aux_loss_scale = float(
                self.action_phase_loss_params.get('weight', 0.0)
            )

        loss, _ = flow_matcher.compute_loss(
            model=_vel_fn,
            target=x_1,
            start=x_0,
            loss_groups=self.action_group_loss_weights,
            aux_loss_weight=aux_loss_weight,
            aux_loss_transform=self.full_idct,
            aux_loss_scale=aux_loss_scale,
        )
        return loss

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def sample(
        self,
        flow_matcher,
        prev_actions,
        global_cond: torch.Tensor,
        num_steps: int = 6,
        fresh_action_indices=None,
    ) -> torch.Tensor:
        """Generate action trajectory via Euler ODE in DCT space.

        Args:
            flow_matcher : TorchFlowMatcher instance.
            prev_actions : (B, H, Da) or None.  None → standard-normal x_0.
            global_cond  : (B, To * obs_dim) observation condition.
            num_steps    : Number of Euler integration steps.

        Returns:
            actions : (B, H, Da) predicted actions in original (time) domain.
        """
        B, H, Da = global_cond.shape[0], self.horizon, self.trajectory_dim
        device = global_cond.device

        if prev_actions is None:
            x_0 = torch.randn(B, H, Da, device=device)
        else:
            x_0 = self.spectral_prior(prev_actions)
            if fresh_action_indices:
                x_0[:, :, fresh_action_indices] = torch.randn_like(
                    x_0[:, :, fresh_action_indices]
                )

        def _vel_fn(x_t, t, **kw):
            return self.velocity_fn(x_t, t, global_cond=global_cond)

        x_1 = flow_matcher.sample(
            model=_vel_fn,
            shape=(B, H, Da),
            device=device,
            num_steps=num_steps,
            start=x_0,
        )

        # convert back to time domain
        return self.full_idct(x_1)
