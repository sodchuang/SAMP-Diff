"""Flow matcher implementations.

Adapted from A2A Flow Matching (roboverse_learn/il/utils/flow/flow_matchers.py),
re-packaged for the SAMP_Diff_v1 pipeline that uses Freqpolicy's MAE Transformer
as the velocity network backbone instead of SimpleFlowNet.

Dependency: torchcfm  (pip install torchcfm)
"""
import numpy as np
import torch
import torchcfm.conditional_flow_matching as cfm

from diffusion_policy.utils.flow.base_flow_matcher import BaseFlowMatcher


def group_balanced_mse(squared_error: torch.Tensor, groups=None) -> torch.Tensor:
    """Average within each action group, then average the group losses."""
    if not groups:
        return squared_error.mean()

    action_dim = squared_error.shape[-1]
    assigned = set()
    weighted_losses = []
    total_weight = 0.0

    for name, raw_cfg in groups.items():
        cfg = dict(raw_cfg)
        indices = [int(index) for index in cfg["indices"]]
        if not indices:
            raise ValueError(f"loss group {name!r} has no action indices")

        invalid = [index for index in indices if index < 0 or index >= action_dim]
        if invalid:
            raise ValueError(
                f"loss group {name!r} has invalid action indices: {invalid}"
            )
        overlap = assigned.intersection(indices)
        if overlap:
            raise ValueError(
                f"loss group {name!r} overlaps action indices: {sorted(overlap)}"
            )

        weight = float(cfg.get("weight", 1.0))
        if weight < 0:
            raise ValueError(f"loss group {name!r} has negative weight: {weight}")
        assigned.update(indices)
        if weight == 0:
            continue

        index_tensor = torch.as_tensor(indices, device=squared_error.device)
        group_loss = squared_error.index_select(-1, index_tensor).mean()
        weighted_losses.append(group_loss * weight)
        total_weight += weight

    remaining = sorted(set(range(action_dim)).difference(assigned))
    if remaining:
        index_tensor = torch.as_tensor(remaining, device=squared_error.device)
        weighted_losses.append(squared_error.index_select(-1, index_tensor).mean())
        total_weight += 1.0

    if total_weight <= 0:
        raise ValueError("at least one loss group must have a positive weight")
    return torch.stack(weighted_losses).sum() / total_weight


class TorchFlowMatcher(BaseFlowMatcher):
    """Generic wrapper around a torchcfm flow-matching object.

    The wrapped ``self.fm`` object must expose::

        sample_location_and_conditional_flow(x0, x1)
            -> (t, x_t, u_t)

    where
      - t  : sampled time  (B,)   ∈ [0, 1]
      - x_t: interpolated point
      - u_t: target velocity (x_1 - x_0 for straight CFM)
    """

    def __init__(self, fm, num_sampling_steps: int = 6):
        self.fm = fm
        self.num_sampling_steps = num_sampling_steps

    def compute_loss(
        self,
        model,
        target: torch.Tensor,
        start=None,
        loss_groups=None,
        aux_loss_weight=None,
        aux_loss_transform=None,
        aux_loss_scale: float = 1.0,
        **kwargs,
    ):
        """Compute flow-matching training loss.

        Args:
            model   : Callable ``(x_t, t, **kwargs) -> v_t``.
            target  : Ground-truth end-point x_1, shape ``(B, ...)``.
            start   : Optional source distribution x_0.  When *None* a
                      standard-normal sample is used (vanilla CFM); when
                      provided this implements the A2A warm-start strategy.
            **kwargs: Forwarded to ``model`` (e.g. ``global_cond``).

        Returns:
            ``(loss, {'loss': float})``
        """
        x0 = torch.randn_like(target) if start is None else start
        t, x_t, u_t = self.fm.sample_location_and_conditional_flow(x0, target)
        v_t = model(x_t, t, **kwargs)
        error = v_t - u_t
        loss = group_balanced_mse(error ** 2, loss_groups)

        aux_loss = None
        if aux_loss_weight is not None:
            aux_error = error
            if aux_loss_transform is not None:
                aux_error = aux_loss_transform(aux_error)
            aux_weight = aux_loss_weight.to(
                device=aux_error.device,
                dtype=aux_error.dtype,
            )
            aux_weight_sum = aux_weight.sum()
            if aux_weight_sum > 0:
                aux_loss = ((aux_error ** 2) * aux_weight).sum() / aux_weight_sum
                loss = loss + float(aux_loss_scale) * aux_loss

        info = {'loss': loss.item()}
        if aux_loss is not None:
            info['aux_loss'] = aux_loss.item()
        return loss, info

    def sample(
        self,
        model,
        shape,
        device: torch.device,
        num_steps: int = None,
        return_traces: bool = False,
        start=None,
        **kwargs,
    ):
        """Euler-method ODE integration from x_0 to x_1.

        Args:
            model       : Callable ``(x_t, t, **kwargs) -> v_t``.
            shape       : Output shape ``(B, ...)``.
            device      : Target device.
            num_steps   : Number of Euler steps; defaults to
                          ``self.num_sampling_steps``.
            return_traces: If True, also return (traj_history, vel_history).
            start       : Optional warm-start tensor x_0.  When *None* a
                          fresh standard-normal sample is drawn.
            **kwargs    : Forwarded to ``model`` at each step.

        Returns:
            ``x`` (final sample), or ``(x, (traj_history, vel_history))``
            when ``return_traces=True``.
        """
        if num_steps is None:
            num_steps = self.num_sampling_steps
        x = torch.randn(shape, device=device) if start is None else start
        dt = 1.0 / num_steps

        if return_traces:
            traj_history = [x.detach().clone().cpu()]
            vel_history = [np.zeros_like(x.cpu().numpy())]

        for step in range(num_steps):
            t = torch.ones(x.shape[0], device=device) * (step / num_steps)
            v_t = model(x, t, **kwargs)
            x = x + v_t * dt

            if return_traces:
                traj_history.append(x.detach().clone().cpu())
                vel_history.append(v_t.detach().clone().cpu().numpy())

        if return_traces:
            return x, (traj_history, vel_history)
        return x


# ---------------------------------------------------------------------------
# Convenience subclasses — thin wrappers that instantiate the underlying
# torchcfm object and forward all remaining kwargs to it.
# ---------------------------------------------------------------------------

class ConditionalFlowMatcher(TorchFlowMatcher):
    """Standard conditional flow matcher (straight paths)."""

    def __init__(self, num_sampling_steps: int = 6, **kwargs):
        super().__init__(cfm.ConditionalFlowMatcher(**kwargs), num_sampling_steps)


class TargetConditionalFlowMatcher(TorchFlowMatcher):
    """Target-conditional flow matcher."""

    def __init__(self, num_sampling_steps: int = 6, **kwargs):
        super().__init__(cfm.TargetConditionalFlowMatcher(**kwargs), num_sampling_steps)


class SchrodingerBridgeConditionalFlowMatcher(TorchFlowMatcher):
    """Schrödinger-bridge conditional flow matcher."""

    def __init__(self, num_sampling_steps: int = 6, **kwargs):
        super().__init__(
            cfm.SchrodingerBridgeConditionalFlowMatcher(**kwargs),
            num_sampling_steps,
        )


class ExactOptimalTransportConditionalFlowMatcher(TorchFlowMatcher):
    """Exact OT conditional flow matcher."""

    def __init__(self, num_sampling_steps: int = 6, **kwargs):
        super().__init__(
            cfm.ExactOptimalTransportConditionalFlowMatcher(**kwargs),
            num_sampling_steps,
        )
