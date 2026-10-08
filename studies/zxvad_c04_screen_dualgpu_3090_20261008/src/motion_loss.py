"""Training-only pixel weighting. This is an ADL-inspired adaptation, not ADL reproduction."""
import math
import torch
import torch.nn.functional as F


def spatial_lower_median(d):
    # torch.median(dim=...) chooses the LOWER middle value for even lengths.
    # Sort values instead of using the CUDA median-with-indices kernel, which
    # raises under torch.use_deterministic_algorithms(True) in PyTorch 2.4.1.
    flat = d.flatten(1)
    return flat.sort(dim=1, stable=True).values[:, (flat.shape[1]-1)//2, None, None, None]


@torch.no_grad()
def motion_weights(sequence, mode, fraction=.2, local_mass=.5):
    """sequence: B,5,3,H,W in [-1,1]; all five SOURCE frames are allowed.

    Reduce RGB absolute differences; take temporal maximum, suppress spatially
    uniform changes, smooth with a reflected 5x5 average, and select the largest
    ceil(.2*H*W) values including ties. Constant maps revert to uniform weights.
    Equal global/local mass gives mean weight 1, independent of ROI area.
    """
    if mode not in {"baseline", "motion", "shifted"}:
        raise ValueError(mode)
    if sequence.ndim != 5 or sequence.shape[1:3] != (5, 3):
        raise ValueError("expected B,5,3,H,W source clips")
    d = (sequence[:, 1:] - sequence[:, :-1]).abs().mean(2).amax(1, keepdim=True)
    center = spatial_lower_median(d)
    d = (d - center).clamp_min(0)
    d = F.avg_pool2d(F.pad(d, (2, 2, 2, 2), mode="reflect"), 5, stride=1)
    flat = d.flatten(1)
    k = max(1, math.ceil(fraction * flat.shape[1]))
    threshold = flat.topk(k, dim=1).values[:, -1, None, None, None]
    # Do not include zero-valued background when fewer than 20% of pixels move.
    mask = ((d >= threshold) & (d > 1e-6)).float()
    active = ((flat.amax(1) - flat.amin(1)) > 1e-6)[:, None, None, None]
    mask = torch.where(active, mask, torch.ones_like(mask))
    if mode == "shifted":
        mask = torch.roll(mask, (mask.shape[-2] // 2, mask.shape[-1] // 2), (-2, -1))
    weights = (1-local_mass) + local_mass * mask / mask.mean((-2, -1), keepdim=True)
    if mode == "baseline":
        return torch.ones_like(weights), mask
    return weights, mask


def prediction_loss(pred, target, sequence, mode):
    weights, mask = motion_weights(sequence, mode)
    loss = F.mse_loss(pred, target) if mode == "baseline" else (weights * (pred-target).square()).mean()
    return loss, weights, mask
