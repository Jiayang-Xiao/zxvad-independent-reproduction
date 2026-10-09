"""Source-only input interventions with a private, step-addressed random stream.

The future frame is left byte-for-byte unchanged during training.  ``probe=True``
also edits the observed fifth frame, solely for a held-out source diagnostic.
No target-domain frame or label is accepted by this module.
"""
import math
import numpy as np
import torch

MODES = ('PHOTO', 'NOISE', 'SELF', 'STATIC', 'MOVE', 'SHUFFLE')


def _trajectory(rng, height, width, bh, bw, probe):
    # Probe uses a held-out turn and up-to-six-pixel displacement; training uses
    # a straight up-to-four-pixel path.  Content/activation draws remain paired
    # across modes within the corresponding training or diagnostic stream.
    limit = 6 if probe else 4
    candidates = []
    for dx in range(-limit, limit + 1):
        for dy in range(-limit, limit + 1):
            if not (dx or dy):
                continue
            offsets = [(0, 0), (dx, dy), (2*dx, 2*dy),
                       (2*dx-dy, 2*dy+dx), (2*dx-2*dy, 2*dy+2*dx)] if probe else \
                      [(t*dx, t*dy) for t in range(5)]
            xs, ys = zip(*offsets)
            if max(xs)-min(xs) <= width-bw and max(ys)-min(ys) <= height-bh:
                candidates.append(offsets)
    offsets = candidates[int(rng.randint(len(candidates)))] if candidates else [(0, 0)]*5
    xs, ys = zip(*offsets)
    xlo, xhi = -min(xs), width-bw-max(xs)
    ylo, yhi = -min(ys), height-bh-max(ys)
    x, y = int(rng.randint(xlo, xhi+1)), int(rng.randint(ylo, yhi+1))
    return [(x+ox, y+oy) for ox, oy in offsets]


@torch.no_grad()
def edit_clip(clip, donor, mode, area, p, step, probe=False):
    """Return ``(edited_clip, JSON-compatible stats, masks)``.

    ``clip`` is B x 5 x 3 x H x W and ``donor`` is B x 3 x H x W.
    Each active example receives one full rectangular patch per edited frame.
    Geometric draws, activation uniforms and all content parameters are shared
    across modes at fixed step/area/probability; only the stated intervention
    changes.  MOVE/SHUFFLE use identical position sets in a different order.
    """
    if mode not in MODES:
        raise ValueError(f'Unknown input edit: {mode}')
    if clip.ndim != 5 or clip.shape[1:3] != (5, 3):
        raise ValueError('Expected five RGB frames per clip')
    if donor.shape != clip[:, 0].shape:
        raise ValueError('Donor must match the first-frame batch shape')
    if not (0 < area < 1 and 0 <= p <= 1) or int(step) != step or step < 0:
        raise ValueError('Invalid intervention area, probability or step')
    b, _, _, h, w = clip.shape
    bh = min(h, max(1, int(round(h * math.sqrt(area)))))
    bw = min(w, max(1, int(round(w * math.sqrt(area)))))
    rng = np.random.RandomState((17 + 1000003 * (int(step) + 1) + 1700033) % (2 ** 32))
    # Draw the whole activation vector first; p changes its threshold only.
    active = rng.uniform(size=b) < p
    edited, masks = clip.clone(), torch.zeros_like(clip[:, :, :1])
    positions, sources, gains, offsets = [], [], [], []
    edit_frames = 5 if probe else 4
    for i in range(b):
        path = _trajectory(rng, h, w, bh, bw, probe)
        sx, sy = int(rng.randint(w - bw + 1)), int(rng.randint(h - bh + 1))
        gain, offset = float(rng.uniform(.6, 1.4)), float(rng.uniform(-.2, .2))
        # Draw every content parameter for every mode.  Neither the NumPy global
        # RNG nor the Torch RNG is consumed by an intervention.
        noise = torch.as_tensor(rng.normal(0, .2, size=(3, bh, bw)),
                                device=clip.device, dtype=clip.dtype)
        order = [0, 2, 1, 3, 4] if mode == 'SHUFFLE' else list(range(5))
        actual = [path[j] for j in order]
        if mode in ('PHOTO', 'NOISE', 'STATIC'):
            actual = [path[0]] * 5
        positions.append([[x, y, x + bw, y + bh] for x, y in actual])
        sources.append([sx, sy, sx + bw, sy + bh]); gains.append(gain); offsets.append(offset)
        if not active[i]:
            continue
        source = clip[i, 0] if mode == 'SELF' else donor[i]
        patch = source[:, sy:sy + bh, sx:sx + bw].clone()
        for t in range(edit_frames):
            x, y = actual[t]
            region = clip[i, t, :, y:y + bh, x:x + bw]
            if mode == 'PHOTO':
                value = (gain * region + offset).clamp(-1, 1)
            elif mode == 'NOISE':
                value = (region + noise).clamp(-1, 1)
            else:
                value = patch
            edited[i, t, :, y:y + bh, x:x + bw] = value
            masks[i, t, :, y:y + bh, x:x + bw] = 1
    stats = {
        'edit_mode': mode, 'edit_probability': float(p), 'requested_area': float(area),
        'rectangle_area': float(bh * bw / (h * w)),
        'active_examples': int(active.sum()), 'batch_examples': int(b),
        'active_fraction': float(active.mean()), 'edited_frames': edit_frames,
        'mask_area': float(masks[:, :edit_frames].mean()),
        'positions': positions, 'source_boxes': sources,
        'gain': gains, 'offset': offsets, 'active': active.tolist(),
        'future_target_unchanged': bool(torch.equal(edited[:, 4], clip[:, 4])) if not probe else None,
        'content_source': 'recipient_first_frame' if mode == 'SELF' else
                          ('recipient_current_region' if mode in ('PHOTO', 'NOISE') else 'donor_first_frame'),
        'private_rng_seed': int((17 + 1000003 * (int(step) + 1) + 1700033) % (2 ** 32)),
    }
    return edited, stats, masks
