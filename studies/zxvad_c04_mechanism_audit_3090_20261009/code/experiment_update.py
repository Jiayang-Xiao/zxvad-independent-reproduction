"""Fixed c04 architecture: one normalcy or input-intervention family per arm.

The c04 baseline aliases delegate directly to the frozen original updater.
Variants retain a single G forward and four N BatchNorm passes per step.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
import baseline_model as base
from edits import MODES, edit_clip
from spec import CONFIG, ORDER, SEEDS, parse, settings

RECIPES = tuple(CONFIG)
NC_RECIPES = tuple(recipe for recipe in RECIPES if recipe.startswith('NC_'))
CF_RECIPES = tuple(recipe for recipe in RECIPES if recipe.startswith('CF_'))


def make_models(arm, device):
    parse(arm)
    return base.make_models('c04', device)


def make_generator(arm, device):
    parse(arm)
    return base.Generator('dot').to(device)


def _cf_config(recipe):
    config = CONFIG[recipe]['cf']
    return config['mode'], config['area'], config['p']


def one_update(clip, donor, arm, step, models, opts, loss_fn, aug, audit=False):
    recipe, _ = parse(arm)
    if recipe in ('B', 'B0', 'B1'):
        return base.one_update(clip, donor, 'c04', step, models, opts, loss_fn, aug, audit=audit)
    G, D, N, O, arc = (models[k] for k in ('G', 'D', 'N', 'O', 'Arc'))
    target = clip[:, 4]
    edit_stats = None
    if recipe in CF_RECIPES:
        mode, area, probability = _cf_config(recipe)
        input_clip, edit_stats, _ = edit_clip(clip, donor, mode, area, probability, step)
    else:
        input_clip = clip
    pred, address = G(input_clip[:, :4].flatten(1, 2))
    with torch.no_grad():
        omask = O(donor)
        # The default c04 pseudo-frame uses the clean recipient X0 and donor.
        # Input-intervention arms never alter this normalcy construction.
        normal_image = pred.detach()
        pseudo, mask, boxes = base.paste(clip[:, 0], donor, omask, step)
        if recipe == 'NC_Y':
            normal_image = target.detach()
        elif recipe == 'NC_X':
            normal_image = clip[:, 0].detach()
        elif recipe == 'NC_MIX50':
            if clip.shape[0] % 2:
                raise ValueError('NC_MIX50 requires an even batch size')
            normal_image = pred.detach().clone()
            normal_image[clip.shape[0] // 2:] = target.detach()[clip.shape[0] // 2:]
        if recipe in ('NC_Y', 'NC_X', 'NC_PRED', 'NC_MIX50'):
            # Same paste seed, O support and box draws; the negative is paired
            # with exactly the positive image rather than a different base.
            pseudo, mask, boxes = base.paste(normal_image, donor, omask, step)
        elif recipe == 'NC_ORIGIN':
            pseudo, mask = clip[:, 0].detach().clone(), torch.zeros_like(mask)
    D.requires_grad_(False); N.requires_grad_(False)
    n_pred, _ = N(pred)
    d_pred, _ = D(pred)
    bb, parts = loss_fn(pred, target, address)
    guidance = .5 * F.mse_loss(n_pred, torch.ones_like(n_pred))
    guidance_coefficient = 0. if recipe == 'NC_OFF' else .5
    lg = bb + .05 * (.5 * F.mse_loss(d_pred, torch.ones_like(d_pred))) + guidance_coefficient * guidance
    guidance_norm = None
    if audit:
        grads = torch.autograd.grad(guidance, tuple(G.parameters()), retain_graph=True, allow_unused=True)
        guidance_norm = sum(g.abs().sum().item() for g in grads if g is not None)
        if not (guidance_norm > 0 and math.isfinite(guidance_norm)):
            raise RuntimeError('Raw N-to-G gradient disconnected')
    opts['G'].zero_grad(set_to_none=True); lg.backward(); opts['G'].step()
    D.requires_grad_(True); N.requires_grad_(True)
    real, _ = D(target); fake, _ = D(pred.detach())
    ld = .5 * F.mse_loss(real, torch.ones_like(real)) + .5 * F.mse_loss(fake, torch.zeros_like(fake))
    opts['D'].zero_grad(set_to_none=True); ld.backward(); opts['D'].step()
    normal, normal_feature = N(normal_image)
    abnormal, abnormal_feature = N(pseudo)
    attention_mode = base.CANDIDATES['c04']['attention']
    a = base.classifier_attention(normal, normal_feature, attention_mode)
    ap = base.classifier_attention(abnormal, abnormal_feature, attention_mode)
    ng = .5 * F.mse_loss(normal, torch.ones_like(normal)) + .5 * F.mse_loss(abnormal, torch.zeros_like(abnormal))
    rn = .5 * F.mse_loss(normal - abnormal.mean(), torch.ones_like(normal)) + .5 * F.mse_loss(abnormal - normal.mean(), -torch.ones_like(abnormal))
    resized_mask = F.interpolate(mask, a.shape[-2:], mode='nearest')
    aa = .5 * F.mse_loss(a, torch.ones_like(a)) + .5 * F.mse_loss(ap, resized_mask)
    devices = [clip.device.index] if clip.is_cuda else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(17 + step + 700001)
        if clip.is_cuda:
            torch.cuda.manual_seed_all(17 + step + 700001)
        transformed = aug(((normal_image + 1) / 2).clamp(0, 1)).clamp(0, 1) * 2 - 1
    aug_logits, aug_feature = N(transformed)
    av = base.classifier_attention(aug_logits, aug_feature, attention_mode)
    labels = torch.cat((torch.ones(2 * clip.shape[0], device=clip.device, dtype=torch.long),
                        torch.zeros(clip.shape[0], device=clip.device, dtype=torch.long)))
    raa = arc(torch.cat((a.flatten(1), av.flatten(1), ap.flatten(1))), labels)
    ln = ng + .01 * rn + aa + raa
    if audit:
        grads = torch.autograd.grad(ln, tuple(G.parameters()), retain_graph=True, allow_unused=True)
        if not all(g is None for g in grads):
            raise RuntimeError('N update connected to G')
        if pseudo.requires_grad or mask.requires_grad or any(p.grad is not None for p in O.parameters()):
            raise RuntimeError('Detached pseudo-label invariant failed')
        expected_base = normal_image if recipe in ('NC_Y', 'NC_X', 'NC_PRED', 'NC_MIX50') else clip[:, 0]
        if not torch.equal((pseudo - expected_base) * (1 - mask), torch.zeros_like(pseudo)):
            raise RuntimeError('Paste altered an unmasked pixel')
        if edit_stats is not None and not edit_stats['future_target_unchanged']:
            raise RuntimeError('Input intervention altered the future training target')
    values = {'g': lg, 'd': ld, 'n': ln, 'normalcy': ng, 'relative_normalcy': rn,
              'attention_affirmation': aa, 'relative_attention': raa, **parts}
    result = {key: float(value.detach()) for key, value in values.items()}
    if not all(math.isfinite(v) for v in result.values()):
        raise RuntimeError('Nonfinite loss')
    result.update(object_mask_area=float(omask.mean()), paste_mask_area=float(mask.mean()),
                  n_attention_mean=float(a.detach().mean()),
                  n_attention_zero_fraction=float((a.detach() == 0).float().mean()),
                  n_bn_updates=int(next(m for m in N.modules() if isinstance(m, nn.BatchNorm2d)).num_batches_tracked),
                  boxes=boxes, guidance_coefficient=guidance_coefficient)
    if guidance_norm is not None:
        result['raw_guidance_gradient_l1'] = guidance_norm
    if edit_stats is not None:
        result['input_edit'] = edit_stats
    return result
