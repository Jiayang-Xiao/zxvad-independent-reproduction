"""Fast CPU-only update regressions; these fixtures are not research results.

Run with the project's pinned Python environment: python regression.py.
This file writes no experiment artifacts and never allocates a CUDA tensor.
"""
from pathlib import Path
import copy
import json
import random
import sys
import time
sys.dont_write_bytecode = True

ROOT = Path(__file__).resolve().parent
# Only delivered project modules are added; site packages belong to the runtime.
sys.path.insert(0, str(ROOT / "frozen_original" / "src"))
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import baseline_model as base
import experiment_update as update

torch.set_num_threads(1)


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def exact(left, right, path="state"):
    """Compare nested model/Adam/RNG states, including tensors and NumPy arrays."""
    if torch.is_tensor(left):
        require(torch.is_tensor(right) and left.dtype == right.dtype
                and left.shape == right.shape and torch.equal(left, right),
                "Exact tensor parity failed: " + path)
    elif isinstance(left, np.ndarray):
        require(isinstance(right, np.ndarray) and left.dtype == right.dtype
                and np.array_equal(left, right), "Exact array parity failed: " + path)
    elif isinstance(left, dict):
        require(isinstance(right, dict) and left.keys() == right.keys(),
                "Dictionary keys differ: " + path)
        for key in left:
            exact(left[key], right[key], path + "/" + str(key))
    elif isinstance(left, (tuple, list)):
        require(type(left) is type(right) and len(left) == len(right),
                "Sequence differs: " + path)
        for index, (a, b) in enumerate(zip(left, right)):
            exact(a, b, path + "/" + str(index))
    else:
        require(type(left) is type(right) and left == right,
                "Exact scalar parity failed: " + path)


def rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state().clone()}


def reset_rng():
    random.seed(1909)
    np.random.seed(1909)
    torch.manual_seed(1909)


class TinyG(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(12, 3, 1)
        self.address = nn.Parameter(torch.ones(4))
        self.calls = 0
        self.outputs = []

    def forward(self, x):
        self.calls += 1
        pred = self.conv(x).tanh()
        self.outputs.append(pred.detach().clone())
        return pred, self.address.softmax(0).expand(x.shape[0], -1)


class TinyPatch(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 2, 1)
        self.bn = nn.BatchNorm2d(2)
        self.score = nn.Conv2d(2, 1, 1)
        self.inputs = []

    def forward(self, x):
        self.inputs.append(x.detach().clone())
        hidden = F.leaky_relu(self.bn(self.conv(x)), .2)
        return self.score(hidden), hidden


class TinyObject(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(()), requires_grad=False)

    def forward(self, x):
        mask = torch.zeros_like(x[:, :1])
        mask[:, :, 4:12, 4:12] = 1
        return mask


class TinyArc(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(256, 2)

    def forward(self, x, labels):
        return F.cross_entropy(self.linear(x), labels)


class TinyLoss(nn.Module):
    def forward(self, pred, target, address):
        mse = F.mse_loss(pred, target)
        zero = mse * 0
        entropy = (-address * (address + 1e-12).log()).sum(1).mean()
        return mse + .0025 * entropy, {
            "mse": mse, "ssim_loss": zero, "gradient": zero,
            "memory_entropy": zero}


class TinyAug(nn.Module):
    """Exercise the real updater's Torch RNG fork with a small random transform."""
    def forward(self, x):
        gain = .8 + .2 * torch.rand((x.shape[0], 1, 1, 1), device=x.device)
        return (x * gain + .02 * torch.rand_like(x)).clamp(0, 1)


def tiny_models():
    # Construct fixtures without consuming the updater's global RNG streams.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(321)
        models = {"G": TinyG(), "D": TinyPatch(), "N": TinyPatch(),
                  "O": TinyObject(), "Arc": TinyArc()}
    models["O"].eval()
    opts = {
        "G": torch.optim.Adam(models["G"].parameters(), lr=.0002, betas=(.5, .999)),
        "D": torch.optim.Adam(models["D"].parameters(), lr=.00002, betas=(.5, .999)),
        "N": torch.optim.Adam(list(models["N"].parameters())
                              + list(models["Arc"].parameters()),
                              lr=.00002, betas=(.5, .999))}
    return models, opts, TinyLoss(), TinyAug()


def parameters(model):
    return [p.detach().clone() for p in model.parameters()]


def changed(before, model):
    return any(not torch.equal(a, b.detach())
               for a, b in zip(before, model.parameters()))


def full_state(fixture):
    models, opts = fixture[:2]
    return copy.deepcopy({
        "models": {key: model.state_dict() for key, model in models.items()},
        "optimizers": {key: opt.state_dict() for key, opt in opts.items()},
        "gradients": {key: [None if p.grad is None else p.grad.detach().clone()
                            for p in model.parameters()]
                      for key, model in models.items()},
        "requires_grad": {key: [p.requires_grad for p in model.parameters()]
                          for key, model in models.items()},
        "training": {key: [module.training for module in model.modules()]
                     for key, model in models.items()},
        "rng": rng_state()})


def adam_steps(opts, step):
    expected = step + 1
    for key, opt in opts.items():
        params = [p for group in opt.param_groups for p in group["params"]]
        require(len(opt.state) == len(params), "Adam state count differs: " + key)
        for parameter in params:
            state = opt.state.get(parameter, {})
            require("step" in state, "Missing Adam parameter step: " + key)
            value = state["step"]
            count = int(value.item()) if torch.is_tensor(value) else int(value)
            require(count == expected,
                    f"Adam step differs: {key}, {count}, expected {expected}")


def check_update(fixture, clip, donor, arm, step, original=False):
    models, opts = fixture[:2]
    old_n = parameters(models["N"])
    old_arc = parameters(models["Arc"])
    old_o = copy.deepcopy(models["O"].state_dict())
    old_calls = models["G"].calls
    old_bn = int(models["N"].bn.num_batches_tracked)
    before_rng = rng_state()
    old_clip = clip.detach().clone()
    old_donor = donor.detach().clone()
    if original:
        metrics = base.one_update(clip, donor, "c04", step, *fixture, audit=True)
    else:
        metrics = update.one_update(clip, donor, arm, step, *fixture, audit=True)
        require(metrics["adam_steps"] == {"G": step + 1, "D": step + 1,
                                          "N": step + 1},
                "Updater optimizer receipt differs: " + arm)
        require(metrics["N_parameters_changed"]
                and metrics["Arc_parameters_changed"],
                "Missing N/Arc parameter-change receipt: " + arm)
    adam_steps(opts, step)
    require(changed(old_n, models["N"]), "N weights unchanged: " + arm)
    require(changed(old_arc, models["Arc"]), "Arc weights unchanged: " + arm)
    require(models["G"].calls == old_calls + 1, "Not one G forward: " + arm)
    require(int(models["N"].bn.num_batches_tracked) == old_bn + 4
            and metrics["n_bn_updates"] == 4 * (step + 1)
            and len(models["N"].inputs) == 4 * (step + 1),
            "Not four N BN passes per update: " + arm)
    exact(before_rng, rng_state(), arm + "/global_rng_preserved")
    exact(old_clip, clip, arm + "/recipient_input_unchanged")
    exact(old_donor, donor, arm + "/donor_input_unchanged")
    exact(old_o, models["O"].state_dict(), arm + "/O_frozen")
    require(all(not p.requires_grad and p.grad is None
                for p in models["O"].parameters()), "O gradient changed: " + arm)
    require(torch.equal(models["D"].inputs[3 * step + 1], clip[:, 4]),
            "D real target changed: " + arm)
    # Prove the declared source routing directly, rather than assuming that
    # different sample origins must create different rounded Adam parameters.
    actual_prediction = models["G"].outputs[step]
    guide_input = models["N"].inputs[4 * step]
    exact(guide_input, actual_prediction, arm + "/guidance_uses_actual_G_prediction")
    normal_input = models["N"].inputs[4 * step + 1]
    negative_input = models["N"].inputs[4 * step + 2]
    expected_normal = actual_prediction
    if arm == "NC_Y_s17":
        expected_normal = clip[:, 4]
    elif arm == "NC_X_s17":
        expected_normal = clip[:, 0]
    elif arm == "NC_MIX50_s17":
        expected_normal = actual_prediction.clone()
        expected_normal[clip.shape[0] // 2:] = clip[:, 4][clip.shape[0] // 2:]
    exact(normal_input, expected_normal, arm + "/normal_image_source")
    paired = arm in ("NC_Y_s17", "NC_X_s17", "NC_PRED_s17", "NC_MIX50_s17")
    with torch.no_grad():
        if arm == "NC_ORIGIN_s17":
            expected_negative = clip[:, 0]
        else:
            paste_base = expected_normal if paired else clip[:, 0]
            expected_negative, _, _ = base.paste(paste_base, donor, models["O"](donor), step)
    exact(negative_input, expected_negative, arm + "/negative_image_source_and_paste")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(17 + step + 700001)
        expected_augmented = fixture[3](((expected_normal + 1) / 2).clamp(0, 1)).clamp(0, 1) * 2 - 1
    exact(models["N"].inputs[4 * step + 3], expected_augmented,
          arm + "/augmented_normal_image_source")
    if not original and arm not in ("B0_s17", "B1_s17", "B_s17"):
        require(metrics["guidance_coefficient"] == (0. if arm == "NC_OFF_s17" else .5),
                "Wrong normalcy-to-generator guidance coefficient: " + arm)
    if "input_edit" in metrics:
        require(metrics["input_edit"]["future_target_unchanged"] is True,
                "CF future target changed: " + arm)
    exact(before_rng, rng_state(), arm + "/validation_helpers_preserve_global_rng")
    return metrics


def baseline_parity(clips, donors):
    reset_rng()
    reference = tiny_models()
    expected = []
    for step, (clip, donor) in enumerate(zip(clips, donors)):
        metrics = check_update(reference, clip, donor, "original", step, True)
        expected.append((metrics, full_state(reference)))
    for arm in ("B0_s17", "B1_s17", "B_s17"):
        reset_rng()
        candidate = tiny_models()
        for step, (clip, donor) in enumerate(zip(clips, donors)):
            metrics = check_update(candidate, clip, donor, arm, step)
            # Original loss receipt is a strict subset of the guarded receipt.
            exact(expected[step][0], {key: metrics[key]
                                     for key in expected[step][0]},
                  arm + "/original_metrics/" + str(step))
            exact(expected[step][1], full_state(candidate),
                  arm + "/full_original_parity/" + str(step))
    return expected


def all_arms(clips, donors):
    retained = {}
    for arm in update.ORDER:
        reset_rng()
        fixture = tiny_models()
        trajectory = []
        for step, (clip, donor) in enumerate(zip(clips, donors)):
            check_update(fixture, clip, donor, arm, step)
            if arm in ("NC_Y_s17", "NC_X_s17"):
                trajectory.append({key: parameters(fixture[0][key])
                                   for key in ("G", "N")})
        if trajectory:
            retained[arm] = trajectory
    # Cross-arm divergence is a diagnostic, not an implementation invariant.
    # A normalized Adam update can round to identical FP32 parameters even
    # when N has learned and the upstream guidance differs.
    rows = []
    for step in range(3):
        row = {"updates": step + 1, "max_absolute_parameter_delta": {}, "parameters_exact": {}}
        for key in ("G", "N"):
            left = retained["NC_Y_s17"][step][key]
            right = retained["NC_X_s17"][step][key]
            row["max_absolute_parameter_delta"][key] = max(float((a - b).abs().max()) for a, b in zip(left, right))
            row["parameters_exact"][key] = all(torch.equal(a, b) for a, b in zip(left, right))
        rows.append(row)
    return rows


def controlled_guidance_influence(clip, donor):
    """Large deliberate N-logit change tests its causal G-gradient influence.

    NC_OFF is the exact negative control. This is a CPU toy unit test and is
    never applied to research models, hyperparameters, data or checkpoints.
    """
    gradients = {}
    for arm in ("NC_Y_s17", "NC_OFF_s17"):
        for perturb in (False, True):
            reset_rng()
            fixture = tiny_models()
            if perturb:
                with torch.no_grad():
                    fixture[0]["N"].score.bias.add_(1.0)
            check_update(fixture, clip, donor, arm, 0)
            gradients[(arm, perturb)] = [None if p.grad is None else p.grad.detach().clone()
                                         for p in fixture[0]["G"].parameters()]
    exact(gradients[("NC_OFF_s17", False)], gradients[("NC_OFF_s17", True)],
          "guidance_off_G_gradient_unchanged_under_N_logit_shift")
    left, right = gradients[("NC_Y_s17", False)], gradients[("NC_Y_s17", True)]
    require(all((a is None) == (b is None) for a, b in zip(left, right)),
            "Guidance perturbation changed gradient topology")
    deltas = [float((a - b).abs().max()) for a, b in zip(left, right) if a is not None]
    require(deltas and all(torch.isfinite(a).all() and torch.isfinite(b).all()
                          for a, b in zip(left, right) if a is not None),
            "Nonfinite controlled guidance gradients")
    maximum = max(deltas)
    require(maximum > 1e-6, "Deliberate N-logit shift did not affect active G guidance gradient")
    return {"status": "PASS", "kind": "CONTROLLED_CPU_TOY_N_LOGIT_SHIFT_NOT_RESEARCH_EDIT",
            "active_G_gradient_max_absolute_change": maximum,
            "disabled_G_gradients_exactly_unchanged": True}



def zero_probability_parity(clips, donors, expected):
    saved_config = update.CONFIG
    # Never mutate the shared protocol or public CONFIG mapping in place.
    private_config = copy.deepcopy(saved_config)
    for recipe in update.CF_RECIPES:
        private_config[recipe]["cf"]["p"] = 0.
    update.CONFIG = private_config
    try:
        for recipe in update.CF_RECIPES:
            arm = recipe + "_s17"
            reset_rng()
            fixture = tiny_models()
            for step, (clip, donor) in enumerate(zip(clips, donors)):
                metrics = check_update(fixture, clip, donor, arm, step)
                edit = metrics["input_edit"]
                require(edit["active_examples"] == 0 and edit["mask_area"] == 0,
                        "Zero-probability edit was active: " + arm)
                exact(expected[step][0], {key: metrics[key]
                                         for key in expected[step][0]},
                      arm + "/p0_metrics/" + str(step))
                exact(expected[step][1], full_state(fixture),
                      arm + "/p0_full_original_parity/" + str(step))
    finally:
        update.CONFIG = saved_config
    require(update.CONFIG is saved_config, "Private p=0 CONFIG was not restored")


def skipped_n_step_mutation(clips, donors):
    """Reproduce the released defect in memory; production code is never edited."""
    arms = ("B0_s17", "NC_Y_s17", "NC_OFF_s17", "CF_MOVE_A10_P25_s17")
    errors = {}
    for arm in arms:
        reset_rng()
        fixture = tiny_models()
        old_n = parameters(fixture[0]["N"])
        old_arc = parameters(fixture[0]["Arc"])
        fixture[1]["N"].step = lambda *args, **kwargs: None
        try:
            # audit=False proves the every-update guard catches the defect,
            # independently of optional expensive parameter snapshots.
            update.one_update(clips[0], donors[0], arm, 0, *fixture, audit=False)
        except RuntimeError as error:
            text = str(error)
            require(("Optimizer did not initialize all parameter states: N" in text)
                    or ("Optimizer N did not step exactly once" in text),
                    "Mutation failed for an unrelated reason: " + text)
            require(not changed(old_n, fixture[0]["N"])
                    and not changed(old_arc, fixture[0]["Arc"]),
                    "Mutation unexpectedly changed N/Arc parameters")
            require(int(fixture[0]["N"].bn.num_batches_tracked) == 4,
                    "Mutation did not reproduce BN-only N updates")
            errors[arm] = text
        else:
            raise RuntimeError("Skipped N optimizer step was accepted: " + arm)
    # Also reject a skipped step after Adam state already exists.
    arm = "NC_Y_s17"
    reset_rng()
    fixture = tiny_models()
    check_update(fixture, clips[0], donors[0], arm, 0)
    fixture[1]["N"].step = lambda *args, **kwargs: None
    try:
        update.one_update(clips[1], donors[1], arm, 1, *fixture, audit=False)
    except RuntimeError as error:
        require("Optimizer N did not step exactly once" in str(error),
                "Initialized-state mutation failed for an unrelated reason")
        errors[arm + "/second_step"] = str(error)
    else:
        raise RuntimeError("Skipped second N optimizer step was accepted")
    return errors


def optimizer_coverage_mutations(clips, donors):
    """Missing, duplicate and foreign identities must fail the runtime guard."""
    cases = (
        ("missing_G_parameter", "G", "missing", "G", "conv.weight"),
        ("missing_D_parameter", "D", "missing", "D", "conv.weight"),
        ("missing_N_parameter", "N", "missing", "N", "conv.weight"),
        ("missing_Arc_parameter", "N", "missing", "Arc", "linear.weight"),
        ("duplicate_N_parameter", "N", "duplicates", "N", "conv.weight"),
        ("foreign_N_parameter", "N", "foreign", None, None))
    errors = {}
    for name, optimizer_key, fault, model_key, parameter_name in cases:
        reset_rng()
        fixture = tiny_models()
        group = fixture[1][optimizer_key].param_groups[0]
        if fault == "foreign":
            group["params"].append(nn.Parameter(torch.ones(1)))
        else:
            parameter = dict(fixture[0][model_key].named_parameters())[parameter_name]
            if fault == "missing":
                group["params"] = [p for p in group["params"] if p is not parameter]
            else:
                group["params"].append(parameter)
        try:
            update.one_update(clips[0], donors[0], "NC_Y_s17", 0,
                              *fixture, audit=False)
        except RuntimeError as error:
            message = str(error)
            require(f"Optimizer {optimizer_key} parameter identity coverage mismatch"
                    in message and fault + "=1" in message,
                    "Coverage mutation failed for an unrelated reason: " + message)
            errors[name] = message
        else:
            raise RuntimeError("Optimizer coverage mutation was accepted: " + name)
    return errors



def main():
    started = time.perf_counter()
    require(len(update.ORDER) == 32 and len(update.CF_RECIPES) == 24,
            "Regression must cover the complete declared schedule")
    generator = torch.Generator(device="cpu").manual_seed(1909)
    clips = [torch.rand((8, 5, 3, 16, 16), generator=generator) * 2 - 1
             for _ in range(3)]
    donors = [torch.rand((8, 3, 16, 16), generator=generator) * 2 - 1
              for _ in range(3)]
    expected = baseline_parity(clips, donors)
    trajectories = all_arms(clips, donors)
    guidance = controlled_guidance_influence(clips[0], donors[0])
    zero_probability_parity(clips, donors, expected)
    errors = skipped_n_step_mutation(clips, donors)
    errors.update(optimizer_coverage_mutations(clips, donors))
    result = {
        "status": "PASS",
        "kind": "CPU_TINY_UPDATE_REGRESSION_NOT_RESEARCH_RESULTS",
        "planned_arms": 32, "consecutive_updates_per_arm": 3,
        "zero_probability_CF_arms": 24, "torch_threads": torch.get_num_threads(),
        "checks": [
            "all_G_D_N_Adam_parameter_steps_1_2_3",
            "N_and_Arc_actual_parameter_change_each_update",
            "one_G_forward_four_N_BN_passes_each_update",
            "original_baseline_exact_models_optimizers_gradients_RNG_three_steps",
            "all_CF_p0_exact_original_full_state_three_steps_CONFIG_restored",
            "all_four_N_inputs_exact_actual_G_positive_negative_augmented_source_routes",
            "controlled_N_logit_shift_affects_active_G_gradient_and_not_NC_OFF",
            "NC_Y_X_trajectory_difference_recorded_without_pass_fail_assumption",
            "O_frozen_D_future_target_preserved_CF_clean_future_preserved",
            "fresh_and_initialized_skipped_N_step_mutations_rejected_without_audit",
            "optimizer_identity_coverage_missing_G_D_N_Arc_duplicate_foreign_rejected"],
        "mutation_rejections": errors,
        "NC_Y_X_trajectory_diagnostic": trajectories,
        "controlled_guidance_influence": guidance,
        "runtime": {"python": sys.version.split()[0], "torch": torch.__version__, "numpy": np.__version__},
        "wall_seconds": round(time.perf_counter() - started, 3),
        "gpu_training_executed": False, "experiment_artifacts_written": False}
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(json.dumps({"status": "FAIL",
                          "kind": "CPU_TINY_UPDATE_REGRESSION_NOT_RESEARCH_RESULTS",
                          "error": str(error)}, ensure_ascii=False), file=sys.stderr)
        raise



