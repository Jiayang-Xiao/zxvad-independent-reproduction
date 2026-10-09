"""Optimizer-state invariants which fail if coverage or updates are incomplete."""


def parameter_snapshot(models):
    return {key: [p.detach().clone() for p in models[key].parameters()]
            for key in ("N", "Arc")}


def finish_update(metrics, models, opts, step, before=None):
    import torch
    expected = int(step) + 1
    counts, entries = {}, {}
    ownership = {"G": ("G",), "D": ("D",), "N": ("N", "Arc")}
    for key, model_keys in ownership.items():
        opt = opts[key]
        parameters = [p for group in opt.param_groups for p in group["params"]]
        expected_parameters = {
            id(parameter): model_key + "." + name
            for model_key in model_keys
            for name, parameter in models[model_key].named_parameters()}
        identities = [id(parameter) for parameter in parameters]
        actual_ids = set(identities)
        expected_ids = set(expected_parameters)
        duplicate_count = len(identities) - len(actual_ids)
        missing = sorted(expected_parameters[identity]
                         for identity in expected_ids - actual_ids)
        foreign_count = len(actual_ids - expected_ids)
        if missing or foreign_count or duplicate_count:
            raise RuntimeError(
                f"Optimizer {key} parameter identity coverage mismatch "
                f"(missing={len(missing)}, foreign={foreign_count}, "
                f"duplicates={duplicate_count}); missing_names={missing}")
        if not all(parameter.requires_grad for parameter in parameters):
            raise RuntimeError("Optimizer parameters remain frozen: " + key)
        states = [opt.state.get(parameter, {}) for parameter in parameters]
        if not all("step" in state for state in states):
            raise RuntimeError("Optimizer did not initialize all parameter states: " + key)
        steps = sorted({
            int(state["step"].item()) if torch.is_tensor(state["step"])
            else int(state["step"]) for state in states})
        if steps != [expected]:
            raise RuntimeError(
                f"Optimizer {key} did not step exactly once: {steps}, expected{expected}")
        counts[key] = expected
        entries[key] = len(states)
    result = dict(metrics)
    result["adam_steps"] = counts
    result["adam_state_entries"] = entries
    if before is not None:
        for key in ("N", "Arc"):
            changed = any(not torch.equal(old, parameter.detach())
                          for old, parameter in zip(before[key], models[key].parameters()))
            if not changed:
                raise RuntimeError(key + " parameters did not change in disposable audit update")
            result[key + "_parameters_changed"] = True
    return result

