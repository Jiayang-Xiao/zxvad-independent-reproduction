"""Disposable, source-only three-step GPU parity for the new updater.

The zero-probability CF arm deliberately exercises the new nonbaseline branch.
The direct reference and wrapped path must agree after every complete update,
including N/Arc optimization.  This module never writes a file or a checkpoint.
"""
import copy
import gc
import hashlib
import os
from pathlib import Path


ARM = 'CF_PHOTO_A10_P25_s17'
RECIPE = 'CF_PHOTO_A10_P25'
STEPS = 3
BATCH = 8
KIND = 'DISPOSABLE_SOURCE_GPU_OPTIMIZER_REGRESSION'
COMPARISON = 'direct frozen c04 versus new CF_PHOTO_A10_P25 with private p=0'


def _require(condition, message):
    if not condition:
        raise RuntimeError('Optimizer regression: ' + message)


def binding(pipeline):
    """No preflight hash: the caller will bind this receipt into preflight."""
    prepared_path = pipeline.OUT / 'prepared.json'
    prepared = pipeline.read_json(prepared_path)
    preflight = pipeline.read_json(pipeline.preflight_path())
    lane = str(pipeline.LANE)
    _require(lane in ('0', '1'), 'explicit physical-card lane required')
    _require(preflight['physical_gpu'] == int(lane), 'preflight physical card changed')
    return {
        'study': pipeline.STUDY,
        'prepared_sha256': pipeline.sha(prepared_path),
        'source_plan_sha256': pipeline.sha(pipeline.OUT / 'source_plan.npy'),
        'source_split_sha256': pipeline.sha(pipeline.OUT / 'source_split.json'),
        'release': prepared['release'],
        'physical_gpu': int(lane),
        'hardware_identity': preflight['environment']['hardware_identity'],
    }


def validate_receipt(receipt, pipeline):
    """Validate a committed source gate before reuse; never rewrite it."""
    _require(receipt.get('binding') == binding(pipeline), 'receipt binding changed')
    _require(receipt.get('status') == 'PASS' and receipt.get('kind') == KIND,
             'receipt is not a successful real-GPU regression')
    _require(receipt.get('steps') == STEPS and receipt.get('batch_size') == BATCH
             and receipt.get('seed') == 17 and receipt.get('comparison') == COMPARISON,
             'receipt comparison/budget changed')
    _require(receipt.get('regression_arm') == ARM and receipt.get('private_edit_probability') == 0.0, 'private zero-probability comparison changed')
    _require(receipt.get('source_only') is True and receipt.get('target_used') is False
             and receipt.get('disposable') is True
             and receipt.get('checkpoint_written') is False
             and receipt.get('models_discarded') is True,
             'receipt must describe disposable source-only updates')
    _require(receipt.get('full_state_exact') is True
             and receipt.get('original_metrics_exact') is True,
             'receipt parity was not exact')
    rows = receipt.get('source_plan_first8')
    _require(isinstance(rows, list) and len(rows) == BATCH
             and all(isinstance(row, list) and len(row) == 4 for row in rows),
             'receipt source panel is incomplete')
    import numpy as np
    plan = np.load(pipeline.OUT / 'source_plan.npy', allow_pickle=False)
    _require(rows == plan[:BATCH].tolist(), 'receipt source panel changed')
    digest = receipt.get('source_batch_sha256')
    _require(isinstance(digest, str) and len(digest) == 64
             and all(character in '0123456789abcdef' for character in digest),
             'missing source pixel-batch digest')
    records = receipt.get('step_records', [])
    _require(isinstance(records, list) and len(records) == STEPS,
             'three per-step records required')
    for step, record in enumerate(records):
        _require(record.get('step') == step and record.get('N_BN_updates') == 4 * (step + 1),
                 'recorded step or BN count changed')
        left = record.get('direct_full_state_sha256')
        right = record.get('wrapped_full_state_sha256')
        _require(isinstance(left, str) and len(left) == 64
                 and all(character in '0123456789abcdef' for character in left)
                 and left == right and record.get('full_state_exact') is True
                 and record.get('original_metrics_exact') is True,
                 'per-step full model/Adam/Torch RNG parity failed')
        for mode in ('direct', 'wrapped'):
            _require(record.get(mode + '_adam_steps') == {'G': step + 1, 'D': step + 1, 'N': step + 1},
                     'per-step optimizer counters changed')
            _require(record.get(mode + '_N_parameters_changed') is True
                     and record.get(mode + '_Arc_parameters_changed') is True,
                     'N/Arc parameter update was not established')
        keys = record.get('original_metric_keys')
        _require(isinstance(keys, list) and keys and all(isinstance(key, str) for key in keys),
                 'original metric comparison missing')
    _require(receipt.get('full_state_sha256') == records[-1]['direct_full_state_sha256'],
             'final state digest changed')
    return receipt


def _three_updates(pipeline, clip, donor, wrapped):
    """Sequential fresh models keep peak memory near one ordinary training fit."""
    import torch
    import baseline_model as base
    import experiment_update as update
    from parity import state_digest
    from train_audit import parameter_snapshot, finish_update

    models = opts = loss_fn = aug = before = None
    records = []
    try:
        pipeline.seed_everything(17)
        if wrapped:
            models, opts, loss_fn, aug = update.make_models(ARM, clip.device)
        else:
            models, opts, loss_fn, aug = base.make_models('c04', clip.device)
        for step in range(STEPS):
            if wrapped:
                # The wrapped update already snapshots N/Arc and audits all Adam counters.
                metrics = update.one_update(clip, donor, ARM, step, models, opts,
                                            loss_fn, aug, audit=True)
            else:
                before = parameter_snapshot(models)
                original = base.one_update(clip, donor, 'c04', step, models, opts,
                                           loss_fn, aug, audit=True)
                metrics = finish_update(original, models, opts, step, before)
                before = None
            _require(metrics['n_bn_updates'] == 4 * (step + 1), 'four N BN passes per step required')
            if clip.is_cuda:
                torch.cuda.synchronize(clip.device)
            records.append({'step': step, 'metrics': metrics,
                            'state_sha256': state_digest(models, opts)})
        return records
    finally:
        del before, models, opts, loss_fn, aug
        gc.collect()
        if clip.is_cuda:
            torch.cuda.empty_cache()


def run(pipeline):
    """Reuse a bound receipt or run three disposable updates on one source batch.

    The caller owns atomic receipt writing and preflight binding.  Targets and
    research checkpoint paths are never opened by this gate.
    """
    import numpy as np
    import torch
    import experiment_update as update

    lane = str(pipeline.LANE)
    _require(torch.cuda.is_available(), 'real CUDA required; CPU checks cannot create a GPU receipt')
    _require(os.environ.get('CUDA_VISIBLE_DEVICES') == lane,
             'only the assigned physical card may be visible')
    current_binding = binding(pipeline)
    receipt_path = pipeline.OUT / 'workers' / ('gpu' + lane) / 'optimizer_regression.json'
    if receipt_path.exists():
        return validate_receipt(pipeline.read_json(receipt_path), pipeline)

    prepared = pipeline.read_json(pipeline.OUT / 'prepared.json')
    _require(prepared['plan_sha256'] == current_binding['source_plan_sha256'], 'source plan binding changed')
    split = pipeline.read_json(pipeline.OUT / 'source_split.json')
    train, heldout = set(split['training_indices']), set(split['validation_indices'])
    _require(len(train) == 297 and len(heldout) == 33 and not train & heldout
             and train | heldout == set(range(330)), '297/33 source isolation required')
    plan = np.load(pipeline.OUT / 'source_plan.npy', allow_pickle=False)
    _require(plan.shape == (40000, 4) and np.issubdtype(plan.dtype, np.integer), 'frozen 40000-row plan required')
    panel = plan[:BATCH]
    _require(set(panel[:, 0]) <= train and set(panel[:, 2]) <= train,
             'regression recipients and donors must come from the 297 training videos')
    source = pipeline.catalog(prepared['data']['source_frames'])
    _require(len(source) == 330, 'canonical source catalogue required')
    for recipient, start, donor_index, donor_start in panel:
        _require(0 <= start <= len(source[recipient][1]) - 5
                 and 0 <= donor_start < len(source[donor_index][1]), 'source panel frame outside video')
    dataset = pipeline.SourceClips(source, plan)
    clips, donors = zip(*(dataset[index] for index in range(BATCH)))
    clip_array, donor_array = np.stack(clips), np.stack(donors)
    batch_hash = hashlib.sha256(clip_array.tobytes() + donor_array.tobytes()).hexdigest()
    device = torch.device('cuda:0')
    clip = torch.from_numpy(clip_array).to(device)
    donor = torch.from_numpy(donor_array).to(device)

    # Keep the published configuration object and every nested value untouched.
    original_config = update.CONFIG
    private_config = copy.deepcopy(original_config)
    _require(RECIPE in private_config and private_config[RECIPE]['cf']['mode'] == 'PHOTO',
             'declared nonbaseline regression arm required')
    private_config[RECIPE]['cf']['p'] = 0.0
    prior_rng = pipeline.rng_state()
    try:
        update.CONFIG = private_config
        direct = _three_updates(pipeline, clip, donor, wrapped=False)
        wrapped = _three_updates(pipeline, clip, donor, wrapped=True)
        step_records = []
        helper_keys = {'adam_steps', 'adam_state_entries', 'N_parameters_changed', 'Arc_parameters_changed'}
        for reference, candidate in zip(direct, wrapped):
            step = reference['step']
            original_keys = sorted(set(reference['metrics']) - helper_keys)
            _require(all(key in candidate['metrics'] and reference['metrics'][key] == candidate['metrics'][key]
                         for key in original_keys), 'original metrics differ at step ' + str(step))
            _require(reference['state_sha256'] == candidate['state_sha256'],
                     'full model/Adam/Torch RNG state differs at step ' + str(step))
            record = {'step': step, 'N_BN_updates': reference['metrics']['n_bn_updates'],
                      'direct_full_state_sha256': reference['state_sha256'],
                      'wrapped_full_state_sha256': candidate['state_sha256'],
                      'full_state_exact': True, 'original_metrics_exact': True,
                      'original_metric_keys': original_keys}
            for mode, value in (('direct', reference), ('wrapped', candidate)):
                for key in ('adam_steps', 'adam_state_entries', 'N_parameters_changed', 'Arc_parameters_changed'):
                    record[mode + '_' + key] = value['metrics'][key]
            step_records.append(record)
        receipt = {'status': 'PASS', 'kind': KIND, 'binding': current_binding,
                   'steps': STEPS, 'batch_size': BATCH, 'seed': 17,
                   'comparison': COMPARISON, 'regression_arm': ARM,
                   'private_edit_probability': 0.0,
                   'source_plan_first8': panel.tolist(), 'source_batch_sha256': batch_hash,
                   'full_state_sha256': step_records[-1]['direct_full_state_sha256'],
                   'full_state_exact': True, 'original_metrics_exact': True,
                   'step_records': step_records,
                   'source_only': True, 'target_used': False, 'disposable': True,
                   'checkpoint_written': False, 'models_discarded': True,
                   'state_scope': 'All G/D/N/O/Arc state_dict tensors, all Adam states, CPU and visible CUDA Torch RNG',
                   'scope': 'First eight frozen297-plan samples repeated for three disposable updates; not research training'}
        return validate_receipt(receipt, pipeline)
    finally:
        update.CONFIG = original_config
        pipeline.restore_rng(prior_rng)
        del dataset, clip, donor, clip_array, donor_array
        gc.collect()
        torch.cuda.empty_cache()
