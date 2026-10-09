"""Frozen source diagnostics and target readouts; never updates a model on targets.

Source normal clips alone determine median/IQR calibration.  All readouts and
source interventions are declared before target access.  Synthetic source labels
are mechanism diagnostics, not an additional target performance criterion.
"""
import gc
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

import spec

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'outputs'
PROBE_AREAS = (.1,)
PROBE_BATCH = 16
PROBE_ITEMS = 256
FUSION_WEIGHT = .25
CALIBRATION_FLOOR = 1e-6
LOCAL_TILE = 8
LOCAL_FRACTION = .1


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temp.replace(path)


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def save_npz(path, arrays):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    with temp.open('wb') as stream:
        np.savez_compressed(stream, **arrays)
    temp.replace(path)


def logits(model, image):
    result = model(image)
    result = result[0] if isinstance(result, (tuple, list)) else result
    return result.flatten(1).mean(1)


def readouts(prediction, target, normalcy, prediction_logit=None, observation_logit=None):
    """Return FP32 equations first, CPU float64 arrays only after scoring.

    ``raw`` exactly retains the original negative-PSNR convention: normalized
    [0,1] residual then 10*log10(MSE).  Higher is more anomalous for every score.
    NC_GAP equals N(prediction) - N(observation); NC equals -N(observation).
    """
    pixel_residual = ((prediction - target) / 2).square()
    mse = pixel_residual.mean((1, 2, 3))
    residual = pixel_residual.mean(1, keepdim=True)
    raw = 10 * torch.log10(mse.clamp_min(1e-12))
    tiles = F.avg_pool2d(residual, LOCAL_TILE, stride=LOCAL_TILE).flatten(1)
    k = max(1, math.ceil(LOCAL_FRACTION * tiles.shape[1]))
    local = tiles.sort(dim=1, descending=True, stable=True).values[:, :k].mean(1)
    observed = logits(normalcy, target) if observation_logit is None else observation_logit
    predicted = logits(normalcy, prediction) if prediction_logit is None else prediction_logit
    return {'raw': raw, 'raw_nc': -observed, 'raw_nc_pred': -predicted,
            'raw_nc_gap': predicted - observed, 'raw_local': local, 'mse': mse}


def calibration(values):
    result = {}
    for key in ('raw', 'raw_nc', 'raw_local'):
        array = np.asarray(values[key], dtype=np.float64)
        require(array.ndim == 1 and len(array) > 0 and np.isfinite(array).all(), 'Invalid source calibration: ' + key)
        q25, q75 = np.quantile(array, [.25, .75], method='linear')
        result[key] = {'median': float(np.median(array)), 'iqr': float(max(q75 - q25, CALIBRATION_FLOOR))}
    return result


def fusion(values, calibrated):
    def z(key):
        return (values[key] - calibrated[key]['median']) / calibrated[key]['iqr']
    return z('raw') + FUSION_WEIGHT * z('raw_nc')


def arrays(values):
    return {key: value.detach().cpu().double().numpy() if torch.is_tensor(value)
            else np.asarray(value, dtype=np.float64) for key, value in values.items()}


def auc(label, score):
    label, score = np.asarray(label), np.asarray(score)
    require(label.ndim == score.ndim == 1 and label.shape == score.shape
            and np.isfinite(score).all() and np.isin(label, [0, 1]).all(), 'Invalid synthetic AUROC data')
    order = np.argsort(score, kind='stable')
    s, y = score[order], label[order]
    starts = np.r_[0, 1 + np.flatnonzero(s[1:] != s[:-1])]
    pos = np.add.reduceat(y.astype(np.float64), starts)
    neg = np.add.reduceat((1 - y).astype(np.float64), starts)
    require(pos.sum() * neg.sum() > 0, 'Both classes required for synthetic AUROC')
    return float((pos * (np.cumsum(neg) - .5 * neg)).sum() / (pos.sum() * neg.sum()))


def bn_snapshot(model):
    return {name: (module.running_mean.detach().clone(), module.running_var.detach().clone(),
                   module.num_batches_tracked.detach().clone())
            for name, module in model.named_modules() if isinstance(module, torch.nn.modules.batchnorm._BatchNorm)}


def assert_bn_unchanged(model, before):
    after = bn_snapshot(model)
    require(before.keys() == after.keys(), 'Normalcy BN structure changed')
    require(all(torch.equal(a, b) for name in before for a, b in zip(before[name], after[name])),
            'Normalcy BN buffers changed during frozen inference')


def load_models(pipeline, candidate, device, object_extractor=False):
    # G/N/O have their original architecture; constructing extra optimizers,
    # discriminator and ArcFace is unnecessary for frozen diagnostic inference.
    import baseline_model as base
    done = read(OUT / candidate / 'completed.json')
    cfg = read(OUT / candidate / 'config.json')
    path = OUT / candidate / 'last.pt'
    require(done['step'] == 5000 and sha(path) == done['checkpoint_sha256'], 'Incomplete/final checkpoint changed: ' + candidate)
    require(done['config_hash'] == pipeline.digest(cfg), 'Checkpoint config binding changed: ' + candidate)
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    require(checkpoint['step'] == 5000 and checkpoint['config_hash'] == done['config_hash'], 'Loaded checkpoint identity changed')
    models = {'G': base.Generator('dot'), 'N': base.PatchGAN()}
    if object_extractor:
        models['O'] = base.ObjectExtractor()
    for key, model in models.items():
        model.load_state_dict(checkpoint['models'][key], strict=True)
        model.requires_grad_(False)
        model.to(device).eval()
    del checkpoint
    return models, done


def source_binding(candidate):
    return {'study': spec.STUDY, 'candidate': candidate, 'checkpoint_sha256':
            read(OUT / candidate / 'completed.json')['checkpoint_sha256'],
            'prepared_sha256': sha(OUT / 'prepared.json'),
            'source_probe_plan_sha256': sha(OUT / 'source_probe_plan.npy'),
            'source_split_sha256': sha(OUT / 'source_split.json'),
            'source_only': True, 'batch_size': PROBE_BATCH, 'clips': PROBE_ITEMS,
            'probe_areas': list(PROBE_AREAS), 'probe_modes': list(spec.MODES),
            'local_tile': LOCAL_TILE, 'local_top_fraction': LOCAL_FRACTION,
            'fusion_weight': FUSION_WEIGHT, 'calibration_floor': CALIBRATION_FLOOR,
            'calibration_fit': 'clean heldout normal future, median and linear-quantile IQR only',
            'target_used': False, 'normalcy_eval': True}


def heldout_names(split):
    # Accept the declared split artifact schemas, never infer from targets.
    for key in ('validation_names', 'validation_videos', 'heldout_names', 'heldout_videos'):
        if key in split:
            return set(split[key])
    if 'validation' in split:
        return set(split['validation'])
    raise RuntimeError('Missing explicit heldout source names')


def audit_probe_plan(pipeline):
    prepared = read(OUT / 'prepared.json')
    source = pipeline.catalog(prepared['data']['source_frames'])
    plan = np.load(OUT / 'source_probe_plan.npy', allow_pickle=False)
    split = read(OUT / 'source_split.json')
    heldout = heldout_names(split)
    require(len(heldout) == 33 and plan.shape == (PROBE_ITEMS, 4) and np.issubdtype(plan.dtype, np.integer), 'Expected 33-video/256-clip source panel')
    for vi, start, donor, donor_start in plan:
        require(0 <= vi < len(source) and 0 <= donor < len(source), 'Invalid source panel video')
        require(source[vi][0] in heldout and source[donor][0] in heldout, 'Source probe recipient/donor is not heldout')
        require(0 <= start <= len(source[vi][1]) - 5 and 0 <= donor_start < len(source[donor][1]), 'Invalid source panel start')
    return source, plan


def source_probe(pipeline, lane, batch_size=16):
    require(batch_size == PROBE_BATCH, 'Frozen source batch must be 16')
    require(lane in spec.QUEUES, 'Unknown physical GPU lane')
    # The source panel also waits for every weight fit.  No target evaluation is
    # performed in this function, and the panel cannot feed back into a fit.
    require(all((OUT / arm / 'completed.json').is_file() and
                read(OUT / arm / 'completed.json')['step'] == 5000 for arm in spec.ORDER),
            'Every declared final fit is required before source diagnostics')
    source, plan = audit_probe_plan(pipeline)
    pipeline.seed_everything(17)
    device = getattr(pipeline, 'DEVICE', 'cuda:0')
    from edits import edit_clip
    import baseline_model as base
    for candidate in spec.QUEUES[lane]:
        dst = OUT / candidate
        binding = source_binding(candidate)
        summary_path, scores_path, calibration_path = dst / 'source_probe.json', dst / 'source_probe.npz', dst / 'calibration.json'
        if all(path.is_file() for path in (summary_path, scores_path, calibration_path)):
            previous = read(summary_path)
            require(previous.get('binding') == binding and previous.get('scores_sha256') == sha(scores_path)
                    and previous.get('calibration_sha256') == sha(calibration_path)
                    and sha(dst / 'last.pt') == binding['checkpoint_sha256'], 'Existing source probe binding changed: ' + candidate)
            continue
        # A crash after an atomic array/calibration write but before the final
        # receipt leaves an uncommitted panel.  Recompute that panel from the
        # same final weights; never replace a committed source receipt.
        require(not summary_path.exists(), 'Committed source probe is incomplete: ' + candidate)
        if calibration_path.exists():
            require(read(calibration_path).get('binding') == binding, 'Partial source calibration binding changed')
        pipeline.status('SOURCE_DIAGNOSTICS', candidate=candidate, clips=PROBE_ITEMS)
        models, done = load_models(pipeline, candidate, device, object_extractor=True)
        G, N, O = models['G'], models['N'], models['O']
        before = bn_snapshot(N)
        result = {}
        def extend(key, value):
            result.setdefault(key, []).extend(np.asarray(value).reshape(-1).tolist())
        with torch.inference_mode():
            for start in range(0, len(plan), batch_size):
                rows = plan[start:start + batch_size]
                clips, donors = [], []
                for vi, t, dv, dt in rows:
                    clips.append(np.stack([pipeline.read_frame(p) for p in source[vi][1][t:t + 5]]))
                    donors.append(pipeline.read_frame(source[dv][1][dt]))
                clip = torch.from_numpy(np.stack(clips)).to(device)
                donor = torch.from_numpy(np.stack(donors)).to(device)
                target = clip[:, 4]
                pred, _ = G(clip[:, :4].flatten(1, 2))
                pred_logit = logits(N, pred)
                target_logit = logits(N, target)
                normal = arrays(readouts(pred, target, N, pred_logit, target_logit))
                for key, value in normal.items():
                    extend('clean_' + key, value)
                extend('video', rows[:, 0]); extend('frame', rows[:, 1] + 4)
                copy_mse = ((clip[:, 3] - target) / 2).square().mean((1, 2, 3))
                extend('copy_last_mse', arrays({'value': copy_mse})['value'])
                reversed_clip = clip[:, :4].clone()
                # Reverse earlier observations while holding the most recent
                # input and the future fixed, so copy-last is not confounded.
                reversed_clip[:, :3] = reversed_clip[:, :3].flip(1)
                reversed_pred, _ = G(reversed_clip.flatten(1, 2))
                extend('history_output_delta', arrays({'value': ((pred - reversed_pred) / 2).square().mean((1, 2, 3))})['value'])
                extend('history_mse_delta', arrays({'value': ((reversed_pred - target) / 2).square().mean((1, 2, 3)) - ((pred - target) / 2).square().mean((1, 2, 3))})['value'])
                mask = O(donor)
                pasted_y, _, _ = base.paste(target, donor, mask, 900000 + start // batch_size)
                pasted_p, _, _ = base.paste(pred, donor, mask, 900000 + start // batch_size)
                for key, value in {'Y': -target_logit, 'P': -pred_logit, 'X': -logits(N, clip[:, 0]),
                                   'TY': -logits(N, pasted_y), 'TP': -logits(N, pasted_p)}.items():
                    extend('origin_' + key, arrays({'value': value})['value'])
                for area in PROBE_AREAS:
                    for mode in spec.MODES:
                        edited, stats, masks = edit_clip(clip, donor, mode, area, 1.,
                                                        700000 + start // batch_size, probe=True)
                        epred, _ = G(edited[:, :4].flatten(1, 2))
                        elogit = logits(N, epred)
                        clean_future = arrays(readouts(epred, target, N, elogit, target_logit))
                        edited_future = arrays(readouts(epred, edited[:, 4], N, elogit))
                        prefix = f'{mode}_A{round(100 * area)}'
                        for key, value in clean_future.items():
                            extend(prefix + '_recovery_' + key, value)
                        for key, value in edited_future.items():
                            extend(prefix + '_edited_' + key, value)
                        extend(prefix + '_output_delta', arrays({'value': ((epred - pred) / 2).square().mean((1, 2, 3))})['value'])
                        extend(prefix + '_input_sensitivity', edited_future['raw_nc'] - normal['raw_nc'])
                        extend(prefix + '_input_edit_energy', arrays({'value': ((edited[:, :4] - clip[:, :4]) / 2).square().mean((1, 2, 3, 4))})['value'])
                        extend(prefix + '_future_edit_energy', arrays({'value': ((edited[:, 4] - target) / 2).square().mean((1, 2, 3))})['value'])
                        extend(prefix + '_support_area', arrays({'value': masks[:, :4].mean((1, 2, 3, 4))})['value'])
        assert_bn_unchanged(N, before)
        result = {key: np.asarray(value, dtype=np.int64 if key in ('video', 'frame') else np.float64) for key, value in result.items()}
        require(all(value.shape == (PROBE_ITEMS,) and np.isfinite(value).all() for value in result.values()), 'Nonfinite/incomplete source panel')
        cal = calibration({key: result['clean_' + key] for key in ('raw', 'raw_nc', 'raw_local')})
        cal_record = {'binding': binding, 'statistics': cal, 'normalcy_bn_unchanged': True}
        result['clean_raw_fusion'] = fusion({key: result['clean_' + key] for key in ('raw', 'raw_nc')}, cal)
        label = np.r_[np.zeros(PROBE_ITEMS, dtype=np.int64), np.ones(PROBE_ITEMS, dtype=np.int64)]
        conditions = {}
        for area in PROBE_AREAS:
            for mode in spec.MODES:
                prefix = f'{mode}_A{round(100 * area)}'
                result[prefix + '_edited_raw_fusion'] = fusion({key: result[prefix + '_edited_' + key] for key in ('raw', 'raw_nc')}, cal)
                result[prefix + '_recovery_raw_fusion'] = fusion({key: result[prefix + '_recovery_' + key] for key in ('raw', 'raw_nc')}, cal)
                conditions[prefix] = {'kind': 'benign photometric corruption control' if mode == 'PHOTO' else 'synthetic edited future; not real anomaly',
                                     'recovery_mse': float(result[prefix + '_recovery_mse'].mean()),
                                     'recovery_ratio_to_clean': float(result[prefix + '_recovery_mse'].mean() / max(result['clean_mse'].mean(), 1e-12)),
                                     'prediction_output_delta': float(result[prefix + '_output_delta'].mean()),
                                     'input_nc_sensitivity': float(result[prefix + '_input_sensitivity'].mean()),
                                     'input_edit_energy': float(result[prefix + '_input_edit_energy'].mean()),
                                     'future_edit_energy': float(result[prefix + '_future_edit_energy'].mean()),
                                     'support_area': float(result[prefix + '_support_area'].mean()),
                                     'synthetic_future_AUROC': {key: auc(label, np.r_[result['clean_' + key], result[prefix + '_edited_' + key]])
                                                               for key in ('raw', 'raw_nc', 'raw_nc_gap', 'raw_local', 'raw_fusion')}}
        origin = {'same_Y_event_AUROC': auc(label, np.r_[result['origin_Y'], result['origin_TY']]),
                  'same_P_event_AUROC': auc(label, np.r_[result['origin_P'], result['origin_TP']]),
                  'no_event_real0_pred1_AUROC': auc(label, np.r_[result['origin_Y'], result['origin_P']]),
                  'no_event_first0_future1_AUROC': auc(label, np.r_[result['origin_X'], result['origin_Y']]),
                  'no_event_pred_minus_real_score': float((result['origin_P'] - result['origin_Y']).mean()),
                  'mean_scores': {key: float(result['origin_' + key].mean()) for key in ('Y', 'P', 'X', 'TY', 'TP')}}
        origin['no_event_origin_separability_AUROC'] = max(origin['no_event_real0_pred1_AUROC'],
                                                         1 - origin['no_event_real0_pred1_AUROC'])
        save_npz(scores_path, result)
        write(calibration_path, cal_record)
        write(summary_path, {'binding': binding, 'normalcy_bn_unchanged': True,
                             'clean_mse': float(result['clean_mse'].mean()),
                             'copy_last_mse': float(result['copy_last_mse'].mean()),
                             'reverse_history_output_delta': float(result['history_output_delta'].mean()),
                             'time_sensitivity': float(result['history_output_delta'].mean()),
                             'source_MOVE_AUROC': conditions['MOVE_A10']['synthetic_future_AUROC']['raw'],
                             'reverse_history_mse_delta': float(result['history_mse_delta'].mean()),
                             'conditions': conditions, 'origin': origin,
                             'scores_sha256': sha(scores_path), 'calibration_sha256': sha(calibration_path),
                             'interpretation': 'Source heldout synthetic mechanism diagnostics; only target AUROC evaluates cross-domain performance'})
        del models, G, N, O
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(candidate, 'source-only panel completed; normalcy BN unchanged', flush=True)


def lane_probe(pipeline, batch_size=16):
    return source_probe(pipeline, pipeline.LANE, batch_size)


def validate_freeze(pipeline, batch_size):
    freeze = read(OUT / 'freeze.json')
    require(batch_size == 16 and freeze['evaluation_batch'] == batch_size, 'Target batch must match frozen 16')
    require(freeze['prepared_sha256'] == sha(OUT / 'prepared.json') and
            freeze['release'] == pipeline.release_identity(), 'Target evaluation identity changed')
    require(set(freeze['models']) == set(spec.ORDER), 'All 32 final weights must be frozen')
    validation_path = OUT / 'source_validation.json'
    require(validation_path.is_file() and
            freeze.get('source_validation_sha256') == sha(validation_path),
            'Missing/floating source validation manifest')
    validation = read(validation_path)
    require(validation.get('study') == spec.STUDY and
            validation.get('prepared_sha256') == freeze['prepared_sha256'] and
            validation.get('source_split_sha256') == sha(OUT / 'source_split.json') and
            validation.get('source_probe_plan_sha256') == sha(OUT / 'source_probe_plan.npy') and
            set(validation.get('artifacts', {})) == set(spec.ORDER),
            'Source validation identity changed')
    for arm in spec.ORDER:
        require(freeze['models'][arm]['step'] == 5000 and
                (OUT / arm / 'source_probe.json').is_file() and (OUT / arm / 'calibration.json').is_file(),
                'Every final fit and source diagnostic must precede targets')
        probe = read(OUT / arm / 'source_probe.json')
        records = validation['artifacts'][arm]
        require(set(records) == {'source_probe.json', 'source_probe.npz', 'calibration.json', 'completed.json'} and
                all(records[name] == sha(OUT / arm / name) for name in records),
                'Frozen source artifact manifest changed: ' + arm)
        require(probe['binding'] == source_binding(arm) and probe['scores_sha256'] == sha(OUT / arm / 'source_probe.npz')
                and probe['calibration_sha256'] == sha(OUT / arm / 'calibration.json'), 'Source readout binding changed')
    # The source selector receipt is produced once, before freeze, by the
    # coordinator.  Its hash belongs to the freeze and may not change on targets.
    selection = OUT / 'source_selection.json'
    require(selection.is_file() and freeze.get('source_selection_sha256') == sha(selection), 'Missing/floating source selection receipt')
    return freeze


def lane_evaluate(pipeline, batch_size=16):
    lane = pipeline.LANE
    require(lane in spec.QUEUES, 'Unknown target-evaluation GPU lane')
    freeze = validate_freeze(pipeline, batch_size)
    require(freeze['worker_preflights'][lane] == sha(pipeline.preflight_path()), 'Target preflight binding changed')
    data_config = read(ROOT / 'data_config.json')
    _, targets = pipeline.check_data(data_config)
    for name, (videos, y) in targets.items():
        require(pipeline.digest(pipeline.metadata(videos)) == freeze['targets'][name]['frame_metadata_sha256']
                and sha(data_config['targets'][name]['labels']) == freeze['targets'][name]['label_sha256'], 'Target data changed: ' + name)
    frozen_sha = sha(OUT / 'freeze.json')
    pipeline.seed_everything(17)
    device = getattr(pipeline, 'DEVICE', 'cuda:0')
    for candidate in spec.QUEUES[lane]:
        models, done = load_models(pipeline, candidate, device)
        require(done == freeze['models'][candidate], 'Target checkpoint completion differs from freeze')
        G, N = models['G'], models['N']
        before = bn_snapshot(N)
        calibration_path = OUT / candidate / 'calibration.json'
        cal_record = read(calibration_path)
        require(cal_record['binding'] == source_binding(candidate), 'Target calibration binding changed')
        cal = cal_record['statistics']
        for name, (videos, y) in targets.items():
            dst = OUT / candidate / (name + '.npz')
            sidecar = dst.with_suffix('.json')
            expected_meta = {'freeze_sha256': frozen_sha, 'calibration_sha256': sha(calibration_path),
                             'source_selection_sha256': sha(OUT / 'source_selection.json'),
                             'normalcy_eval': True, 'normalcy_bn_unchanged': True,
                             'scored_frames': spec.TARGETS[name]}
            if dst.exists() and sidecar.exists():
                previous = read(sidecar)
                require(all(previous.get(key) == value for key, value in expected_meta.items())
                        and previous.get('scores_sha256') == sha(dst), 'Existing target readouts changed')
                continue
            require(not sidecar.exists(), 'Committed target readout array missing')
            pipeline.status('EVALUATING', candidate=candidate, target=name)
            result = {key: [] for key in ('raw', 'raw_nc', 'raw_nc_pred', 'raw_nc_gap', 'raw_local', 'raw_fusion', 'label', 'video', 'frame')}
            offset = 0
            with torch.inference_mode():
                for vi, (video, files) in enumerate(videos):
                    window = [pipeline.read_frame(path) for path in files[:4]]
                    inputs, observations, times = [], [], []
                    def flush():
                        x = torch.from_numpy(np.stack(inputs)).to(device)
                        target = torch.from_numpy(np.stack(observations)).to(device)
                        pred, _ = G(x)
                        score = arrays(readouts(pred, target, N))
                        score['raw_fusion'] = fusion(score, cal)
                        for key in ('raw', 'raw_nc', 'raw_nc_pred', 'raw_nc_gap', 'raw_local', 'raw_fusion'):
                            result[key].extend(score[key].tolist())
                        result['label'].extend(int(y[offset + t]) for t in times)
                        result['video'].extend([vi] * len(times))
                        result['frame'].extend(times)
                        inputs.clear(); observations.clear(); times.clear()
                    for t in range(4, len(files)):
                        observed = pipeline.read_frame(files[t])
                        inputs.append(np.stack(window).reshape(12, 256, 256))
                        observations.append(observed); times.append(t)
                        window = window[1:] + [observed]
                        if len(inputs) == batch_size:
                            flush()
                    if inputs:
                        flush()
                    offset += len(files)
                    print(candidate, name, video, len(result['raw']), 'frames; frozen G/N', flush=True)
                    pipeline.status('EVALUATING', candidate=candidate, target=name,
                                    scored_frames=len(result['raw']), expected_frames=spec.TARGETS[name])
            result = {key: np.asarray(value, dtype=np.int64 if key in ('label', 'video', 'frame') else np.float64)
                      for key, value in result.items()}
            result['video_names'] = np.asarray([name for name, files in videos])
            require(len(result['raw']) == spec.TARGETS[name] and all(np.isfinite(result[key]).all()
                    for key in ('raw', 'raw_nc', 'raw_nc_pred', 'raw_nc_gap', 'raw_local', 'raw_fusion')), 'Incomplete/nonfinite target scores')
            assert_bn_unchanged(N, before)
            save_npz(dst, result)
            write(sidecar, {**expected_meta, 'scores_sha256': sha(dst)})
        assert_bn_unchanged(N, before)
        del models, G, N
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
