"""CPU audit of every declared final-model readout, including negative controls.

Only target AUROC is a performance metric. Source synthetic diagnostics and
cross-card score drift are reported separately and never rank target winners.
"""
import csv
import io
from pathlib import Path

import numpy as np

import coordinate
import spec
import study_support as support

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'outputs'
FIELDS = {'PSNR': 'raw', 'NC': 'raw_nc', 'NC_GAP': 'raw_nc_gap',
          'LOCAL': 'raw_local', 'FUSION': 'raw_fusion',
          'PSNR_FROZEN': 'raw', 'FUSION_FROZEN': 'raw_fusion'}
SCORES = ('raw', 'raw_nc', 'raw_nc_pred', 'raw_nc_gap', 'raw_local', 'raw_fusion')
IDENTITIES = ('label', 'video', 'frame', 'video_names')


def normalize(raw, video):
    result = np.zeros(len(raw), dtype=np.float64)
    for v in np.unique(video):
        use = video == v
        values = raw[use]
        span = values.max() - values.min()
        if span > 0:
            result[use] = (values - values.min()) / span
    return result


def auroc(y, score):
    y, score = np.asarray(y), np.asarray(score)
    support.require(y.shape == score.shape and y.ndim == 1 and len(y) > 0,
                    'AUROC requires matching nonempty vectors')
    support.require(np.isin(y, [0, 1]).all() and np.isfinite(score).all(), 'Invalid AUROC scores/labels')
    order = np.argsort(score, kind='stable')
    s, labels = score[order], y[order]
    starts = np.r_[0, 1 + np.flatnonzero(s[1:] != s[:-1])]
    positive = np.add.reduceat(labels.astype(np.float64), starts)
    negative = np.add.reduceat((1 - labels).astype(np.float64), starts)
    denominator = positive.sum() * negative.sum()
    support.require(denominator > 0, 'Both classes are required for AUROC')
    return float((positive * (np.cumsum(negative) - .5 * negative)).sum() / denominator)


def atomic_text(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(content, encoding='utf-8', newline='')
    temp.replace(path)


def write_csv(path, fields, rows):
    buffer = io.StringIO(newline='')
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator='\n')
    writer.writeheader()
    writer.writerows(rows)
    atomic_text(path, buffer.getvalue())


def finite_vector(value, count, name):
    support.require(value.shape == (count,) and np.issubdtype(value.dtype, np.number)
                    and np.isrealobj(value) and np.isfinite(value).all(), 'Invalid vector: ' + name)


def source_calibration(arm, done, prepared_sha):
    dst = OUT / arm
    probe, calibration = support.read(dst / 'source_probe.json'), support.read(dst / 'calibration.json')
    binding = probe.get('binding', {})
    support.require(binding == calibration.get('binding') and binding.get('study') == spec.STUDY
                    and binding.get('candidate') == arm and binding.get('checkpoint_sha256') == done['checkpoint_sha256']
                    and binding.get('prepared_sha256') == prepared_sha
                    and binding.get('source_probe_plan_sha256') == support.sha(OUT / 'source_probe_plan.npy')
                    and binding.get('source_split_sha256') == support.sha(OUT / 'source_split.json')
                    and binding.get('source_only') is True and binding.get('target_used') is False
                    and binding.get('clips') == 256 and binding.get('batch_size') == 16
                    and binding.get('normalcy_eval') is True and binding.get('fusion_weight') == .25
                    and binding.get('local_tile') == 8 and binding.get('local_top_fraction') == .1,
                    'Source calibration identity differs: ' + arm)
    support.require(probe.get('normalcy_bn_unchanged') is True and calibration.get('normalcy_bn_unchanged') is True
                    and probe.get('scores_sha256') == support.sha(dst / 'source_probe.npz')
                    and probe.get('calibration_sha256') == support.sha(dst / 'calibration.json'),
                    'Source panel receipt changed: ' + arm)
    statistics = calibration.get('statistics', {})
    support.require(set(statistics) == {'raw', 'raw_nc', 'raw_local'}, 'Calibration component set differs')
    with np.load(dst / 'source_probe.npz', allow_pickle=False) as values:
        support.require('video' in values.files and 'frame' in values.files, 'Missing source panel identities')
        for key in values.files:
            finite_vector(values[key], 256, arm + '/' + key)
        panel = np.load(OUT / 'source_probe_plan.npy', allow_pickle=False)
        support.require(np.array_equal(values['video'], panel[:, 0])
                        and np.array_equal(values['frame'], panel[:, 1] + 4), 'Source panel alignment changed')
        for key, record in statistics.items():
            clean = values['clean_' + key]
            q25, q75 = np.quantile(clean, [.25, .75], method='linear')
            expected = {'median': float(np.median(clean)), 'iqr': float(max(q75 - q25, 1e-6))}
            support.require(record == expected, 'Source-only median/IQR calibration differs: ' + arm + '/' + key)
    return statistics, probe


def audit_fit(arm, freeze, release):
    lane = spec.lane_for(arm)
    dst = OUT / arm
    config, done, state = [support.read(dst / name) for name in ('config.json', 'completed.json', 'state.json')]
    recipe, seed = spec.parse(arm)
    support.require(config.get('study') == spec.STUDY and config.get('candidate') == arm
                    and config.get('recipe') == recipe and config.get('physical_gpu') == int(lane)
                    and config.get('seed') == seed == 17 and config.get('iterations') == 5000
                    and config.get('batch_size') == 8, 'Final fit setting differs: ' + arm)
    support.require(done.get('phase') == 'TRAINED' and done.get('step') == 5000
                    and done.get('config_hash') == support.digest(config) and freeze['models'].get(arm) == done,
                    'Incomplete/unbound final fit: ' + arm)
    support.require(state.get('phase') == 'TRAINED' and state.get('step') == 5000
                    and state.get('checkpoint_step') == 5000 and state.get('config_hash') == done['config_hash'],
                    'Final training state differs: ' + arm)
    support.require(config.get('release') == freeze['release'] == release
                    and config.get('prepared_sha256') == freeze['prepared_sha256'] == support.sha(OUT / 'prepared.json')
                    and config.get('preflight_sha256') == freeze['worker_preflights'][lane]
                    == support.sha(OUT / 'workers' / ('gpu' + lane) / 'preflight.json'), 'Fit identity binding changed: ' + arm)
    preflight = support.read(OUT / 'workers' / ('gpu' + lane) / 'preflight.json')
    support.require(config.get('hardware_identity') == preflight['environment']['hardware_identity'], 'GPU hardware identity differs')
    statistics, probe = source_calibration(arm, done, freeze['prepared_sha256'])
    return config, statistics, probe


def fused(raw, nc, calibration):
    return ((raw - calibration['raw']['median']) / calibration['raw']['iqr']
            + .25 * (nc - calibration['raw_nc']['median']) / calibration['raw_nc']['iqr'])


def audit_scores(arm, target, freeze, calibration):
    count = spec.TARGETS[target]
    path = OUT / arm / (target + '.npz')
    meta = support.read(path.with_suffix('.json'))
    support.require(meta.get('freeze_sha256') == support.sha(OUT / 'freeze.json')
                    and meta.get('scores_sha256') == support.sha(path) and meta.get('scored_frames') == count
                    and meta.get('calibration_sha256') == support.sha(OUT / arm / 'calibration.json')
                    and meta.get('source_selection_sha256') == support.sha(OUT / 'source_selection.json')
                    and meta.get('normalcy_eval') is True and meta.get('normalcy_bn_unchanged') is True,
                    'Target sidecar binding differs: ' + arm + '/' + target)
    with np.load(path, allow_pickle=False) as data, np.load(ROOT / 'reference' / ('c04_' + target + '.npz'), allow_pickle=False) as reference:
        support.require(set(data.files) == set(SCORES) | set(IDENTITIES), 'Unexpected target score fields')
        for key in IDENTITIES:
            support.require(np.array_equal(data[key], reference[key]), 'Canonical alignment differs: ' + arm + '/' + target + '/' + key)
        arrays = {key: data[key].copy() for key in data.files}
    for key in SCORES:
        finite_vector(arrays[key], count, arm + '/' + target + '/' + key)
    label, video, frame = arrays['label'], arrays['video'], arrays['frame']
    support.require(label.shape == video.shape == frame.shape == (count,) and np.isin(label, [0, 1]).all(), 'Invalid canonical label/index length')
    target_freeze = freeze['targets'][target]
    support.require(arrays['video_names'].tolist() == target_freeze['video_names']
                    and len(target_freeze['video_total_frames']) == len(arrays['video_names'])
                    and sum(n - 4 for n in target_freeze['video_total_frames']) == count,
                    'First-four-frame exclusion/video identity changed')
    for vi, total in enumerate(target_freeze['video_total_frames']):
        support.require(np.array_equal(frame[video == vi], np.arange(4, total)), 'Noncanonical temporal indices')
    support.require(np.allclose(arrays['raw_fusion'], fused(arrays['raw'], arrays['raw_nc'], calibration), rtol=1e-12, atol=1e-10), 'Frozen fusion equation differs')
    # The gap is computed in FP32 before CPU conversion; subtraction below is
    # float64 and may differ by the rounding of one FP32 subtraction.
    support.require(np.allclose(arrays['raw_nc_gap'], arrays['raw_nc'] - arrays['raw_nc_pred'], rtol=1e-5, atol=2e-6), 'NC gap sign/equation differs')
    values = {}
    for readout in spec.READOUTS:
        raw = arrays[FIELDS[readout]]
        score = raw if readout.endswith('_FROZEN') else normalize(raw, video)
        values[readout] = auroc(label, score)
    return values, arrays, meta


def drift(a, b):
    support.require(a.shape == b.shape, 'Cross-card arrays do not align')
    delta = np.abs(a - b)
    return {'array_equal': bool(np.array_equal(a, b)), 'different_elements': int(np.count_nonzero(a != b)),
            'max_abs_difference': float(delta.max()), 'mean_abs_difference': float(delta.mean())}


def chosen_arms(selection):
    # The selector's explicit declaration, not a target-ranked maximum, defines
    # this informational flag. An empty list is a legitimate source decision.
    arms = selection.get('chosen_arms')
    support.require(isinstance(arms, list) and len(set(arms)) == len(arms) and set(arms) <= set(spec.ORDER), 'Invalid source-selected arm declaration')
    return set(arms)


def generate_report():
    support.validate_parent()
    support.validate_prepared()
    for lane in spec.QUEUES:
        support.validate_preflight(lane)
    support.require(coordinate.completed_noop(), 'All 32 fits, source probes, frozen choices and targets must complete before reporting')
    release = coordinate.release_identity()
    freeze, prepared = support.read(OUT / 'freeze.json'), support.read(OUT / 'prepared.json')
    support.require(freeze.get('study') == spec.STUDY and freeze.get('release') == release
                    and freeze.get('prepared_sha256') == support.sha(OUT / 'prepared.json')
                    and freeze.get('evaluation_batch') == 16 and set(freeze.get('models', {})) == set(spec.ORDER)
                    and set(freeze.get('targets', {})) == set(spec.TARGETS), 'Freeze differs from declared 32-fit protocol')
    for target in spec.TARGETS:
        declared = prepared.get('data', {}).get('targets', {}).get(target)
        if declared is not None:
            support.require(freeze['targets'][target].get('frame_metadata_sha256') == declared['metadata_sha256']
                            and freeze['targets'][target].get('label_sha256') == declared['labels_sha256'],
                            'Target metadata differs between preparation and freeze: ' + target)
    selection = support.read(OUT / 'source_selection.json')
    support.require(freeze.get('source_selection_sha256') == support.sha(OUT / 'source_selection.json')
                    and freeze.get('source_validation_sha256') == support.sha(OUT / 'source_validation.json')
                    and selection.get('source_validation_sha256') == support.sha(OUT / 'source_validation.json')
                    and selection.get('study') == spec.STUDY and selection.get('metric') == 'AUROC only',
                    'Source selection/validation changed after target freeze')
    source_validation = support.read(OUT / 'source_validation.json')
    artifacts = source_validation.get('artifacts', {})
    support.require(set(artifacts) == set(spec.ORDER), 'All32 source validation artifact receipts are required')
    for arm in spec.ORDER:
        support.require(set(artifacts[arm]) == {'source_probe.json', 'source_probe.npz', 'calibration.json', 'completed.json'},
                        'Unexpected source validation artifact set')
        for name, value in artifacts[arm].items():
            support.require(support.sha(OUT / arm / name) == value, 'Frozen source validation artifact changed: ' + arm + '/' + name)
    selected = chosen_arms(selection)
    split = support.read(OUT / 'source_split.json')
    training = split.get('training_names', split.get('training_videos', []))
    validation = split.get('validation_names', split.get('validation_videos', []))
    support.require(len(training) == 297 and len(validation) == 33 and not (set(training) & set(validation)), 'Source partition differs from 297/33 heldout protocol')
    rows, values, score_arrays, source_probes = [], {}, {}, {}
    for arm in spec.ORDER:
        config, calibration, source_probes[arm] = audit_fit(arm, freeze, release)
        values[arm], score_arrays[arm] = {}, {}
        for target, count in spec.TARGETS.items():
            values[arm][target], score_arrays[arm][target], meta = audit_scores(arm, target, freeze, calibration)
            for readout in spec.READOUTS:
                rows.append({'study': spec.STUDY, 'arm': arm, 'recipe': spec.parse(arm)[0], 'seed': 17,
                             'physical_gpu': int(spec.lane_for(arm)), 'checkpoint_step': 5000,
                             'target': target, 'frames': count, 'readout': readout, 'AUROC': values[arm][target][readout],
                             'primary': readout == 'PSNR', 'source_chosen': arm in selected,
                             'scores_sha256': meta['scores_sha256'], 'freeze_sha256': meta['freeze_sha256'],
                             'calibration_sha256': meta['calibration_sha256']})
    support.require(len(rows) == 672 and sum(r['primary'] for r in rows) == 96, 'Expected all 672 rows including 96 primary PSNR rows')
    macros = {arm: {r: float(np.mean([values[arm][target][r] for target in spec.TARGETS])) for r in spec.READOUTS} for arm in spec.ORDER}
    same_gpu = []
    for arm in spec.ORDER:
        lane = spec.lane_for(arm)
        baseline = spec.BASELINE[lane]
        for readout in spec.READOUTS:
            for target in (*spec.TARGETS, 'macro'):
                a = macros[arm][readout] if target == 'macro' else values[arm][target][readout]
                b = macros[baseline][readout] if target == 'macro' else values[baseline][target][readout]
                same_gpu.append({'arm': arm, 'baseline': baseline, 'physical_gpu': int(lane), 'readout': readout,
                                 'target': target, 'arm_AUROC': a, 'baseline_AUROC': b, 'delta_pp': 100 * (a - b)})
    controls = []
    for lane, area in (('0', 10), ('1', 20)):
        for probability in (25, 50):
            move = f'CF_MOVE_A{area}_P{probability}_s17'
            for mode in ('STATIC', 'SHUFFLE', 'NOISE', 'SELF', 'PHOTO'):
                control = f'CF_{mode}_A{area}_P{probability}_s17'
                support.require(spec.lane_for(move) == spec.lane_for(control) == lane, 'CF controls must match physical GPU')
                for readout in spec.READOUTS:
                    for target in (*spec.TARGETS, 'macro'):
                        m = macros[move][readout] if target == 'macro' else values[move][target][readout]
                        c = macros[control][readout] if target == 'macro' else values[control][target][readout]
                        controls.append({'physical_gpu': int(lane), 'area': area / 100, 'probability': probability / 100,
                                         'move_arm': move, 'control_arm': control, 'control_mode': mode, 'readout': readout,
                                         'target': target, 'MOVE_AUROC': m, 'control_AUROC': c, 'MOVE_minus_control_pp': 100 * (m - c)})
    cross_card = {'kind': 'SINGLE_SEED_CROSS_CARD_SCORE_DIAGNOSTIC', 'performance_metric': False,
                  'scope': 'B0 versus B1 final score arrays only; not proof of identical training trajectories', 'targets': {}}
    for target in spec.TARGETS:
        a, b = score_arrays['B0_s17'][target], score_arrays['B1_s17'][target]
        cross_card['targets'][target] = {'scores': {key: drift(a[key], b[key]) for key in SCORES},
                                        'readout_AUROC_delta_pp': {r: 100 * (values['B1_s17'][target][r] - values['B0_s17'][target][r]) for r in spec.READOUTS}}
    primary_macros = {arm: macros[arm]['PSNR'] for arm in spec.ORDER}
    primary_delta = {arm: 100 * (primary_macros[arm] - primary_macros[spec.BASELINE[spec.lane_for(arm)]]) for arm in spec.ORDER}
    summary = {'status': 'COMPLETED', 'study': spec.STUDY, 'author_confirmed': False,
               'kind': 'SINGLE_SEED_SOURCE_HELDOUT_MATCHED_CONTROL_MECHANISM_SCREEN', 'metric': 'AUROC only',
               'new_fits': 32, 'seeds': [17], 'targets': list(spec.TARGETS), 'target_frames': spec.TARGETS,
               'primary_readout': 'PSNR', 'readouts': list(spec.READOUTS), 'primary_AUROC_rows': 96,
               'total_AUROC_rows': 672, 'same_gpu_contrast_rows': len(same_gpu), 'CF_control_contrast_rows': len(controls),
               'rows_are_not_independent_training_trials': True, 'rows': rows,
               'target_AUROC': values, 'macro_AUROC': primary_macros, 'macro_by_readout': macros,
               'primary_same_gpu_delta_pp': primary_delta, 'same_gpu_baselines': spec.BASELINE,
               'source_chosen_arms': sorted(selected), 'source_selection': selection,
               'source_selection_sha256': support.sha(OUT / 'source_selection.json'),
               'source_validation_sha256': support.sha(OUT / 'source_validation.json'),
               'source_diagnostics': source_probes, 'source_training_videos': 297, 'source_validation_videos': 33,
               'paper_all330_protocol_aligned': False, 'source_selection_frozen_before_target': True,
               'code_release': release, 'freeze_sha256': support.sha(OUT / 'freeze.json'),
               'run_completion_sha256': support.sha(OUT / 'run_completion.json'),
               'parent_binding_sha256': support.sha(OUT / 'parent_binding.json'),
               'no_target_adaptation': True, 'no_target_checkpoint_selection': True,
               'target_feedback_used_for_design': True, 'target_blind': False, 'target_winner_selected': False,
               'interpretation': 'Every negative result and matched control is retained. Source synthetic diagnostics do not establish real-anomaly mechanism, stable improvement, novelty or SOTA. This heldout297-source experiment cannot be compared as an all330 paper reproduction.'}
    fields = ['study', 'arm', 'recipe', 'seed', 'physical_gpu', 'checkpoint_step', 'target', 'frames', 'readout', 'AUROC', 'primary', 'source_chosen', 'scores_sha256', 'freeze_sha256', 'calibration_sha256']
    write_csv(OUT / 'results.csv', fields, rows)
    write_csv(OUT / 'primary_results.csv', fields, [row for row in rows if row['primary']])
    write_csv(OUT / 'same_gpu_contrasts.csv', ['arm', 'baseline', 'physical_gpu', 'readout', 'target', 'arm_AUROC', 'baseline_AUROC', 'delta_pp'], same_gpu)
    write_csv(OUT / 'cf_control_contrasts.csv', ['physical_gpu', 'area', 'probability', 'move_arm', 'control_arm', 'control_mode', 'readout', 'target', 'MOVE_AUROC', 'control_AUROC', 'MOVE_minus_control_pp'], controls)
    support.write(OUT / 'cross_card_diagnostics.json', cross_card)
    support.write(OUT / 'summary.json', summary)
    lines = ['# c04机制审核：单种子配对筛查', '',
             '仅以目标域AUROC评价性能；32次最终5000step训练，96条主PSNR结果，672条完整读出结果。指标行不是独立训练次数。',
             '源域按视频划分297训练/33验证，本轮不是论文all330协议对齐复现。N使用冻结eval状态；无目标域模型或BN更新。', '',
             '| 方案 | GPU | 源选择 | Ped1 % | Ped2 % | Avenue % | Macro % | 同卡Δ pp |',
             '|---|---:|---|---:|---:|---:|---:|---:|']
    for arm in spec.ORDER:
        cells = [f'{100 * values[arm][target]["PSNR"]:.4f}' for target in spec.TARGETS]
        lines.append('| ' + ' | '.join([arm, spec.lane_for(arm), '是' if arm in selected else '否', *cells,
                                        f'{100 * primary_macros[arm]:.4f}', f'{primary_delta[arm]:+.4f}']) + ' |')
    lines += ['', '## MOVE与同面积、同概率、同卡控制组的主读出比较', '',
              '| MOVE | 控制 | Ped1 Δ pp | Ped2 Δ pp | Avenue Δ pp | Macro Δ pp |', '|---|---|---:|---:|---:|---:|']
    for lane, area in (('0', 10), ('1', 20)):
        for probability in (25, 50):
            move = f'CF_MOVE_A{area}_P{probability}_s17'
            for mode in ('STATIC', 'SHUFFLE', 'NOISE', 'SELF', 'PHOTO'):
                control = f'CF_{mode}_A{area}_P{probability}_s17'
                delta = [100 * (values[move][t]['PSNR'] - values[control][t]['PSNR']) for t in spec.TARGETS]
                delta.append(100 * (primary_macros[move] - primary_macros[control]))
                lines.append('| ' + ' | '.join([move, control, *[f'{d:+.4f}' for d in delta]]) + ' |')
    lines += ['', '七种读出以及所有正负结果完整保留在results.csv；同卡baseline比较见same_gpu_contrasts.csv，全部时序控制比较见cf_control_contrasts.csv。',
              '源域继续候选在目标评测前冻结，见source_selection.json。该标记不由目标最高AUROC决定。源域人工合成AUROC只是机制诊断，不是额外跨域性能依据。',
              'B0/B1跨卡原始分数漂移单独保存，不参与性能排名。单种子、开发目标反馈与源域合成数据不支持稳定提升、显著性、首创或SOTA声明。', '']
    atomic_text(OUT / 'report.md', '\n'.join(lines))
    print('COMPLETED: 32 final fits, 96 primary rows, all 672 AUROC readout rows; source-selected arms=' + str(sorted(selected)), flush=True)
    return summary


if __name__ == '__main__':
    generate_report()
