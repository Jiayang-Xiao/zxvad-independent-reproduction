"""Whitelist export of all fixed code and all completed mechanism-screen evidence.

No checkpoint weights, datasets, environments, credentials or publish checkouts
are members. Public logs are captured once; immutable scientific files continue
to be compared byte for byte on every retry. ZIP timestamps are explicit1980.
"""
import argparse
import hashlib
import zipfile
from pathlib import Path

import spec
import study_support as s

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'outputs'
SNAPSHOT = OUT / 'public_snapshot'


def members():
    release = s.read(ROOT / 'release_manifest.json')
    for name, expected in release['files'].items():
        s.require(s.sha(ROOT / name) == expected, 'Frozen release code changed: ' + name)
    result = {'code/' + name: ROOT / name for name in release['files']}
    result['code/release_manifest.json'] = ROOT / 'release_manifest.json'
    result['README.md'] = ROOT / 'README.md'
    for name in ('data_config.json', 'runtime_python.txt'):
        result['data_preparation/' + name] = ROOT / name
    result['data_preparation/import_record.json'] = ROOT / 'data_view' / 'import_record.json'
    contract = s.read(ROOT / 'parent_contract.json')
    for name in contract['files']:
        result['parent_reference/' + name] = ROOT / 'parent_reference' / name
    names = ['prepared.json', 'source_manifest.json', 'source_plan.npy', 'source_probe_plan.npy', 'source_split.json',
             'parent_binding.json', 'freeze.json', 'cross_card_parity.json', 'source_validation.json', 'source_selection.json', 'run_completion.json',
             'results.csv', 'primary_results.csv', 'same_gpu_contrasts.csv', 'cf_control_contrasts.csv',
             'summary.json', 'report.md', 'cross_card_diagnostics.json', 'execution.log']
    for lane, arms in spec.QUEUES.items():
        prefix = 'workers/gpu' + lane + '/'
        names += [prefix + name for name in ('preflight.json', 'optimizer_regression.json', 'state.json', 'execution.log',
                  'preflight_exit.json', 'fit_exit.json', 'probe_exit.json', 'evaluate_exit.json')]
        for phase in ('preflight', 'fit', 'probe', 'evaluate'):
            receipt = s.read(OUT / prefix / (phase + '_exit.json'))
            attempt = receipt.get('attempt_id')
            s.require(isinstance(attempt, str) and attempt and Path(attempt).name == attempt
                      and '/' not in attempt and '\\' not in attempt, 'Unsafe/missing worker attempt receipt')
            name = prefix + 'attempts/' + attempt + '_exit.json'
            s.require(s.sha(OUT / name) == s.sha(OUT / prefix / (phase + '_exit.json')),
                      'Current worker attempt differs from committed receipt')
            names.append(name)
        for arm in arms:
            names += [arm + '/' + name for name in ('config.json', 'completed.json', 'state.json', 'train.jsonl',
                      'calibration.json', 'source_probe.npz', 'source_probe.json')]
            names += [arm + '/' + target + suffix for target in spec.TARGETS for suffix in ('.npz', '.json')]
    names += ['coordinator/prepare_exit.json', 'coordinator/freeze_exit.json', 'coordinator/selection_exit.json']
    for name in names:
        result['evidence/' + name] = OUT / name
    for name, path in result.items():
        p = Path(name)
        s.require(not p.is_absolute() and '..' not in p.parts and all(part not in p.parts for part in
                  ('.private', '.git', '.publish', '.venv', '__pycache__')), 'Unsafe public path: ' + name)
        s.require(p.suffix.lower() not in ('.pt', '.pth', '.ckpt', '.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'),
                  'Weights/images cannot be exported: ' + name)
        s.require(path.is_file(), 'Missing completed public artifact: ' + str(path))
    return result


def export_header():
    contract = s.read(ROOT / 'parent_contract.json')
    return {'study': spec.STUDY, 'status': 'COMPLETED', 'kind': '32_SINGLE_SEED_MATCHED_CONTROL_FINAL_FITS',
            'new_fits': 32, 'primary_AUROC_rows': 96, 'total_AUROC_rows': 672,
            'source_training_videos': 297, 'source_validation_videos': 33, 'paper_all330_protocol_aligned': False,
            'weights_included': False, 'images_included': False, 'parent_commit': contract['parent_commit'],
            'release_identity': s.digest(s.read(ROOT / 'release_manifest.json')),
            'freeze_sha256': s.sha(OUT / 'freeze.json'),
            'source_selection_sha256': s.sha(OUT / 'source_selection.json'),
            'source_validation_sha256': s.sha(OUT / 'source_validation.json')}


def verify_snapshot():
    manifest = s.read(OUT / 'export_manifest.json')
    s.require(manifest['study'] == spec.STUDY and manifest['status'] == 'COMPLETED', 'Invalid export receipt')
    sources = members()
    s.require(set(manifest['files']) == set(sources), 'Public manifest differs from exact whitelist')
    s.require({k: v for k, v in manifest.items() if k != 'files'} == export_header(), 'Public export identity changed')
    for name, value in manifest['files'].items():
        s.require(s.sha(SNAPSHOT / name) == value, 'Public snapshot changed: ' + name)
    expected = set(manifest['files']) | {'export_manifest.json'}
    actual = {p.relative_to(SNAPSHOT).as_posix() for p in SNAPSHOT.rglob('*') if p.is_file()}
    s.require(actual == expected, 'Unexpected/missing snapshot files')
    s.require(s.sha(SNAPSHOT / 'export_manifest.json') == s.sha(OUT / 'export_manifest.json'), 'Export manifest copies differ')
    for name, source in sources.items():
        if not name.endswith('execution.log'):
            s.require(s.sha(source) == manifest['files'][name], 'Completed live evidence differs: ' + name)
    return manifest


def snapshot_completed():
    from report import generate_report
    summary = generate_report()
    s.require(summary['status'] == 'COMPLETED' and summary['total_AUROC_rows'] == 672, 'Cannot export incomplete results')
    if (OUT / 'export_manifest.json').exists():
        return verify_snapshot()
    SNAPSHOT.mkdir(parents=True, exist_ok=True)
    sources = members()
    files = {}
    for name, source in sorted(sources.items()):
        dest = SNAPSHOT / name
        if dest.exists() and name.endswith('execution.log'):
            pass
        else:
            s.copy_exact(source, dest, s.sha(source))
        files[name] = s.sha(dest)
    manifest = {**export_header(), 'files': files}
    s.write(SNAPSHOT / 'export_manifest.json', manifest)
    s.copy_exact(SNAPSHOT / 'export_manifest.json', OUT / 'export_manifest.json', s.sha(SNAPSHOT / 'export_manifest.json'))
    return verify_snapshot()


def bundle(manifest):
    path, temp = OUT / 'review_bundle.zip', OUT / 'review_bundle.zip.incoming'
    names = sorted(set(manifest['files']) | {'export_manifest.json'})
    with zipfile.ZipFile(temp, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for name in names:
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, (SNAPSHOT / name).read_bytes())
    with zipfile.ZipFile(temp) as archive:
        s.require(archive.testzip() is None and len(archive.namelist()) == len(set(archive.namelist()))
                  and set(archive.namelist()) == set(names), 'Bundle member integrity mismatch')
        for name, value in manifest['files'].items():
            s.require(hashlib.sha256(archive.read(name)).hexdigest() == value, 'Bundled bytes differ')
    temp.replace(path)
    s.write(OUT / 'export_completion.json', {'status': 'COMPLETED', 'study': spec.STUDY,
            'export_manifest_sha256': s.sha(OUT / 'export_manifest.json'), 'review_bundle_sha256': s.sha(path),
            'public_files': len(names), 'weights_included': False, 'images_included': False})
    return path


def materialize(destination, manifest):
    destination = Path(destination)
    allowed = set(manifest['files']) | {'export_manifest.json'}
    if destination.exists():
        actual = {p.relative_to(destination).as_posix() for p in destination.rglob('*') if p.is_file()}
        s.require(actual <= allowed, 'Destination contains foreign files; refuse mixing studies')
    for name in sorted(allowed):
        s.copy_exact(SNAPSHOT / name, destination / name, s.sha(SNAPSHOT / name))
    actual = {p.relative_to(destination).as_posix() for p in destination.rglob('*') if p.is_file()}
    s.require(actual == allowed, 'Public destination whitelist mismatch')
    print(f'Public evidence whitelist materialized: {len(allowed)} files', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--destination', type=Path)
    args = parser.parse_args()
    manifest = snapshot_completed()
    path = bundle(manifest)
    if args.destination:
        materialize(args.destination, manifest)
    print('Review bundle: ' + str(path), flush=True)


if __name__ == '__main__':
    main()
