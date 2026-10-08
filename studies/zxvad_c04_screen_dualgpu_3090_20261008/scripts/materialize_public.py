"""Validate completed evidence and materialize a new public study directory."""
import argparse,hashlib,json,sys,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from common import sha,read_json,release_identity
ap=argparse.ArgumentParser();ap.add_argument('--destination',required=True);args=ap.parse_args()
record=read_json(ROOT/'outputs'/'export_record.json');bundle=ROOT/'outputs'/'review_bundle.zip'
if sha(bundle)!=record['bundle_sha256']:raise SystemExit('Bundle changed')
with zipfile.ZipFile(bundle) as z:
    manifest=json.loads(z.read('export_manifest.json'))
    if manifest['release_identity']!=release_identity():raise SystemExit('Export differs from code release')
    if manifest['summary_sha256']!=record['summary_sha256']:raise SystemExit('Summary binding changed')
    if set(z.namelist())!=set(manifest['files'])|{'export_manifest.json'}:raise SystemExit('Bundle member list differs')
    for name,expected in manifest['files'].items():
        if hashlib.sha256(z.read(name)).hexdigest()!=expected:raise SystemExit('Bundle member changed: '+name)
    destination=Path(args.destination).resolve()
    if destination.exists():
        old=destination/'export_manifest.json'
        if not old.exists() or json.loads(old.read_text())!=manifest:raise SystemExit('Existing public study identity differs; refusing overwrite')
        for name,expected in manifest['files'].items():
            if not (destination/name).is_file() or sha(destination/name)!=expected:raise SystemExit('Existing public study file changed: '+name)
    for name in z.namelist():
        relative=Path(name)
        if relative.is_absolute() or '..' in relative.parts:raise SystemExit('Unsafe archive path')
        target=destination/relative;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(z.read(name))
print('Completed public study materialized:',destination)
