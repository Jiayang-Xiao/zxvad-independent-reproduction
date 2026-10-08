import hashlib,json
from pathlib import Path
root=Path(__file__).resolve().parents[1]
manifest=json.loads((root/'release_manifest.json').read_text())
for name,expected in manifest['files'].items():
    actual=hashlib.sha256((root/name).read_bytes()).hexdigest()
    if actual!=expected:raise SystemExit('Release file differs: '+name)
print('Release hashes: PASS ('+str(len(manifest['files']))+' files)')
