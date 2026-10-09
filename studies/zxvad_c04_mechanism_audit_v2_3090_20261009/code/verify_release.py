"""Check every published implementation byte without importing CUDA libraries."""
import json
from pathlib import Path
import study_support as s
ROOT=Path(__file__).resolve().parent
def verify():
    m=s.read(ROOT/'release_manifest.json')
    for name,h in m['files'].items():
        p=Path(name)
        s.require(not p.is_absolute() and '..' not in p.parts,'Invalid release path')
        s.require(s.sha(ROOT/p)==h,'Release file changed: '+name)
    s.verify_original_code()
    print(f'Release hashes and original41 implementation files: PASS ({len(m["files"])} files)',flush=True)
    return s.digest(m)
if __name__=='__main__':verify()
