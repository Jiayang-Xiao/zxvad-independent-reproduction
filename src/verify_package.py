"""Reject modified implementation/recipe before a server run."""
import hashlib
import json
from pathlib import Path

if __name__=="__main__":
    here=Path(__file__).resolve().parent
    expected=json.loads((here/"package_manifest.json").read_text(encoding="utf-8"))
    for name,value in expected.items():
        if hashlib.sha256((here/name).read_bytes()).hexdigest()!=value:
            raise RuntimeError("Package changed: "+name)
    print(f"Verified {len(expected)} packaged files.")
