"""Metadata-only repair for the frozen V2 release; never trains or scores models."""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path

EXPECTED_RELEASE = "fd4a06f79fb2e73173d5a934d26a53de7bb4d35b6265b9df89b2834e1019ac67"
SAFE_MTIME_NS = 315619200 * 1_000_000_000  # 1980-01-02 UTC, safe across time zones.


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def repair(root):
    root = Path(root).resolve()
    manifest_path = root / "release_manifest.json"
    if sha(manifest_path) != EXPECTED_RELEASE:
        raise RuntimeError("Unexpected release; refusing this recovery")
    manifest = read(manifest_path)
    out = root / "outputs"
    summary = read(out / "summary.json")
    if summary.get("status") != "COMPLETED" or len(summary.get("rows", [])) != 12:
        raise RuntimeError("Completed twelve-row score evidence required")
    wanted = {(c, t) for c in ("c01", "c02", "c03", "c04") for t in ("ped1", "ped2", "avenue")}
    if {(r["candidate"], r["target"]) for r in summary["rows"]} != wanted:
        raise RuntimeError("Incomplete candidate/target evidence")
    for candidate in ("c01", "c02", "c03", "c04"):
        state = read(out / candidate / "state.json")
        completed = read(out / candidate / "completed.json")
        if state.get("phase") != "TRAINED" or state.get("step") != 5000 or completed.get("step") != 5000:
            raise RuntimeError("A fit is not finished: " + candidate)
        for target in ("ped1", "ped2", "avenue"):
            for suffix in (".npz", ".json"):
                if not (out / candidate / (target + suffix)).is_file():
                    raise RuntimeError("Missing target evidence: " + candidate + "/" + target + suffix)

    expected = dict(manifest["files"])
    expected["release_manifest.json"] = EXPECTED_RELEASE
    paths = []
    for name, digest in sorted(expected.items()):
        path = root / name
        if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(root):
            raise RuntimeError("Unexpected release path: " + name)
        if sha(path) != digest:
            raise RuntimeError("Release content changed: " + name)
        paths.append((name, path, digest))

    record_path = out / "zip_timestamp_recovery.json"
    if record_path.exists():
        record = read(record_path)
        if record.get("release_manifest_sha256") != EXPECTED_RELEASE or record.get("kind") != "ZIP_TIMESTAMP_METADATA_ONLY":
            raise RuntimeError("Existing recovery record differs")
    else:
        record = {
            "kind": "ZIP_TIMESTAMP_METADATA_ONLY",
            "release_manifest_sha256": EXPECTED_RELEASE,
            "repair_script_sha256": sha(Path(__file__)),
            "observed_pipeline_state_before": read(out / "state.json"),
            "observed_exit_before": (out / "last_exit.txt").read_text().strip(),
            "safe_mtime_ns": SAFE_MTIME_NS,
            "file_bytes_changed": False,
            "model_training_or_inference_performed": False,
            "changes": [],
        }
    for name, path, digest in paths:
        stat = path.stat()
        if time.localtime(stat.st_mtime).tm_year < 1980:
            record["changes"].append({
                "file": name,
                "before_mtime_ns": stat.st_mtime_ns,
                "after_mtime_ns": SAFE_MTIME_NS,
                "sha256_unchanged": digest,
            })
            temporary = record_path.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
            temporary.replace(record_path)
            os.utime(path, ns=(stat.st_atime_ns, SAFE_MTIME_NS))
        if time.localtime(path.stat().st_mtime).tm_year < 1980 or sha(path) != digest:
            raise RuntimeError("Metadata repair verification failed: " + name)
    temporary = record_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    temporary.replace(record_path)
    print("ZIP timestamp repair: PASS. Frozen file bytes unchanged; no training/inference.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root")
    repair(parser.parse_args().root)
