"""Deterministic manifests and I/O shared by all three arms."""
import hashlib
import json
import random
from pathlib import Path
import numpy as np
from PIL import Image

VARIANTS = ("baseline", "motion", "shifted")
TARGETS = ("ped1", "ped2", "avenue")
EXPECTED = {"ped1": (36, 7056), "ped2": (12, 1962), "avenue": (21, 15240)}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1024*1024), b""):
            h.update(b)
    return h.hexdigest()


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def save_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    tmp.replace(path)


def frames(root):
    return [(v.name, sorted(v.glob("*.jpg"))) for v in sorted(Path(root).iterdir()) if v.is_dir()]


def identity(videos):
    # Metadata identity, NOT a cryptographic content hash of the image bytes.
    return [{"video": name, "frames": [[p.name, p.stat().st_size, p.stat().st_mtime_ns] for p in fs]}
            for name, fs in videos]


def read_frame(path):
    with Image.open(path) as im:
        arr = np.asarray(im.convert("RGB").resize((256, 256), Image.Resampling.BILINEAR), dtype=np.float32)
    return (arr / 127.5 - 1.0).transpose(2, 0, 1).copy()


def sample_batch(videos, step, seed=17):
    # Each update has its own sampler stream; recovery cannot skip/repeat clips.
    rng = random.Random(seed * 1000003 + step)
    clips = []
    for _ in range(8):
        _, fs = videos[rng.randrange(len(videos))]
        start = rng.randrange(len(fs)-4)
        clips.append(np.stack([read_frame(p) for p in fs[start:start+5]]))
    return np.stack(clips)


def code_identity():
    here = Path(__file__).resolve().parent
    names = json.loads((here/"package_manifest.json").read_text())
    return {n: sha(here/n) for n in sorted(names) if n.endswith(".py")}
