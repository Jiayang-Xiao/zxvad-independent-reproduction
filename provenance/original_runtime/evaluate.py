"""Freeze all three fits before target inference; identical global -PSNR for every arm."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from baseline_zxvad import FutureFrameGenerator
from common import VARIANTS,TARGETS,EXPECTED,code_identity,digest,frames,identity,read_frame,save_json,sha


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-root",required=True)
    ap.add_argument("--output",required=True)
    ap.add_argument("--device",default="cuda:0")
    args = ap.parse_args()
    root,out = Path(args.target_root),Path(args.output)
    freeze = {"study":"ZXVAD_MNA_V1","code":code_identity(),"models":{},"targets":{},"configs":{},"recipe_sha256":sha(Path(__file__).parent/"protocol.json"),"design_freeze_sha256":sha(Path(__file__).parent/"design_freeze.json")}
    preflight=json.loads((out/'preflight.json').read_text())
    if preflight['status']!='PASS' or preflight['code_sha256']!=freeze['code'] or preflight['recipe_sha256']!=freeze['recipe_sha256']:
        raise RuntimeError('Missing or changed source-only preflight provenance')
    freeze['preflight_sha256']=sha(out/'preflight.json')
    source_ids = set()
    for variant in VARIANTS:
        done = json.loads((out/variant/"completed.json").read_text())
        config = json.loads((out/variant/"config.json").read_text())
        if done["step"]!=5000 or done["checkpoint_sha256"]!=sha(out/variant/"last.pt") or config["code_sha256"]!=freeze["code"]:
            raise RuntimeError("Unfinished or modified fit: "+variant)
        if digest(config)!=done["config_hash"] or config["recipe_sha256"]!=freeze["recipe_sha256"]:
            raise RuntimeError("Configuration digest mismatch: "+variant)
        freeze["configs"][variant]=config
        source_ids.add(config["source_metadata_sha256"])
        freeze["models"][variant]=done
    if source_ids!={preflight["source_metadata_sha256"]}: raise RuntimeError("Source identity differs from preflight")
    catalogs = {}
    for dataset in TARGETS:
        videos=frames(root/dataset/"testing"/"frames")
        y = np.load(root/f"frame_labels_{dataset}.npy",allow_pickle=False).reshape(-1)
        nvideos,nscored = EXPECTED[dataset]
        if len(videos)!=nvideos or sum(len(fs)-4 for _,fs in videos)!=nscored or sum(len(fs) for _,fs in videos)!=len(y):
            raise RuntimeError("Canonical frame/label count mismatch: "+dataset)
        if not np.isin(y,[0,1]).all(): raise RuntimeError("Non-binary labels")
        catalogs[dataset]=(videos,y)
        freeze["targets"][dataset]={"frame_metadata_sha256":digest(identity(videos)),
            "label_sha256":sha(root/f"frame_labels_{dataset}.npy"),"counts":[nvideos,nscored]}
    freeze_path=out/"freeze.json"
    if freeze_path.exists() and json.loads(freeze_path.read_text())!=freeze:
        raise RuntimeError("Freeze identity changed; results cannot be overwritten")
    save_json(freeze_path,freeze)
    device=torch.device(args.device)
    if device.type!="cuda" or not torch.cuda.is_available(): raise RuntimeError("CUDA required")
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True)
    for variant in VARIANTS:
        model=FutureFrameGenerator(input_frames=4,mem_dim=2000).to(device).eval()
        ck=torch.load(out/variant/"last.pt",map_location=device,weights_only=False)
        if ck["config_hash"]!=freeze["models"][variant]["config_hash"] or ck["step"]!=5000:
            raise RuntimeError("Checkpoint identity mismatch")
        model.load_state_dict(ck["models"]["G"])
        del ck
        for dataset,(videos,y) in catalogs.items():
            dst=out/variant/(dataset+".npz")
            meta=dst.with_suffix(".json")
            if dst.exists() and meta.exists():
                old=json.loads(meta.read_text())
                if old["freeze_sha256"]==sha(freeze_path) and old["scores_sha256"]==sha(dst):
                    continue
                raise RuntimeError("Score identity mismatch")
            raw,gt,vids,indices=[],[],[],[]
            offset=0
            with torch.inference_mode():
                for v,(name,fs) in enumerate(videos):
                    # Sliding CPU frame buffer; four inputs and one observed target.
                    window=[read_frame(p) for p in fs[:4]]
                    for t in range(4,len(fs)):
                        target=read_frame(fs[t])
                        x=torch.from_numpy(np.stack(window).reshape(1,12,256,256)).to(device)
                        pred,_=model(x)
                        tgt=torch.from_numpy(target[None]).to(device)
                        mse=((pred-tgt)/2).square().mean().item()
                        raw.append(10*np.log10(max(mse,1e-12)))
                        gt.append(y[offset+t]);vids.append(v);indices.append(t)
                        window=window[1:]+[target]
                    offset+=len(fs)
                    print(f"{variant} {dataset} {name} {len(raw)} frames",flush=True)
            tmp=dst.with_suffix(".npz.tmp")
            with tmp.open("wb") as f:
                np.savez_compressed(f,raw=np.asarray(raw),label=np.asarray(gt),video=np.asarray(vids),
                    frame=np.asarray(indices),video_names=np.asarray([v[0] for v in videos]))
            tmp.replace(dst)
            save_json(meta,{"freeze_sha256":sha(freeze_path),"scores_sha256":sha(dst)})


if __name__=="__main__": main()
