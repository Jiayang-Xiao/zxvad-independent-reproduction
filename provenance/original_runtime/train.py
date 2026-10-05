"""Three original zxVAD fits; only the normal attention supervision target differs."""
import argparse
import json
import os
import random
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from baseline_zxvad import (ArcFaceLoss, FutureFrameGenerator, NormalcyAugmentation,
    NormalcyClassifier, PatchDiscriminator, UntrainedAnomalySynthesizer,
    gradient_loss, paste_object, resample_attention, scda_attention, ssim_loss)
from common import VARIANTS, code_identity, digest, frames, identity, sample_batch, save_json, sha
from normal_attention import normal_attention_target, target_diagnostics


def one_update(batch, models, opts, aug, variant, audit=False):
    G, D, N, O, arcface = (models[k] for k in ("G", "D", "N", "O", "arcface"))
    device = batch.device
    x, target = batch[:,:4].reshape(8,12,256,256), batch[:,4]
    v_hat, att = G(x)
    l_mse = F.mse_loss(v_hat, target)
    l_bb = l_mse + ssim_loss(v_hat,target) + gradient_loss(v_hat,target) + 0.0025*G.memory.entropy_loss(att)
    obj_start = random.randrange(4)*3
    obj_frame = x[:,obj_start:obj_start+3]
    with torch.no_grad():
        v_tilde, mask = paste_object(x[:,:3],obj_frame,O.attention_mask(obj_frame))
    logit_hat, _ = N(v_hat)
    # Preserve upstream's extra normalcy forward and BatchNorm update.
    N(v_tilde)
    g_adv = 0.5*F.mse_loss(D(v_hat),torch.ones_like(D(v_hat)))
    l_g = l_bb + 0.05*g_adv + 0.5*(0.5*F.mse_loss(logit_hat,torch.ones_like(logit_hat)))
    if not torch.isfinite(l_g): raise RuntimeError("Nonfinite G update")
    if audit:
        signal = 0.5*(0.5*F.mse_loss(logit_hat,torch.ones_like(logit_hat)))
        grads = torch.autograd.grad(signal, tuple(G.parameters()), retain_graph=True, allow_unused=True)
        active = [g for g in grads if g is not None]
        assert active and all(torch.isfinite(g).all() for g in active)
        assert sum(g.abs().sum().item() for g in active) > 0, "N-to-G gradient disconnected"
        del grads, active, signal
    opts["G"].zero_grad(); l_g.backward(); opts["G"].step()
    d_real, d_fake = D(target), D(v_hat.detach())
    l_d = 0.5*F.mse_loss(d_real,torch.ones_like(d_real)) + 0.5*F.mse_loss(d_fake,torch.zeros_like(d_fake))
    if not torch.isfinite(l_d): raise RuntimeError("Nonfinite D update")
    opts["D"].zero_grad(); l_d.backward(); opts["D"].step()
    logit_hat, feat_hat = N(v_hat.detach())
    logit_tilde, feat_tilde = N(v_tilde.detach())
    a_hat, a_tilde = scda_attention(feat_hat), scda_attention(feat_tilde)
    l_n = 0.5*F.mse_loss(logit_hat,torch.ones_like(logit_hat)) + 0.5*F.mse_loss(logit_tilde,torch.zeros_like(logit_tilde))
    l_rn = 0.5*F.mse_loss(logit_hat-logit_tilde.mean(),torch.ones_like(logit_hat)) + 0.5*F.mse_loss(logit_tilde-logit_hat.mean(),-torch.ones_like(logit_tilde))
    if variant == "baseline":
        l_aa = 0.5*F.mse_loss(a_hat,torch.ones_like(a_hat)) + 0.5*F.mse_loss(a_tilde,resample_attention(mask.detach(),feat_tilde))
    else:
        normal_target = normal_attention_target(batch,variant)
        l_aa = 0.5*F.mse_loss(a_hat,normal_target) + 0.5*F.mse_loss(a_tilde,resample_attention(mask.detach(),feat_tilde))
    _, feat_aug = N(aug(v_hat.detach()))
    att_vecs = torch.cat([a_hat.flatten(1),scda_attention(feat_aug).flatten(1),a_tilde.flatten(1)])
    labels = torch.cat([torch.ones(16,device=device,dtype=torch.long),torch.zeros(8,device=device,dtype=torch.long)])
    l_nc = l_n + 0.01*l_rn + l_aa + arcface(att_vecs,labels)
    if not torch.isfinite(l_nc): raise RuntimeError("Nonfinite N update")
    if audit:
        assert not v_tilde.requires_grad
        grads = torch.autograd.grad(l_nc, tuple(G.parameters()), retain_graph=True, allow_unused=True)
        assert all(g is None for g in grads), "Classifier update leaked into G"
        assert torch.count_nonzero((v_tilde-x[:,:3])*(1-mask)) == 0
    opts["N"].zero_grad(); l_nc.backward(); opts["N"].step()
    return l_g.detach(), l_d.detach(), l_nc.detach(), l_mse.detach()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--variant", choices=VARIANTS, required=True)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required; CPU training is not an implicit fallback")
    out = Path(args.output) / args.variant
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    random.seed(17); np.random.seed(17); torch.manual_seed(17); torch.cuda.manual_seed_all(17)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    videos = frames(args.source)
    if len(videos) != 330 or any(len(fs) < 5 for _, fs in videos):
        raise RuntimeError("Expected all 330 usable ShanghaiTech normal training videos")
    source_id = identity(videos)
    config = {"study": "ZXVAD_MNA_V1", "variant": args.variant, "seed": 17,
        "iterations": 5000, "batch_size": 8, "source_metadata_sha256": digest(source_id),
        "code_sha256": code_identity(), "recipe": json.loads((Path(__file__).parent/"protocol.json").read_text()),
        "recipe_sha256": sha(Path(__file__).parent/"protocol.json"),
        "torch": torch.__version__, "cuda": torch.version.cuda, "numpy": np.__version__,
        "device": torch.cuda.get_device_name(device)}
    cfg_hash = digest(config)
    config_path = out / "config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise RuntimeError("Existing run identity differs; do not overwrite or resume it")
    save_json(config_path, config)
    save_json(out / "source_manifest.json", source_id)
    if (out / "completed.json").exists():
        done = json.loads((out / "completed.json").read_text())
        if done["config_hash"] != cfg_hash or done["checkpoint_sha256"] != sha(out / "last.pt"):
            raise RuntimeError("Completed-run identity mismatch")
        print(f"Already complete: {args.variant}", flush=True)
        return
    G = FutureFrameGenerator(input_frames=4, mem_dim=2000).to(device)
    D = PatchDiscriminator().to(device)
    N = NormalcyClassifier().to(device)
    O = UntrainedAnomalySynthesizer("resnet50").to(device)
    arcface = ArcFaceLoss(in_features=64).to(device)
    aug = NormalcyAugmentation()
    models = {"G": G, "D": D, "N": N, "O": O, "arcface": arcface}
    opts = {"G": torch.optim.Adam(G.parameters(), lr=2e-4, betas=(0.5,0.999)),
            "D": torch.optim.Adam(D.parameters(), lr=2e-5, betas=(0.5,0.999)),
            "N": torch.optim.Adam(list(N.parameters())+list(arcface.parameters()), lr=2e-5, betas=(0.5,0.999))}
    ck_path = out / "last.pt"
    start, elapsed = 0, 0.0
    if ck_path.exists():
        ck = torch.load(ck_path, map_location=device, weights_only=False)
        if ck["config_hash"] != cfg_hash:
            raise RuntimeError("Checkpoint config differs")
        for key, model in models.items(): model.load_state_dict(ck["models"][key])
        for key, opt in opts.items(): opt.load_state_dict(ck["optimizers"][key])
        random.setstate(ck["python_rng"]); np.random.set_state(ck["numpy_rng"])
        torch.set_rng_state(ck["torch_rng"].cpu())
        torch.cuda.set_rng_state_all([x.cpu() for x in ck["cuda_rng"]])
        start, elapsed = ck["step"], ck["elapsed_seconds"]
        del ck
    log_path = out / "train.jsonl"
    if log_path.exists():
        records = [json.loads(s) for s in log_path.read_text().splitlines() if s.strip()]
        kept = [r for r in records if r["step"] <= start]
        if len(kept) != len(records):
            archive = out / ("uncommitted_log_" + str(time.time_ns()) + ".jsonl")
            log_path.replace(archive)
            log_path.write_text("".join(json.dumps(r)+"\n" for r in kept))
    for model in (G,D,N,arcface): model.train()
    O.eval()
    began = time.monotonic()

    def checkpoint(step):
        torch.cuda.synchronize(device)
        ck = {"step": step, "config_hash": cfg_hash,
              "models": {k:m.state_dict() for k,m in models.items()},
              "optimizers": {k:o.state_dict() for k,o in opts.items()},
              "python_rng": random.getstate(), "numpy_rng": np.random.get_state(),
              "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all(),
              "elapsed_seconds": elapsed + time.monotonic()-began}
        tmp = out / "last.pt.tmp"
        torch.save(ck,tmp); tmp.replace(ck_path)
        save_json(out / "state.json", {"status": "TRAINING" if step < 5000 else "TRAINED",
            "step": step, "config_hash": cfg_hash, "elapsed_seconds": ck["elapsed_seconds"]})

    print(f"Starting {args.variant} at step {start}/5000", flush=True)
    for it in range(start,5000):
        batch = torch.from_numpy(sample_batch(videos,it)).to(device)
        l_g, l_d, l_nc, l_mse = one_update(batch, models, opts, aug, args.variant)
        if it == start or (it+1)%50 == 0:
            with torch.no_grad():
                entry = {"step":it+1,"g":l_g.item(),"d":l_d.item(),"n":l_nc.item(),
                    "mse":l_mse.item(), **target_diagnostics(batch,args.variant), "n_bn_updates":int(next(m for m in N.modules() if isinstance(m,torch.nn.BatchNorm2d)).num_batches_tracked.item()), "elapsed_seconds":elapsed+time.monotonic()-began}
            with (out/"train.jsonl").open("a") as f: f.write(json.dumps(entry)+"\n")
            print(args.variant,json.dumps(entry),flush=True)
        if (it+1)%250 == 0: checkpoint(it+1)
    checkpoint(5000)
    save_json(out/"completed.json",{"status":"TRAINED","step":5000,"seed":17,
        "config_hash":cfg_hash,"checkpoint_sha256":sha(ck_path),
        "elapsed_seconds":elapsed+time.monotonic()-began})


if __name__ == "__main__": main()
