# zxVAD independent reproduction V2 on RTX5090

This is an independently written, paper-guided implementation audit, not the unavailable authors' code. It studies four preset interpretations of N attention and the paper/code memory-addressing conflict. The common architecture follows directly cited works and the explicit zxVAD settings. All implementation assumptions are public in `protocol.json` and `docs/reproduction_ledger.md`.

Each candidate trains only on all330 normal ShanghaiTech training videos for5000 updates, batch8, seed17. D and N are PatchGAN. G has8,728,195 parameters including2000x512 memory. After all four fits, their fixed final checkpoint is tested on Ped1/Ped2/Avenue. AUROC is the only performance measure. All twelve rows are reported; no target training, checkpoint search, test-best early stopping, smoothing or score fusion is performed.

| Candidate | N attention | Memory |
|---|---|---|
| c01 | Final score, raw channel sum | Cosine |
| c02 | Final score, ReLU/spatial-max normalization | Cosine |
| c03 | Last hidden feature, sum/ReLU/spatial-max | Cosine |
| c04 | Same attention as c03 | Official MemAE dot product |

The published earlier legacy implementation remains a separate identity. This study does not combine any of its ten optimization experiments. Parameter matching8.73M does not uniquely prove the original architecture. FPN paper/code differ on transpose-convolution kernel; MemAE paper/code differ on cosine versus dot addressing. N attention layer and normalization, O mask scaling, update order, SSIM range and sampling are unconfirmed choices, not facts recovered from the authors. A target-observed choice among candidates is a development choice.

## Server execution

Codex is not required on the server. A supplied self-contained Bash bootstrap creates a new workspace and dispatches `scripts/run.sh` with nohup. It installs a private torch2.7.1/torchvision0.22.1/CUDA12.8 venv, performs a real CUDA compatibility check, checks data and runs the pipeline. It does not replace the server's base environment.

```bash
bash scripts/launch.sh
bash scripts/progress.sh
tail -n 30 -f outputs/execution.log
```

If training data or canonical target frames/labels are absent, the job records`WAITING_FOR_DATA` and exits20. Existing ShanghaiTech testing frames do not meet the training-data requirement. Configure correct paths and relaunch the same script; no completed identity is overwritten.

```bash
/environment/miniconda3/bin/python scripts/configure_data.py --data-root /path/to/MPN/data --source /path/to/MPN/data/shanghai/training/frames
bash scripts/launch.sh
```

Expected target structure below the chosen data root:

```text
ped1/testing/frames/<video>/*.{jpg,png,tif,...}
ped2/testing/frames/<video>/*.{jpg,png,tif,...}
avenue/testing/frames/<video>/*.{jpg,png,...}
frame_labels_ped1.npy
frame_labels_ped2.npy
frame_labels_avenue.npy
```

Source must have330 usable video folders. Targets require36/12/21 folders with7200/2010/15324 total frames and7056/1962/15240 scored frames after first4 removal. Labels are flat or flattenable binary arrays,0normal/1abnormal, ordered by lexicographically sorted videos and frames. Counts cannot by themselves prove the semantic alignment of an arbitrary external label file. Prefer the previously audited MPN-derived data/labels together.

Training snapshots include optimizer/RNG and are saved atomically every250 steps. All fits start fresh after disposable source-only preflight; preflight checks shapes, parameter count, connected guide gradient, detached N update, frozen O and inference batching consistency. `outputs/source_plan.npy` is common to all candidates. No target scores are computed before all fits complete.

`nohup` permits SSH disconnection after dispatch; the server must remain running. No tmux session is required. Tail's Ctrl-C stops monitoring only. A later relaunch restores the last250-step checkpoint after archiving any uncommitted log tail. It resumes within this same frozen runtime/data/code identity; changing identity requires a new workspace.

## Evidence and publication

Final outputs are`results.csv`,`summary.json`,`report.md`,`freeze.json`, raw target NPZ scores, configs, training logs and`review_bundle.zip`. No fabricated or CPU-fixture AUROC is a research result. Bootstrap intervals are not reported. All model checkpoints remain private server files, and no datasets/credentials/private email are published.

After the job completes, run interactively:

```bash
bash scripts/publish.sh
```

The script authenticates GitHub if needed and publishes code plus completed score evidence in the existing independent repository on a new`repro-v2-5090-20261007`branch, under`studies/zxvad_repro_v2_5090_20261007`. It leaves main unchanged, uses a normal push, and refuses mismatched existing study evidence. Keep the connection until Branch/Commit are printed, then return the commit for review.

Local validation covers CPU code/math/gradient/resume/export and Bash delivery plumbing. The production environment and actual data/CUDA results must pass on the server; no paper-level performance is promised.
