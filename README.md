# zxVAD independent implementation and reproduction review

An independent implementation of **Cross-Domain Video Anomaly Detection without Target Domain Adaptation** (WACV 2023), prepared for a technical reproduction review. This repository contains no official author code. The current `legacy_v1` implementation has known differences from the paper and has not been validated as a faithful reproduction.

The initial publication preserves the baseline computations used in our earlier experiments. It provides a clean directory, a baseline-only training entry point, an evaluation entry point, and explicit provenance. It does not introduce a new PatchGAN N or attribute old scores to a corrected model. Future author-confirmed corrections will receive a separate version and new experiments.

## Start here

- [Complete implementation description and reproduction questions](docs/author_review.md)
- [Versioned configuration](src/protocol.json)
- [Model and losses](src/baseline_zxvad.py), [training update](src/train.py), [evaluation](src/evaluate.py)
- [Legacy baseline evidence](evidence/legacy_baseline/README.md), [AUROC table](evidence/legacy_baseline/results.csv)
- [Code provenance and transformations](provenance/README.md)

Primary sources: [paper](https://www.merl.com/publications/docs/TR2023-001.pdf), [supplement](https://openaccess.thecvf.com/content/WACV2023/supplemental/Aich_Cross-Domain_Video_Anomaly_WACV_2023_supplemental.pdf).

## Baseline results

| Target | Legacy baseline AUROC (%) | Paper Table 4 SHT/SHT (%) |
|---|---:|---:|
| Ped1 | 74.0423 | 76.14 |
| Ped2 | 93.2211 | 95.78 |
| Avenue | 73.7176 | 82.28 |
| Equal-target mean (derived) | 80.3270 | 84.7333 |

Legacy scores are from an already completed, audited run, not a fresh run of this newly packaged repository. The released model and baseline loss-update computations are checked against that run's archived code. Packaging/configuration identities differ; fresh-run logs must be distinguished from legacy evidence. The comparison with the paper remains provisional until all evaluation details are aligned. AUROC is the only performance metric. No bootstrap interval is reported.

Known differences: scalar ResNet18 N instead of PatchGAN; raw-dot memory instead of cosine addressing; torchvision instead of Kornia augmentation; whole-box relocation instead of corner clipping. See the review document for unresolved choices and actual equations.

## Requirements and data

Use the existing server environment first. The audited run recorded Python 3.10 from earlier server diagnostics, torch 2.4.1, CUDA build 12.1, NumPy 2.2.6, and NVIDIA RTX 3090. torchvision and Pillow are required, but their historical versions were not recorded. This repository does not invent a dependency lock. `scripts/collect_environment.py` records the publication host's current environment separately; it does not reconstruct the historical environment.

Provide datasets outside the repository, without redistributing video frames:

```text
SOURCE/
  <video>/*.jpg                 # all 330 normal ShanghaiTech training videos
TARGET_ROOT/
  ped1/testing/frames/<video>/*.jpg
  ped2/testing/frames/<video>/*.jpg
  avenue/testing/frames/<video>/*.jpg
  frame_labels_ped1.npy
  frame_labels_ped2.npy
  frame_labels_avenue.npy
```

Each label file is a flat binary array covering **all** test frames, ordered by lexicographically sorted video directory and frame filename. Anomaly=1. The evaluator uses label offset+t while scoring t>=4. Canonical counts are 36/12/21 videos and 7056/1962/15240 scored frames. Frame-image contents and label conversion from the original datasets have not been independently certified by this release's local audit.

## Verify without GPU or datasets

```bash
python scripts/verify_repository.py
python scripts/audit_legacy_scores.py
```

The first command verifies packaged hashes and AST equivalence of model/loss/preprocessing computations against archived code. The second recomputes AUROC from public frame scores with NumPy. Local verification of this publication does not execute PyTorch/CUDA training.

## Fresh training of legacy_v1

This optional command launches one 5000-step baseline fit and three target evaluations. It reproduces the current independent implementation's choices; it is not a paper-aligned corrected experiment.

```bash
REPRO_PYTHON=/home/xjy/.conda/envs/aris-torch/bin/python \
REPRO_SOURCE=/home/xjy/strict_zs_cross_domain_vad/third_party/MPN/data/shanghai/training/frames \
REPRO_TARGET_ROOT=/home/xjy/strict_zs_cross_domain_vad/third_party/MPN/data \
bash scripts/launch_baseline.sh
```

Defaults: outputs/legacy_v1_seed17, GPU 0, tmux session zxvad-repro-legacy-v1. CUDA preflight checks the source update and gradients before training. Training checkpoints include optimizer/RNG state every 250 updates. Both G/D/N and ArcFace are random initialized; O is random initialized and frozen. No pretrained weights are downloaded. Training uses only normal source data; target labels are used only in evaluation. Targets have been observed during earlier development experiments, and no untouched-test claim is made.

After the script reports the detached tmux session, the SSH connection may be closed while the server remains running. Attach with `tmux attach -t zxvad-repro-legacy-v1`; detach with Ctrl-B then D. Check outputs/legacy_v1_seed17/execution.log, baseline/state.json, last_exit.txt and results.csv. A successful end has last_exit.txt=`0`, state TRAINED step5000, and summary status COMPLETED. A running loss log alone is not proof of completion.

## Publication and later updates

`scripts/publish_initial.sh` creates the public repository when authenticated GitHub CLI is available, then uses the existing Git SSH authentication for an ordinary push. `scripts/publish_run.sh` exports a completed fresh baseline's compact evidence before committing and pushing it. Neither command sends an email. Credentials, datasets, model checkpoints, private correspondence and workstation cache files are not part of the publication.

Maintain the original code/result identity when incorporating author feedback. Record the source of each confirmed detail in docs/author_review.md; implement corrected paper alignment as a separately frozen version rather than overwriting legacy_v1 evidence.
