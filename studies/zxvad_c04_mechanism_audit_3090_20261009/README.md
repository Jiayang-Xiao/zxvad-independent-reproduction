# c04 mechanism audit, 3090, seed17

Fixed author-unconfirmed c04 independent implementation. This study audits normalcy image-origin shortcuts and counterfactual past-input training. It does not claim official zxVAD reproduction or SOTA.

32 final5000-step fits,16 sequential fits on each physical3090. The source is partitioned by video:297 training /33 heldout, with both recipients and donors isolated. All fits, all source probes and source-only choices finish before target evaluation. This is not the paper all330 training protocol. Original41 code files remain byte-frozen in frozen_original; crosscheck adapts process bindings and uses the separately audited experiment_update.

NC: two baselines, four source-matched functional arms, a deliberately false-label origin diagnostic and a guidance-off ablation. CF: PHOTO, NOISE, SELF, STATIC, MOVE and SHUFFLE at two areas and two probabilities. Every family preserves one generator forward and four normalcy BN passes per step. No combinations or multi-seed fits.

Primary performance is pooled frame AUROC of per-video minmax negative PSNR. NC, NC_GAP, LOCAL and source-calibrated fusion, plus fixed-scale scores, are declared diagnostic readouts;96 primary rows and672 readout rows are not672 training runs. N.eval and unchanged BN buffers are enforced on inference. No target parameter updates or checkpoint choice. Prior target feedback informed design; these are development comparisons, not blind results.

See plan.zh.md, docs/factual_review.md, protocol.json and spec.py. Source synthetic preferences do not prove real anomaly generalization. The pseudo-anomaly-to-clean objective and moving patches have prior art (BMVC2021, Learning Not to Reconstruct Anomalies).

Server root: /home/xjy/zxvad-c04-mechanism-audit-v1. Existing completed parent and pinned interpreter are required. Offline deployment requires no network or Codex on the server. Use bash launch.sh; bash progress.sh; after COMPLETED/exit0, bash publish.sh with the established local proxy tunnel. Rerunning matching deployment/launch resumes250-step checkpoints without changing the release. An occupied GPU waits without stopping other work. Need25GiB free disk.

Public branch: c04-mechanism-audit-3090-20261009. Publication contains frozen code, plans, calibration, all raw scores and both positive/negative results; no images, weights, environments or credentials. Each new candidate compares with its same-card fresh baseline. Parent all330 scores are not the paired baseline.

Local tests are CPU synthetic implementation diagnostics, not measured GPU results. Real physical-card parity and GPU preflight are mandatory before fresh fits.
