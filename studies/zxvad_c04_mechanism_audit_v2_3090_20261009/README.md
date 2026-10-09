# c04 mechanism audit V2 corrective rerun, 3090, seed17

V1 commit61f9232 omitted N/Arc optimizer updates in all30 nonbaseline arms. Those outputs are INVALID_IMPLEMENTATION; two directly delegated baselines remain valid. V2 corrects this defect and reruns the unchanged32-arm plan from step0 in a fresh directory, including paired baselines. It never resumes V1 checkpoints or selects parameters from invalid target results. See docs/v1_failure_review.md.

Fixed author-unconfirmed c04 independent implementation. This study audits normalcy image-origin shortcuts and counterfactual past-input training. It does not claim official zxVAD reproduction or SOTA.

32 final5000-step fits,16 sequential fits on each physical3090. The source is partitioned by video:297 training /33 heldout, with both recipients and donors isolated. All fits, all source probes and source-only choices finish before target evaluation. This is not the paper all330 training protocol. Original41 code files remain byte-frozen in frozen_original; crosscheck adapts process bindings and uses the separately audited experiment_update.

NC: two baselines, four source-matched functional arms, a deliberately false-label origin diagnostic and a guidance-off ablation. CF: PHOTO, NOISE, SELF, STATIC, MOVE and SHUFFLE at two areas and two probabilities. Every family preserves one generator forward and four normalcy BN passes per step. No combinations or multi-seed fits.

Primary performance is pooled frame AUROC of per-video minmax negative PSNR. NC, NC_GAP, LOCAL and source-calibrated fusion, plus fixed-scale scores, are declared diagnostic readouts;96 primary rows and672 readout rows are not672 training runs. N.eval and unchanged BN buffers are enforced on inference. No target parameter updates or checkpoint choice. Prior target feedback informed design; these are development comparisons, not blind results.

See plan.zh.md, docs/factual_review.md, protocol.json and spec.py. Source synthetic preferences do not prove real anomaly generalization. The pseudo-anomaly-to-clean objective and moving patches have prior art (BMVC2021, Learning Not to Reconstruct Anomalies).

Server root: /home/xjy/zxvad-c04-mechanism-audit-v2. Existing completed parent and pinned interpreter are required. Offline deployment requires no network or Codex on the server. Use bash launch.sh; bash progress.sh; after COMPLETED/exit0, bash publish.sh with the established local proxy tunnel. Rerunning matching deployment/launch resumes250-step checkpoints without changing the release. An occupied GPU waits without stopping other work. Need25GiB free disk.

Public branch: c04-mechanism-audit-v2-3090-20261009. Publication contains frozen code, plans, calibration, all raw scores and both positive/negative results; no images, weights, environments or credentials. Each new candidate compares with its same-card fresh baseline. Parent all330 scores are not the paired baseline.

Local tests are CPU synthetic implementation diagnostics, not measured GPU results. CPU regression covers32×3 updates,24 zero-probability CF×3 full-state comparisons and deliberate missing-step mutations. Real physical-card three-step new-updater parity, N/Arc parameter changes and all Adam counters are mandatory before fresh fits; their receipt is bound into preflight, completion and export. Every formal step checks optimizer coverage and counters. Local CPU fixtures do not establish actual CUDA performance.

Verification portability repair: an early CPU gate wrongly required NC_Y and NC_X G parameters to differ after two updates. Cross-arm tiny FP32 differences are now recorded diagnostically. The revised regression directly verifies all four N input routes, actual N/Arc learning, complete optimizer coverage/steps and a controlled N-logit intervention on actual G gradients (NC_OFF negative control). Missing-step mutations and zero-edit full-state parity remain mandatory. No research training code, hyperparameters, data split, queues, plan arrays, GPU gate or metric changed. See docs/regression_gate_repair.md.
