# Paired c04 T/N/Q factorial on RTX3090

Independent zxVAD implementation, author details remain unconfirmed. Baseline c04 is frozen from commit `3eb129ec5362ebd0575f9531d6ab178f4149e879` of Jiayang-Xiao/zxvad-independent-reproduction. Its full model source is byte-identical in src/baseline_model.py. Original source snapshots and SHA identities are in provenance/.

Eight arms B,T,N,Q,TN,TQ,NQ,TNQ each train from scratch, seed17,5000 updates,batch8, same frozen source/donor plan. All fits finish before all24 final target evaluations. B directly calls the unchanged c04 updater; T augments source appearance coherently; N multiplies generator normalcy guidance by0.9001; Q adds0.1 query compactness toward detached nearest memory prototypes. No target-time modules or score changes.

This is prospective development testing chosen using previous target feedback. Historical controls remain controls in the old records. A higher old control AUROC does not establish a new causal mechanism, combination gain or official reproduction. Baseline O masks were nearly full in V2; this batch keeps O unchanged. c04 still fell4.23pp short on Ped2 compared with the paper; no SOTA claim is made.

Default server root /home/xjy/zxvad-c04-combinations-v1, private venv, physical GPU1. Existing lab datasets are read through symlinks; no frames are copied or changed. Target scored labels/mappings are taken from the previously audited V2 artifacts, including Avenue; old lab Avenue annotations are not used. Unscored first4 labels are explicitly missing(-1). Source video counts match V2; image-byte identity between hosts is not assumed.

Run `bash scripts/launch.sh`; check `bash scripts/progress.sh`; after COMPLETED and exit0, run `bash scripts/publish.sh` interactively. A rerun resumes last full checkpoint every250steps with optimizer/RNG state. All raw24 vectors, configuration, source plan, logs, environment and freeze records are published on a new branch; images, checkpoint bytes, credentials and environment packages are excluded. Actual checkpoints remain on server.

See protocol.json and docs/design.md. AUROC is the sole performance metric; all domains and all arms are retained. Seed17 is a screening experiment, not evidence of stability. Local CPU tests are engineering checks; the actual server GPU parity/preflight must also pass before research fits.
