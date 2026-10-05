# Legacy baseline evidence

These files are the baseline arm of the completed zxVAD MNA run, published at commit056cdb69a6732f80f688455e143eb561d6c67837 in the prior experiment repository. They are **not new results**. The baseline computational update is the original fixed baseline. The original config contains the three-arm MNA study recipe; variant=baseline identifies the arm and uses the all-ones attention target. Nonbaseline motion diagnostics in train.jsonl are retained as original log fields and are not changes to its baseline loss.

The public initial snapshot has a different package/config identity. Its model, baseline update, preprocessing and frame scorer are statically checked against these archived computations. A fresh run is required to assess numerical agreement; static preservation is not a GPU replay.

NPZ contains raw negative PSNR, binary label, video index, frame index, and video names. Results are recomputed from these arrays. Checkpoint hashes and legacy completion records are retained, but model checkpoint/image bytes are not included or replayed here. .json score companions refer to the historical freeze and are provenance records, not a freeze of the new publication. Target identity metadata uses names/sizes/mtimes and label hashes; it does not certify raw image contents or original label conversion.

Reference: https://github.com/Jiayang-Xiao/strict-zs-cross-domain-vad/tree/056cdb69a6732f80f688455e143eb561d6c67837/experiments/zxvad_mna_v1
