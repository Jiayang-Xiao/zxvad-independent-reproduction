# Reproduction evidence and unresolved assumptions

This independently written study reconstructs plausible zxVAD implementations. It is not author-confirmed code. No claim of an exact reproduction follows from parameter-count matching or a high target AUROC.

## Direct zxVAD evidence

The [main paper, author institution copy](https://www.merl.com/publications/docs/TR2023-001.pdf) and [WACV2023 supplement](https://openaccess.thecvf.com/content/WACV2023/supplemental/Aich_Cross-Domain_Video_Anomaly_WACV_2023_supplemental.pdf) specify four-frame prediction at256x256, PatchGAN D/N, random frozen ResNet50 object localization, normal source-only training, losses and coefficients,5000 iterations/batch8, Adam(.5,.999), G2e-4/D,N2e-5, memory shrink.0005/entropy.0025, Kornia augmentation parameters, ArcFace28.6/64, and per-video PSNR normalization. N removes its last sigmoid. Table4 G=SHT/O=SHT reports Ped1/Ped2/Avenue76.14/95.78/82.28%. The equal-target macro84.7333% is our derived summary, not a separate paper-reported metric.

Neither paper nor the author correspondence supplies the complete layer table, exact N attention tensor/scaling, object-mask scaling, BatchNorm policy, random seeds, sampling implementation or all update-order details. Those remain assumptions.

## Reference implementations and boundaries

| Detail | Primary reference | Selected implementation / limitation |
|---|---|---|
| G skeleton and gradient loss | [FPN author code at prepublication snapshot](https://github.com/StevenLiuWen/ano_pred_cvpr2018/tree/f3043d00c64b83b1ae6ffd4a5e0f4d4a9bda91ac), [FPN paper](https://arxiv.org/abs/1712.09867) | Four DoubleConv levels, no BN, skip concatenation,2x2 transpose convolutions, single output convolution; SAME zero-padded absolute gradient difference. FPN paper says3x3 transpose convolutions, conflicting with its code. |
| Memory | [MemAE author code](https://github.com/donggong1/memae-anomaly-detection/tree/ceece7714fb241e82ef3f3785d3d1ed86c28113e), [MemAE paper](https://arxiv.org/abs/1904.02639) | Reference K=2000 and512 channels; cosine paper branch for c01-c03, dot-product code branch for c04. K=2000 is not explicitly disclosed by zxVAD. Both branches retain zxVAD shrink and entropy weights. |
| G parameter clue | zxVAD Table3 + reference layer count | FPN skeleton7,704,195 plus memory1,024,000 equals8,728,195, rounding to paper8.73M. A rounded total cannot uniquely identify the original architecture. Earlier legacy G9,662,211 remains separate. |
| D/N layers | [pix2pix author PyTorch implementation](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/da39a525eb793614807db4330cfe9b2157bbe33a/models/networks.py) | Three-channel basic PatchGAN: k4/p1, channels64/128/256/512/1, strides2/2/2/1/1, score30x30 and hidden31x31. FPN's TensorFlow PatchGAN uses a different border geometry; the exact zxVAD port is unknown. |
| Attention and O | [SCDA paper](https://arxiv.org/abs/1607.06189), [Tobias author code](https://github.com/CupidJay/Tobias/tree/f2f57d62bb50a73d9c411d7da96df4087a19e871) | Channel aggregation supplies a reference, not the missing normalization/layer definition. SCDA's mean threshold and largest component are not imported. Our O is channel-sum/spatial-max then>.1, frozen eval BN, random weights, RGB[-1,1]; this scaling and BN policy are unconfirmed. |
| ArcFace | [PML v1.6.3 source](https://github.com/KevinMusgrave/pytorch-metric-learning/tree/d3feecec5019ad3e222c055531865c18b1af71e8), [ArcFace paper](https://arxiv.org/abs/1801.07698) |28.6 means degrees in this library. Two trainable class centres share the N optimizer. Other distance/reducer behavior follows that version's defaults. |
| Relative normalcy | [RelativisticGAN author code](https://github.com/AlexiaJM/RelativisticGAN/tree/2646c0b27cfd42c20fcecaf4005c3d67322fdf4a) | No sigmoid, means over batch and spatial scores, gradients flow through means. The specified zxVAD loss weights override reference recipes. |
| SSIM | [Original authors' implementation](https://www.cns.nyu.edu/~lcv/ssim/) | Our choice: images rescaled[0,1], L1,11x11 Gaussian sigma1.5, constants.01/.03, valid window, mean channels/pixels/batch. zxVAD does not specify all these details. |
| Sampling/evaluation | FPN code above; [MPN author code](https://github.com/ktr-hubrt/MPN/tree/a941cec58a5e06d1e4a9040910768efc17777d01) | Source videos round-robin with random consecutive five-frame starts, NumPy shuffle buffer1000 and source-only donor plan. Last valid start is included. Not byte-identical to FPN's TensorFlow sampler; FPN uses BGR while we use RGB. SHT raw-video extraction remains unknown. Per-video PSNR minmax follows zxVAD; first4 exclusion and pooled-frame AUROC follow the reference. |

Main reference[20] is Dual Discriminator GAN, whereas supplementary[20] is MemAE. We did not obtain verifiable official code for Dual Discriminator GAN and do not misidentify it as MNAD.

## Fixed four candidates

c01 uses the last N score directly as attention; c02 adds ReLU/spatial-max scaling; c03 changes to the last hidden feature; c04 changes only memory addressing to dot product. All other shared choices are fixed before target scoring. c01 can have attention-loss scale competition, c02 can have zero attention after ReLU, and c03's hidden attention is still an interpretation, not a recovered fact. The source-only preflight records loss/attention behavior and verifies the required gradient paths. It does not optimize candidates against target data.

G then D then N is our disclosed update order. One G prediction is reused detached by D/N. D/N parameter gradients are disabled during G update while BN still trains; N has four BN forwards per actual step, D three, O zero. Augmentation uses the same step-keyed seed across candidates. Other RNG, sampling and checkpoint settings are in protocol.json.

All four fits complete before any target inference. Only final5000-step G is evaluated. No target adaptation, target checkpoint selection, upstream test-best checkpoint search, Gaussian smoothing, feature fusion or earlier optimization is imported. All12 AUROC rows are retained. Selecting a candidate after observing these target results would constitute development selection, not an untouched evaluation. Bootstrap is not reported; raw per-video scores remain available as reference evidence.

Frame metadata identities use sorted file names, sizes and mtimes, not full image-content SHA256. Labels, source plan, code, checkpoints and raw score files have SHA256 bindings. Canonical frame counts and binary label lengths are checked, but counts alone cannot establish semantic label alignment for arbitrary external data. Transfer the previously audited frame/label tree together where possible.

## Execution scope

A private PyTorch2.7.1/cu128 environment is used for Blackwell: [official PyTorch2.7 announcement](https://pytorch.org/blog/pytorch-2-7/). Local verification uses CPU PyTorch2.4.1/torchvision0.19.1 with historical Kornia0.6.9/PML1.6.3, checking architecture, losses, gradients, recovery and score/export provenance. A real CUDA forward/backward and disposable full batch8 source update for all candidates must pass on the server. Neither GPU compatibility nor paper-level AUROC is inferred from CPU fixtures.
