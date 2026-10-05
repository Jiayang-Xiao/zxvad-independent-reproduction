# Technical reproduction review: current legacy_v1 baseline

Prepared 2026-10-05. This document describes **our current implementation**, not the authors' implementation. Items marked pending are questions, not verified paper requirements. The paper/supplement supply the stated size, batch size, iterations, optimizer settings, augmentation forms, cosine memory addressing and PatchGAN N/D; they do not establish our unspecified choices. Original source: [main paper](https://www.merl.com/publications/docs/TR2023-001.pdf), [supplement](https://openaccess.thecvf.com/content/WACV2023/supplemental/Aich_Cross-Domain_Video_Anomaly_WACV_2023_supplemental.pdf).

## Dataset, sampling and initialization

- SHT/SHT: all 330 normal ShanghaiTech training videos for G and O; no target training/adaptation or target-selected checkpoint.
- PIL RGB decoding and bilinear resize to256x256; array/127.5-1; no ImageNet standardization.
- Each iteration samples eight clips. A local sampler RNG is seeded as17*1000003+zero_based_step. Video is uniformly sampled, then the valid clip start uniformly sampled. Five consecutive frames, stride1; first four flattened to12 input channels and fifth used as target. With replacement; no epoch-based traversal or learning-rate schedule.
- Python, NumPy and Torch seeds17; random model initialization in order G,D,N,O,ArcFace. O.eval(); G,D,N,ArcFace.train(). All O parameters frozen. No pretrained weights.
- AdamG2e-4,D/N2e-5,betas(.5,.999); remaining Adam defaults. N optimizer includes two ArcFace class centres. Batch8,5000updates, no AMP, clip or accumulated gradients. TF32 disabled; cuDNN benchmark false/deterministic true; torch deterministic algorithms true; CUBLAS_WORKSPACE_CONFIG=:4096:8.
- Audited run: torch2.4.1,CUDA12.1,NumPy2.2.6,RTX3090. Python3.10 was seen in earlier server diagnostics. Historical torchvision/Pillow versions were not saved. Current publication-host environment, if present, has a distinct label and is not retroactively assigned to the legacy run.

**Pending Q01:** uniform videos versus uniform clips; stride, preprocessing/color/resize and dataset frame extraction/label conventions.

## Generator, discriminator, classifier and memory

G receives Bx12x256x256. Four DoubleConv blocks use two3x3 stride1 padding1 biased convolutions+ReLU each, channels64/128/256/512. Three maxpool2 stages produce a512x32x32 bottleneck. Memory bank2000x512 is learned; initial weights are uniform[-1/sqrt(512),1/sqrt(512)]. Each spatial query uses raw F.linear with memory weights, softmax across2000 slots, hard shrink threshold.0005, L1 renormalization, and a weighted memory read. Transpose convolutions3x3 stride2 padding1 output_padding1 upsample512→256→128→64; concatenation with encoder features followed by DoubleConv512→256,256→128,128→64. Output head64→64→64→3 with3x3/ReLU/ReLU/tanh. No G BatchNorm or dropout.

D is five4x4 stride2 padding1 convolutions3→64→128→256→512→1. Internal128/256/512 stages use bias=false,BN,LeakyReLU(.2); first stage uses bias and LeakyReLU; final convolution has no sigmoid. Output Bx1x8x8. This particular D architecture has not been confirmed as the original D.

N is random torchvision ResNet18 through layer4, yielding Bx512x8x8, followed by global average pooling and a biased512→1 linear layer. It returns the scalar logit and pre-pooling features. This **differs from the paper's PatchGAN N**. No sigmoid.

**Confirmed difference D01:** scalar ResNet18 N versus paper PatchGAN. **D02:** raw dot-product memory versus paper cosine similarity. **Pending Q02:** exact G/D/N layers/initialization and normalization; N score-map shape and SCDA layer/channel/spatial shape; compatible ArcFace dimension; memory insertion points/slots/dimension/initialization/cosine-softmax temperature.

## Attention, synthesis and augmentation

Our SCDA implementation is A=sum_c(abs(feature)); divide each sample's map by its spatial maximum clamped at1e-8. N map8x8 is flattened to64 elements for ArcFace. N attention has a differentiable normalizer; O attention does not require gradients.

O is a random frozen ResNet50 truncated before pooling/fc, kept in eval mode including BN. It receives normalized[-1,1] frames. Its channel-summed absolute features are spatial-max normalized; binary object mask is A>.1, resized by nearest-neighbor to256x256.

One Python random.randrange(4) donor index is drawn **for the whole batch**; each donor is that input frame from the same clip. Base is the first input frame. A donor may equal its base. For each sample beta is uniform[0,1), width/height=max(1,int(256*sqrt(1-beta))); centre x/y uses integer randint0..256 inclusive. Our left/top clamp the full rectangle inside the image, preserving width/height. The whole masked donor is resized bilinearly to that rectangle; mask uses nearest-neighbor. Only true mask pixels replace the base. This **differs from corner clipping specified in the supplement**. The full pasted mask is area-resized to the N attention grid for the attention loss; no foreground bounding-box crop.

Augmented-normal is generated from the detached predicted frame. torchvision Compose applies ColorJitter(.1 each),RandomAffine360,RandomPerspective(.2,p=1), independently per image after converting[-1,1]→[0,1], then back. Interpolation and fill arguments are omitted and follow the installed torchvision version's defaults; historical versions were not captured, so exact historical defaults are not certified. No augmentation is applied to source clip inputs in this baseline.

**Confirmed difference D03:** torchvision versus supplement Kornia AugmentationSequential. **D04:** whole-box relocation versus clipped corners. **Pending Q03:** signed versus absolute SCDA sum, normalization, threshold treatment, mask resize, O BN/input normalization, independent donor/base sampling, foreground crop versus full masked donor, integer/continuous box and rounding details, Kornia version/input range/interpolation/padding/independent or shared draws.

## Exact loss reductions and update sequence

All MSEs use mean over all their tensor elements. Denote predicted frame p=G(x), target y, pseudo frame q, scalar N logits n=N(p),z=N(q), attention A_p/A_q and pasted mask M. Negative/nonfinite losses abort.

```text
L_bb = MSE(p,y) + (1-SSIM(p,y)) + L_gradient(p,y) + .0025*H(memory_attention)
L_G_adv = .5*MSE(D(p),1)
L_N_guide = .5*MSE(N(p),1)
L_G = L_bb + .05*L_G_adv + .5*L_N_guide
L_D = .5*MSE(D(y),1) + .5*MSE(D(detach(p)),0)
L_N = .5*MSE(n,1) + .5*MSE(z,0)
L_RN = .5*MSE(n-mean(z),1) + .5*MSE(z-mean(n),-1)
L_AA = .5*MSE(A_p,1) + .5*MSE(A_q,area_resize(M))
L_NC = L_N + .01*L_RN + L_AA + L_ArcFace
```

H=-sum_slots(att*log(att+1e-12)) averaged over batch/spatial. Gradient loss is the mean absolute difference between absolute horizontal gradients plus the analogous vertical term (alpha1). SSIM uses11x11 Gaussian sigma1.5, zero padding5, per-channel grouped convolution, constants.01^2/.03^2 directly on[-1,1], mean over every element. The data-range/constants and SSIM implementation remain unconfirmed relative to the original code.

ArcFace normalizes flattened attention embeddings and two learned class-centre rows, clamps cosine to[-1+1e-7,1-1e-7], replaces the true-class cosine with cos(acos(cosine)+28.6degrees), scales64, and uses mean cross-entropy. Inputs are predicted normal, augmented-predicted normal, and pseudo; labels1,1,0. The centres are optimized with N; no extra embedding network.

Each step:

1. Generate p and memory attention; synthesize q without gradients from source inputs using frozen O.
2. N(p) for generator guidance; then an unused N(q) forward, both in train mode. D(p) is evaluated twice in the expression constructing L_G_adv, also in train mode. G optimizer zero_grad/backward/step. N/D parameter gradients are accumulated here and cleared before their own updates; their BN buffers still update.
3. D(y),D(detach(p)); D optimizer zero_grad/backward/step.
4. N(detach(p)),N(detach(q)); calculate classification/relative/attention losses. No fresh G prediction after its optimizer step.
5. N(augment(detach(p))) supplies ArcFace attention. N/ArcFace optimizer zero_grad/backward/step.

Total N forwards/BN updates:5 per step, including one unused pseudo forward; first BN counter25000 after5000steps. D BN updates4 per step. N-to-G gradients are connected; N's own update is detached from G. G/D/N optimizer order and reuse are current implementation choices, not facts established by the paper equations.

**Pending Q04:** exact coefficients/reduction axes for PatchGAN, SSIM implementation/range, entropy/gradient conventions and ArcFace centre/input handling; N normal examples (predicted versus real); G/D/N order; regeneration of prediction; freezing parameters versus BN mode; expected forward counts and unused forwards.

## Scoring and evidence

Use only final5000-step G.eval(), full-frame prediction with four consecutive inputs. For every target frame t>=4, mse=mean(((prediction-target)/2)^2); raw=10*log10(max(mse,1e-12)) equals negative PSNR in[0,1]. Larger raw means more anomalous. Video labels for first four frames are excluded. Each video's raw scores are min-max normalized, constant videos become all0. Pool all scored frames **within each target dataset** and compute AUROC using half credit for score ties. Equal-target macro is the mean of three target AUROCs. No score weighting, smoothing, filtering, target adaptation, threshold metric or checkpoint selection.

Legacy score archives hold raw,label,video,frame,video_names. Their rows are36videos7056frames Ped1,12/1962 Ped2,21/15240 Avenue. LegacyAUROC=.7404233708134975/.9322106707096961/.7371757489775154; macro=.803269930166903. Paper fixed SHT/SHT row=.7614/.9578/.8228. The paper-row mean.8473333333333333 is derived here, not the paper's general narrative mean84.26%.

Source metadata manifests identify names/sizes/mtimes, not image-content hashes. Label files are hashed in the legacy evidence provenance. Raw-score consistency is checked; model checkpoints/images were not replayed in the local publication audit. Source-only train data and target evaluation are separate; targets are development-observed following earlier experiments.

**Pending Q05:** final checkpoint, per-video normalization then pooled versus averaged AUROC, first-four-frame handling, smoothing/boundaries, data-specific frame/label conventions. Supplement confirms individual-video PSNR normalization; other conventions need confirmation.

## Correction record

No author confirmation has yet been recorded. Preserve legacy_v1 results and their provenance. After clarification, freeze a separate paper-alignment version with every change and remaining assumption declared. Multi-item restoration of paper behavior is reproduction work, not a one-flaw improvement claim. Subsequent improvements require new paired measurements against the corrected baseline.

| Item | Current status | Author confirmation | Future code/version |
|---|---|---|---|
| D01/N architecture and SCDA | Known difference; exact replacement pending | Pending | Pending |
| D02/memory cosine | Known difference; remaining memory settings pending | Pending | Pending |
| D03/Kornia | Known difference; versions/range/defaults pending | Pending | Pending |
| D04/corner clipping | Known difference; rounding/cropping details pending | Pending | Pending |
| Q01/sampling/data | Unconfirmed implementation choices | Pending | Pending |
| Q02/G,D,N,memory details | Unconfirmed implementation choices | Pending | Pending |
| Q03/attention,synthesis,augmentation | Unconfirmed implementation choices | Pending | Pending |
| Q04/losses and update | Unconfirmed implementation choices | Pending | Pending |
| Q05/evaluation | Normalization principle confirmed; others pending | Pending | Pending |
