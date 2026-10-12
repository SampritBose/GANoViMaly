# GANoViMaly for Video Anomaly Detection 

PyTorch reference implementation inspired by the uploaded paper, **GANoViMaly: A Video Anomaly Detection Framework using Vision Mamba based Generative Adversarial Network**.

## Architecture in this repository

1. **Spatiotemporal attention**: 3D convolutions produce local spatiotemporal features and an attention map.
2. **3D token generation**: features are spatially pooled, projected to token embeddings, and arranged as a sequence.
3. **Vision-Mamba-style generator**: bidirectional gated sequence blocks model temporal context and predict the next frame.
4. **1x1 PatchGAN discriminator**: judges real versus predicted frames.
5. **Objective**: `L_G = alpha * MSE + beta * L_adv + gamma * L1`. Defaults follow the paper's reported strong setting: alpha=1, beta=10, gamma=0.01.
6. **Anomaly score**: prediction PSNR is normalized per video, then `anomaly_score = 1 - regularity_score`.

## Important implementation note

The paper describes Vision Mamba's bidirectional selective state-space mechanism, but does not provide a complete runnable source implementation or all architectural dimensions. To keep this repository installable without custom CUDA extensions, `models.py` implements a **Vision-Mamba-style approximation** using bidirectional gated sequence blocks and depthwise temporal convolutions. It should not be represented as the official Vision Mamba implementation, and the paper's reported AUC/runtime figures are not guaranteed by this code. For exact paper reproduction, replace `BidirectionalSSMBlock` with the official Vim block and match the authors' full training protocol.

The paper reports grayscale frames resized to 256x256, Adam, learning rate 1e-5, 150 epochs, batch size 4, cosine scheduling, and the combined MSE/adversarial/L1 objective.

## Setup

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

## Dataset layout

Use **normal-only** videos for training in the one-class anomaly detection setting.

```text
data/
  train/
    normal_scene_01/
      000001.jpg
      000002.jpg
      ...
    normal_scene_02/
      ...
  test/
    test_video.mp4
```

A directory can contain ordered frame images or video files. For benchmark datasets, adapt the directory parsing and ground-truth label mapping to the official split and annotation format.

## Train

```bash
python train.py --data data/train
```

For a quick smoke test:

```bash
python train.py --data data/train --epochs 1 --batch-size 1
```

Checkpoint is written to `checkpoints/latest.pt`.

## Score a video

```bash
python infer.py --video data/test/test_video.mp4 \
  --checkpoint checkpoints/latest.pt \
  --output outputs/anomaly_scores.csv
```

Optionally set a threshold:

```bash
python infer.py --video path/to/video.mp4 --threshold 0.7
```

The threshold is an application choice, not a threshold specified by the paper. Scores are normalized per video, so they are useful for ranking frames within a video and should not be assumed calibrated across videos.

## Evaluate

Create `labels.csv` with columns `frame_index,label`, where label is 0 for normal and 1 for anomalous frames. Frame indices must match the rows produced by `infer.py`.

```bash
python evaluate.py --scores outputs/anomaly_scores.csv --labels labels.csv
```

## GitHub upload

```bash
git init
git add .
git commit -m "Add GANoViMaly video anomaly detection implementation"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/GANoViMaly.git
git push -u origin main
```

Do not commit datasets, checkpoints, or private data. They are excluded by `.gitignore`.


## Architecture fidelity note

`models.py` now follows the paper's described topology more closely:
- local 3D spatiotemporal attention and 3D non-overlapping spatial patch embedding;
- linearly projected tokens, learnable positional embeddings, and a class token;
- forward and reverse selective SSM paths using `mamba_ssm.Mamba`, with Conv1D,
  gating, and residual connections;
- next-frame prediction decoder; and
- a PatchGAN discriminator with a final 1x1 classifier.

**Important:** the paper does not provide the authors' complete source code or all
layer dimensions and implementation details. Therefore, this is an implementation
of the architecture as described in the paper, not a verified bit-for-bit reproduction
of the authors' original implementation. `mamba-ssm` generally requires a supported
Linux/CUDA/PyTorch environment; installation may not work directly on native Windows.
