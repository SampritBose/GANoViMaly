"""Run frame-level anomaly scoring on a video using prediction error / PSNR."""
import argparse
import csv
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from config import Config
from models import GANoViMaly


def preprocess(frame, size, device):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA)
    x = gray.astype(np.float32) / 255.0
    x = (x - 0.5) / 0.5
    return torch.from_numpy(x).unsqueeze(0).unsqueeze(0).to(device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--checkpoint", default="checkpoints/latest.pt")
    parser.add_argument("--output", default="outputs/anomaly_scores.csv")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Optional anomaly-score threshold after normalization")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    cfg = Config()
    if isinstance(checkpoint, dict) and "config" in checkpoint:
        for key, value in checkpoint["config"].items():
            if hasattr(cfg, key):
                setattr(cfg, key, value)
    model = GANoViMaly(cfg.channels, cfg.feature_dim, cfg.embed_dim,
                       cfg.mamba_layers, cfg.patch_size).to(device)
    weights = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    model.load_state_dict(weights)
    model.eval()

    cap = cv2.VideoCapture(args.video)
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    if len(frames) <= cfg.clip_length:
        raise ValueError(f"Video needs at least {cfg.clip_length + 1} frames.")

    errors = []
    with torch.no_grad():
        for i in range(cfg.clip_length, len(frames)):
            clip = torch.stack(
                [preprocess(frames[j], cfg.image_size, device).squeeze(0)
                 for j in range(i - cfg.clip_length, i)], dim=0
            ).unsqueeze(0)  # [1,T,1,H,W]
            target = preprocess(frames[i], cfg.image_size, device)
            pred = model.generator(clip)
            mse = F.mse_loss(pred, target).item()
            psnr = 10.0 * np.log10(4.0 / max(mse, 1e-10))  # normalized [-1,1] range
            errors.append((i, mse, psnr))

    psnrs = np.array([x[2] for x in errors], dtype=np.float32)
    denom = float(psnrs.max() - psnrs.min())
    regularity = (psnrs - psnrs.min()) / denom if denom > 1e-8 else np.ones_like(psnrs)
    anomaly = 1.0 - regularity
    # Smooth the score curve as described in the paper.
    anomaly = cv2.GaussianBlur(anomaly.reshape(-1, 1), (1, 0), sigmaX=0).reshape(-1)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["frame_index", "mse", "psnr", "regularity_score",
                         "anomaly_score", "predicted_anomaly"])
        for k, (frame_idx, mse, psnr) in enumerate(errors):
            predicted = int(args.threshold is not None and anomaly[k] >= args.threshold)
            writer.writerow([frame_idx, mse, psnr, float(regularity[k]),
                             float(anomaly[k]), predicted])
    print(f"Saved {len(errors)} frame scores to {out}")


if __name__ == "__main__":
    main()
