"""Video clip dataset.

Expected layout:
data/
  train/
    normal_video_001/000001.jpg
    normal_video_001/000002.jpg
    normal_video_002/...
  test/
    video_001/000001.jpg
    video_001/000002.jpg
or video files can be placed in train/ and test/.
Training should contain normal videos only for the one-class setting.
"""
from pathlib import Path
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}


def read_video_frames(path):
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(gray)
    cap.release()
    return frames


def load_frame(path, image_size):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Could not read image: {path}")
    image = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_AREA)
    # Paper preprocessing: grayscale, 256x256, mean/std 0.5.
    image = image.astype(np.float32) / 255.0
    image = (image - 0.5) / 0.5
    return torch.from_numpy(image).unsqueeze(0)


class VideoClipDataset(Dataset):
    def __init__(self, root, clip_length=8, image_size=256, stride=1):
        self.root = Path(root)
        self.clip_length = clip_length
        self.image_size = image_size
        self.clips = []
        if not self.root.exists():
            raise FileNotFoundError(f"Dataset directory not found: {self.root}")
        for item in sorted(self.root.iterdir()):
            if item.is_dir():
                frames = sorted(p for p in item.iterdir() if p.suffix.lower() in IMAGE_EXTS)
                if frames:
                    self._add_frame_clips(frames, stride)
                else:
                    for video in sorted(p for p in item.iterdir() if p.suffix.lower() in VIDEO_EXTS):
                        self._add_video_clips(video, stride)
            elif item.suffix.lower() in VIDEO_EXTS:
                self._add_video_clips(item, stride)
        if not self.clips:
            raise RuntimeError(
                f"No clips found in {self.root}. Use subfolders of ordered frames or video files."
            )

    def _add_frame_clips(self, frames, stride):
        length = self.clip_length + 1
        for start in range(0, len(frames) - length + 1, stride):
            self.clips.append(("images", frames[start:start + length]))

    def _add_video_clips(self, video_path, stride):
        frames = read_video_frames(video_path)
        length = self.clip_length + 1
        for start in range(0, len(frames) - length + 1, stride):
            self.clips.append(("arrays", frames[start:start + length]))

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, index):
        kind, items = self.clips[index]
        tensors = []
        for item in items:
            if kind == "images":
                tensors.append(load_frame(item, self.image_size))
            else:
                frame = cv2.resize(item, (self.image_size, self.image_size),
                                   interpolation=cv2.INTER_AREA)
                frame = frame.astype(np.float32) / 255.0
                frame = (frame - 0.5) / 0.5
                tensors.append(torch.from_numpy(frame).unsqueeze(0))
        clip = torch.stack(tensors, dim=0)  # [T+1,C,H,W]
        return clip[:-1], clip[-1]
