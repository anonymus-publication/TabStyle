# utils/video_io.py

import os
import cv2
import torch
from tqdm import tqdm

def load_video(
    path: str,
    video_dim: tuple[int, int],
    grayscale: bool = False,
    show_tqdm: bool = False
) -> tuple[torch.Tensor, float]:
    """
    Load a video and return it as a torch tensor of shape (T, C, H, W), plus its FPS.
    
    Args:
        path: Path to the video file
        video_dim: Target (H, W) size to resize each frame
        grayscale: If True, load as grayscale (1 channel)
        normalizer: If "imagenet" or "kinetics", and grayscale, repeat channel to match pretrained expectations
        show_tqdm: If True, show progress bar while loading

    Returns:
        frames: Tensor of shape (T, C, H, W)
        fps: Frames per second
    """
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    frames = []

    H, W = video_dim  # semantic clarity

    iterator = range(total) if total else iter(int, 1)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # -------------------- GRAYSCALE PATH --------------------
        if grayscale:
            # Handle both (H,W) and (H,W,3)
            if frame.ndim == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # frame is now (H,W)

            frame = cv2.resize(frame, (W, H))
            frame = torch.from_numpy(frame).float() / 255.0
            frame = frame.unsqueeze(0)  # (1,H,W)
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, video_dim)
            frame = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0

        frames.append(frame)

    cap.release()

    if not frames:
        raise ValueError(f"No readable frames in {path}")

    frames = torch.stack(frames)  # (T, C, H, W)

    return frames, fps
