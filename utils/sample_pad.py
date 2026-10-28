import numpy as np
import torch

def sample_clip_and_pad(frames, clip_len, method="center"):
    """
    Returns a clip of exactly `clip_len` frames:
    - If video is longer: sample a subset
    - If video is shorter: repeat frames from the start (loop) until length matches
    """
    T = frames.shape[0]

    if T >= clip_len:
        if method == "center":
            start = (T - clip_len) // 2
            clip = frames[start:start + clip_len]
        elif method == "random":
            start = np.random.randint(0, T - clip_len + 1)
            clip = frames[start:start + clip_len]
        elif method == "uniform":
            idxs = np.linspace(0, T - 1, clip_len).astype(int)
            clip = frames[idxs]
        else:
            raise ValueError(f"Unknown sampling method: {method}")
        mask = torch.ones(clip_len, dtype=torch.bool)

    else:
        # Loop from the beginning until we reach clip_len
        repeat_times = clip_len // T
        remainder = clip_len % T
        clip = torch.cat([frames] * repeat_times + [frames[:remainder]], dim=0)
        mask = torch.ones(clip_len, dtype=torch.bool)

    return clip, mask
