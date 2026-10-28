import os, random, json, cv2, torch
from tqdm import tqdm

def compute_video_mean_std(
    video_paths,
    sample_size=200,
    size=(112, 112),
    grayscale=False,
    frame_stride=8,
    progress=True,
):
    """
    Returns per-channel mean/std over pixels in [0,1].
    Uses a subset of videos (sample_size) and samples 1/frame_stride frames per video.
    """
    video_subset = random.sample(video_paths, min(sample_size, len(video_paths)))
    C = 1 if grayscale else 3
    sum_c   = torch.zeros(C, dtype=torch.float64)
    sumsq_c = torch.zeros(C, dtype=torch.float64)
    count   = 0

    iterator = tqdm(video_subset, desc="Computing dataset mean/std", disable=not progress)
    for path in iterator:
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            continue
        idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if idx % frame_stride != 0:
                idx += 1
                continue

            if grayscale:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)      # (H,W)
                frame = cv2.resize(frame, size)
                x = torch.from_numpy(frame).float().view(1, -1) / 255.0     # (1, N)
            else:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)       # (H,W,3)
                frame = cv2.resize(frame, size)
                x = torch.from_numpy(frame).float().permute(2,0,1).view(3,-1) / 255.0  # (3, N)

            sum_c   += x.sum(dim=1).to(torch.float64)
            sumsq_c += (x**2).sum(dim=1).to(torch.float64)
            count   += x.shape[1]
            idx     += 1
        cap.release()

    if count == 0:
        raise RuntimeError("No frames read while computing mean/std.")

    mean = (sum_c / count)
    var  = (sumsq_c / count) - mean**2
    std  = torch.sqrt(var.clamp_min(1e-12))

    return mean.tolist(), std.tolist()


def save_stats(mean, std, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump({"mean": mean, "std": std}, f)


def load_stats(path):
    with open(path, "r") as f:
        d = json.load(f)
    return d["mean"], d["std"]
