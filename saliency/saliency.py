import torch
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from pathlib import Path
from scipy.ndimage import gaussian_filter1d   # add this import at top of file
from torchvision.models.optical_flow import raft_large, Raft_Large_Weights

# --- load model once ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
weights = Raft_Large_Weights.DEFAULT
raft_model = raft_large(weights=weights).to(device).eval()

def to_device_batch(batch: dict, device: torch.device, use_video: bool = True, use_tabular: bool = True):
    inputs = {}
    # VIDEO 
    if use_video:
        # Always present in batch["frames"], but only moved if needed
        inputs["video"] = batch["frames"].to(device, non_blocking=True)
    else:
        inputs["video"] = None  # Explicitly mark inactive branch
    # TABULAR 
    if use_tabular:
        tab = batch.get("tabular", None)
        if tab is not None:
            tab = tab.to(device, non_blocking=True)
        inputs["tabular"] = tab
    else:
        inputs["tabular"] = None
    # TARGETS 
    targets = {k: v.to(device) for k, v in batch["targets"].items()}
    # META 
    meta = {
        "video_id": batch.get("video_id", None),
        "video_name": batch.get("video_name", None),
        "fps": batch.get("fps", None),
        "num_frames": batch.get("num_frames", None),
    }
    return inputs, targets, meta

def saliency_to_heatmap(sal):
    """
    sal: (H, W) in [0,1]
    returns: (H, W, 3) RGB heatmap
    """
    sal_uint8 = np.uint8(255 * sal)
    heatmap = cv2.applyColorMap(sal_uint8, cv2.COLORMAP_TURBO)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    return heatmap

def overlay_heatmap(frame, heatmap, alpha=0.3):
    """
    frame: (H,W,3) RGB
        - uint8 [0,255] OR
        - float [0,1]
    heatmap: (H,W,3) uint8 RGB
    """

    # ---- Fix frame intensity ----
    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 1)
        frame = (frame * 255.0).astype(np.uint8)

    # ---- Safety: heatmap to uint8 ----
    if heatmap.dtype != np.uint8:
        heatmap = heatmap.astype(np.uint8)

    # ---- Blend ----
    overlay = cv2.addWeighted(
        frame, 1.0 - alpha,
        heatmap, alpha,
        0
    )

    return overlay

def input_gradient_saliency(model, video, class_idx=None):
    """
    video: (B, C, T, H, W)
    returns: saliency (B, T, H, W)
    """
    model.eval()

    video = video.detach().clone()
    video.requires_grad_(True)

    model.zero_grad(set_to_none=True)

    outputs = model(video, None)   # video-only
    y_out = outputs["Rythm_M24"]    # <-- your binary task name

    # binary logit
    if y_out.shape[-1] == 2:
        score = y_out[:, 1] - y_out[:, 0]
    else:
        score = y_out[:, 0]

    score.sum().backward()

    grad = video.grad               # (B,C,T,H,W)
    sal = grad.abs().mean(dim=1)    # (B,T,H,W)

    # normalize per video
    # sal -= sal.amin(dim=(1,2,3), keepdim=True)
    # sal /= (sal.amax(dim=(1,2,3), keepdim=True) + 1e-6)
    # robust normalization (per video)
    flat = sal.view(sal.size(0), -1)
    p_low  = torch.quantile(flat, 0.05, dim=1).view(-1,1,1,1)
    p_high = torch.quantile(flat, 0.95, dim=1).view(-1,1,1,1)

    sal = torch.clamp(sal, p_low, p_high)
    sal = (sal - p_low) / (p_high - p_low + 1e-6)

    return sal.detach()

def normalize_saliency_per_frame(sal, q_low=0.01, q_high=0.99, eps=1e-6):
    # sal: (B,T,H,W) torch
    B,T,H,W = sal.shape
    sal2 = sal.view(B*T, -1)
    lo = torch.quantile(sal2, q_low, dim=1).view(B,T,1,1)
    hi = torch.quantile(sal2, q_high, dim=1).view(B,T,1,1)
    sal = torch.clamp(sal, lo, hi)
    sal = (sal - lo) / (hi - lo + eps)
    return sal

def select_score(out):
    if out.ndim == 1:
        return out
    elif out.shape[-1] == 2:
        return out[:, 1] - out[:, 0]
    else:
        return out[:, 0]

def integrated_gradients_video(
    model,
    video,                      # (B,C,T,H,W) float tensor on device
    tabular=None,
    task_name="Rythm_M24",
    steps=16,                   # 16–32 typically looks better than 8
    baseline="zeros",           # "zeros" | "mean" | "blur"
    blur_kernel=(1, 11, 11),      # (t,h,w) for blur baseline; t=1 recommended
    blur_sigma=3.0,
    chunk_size=2,               # number of alpha steps processed at once
    normalize="per_frame",      # "per_frame" | "per_video" | "none"
    q_low=0.01,                 # robust contrast (1%–99%)
    q_high=0.99,
    eps=1e-6,
    return_baseline=False,   # NEW
    ):
    """
    Returns:
        sal: (B,T,H,W) float in [0,1] (unless normalize="none")
    Notes:
        - Make sure `video` is the *same tensor* your model normally receives
          (i.e., already preprocessed/normalized the same way).
        - This function computes IG for the logit of `task_name`:
            - if output has 2 logits: uses (logit1 - logit0)
            - else: uses out[:,0]
    """
    model.eval()

    x = video.detach()
    B, C, T, H, W = x.shape

    # ----------- Baseline -----------
    if baseline == "zeros":
        x0 = torch.zeros_like(x)

    elif baseline == "mean":
        # per-sample temporal mean baseline (keeps spatial structure)
        x0 = x.mean(dim=2, keepdim=True).expand_as(x)

    elif baseline == "blur":
        # blur each frame spatially; keep temporal axis intact
        # We'll apply a depthwise-ish blur via conv3d with groups=C
        kt, kh, kw = blur_kernel
        if kt != 1:
            raise ValueError("For ultrasound, recommend kt=1 (no temporal blur).")

        # Build Gaussian kernel (kh x kw), then expand to (C,1,1,kh,kw) for groups=C conv3d
        # Do it in torch for device safety
        yy, xx = torch.meshgrid(
            torch.arange(kh, device=x.device) - (kh - 1) / 2,
            torch.arange(kw, device=x.device) - (kw - 1) / 2,
            indexing="ij",
        )
        kernel2d = torch.exp(-(xx**2 + yy**2) / (2 * blur_sigma**2))
        kernel2d = kernel2d / kernel2d.sum()
        kernel3d = kernel2d.view(1, 1, 1, kh, kw)  # (1,1,1,kh,kw)
        weight = kernel3d.repeat(C, 1, 1, 1, 1)    # (C,1,1,kh,kw)

        pad_h = kh // 2
        pad_w = kw // 2

        x_pad = F.pad(x, (pad_w, pad_w, pad_h, pad_h, 0, 0), mode="reflect")  # pad W,H only
        x0 = F.conv3d(x_pad, weight, bias=None, stride=1, padding=0, groups=C)

    else:
        raise ValueError("baseline must be one of: 'zeros', 'mean', 'blur'")
    
    assert x0.device == x.device, f"Baseline device {x0.device} != input device {x.device}"

    # ----------- Alphas -----------
    # Use steps points in (0,1], exclude 0 to avoid baseline gradient issues
    # alphas = torch.linspace(0.0, 1.0, steps + 1, device=x.device)[1:]
    alphas = torch.linspace(1/(2*steps), 1 - 1/(2*steps), steps, device=x.device)

    total_grad = torch.zeros_like(x)

    # ----------- IG integration loop (chunked over alpha) -----------
    for i in range(0, len(alphas), chunk_size):
        a = alphas[i:i + chunk_size]
        m = a.numel()
        a = a.view(m, 1, 1, 1, 1, 1)

        x_step = (x0.unsqueeze(0) + a * (x.unsqueeze(0) - x0.unsqueeze(0))).detach().contiguous()
        x_step.requires_grad_(True)

        x_step_flat = x_step.view(m * B, C, T, H, W)

        # Repeat tabular to match expanded batch size
        # tab = tabular.repeat_interleave(m, dim=0) if tabular is not None else None
        if tabular is not None:
            tab = tabular.unsqueeze(0).expand(m, B, -1).reshape(m * B, -1)
        else:
            tab = None

        out = model(x_step_flat, tab)[task_name]

        if out.ndim == 1:
            score = out
        elif out.shape[-1] == 2:
            score = out[:, 1] - out[:, 0]
        else:
            score = out[:, 0]

        grad = torch.autograd.grad(
            outputs=score.sum(),
            inputs=x_step,
            create_graph=False,
            retain_graph=False,
            allow_unused=False,
        )[0]

        if grad is None:
            raise RuntimeError(
                "Integrated Gradients: grad is None. "
                "Gradient path is broken (e.g., detach inside model)."
            )

        total_grad += grad.sum(dim=0)
        del x_step, x_step_flat, out, score, grad, tab
        # torch.cuda.empty_cache()   # <-- add this

    avg_grad = total_grad / len(alphas)      # (B,C,T,H,W)
    ig = (x - x0) * avg_grad                 # (B,C,T,H,W)

    # ig: (B,C,T,H,W)
    ig_sum = ig.view(B, -1).sum(dim=1)  # (B,)

    # --------- Check can be deleted later 
    with torch.no_grad():
        out_x = model(x, tabular)[task_name]
        out_x0 = model(x0, tabular)[task_name]
        fx = select_score(out_x)    # (B,)
        fx0 = select_score(out_x0)  # (B,)

    # completeness residual: should be close to 0
    delta = ig_sum - (fx - fx0)      # (B,)
    abs_delta = delta.abs()

    # optional relative error (scale-safe)
    den = (fx - fx0).abs().clamp_min(1e-6)
    rel_err = abs_delta / den

    # print(
    #     f"[IG completeness] abs_delta mean={abs_delta.mean().item():.4e}, "
    #     f"median={abs_delta.median().item():.4e}, "
    #     f"p90={abs_delta.quantile(0.9).item():.4e}, "
    #     f"rel_err mean={rel_err.mean().item():.4e}"
    # )
    # ------------------------------

    # Saliency as channel-aggregated magnitude
    # we are only focusing on importance magnitudde not supportinf or suppressing evidence, so we take absolute value before averaging across channels
    sal = ig.abs().mean(dim=1)               # (B,T,H,W)
    baseline_out = x0.detach()

    # ----------- Robust normalization for contrast -----------
    if normalize == "none":
        sal = sal.detach()
        return (sal, baseline_out) if return_baseline else sal

    elif normalize == "per_video":
        # normalize across T*H*W per sample
        flat = sal.view(B, -1)
        lo = torch.quantile(flat, q_low, dim=1).view(B, 1, 1, 1)
        hi = torch.quantile(flat, q_high, dim=1).view(B, 1, 1, 1)
        sal = torch.clamp(sal, lo, hi)
        sal = (sal - lo) / (hi - lo + eps)
        sal = sal.detach()
        return (sal, baseline_out) if return_baseline else sal

    elif normalize == "per_frame":
        # normalize each frame separately (best for avoiding low contrast + temporal flicker)
        flat = sal.view(B * T, -1)
        lo = torch.quantile(flat, q_low, dim=1).view(B, T, 1, 1)
        hi = torch.quantile(flat, q_high, dim=1).view(B, T, 1, 1)
        sal = torch.clamp(sal, lo, hi)
        sal = (sal - lo) / (hi - lo + eps)
        sal = sal.detach()
        return (sal, baseline_out) if return_baseline else sal

    else:
        raise ValueError("normalize must be one of: 'per_frame', 'per_video', 'none'")

# def compute_optical_flow_to_reference(frames, ref_idx=0):
#     """
#     Compute dense optical flow from each frame to the reference frame.

#     Args:
#         frames:  (T, H, W, 3) uint8 numpy array — original video frames (RGB)
#         ref_idx: which frame to use as reference (default: 0)

#     Returns:
#         flows: (T, H, W, 2) float32 — (dx, dy) displacement fields
#                flows[ref_idx] is zeros (identity)
#     """
#     T, H, W, _ = frames.shape
#     ref_gray = cv2.cvtColor(frames[ref_idx], cv2.COLOR_RGB2GRAY)

#     flows = np.zeros((T, H, W, 2), dtype=np.float32)

#     for t in range(T):
#         if t == ref_idx:
#             continue  # identity — zero displacement

#         cur_gray = cv2.cvtColor(frames[t], cv2.COLOR_RGB2GRAY)

#         # Farneback dense optical flow: cur_gray → ref_gray
#         flow = cv2.calcOpticalFlowFarneback(
#             cur_gray,       # source: current frame
#             ref_gray,       # target: reference frame
#             None,
#             pyr_scale=0.5,
#             levels=4,       # more levels = handles larger motion
#             winsize=15,
#             iterations=3,
#             poly_n=5,
#             poly_sigma=1.2,
#             flags=0,
#         )
#         flows[t] = flow  # (H, W, 2)

#     return flows

def compute_optical_flow_to_reference(frames, ref_idx=0):
    T, H, W, _ = frames.shape

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    ref_gray = cv2.cvtColor(frames[ref_idx], cv2.COLOR_RGB2GRAY)
    ref_gray = clahe.apply(ref_gray)
    ref_gray = cv2.GaussianBlur(ref_gray, (5, 5), 0)

    flows = np.zeros((T, H, W, 2), dtype=np.float32)

    for t in range(T):
        if t == ref_idx:
            continue

        cur_gray = cv2.cvtColor(frames[t], cv2.COLOR_RGB2GRAY)
        cur_gray = clahe.apply(cur_gray)
        cur_gray = cv2.GaussianBlur(cur_gray, (5, 5), 0)

        flow = cv2.calcOpticalFlowFarneback(
            cur_gray,
            ref_gray,
            None,
            pyr_scale=0.5,
            levels=5,
            winsize=25,
            iterations=5,
            poly_n=7,
            poly_sigma=1.5,
            flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN,
        )

        flows[t] = flow

    return flows

def preprocess_frame(frame):
    """
    frame: (H,W,3) RGB uint8 or (H,W) uint8
    return: (1,3,H,W) float tensor
    """
    if frame.ndim == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    else:
        gray = frame

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    frame_3ch = np.stack([gray, gray, gray], axis=-1)

    img = torch.from_numpy(frame_3ch).permute(2, 0, 1).float() / 255.0
    return img.unsqueeze(0).to(device)

def compute_optical_flow_to_reference_raft(frames, ref_idx=0):
    """
    Compute dense optical flow from each frame to a reference frame using RAFT.

    Args:
        frames:  (T, H, W, 3) uint8 numpy array
        ref_idx: reference frame index

    Returns:
        flows: (T, H, W, 2) float32
    """
    T, H, W, _ = frames.shape

    flows = np.zeros((T, H, W, 2), dtype=np.float32)

    # preprocess reference once
    ref_frame = preprocess_frame(frames[ref_idx])

    for t in range(T):
        if t == ref_idx:
            continue

        cur_frame = preprocess_frame(frames[t])

        with torch.no_grad():
            flow = raft_model(cur_frame, ref_frame)[-1]  # (1,2,H,W)

        flow_np = flow[0].permute(1, 2, 0).cpu().numpy()  # (H,W,2)
        flows[t] = flow_np

    return flows

def warp_and_stabilize_saliency(sal_frames, flows, ref_idx=0, temporal_sigma=1.0):
    """
    Warp each saliency frame to align with the reference frame using
    precomputed optical flow, then apply temporal Gaussian smoothing.

    Args:
        sal_frames:     (T, H, W) float32 — per-frame IG saliency in [0, 1]
        flows:          (T, H, W, 2) float32 — flow fields from compute_optical_flow_to_reference
        ref_idx:        reference frame index (must match what was used in flow computation)
        temporal_sigma: sigma for Gaussian smoothing along time axis (0 = no smoothing)

    Returns:
        stabilized: (T, H, W) float32 — warped and smoothed saliency
    """
    T, H, W = sal_frames.shape

    # Build base coordinate grids once
    grid_x, grid_y = np.meshgrid(np.arange(W), np.arange(H))  # (H, W) each

    warped = np.zeros_like(sal_frames)

    for t in range(T):
        flow_t = flows[t]  # (H, W, 2)

        # Absolute remap: where does each pixel in frame t land in reference frame?
        map_x = (grid_x - flow_t[..., 0]).astype(np.float32)
        map_y = (grid_y - flow_t[..., 1]).astype(np.float32)

        warped[t] = cv2.remap(
            sal_frames[t],
            map_x, map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0.0,
        )

    # Temporal Gaussian smoothing along time axis
    if temporal_sigma > 0:
        warped = gaussian_filter1d(warped, sigma=temporal_sigma, axis=0)

    # Re-normalize to [0, 1] after smoothing (smoothing reduces contrast slightly)
    lo, hi = warped.min(), warped.max()
    if hi - lo > 1e-6:
        warped = (warped - lo) / (hi - lo)

    return warped.astype(np.float32)

def generate_saliency(

    model,
    dataloader,
    device,
    task_name="Rythm_M24",
    saliency_out_dir="runs/exp/saliency",
    max_batches=None,
    steps=8,
    baseline="blur",
    blur_kernel=(1, 11, 11),
    blur_sigma=3,
    chunk_size=1,
    normalize="per_video",          # per_video recommended before optical flow
    ref_idx=0,                      # reference frame for optical flow
    temporal_sigma=1.0,             # smoothing strength along time axis
    stabilize=True, 
    ):
    model.eval()

    saliency_out_dir = Path(saliency_out_dir)
    heatmap_dir      = saliency_out_dir / "heatmaps"
    overlay_dir      = saliency_out_dir / "overlays"
    heatmap_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir.mkdir(parents=True, exist_ok=True)

    for batch_idx, batch in enumerate(dataloader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        inputs, targets, meta = to_device_batch(batch, device, model.use_video, model.use_tabular)
        video   = inputs["video"]
        tabular = inputs["tabular"] if model.use_tabular else None

        # --- Process one sample at a time to save GPU memory ---
        sal_list = []
        for b in range(video.shape[0]):
            vid_single = video[b:b+1]
            tab_single = tabular[b:b+1] if tabular is not None else None

            sal_single = integrated_gradients_video(
                model, vid_single,
                tabular=tab_single,
                task_name=task_name,
                steps=steps,
                baseline=baseline,
                blur_kernel=blur_kernel,
                blur_sigma=blur_sigma,
                chunk_size=chunk_size,
                normalize=normalize,
            )
            sal_list.append(sal_single)
            torch.cuda.empty_cache()

        sal = torch.cat(sal_list, dim=0)   # (B, T, H, W)

        video_np = video.detach().cpu().numpy()   # (B, C, T, H, W)

        for i in range(sal.shape[0]):
            vid_name = meta["video_name"][i]

            # --- Prepare original frames for optical flow ---
            frames = video_np[i]                          # (C, T, H, W)
            if frames.shape[0] == 1:
                frames = np.repeat(frames, 3, axis=0)     # (3, T, H, W)
            frames = np.transpose(frames, (1, 2, 3, 0))  # (T, H, W, 3)

            # Convert to uint8 — optical flow expects uint8 grayscale
            frames_uint8 = (
                (frames - frames.min()) /
                (frames.max() - frames.min() + 1e-6) * 255
            ).clip(0, 255).astype(np.uint8)

            sal_np = sal[i].cpu().numpy()   # (T, H, W) in [0, 1]

            if stabilize:
                flows = compute_optical_flow_to_reference(frames_uint8, ref_idx=ref_idx)
                sal_stabilized = warp_and_stabilize_saliency(
                    sal_np, flows,
                    ref_idx=ref_idx,
                    temporal_sigma=temporal_sigma,
                )
            else:
                sal_stabilized = sal_np        # use raw IG maps directly

            # --- Save both raw and stabilized maps ---
            T_len = sal.shape[1]
            for t in range(T_len):
                # Stabilized heatmap + overlay
                heatmap  = saliency_to_heatmap(sal_stabilized[t])
                heatmap_path = heatmap_dir / f"{vid_name}_t{t}.png"
                cv2.imwrite(str(heatmap_path), cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR))

                overlay = overlay_heatmap(frames_uint8[t], heatmap, alpha=0.50)
                overlay_path = overlay_dir / f"{vid_name}_t{t}.png"
                cv2.imwrite(str(overlay_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

        print(f"[Saliency Config] steps={steps}, baseline={baseline}, "
          f"chunk_size={chunk_size}, normalize={normalize}, "
          f"temporal_sigma={temporal_sigma}, stabilize={stabilize}")

        print(f"[Saliency] batch {batch_idx+1} done — {sal.shape[0]} samples saved.")

def smooth_grad_video(
    model,
    video,                      # (B,C,T,H,W)
    tabular=None,
    task_name="Rythm_M24",
    n_samples=25,
    noise_level=0.10,            # relative to input value range/std
    normalize="per_video",
    q_low=0.01,
    q_high=0.99,
    eps=1e-6,
 ):
    """
    SmoothGrad saliency:
        saliency = average over noisy inputs of |d score / d input|

    Returns:
        sal: (B,T,H,W)
    """
    model.eval()

    x = video.detach()
    B, C, T, H, W = x.shape

    # Estimate noise std per sample
    flat = x.view(B, -1)
    x_min = flat.min(dim=1)[0].view(B, 1, 1, 1, 1)
    x_max = flat.max(dim=1)[0].view(B, 1, 1, 1, 1)

    sigma = noise_level * (x_max - x_min + eps)

    total_sal = torch.zeros_like(x)

    for _ in range(n_samples):
        model.zero_grad(set_to_none=True)
        
        noise = torch.randn_like(x) * sigma
        x_noisy = (x + noise).detach().contiguous()
        x_noisy.requires_grad_(True)

        out = model(x_noisy, tabular)[task_name]

        if out.ndim == 1:
            score = out
        elif out.shape[-1] == 2:
            score = out[:, 1] - out[:, 0]
        else:
            score = out[:, 0]

        grad = torch.autograd.grad(
            outputs=score.sum(),
            inputs=x_noisy,
            create_graph=False,
            retain_graph=False,
            allow_unused=False,
        )[0]

        total_sal += grad.abs().detach()

        del x_noisy, out, score, grad, noise

    sal = total_sal / n_samples          # (B,C,T,H,W)
    sal = sal.mean(dim=1)                # (B,T,H,W)

    if normalize == "none":
        return sal.detach()

    elif normalize == "per_video":
        flat = sal.view(B, -1)
        lo = torch.quantile(flat, q_low, dim=1).view(B, 1, 1, 1)
        hi = torch.quantile(flat, q_high, dim=1).view(B, 1, 1, 1)
        sal = torch.clamp(sal, lo, hi)
        sal = (sal - lo) / (hi - lo + eps)
        return sal.detach()

    elif normalize == "per_frame":
        flat = sal.view(B * T, -1)
        lo = torch.quantile(flat, q_low, dim=1).view(B, T, 1, 1)
        hi = torch.quantile(flat, q_high, dim=1).view(B, T, 1, 1)
        sal = torch.clamp(sal, lo, hi)
        sal = (sal - lo) / (hi - lo + eps)
        return sal.detach()

    else:
        raise ValueError("normalize must be one of: 'per_frame', 'per_video', 'none'")

def smooth_integrated_gradients_video(
    model,
    video,                      # (B,C,T,H,W)
    tabular=None,
    task_name="Rythm_M24",
    steps=16,
    n_samples=25,
    noise_level=0.10,
    baseline="blur",
    blur_kernel=(1, 11, 11),
    blur_sigma=3.0,
    chunk_size=1,
    normalize="per_video",
 ):
    """
    Smooth Integrated Gradients:
        1. Add Gaussian noise to input video
        2. Compute Integrated Gradients
        3. Repeat n_samples times
        4. Average the saliency maps

    Returns:
        sal: (B,T,H,W)
    """

    model.eval()

    x = video.detach()
    B = x.shape[0]

    flat = x.view(B, -1)
    x_min = flat.min(dim=1)[0].view(B, 1, 1, 1, 1)
    x_max = flat.max(dim=1)[0].view(B, 1, 1, 1, 1)

    sigma = noise_level * (x_max - x_min + 1e-6)

    total_sal = None

    for _ in range(n_samples):
        noise = torch.randn_like(x) * sigma
        x_noisy = x + noise

        sal = integrated_gradients_video(
            model=model,
            video=x_noisy,
            tabular=tabular,
            task_name=task_name,
            steps=steps,
            baseline=baseline,
            blur_kernel=blur_kernel,
            blur_sigma=blur_sigma,
            chunk_size=chunk_size,
            normalize="none",   # important: normalize only after averaging
        )

        if total_sal is None:
            total_sal = torch.zeros_like(sal)

        total_sal += sal.detach()

        del noise, x_noisy, sal

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    sal_avg = total_sal / n_samples      # (B,T,H,W)

    if normalize == "none":
        return sal_avg.detach()

    elif normalize == "per_video":
        flat = sal_avg.view(B, -1)
        lo = torch.quantile(flat, 0.01, dim=1).view(B, 1, 1, 1)
        hi = torch.quantile(flat, 0.99, dim=1).view(B, 1, 1, 1)

        sal_avg = torch.clamp(sal_avg, lo, hi)
        sal_avg = (sal_avg - lo) / (hi - lo + 1e-6)

        return sal_avg.detach()

    elif normalize == "per_frame":
        B, T, H, W = sal_avg.shape
        flat = sal_avg.view(B * T, -1)

        lo = torch.quantile(flat, 0.01, dim=1).view(B, T, 1, 1)
        hi = torch.quantile(flat, 0.99, dim=1).view(B, T, 1, 1)

        sal_avg = torch.clamp(sal_avg, lo, hi)
        sal_avg = (sal_avg - lo) / (hi - lo + 1e-6)

        return sal_avg.detach()

    else:
        raise ValueError("normalize must be one of: 'per_frame', 'per_video', 'none'")