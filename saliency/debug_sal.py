import os
from pathlib import Path
import torch
import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d
from saliency.saliency import integrated_gradients_video , compute_optical_flow_to_reference, smooth_grad_video, smooth_integrated_gradients_video, compute_optical_flow_to_reference_raft

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

def warp_with_flow(img, flow, sign="minus", border_value=0):
    """
    Warp a 2D or 3D image using a dense flow field.

    Args:
        img:   (H,W) or (H,W,3)
        flow:  (H,W,2), where flow[...,0]=u and flow[...,1]=v
        sign:  "minus" -> x-u, y-v
               "plus"  -> x+u, y+v
        border_value: value outside image

    Returns:
        warped image with same shape as img
    """
    H, W = img.shape[:2]
    grid_x, grid_y = np.meshgrid(np.arange(W), np.arange(H))

    if sign == "minus":
        map_x = (grid_x - flow[..., 0]).astype(np.float32)
        map_y = (grid_y - flow[..., 1]).astype(np.float32)
    elif sign == "plus":
        map_x = (grid_x + flow[..., 0]).astype(np.float32)
        map_y = (grid_y + flow[..., 1]).astype(np.float32)
    else:
        raise ValueError("sign must be 'minus' or 'plus'")

    warped = cv2.remap(
        img,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_value,
    )
    return warped

def compute_alignment_score(ref_frame, warped_frame):
    """
    Mean absolute difference between reference and warped frame.
    Lower is better.
    """
    ref_f = ref_frame.astype(np.float32)
    warped_f = warped_frame.astype(np.float32)
    return np.mean(np.abs(ref_f - warped_f))

def save_sign_comparison_image(
    ref_frame,
    cur_frame,
    warped_minus,
    warped_plus,
    out_path,
    score_minus=None,
    score_plus=None,
    pad=20,          # margin around each image
    gap=20,          # space between panels
    scale=1.5        # upscale factor for visibility
    ):
    """
    Save a comparison image with padding and spacing:

    [ reference | current | warped_minus | warped_plus ]
    """

    def prepare(img):
        # ensure RGB
        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)

        # resize for better visibility
        if scale != 1.0:
            H, W = img.shape[:2]
            img = cv2.resize(img, (int(W * scale), int(H * scale)))

        # add padding around each image
        img = cv2.copyMakeBorder(
            img,
            pad, pad, pad, pad,
            borderType=cv2.BORDER_CONSTANT,
            value=(0, 0, 0)
        )
        return img

    # Prepare images
    ref_p   = prepare(ref_frame)
    cur_p   = prepare(cur_frame)
    minus_p = prepare(warped_minus)
    plus_p  = prepare(warped_plus)

    panels = [ref_p, cur_p, minus_p, plus_p]

    # Add gaps between panels
    gap_block = np.zeros((panels[0].shape[0], gap, 3), dtype=np.uint8)

    canvas = panels[0]
    for p in panels[1:]:
        canvas = np.concatenate([canvas, gap_block, p], axis=1)

    # ---------- Titles ----------
    font = cv2.FONT_HERSHEY_SIMPLEX
    color = (255, 255, 255)
    thickness = 2
    font_scale = 0.7

    titles = [
        "reference",
        "current",
        f"x-u,y-v" + (f" | diff={score_minus:.2f}" if score_minus is not None else ""),
        f"x+u,y+v" + (f" | diff={score_plus:.2f}" if score_plus is not None else ""),
    ]

    # Draw titles centered above each panel
    x_cursor = 0
    for i, (panel, title) in enumerate(zip(panels, titles)):
        h, w, _ = panel.shape

        text_size, _ = cv2.getTextSize(title, font, font_scale, thickness)
        text_w, text_h = text_size

        x_text = x_cursor + (w - text_w) // 2
        y_text = int(20 * scale)  # a bit higher

        cv2.putText(
            canvas,
            title,
            (x_text, y_text),
            font,
            font_scale,
            color,
            thickness,
            cv2.LINE_AA
        )

        x_cursor += w + gap

    # Save
    cv2.imwrite(str(out_path), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))

def compare_flow_signs_on_video(frames_uint8, flows, ref_idx=0, out_dir=None, video_name="video"):
    """
    Compare x-u,y-v vs x+u,y+v on original frames by aligning each frame to the reference.

    Args:
        frames_uint8: (T,H,W,3) uint8
        flows:        (T,H,W,2) float32
        ref_idx:      reference frame index
        out_dir:      if provided, saves side-by-side comparisons
        video_name:   prefix for saved images

    Returns:
        avg_minus, avg_plus, best_sign
    """
    ref = frames_uint8[ref_idx]
    T = frames_uint8.shape[0]

    minus_scores = []
    plus_scores = []

    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

    for t in range(T):
        if t == ref_idx:
            continue

        cur = frames_uint8[t]
        flow = flows[t]

        warped_minus = warp_with_flow(cur, flow, sign="minus", border_value=0)
        warped_plus  = warp_with_flow(cur, flow, sign="plus", border_value=0)

        score_minus = compute_alignment_score(ref, warped_minus)
        score_plus  = compute_alignment_score(ref, warped_plus)

        minus_scores.append(score_minus)
        plus_scores.append(score_plus)

        if out_dir is not None:
            save_sign_comparison_image(
                ref_frame=ref,
                cur_frame=cur,
                warped_minus=warped_minus,
                warped_plus=warped_plus,
                out_path=out_dir / f"{video_name}_t{t:03d}_compare.png",
                score_minus=score_minus,
                score_plus=score_plus,
            )

    avg_minus = float(np.mean(minus_scores)) if len(minus_scores) else 0.0
    avg_plus = float(np.mean(plus_scores)) if len(plus_scores) else 0.0
    best_sign = "minus" if avg_minus <= avg_plus else "plus"

    # print(
    #     f"[Flow sign check] {video_name}: "
    #     f"avg(x-u,y-v)={avg_minus:.4f}, avg(x+u,y+v)={avg_plus:.4f} -> best={best_sign}"
    # )

    return avg_minus, avg_plus, best_sign

def warp_and_stabilize_saliency_sign(
    sal_frames,
    flows,
    sign="minus",
    temporal_sigma=1.0,
 ):
    """
    Warp each saliency frame using the chosen sign convention,
    then apply temporal Gaussian smoothing.

    Args:
        sal_frames:     (T,H,W) float32
        flows:          (T,H,W,2) float32
        sign:           "minus" -> x-u, y-v
                        "plus"  -> x+u, y+v
        temporal_sigma: sigma for temporal smoothing

    Returns:
        stabilized: (T,H,W) float32
    """
    T, H, W = sal_frames.shape
    warped = np.zeros_like(sal_frames, dtype=np.float32)

    for t in range(T):
        warped[t] = warp_with_flow(
            sal_frames[t],
            flows[t],
            sign=sign,
            border_value=0.0,
        )

    if temporal_sigma > 0:
        warped = gaussian_filter1d(warped, sigma=temporal_sigma, axis=0)

    lo, hi = warped.min(), warped.max()
    if hi - lo > 1e-6:
        warped = (warped - lo) / (hi - lo)

    return warped.astype(np.float32)

def flow_to_color(flow, clip_flow=None, convert_to_bgr=False):
    """
    Convert optical flow to a color wheel visualization.

    Args:
        flow: (H, W, 2) float32, where flow[...,0]=u and flow[...,1]=v
        clip_flow: optional float. If provided, magnitude is clipped to this max
                   before normalization for more readable visualization.
        convert_to_bgr: if True, returns BGR for direct cv2.imwrite use.
                        if False, returns RGB.

    Returns:
        color: (H, W, 3) uint8
    """
    u = flow[..., 0]
    v = flow[..., 1]

    mag, ang = cv2.cartToPolar(u, v, angleInDegrees=False)  # ang in [0, 2pi)

    if clip_flow is not None:
        mag = np.clip(mag, 0, clip_flow)

    # Hue in OpenCV HSV is [0,179], representing [0,360) degrees
    hue = ang * 180 / np.pi / 2

    # Normalize magnitude to [0,255]
    if mag.max() > 1e-6:
        val = (mag / mag.max() * 255).astype(np.uint8)
    else:
        val = np.zeros_like(mag, dtype=np.uint8)

    hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
    hsv[..., 0] = hue.astype(np.uint8)      # hue = direction
    hsv[..., 1] = 255                       # full saturation
    hsv[..., 2] = val                       # value = magnitude

    color_bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    if convert_to_bgr:
        return color_bgr
    else:
        return cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)

def save_frame_and_flow_side_by_side(frames_uint8, flows, out_dir, video_name, clip_flow=None):
    """
    Save [frame | flow_color] side by side for each time step.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    T = flows.shape[0]
    for t in range(T):
        frame = frames_uint8[t]
        flow_color = flow_to_color(flows[t], clip_flow=clip_flow, convert_to_bgr=False)

        if frame.ndim == 2:
            frame = np.stack([frame] * 3, axis=-1)

        canvas = np.concatenate([frame, flow_color], axis=1)
        out_path = out_dir / f"{video_name}_t{t:03d}_frame_flow.png"
        cv2.imwrite(str(out_path), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))

def save_flow_color_images(flows, out_dir, video_name, clip_flow=None):
    """
    Save one color-wheel optical flow image per frame.

    Args:
        flows: (T, H, W, 2) float32
        out_dir: folder path
        video_name: string
        clip_flow: optional max magnitude for visualization
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    T = flows.shape[0]
    for t in range(T):
        flow_color = flow_to_color(flows[t], clip_flow=clip_flow, convert_to_bgr=True)
        out_path = out_dir / f"{video_name}_t{t:03d}_flow.png"
        cv2.imwrite(str(out_path), flow_color)

def generate_saliency_debug(
    model,
    dataloader,
    device,
    task_name="Rythm_M24",
    saliency_out_dir="runs/exp/saliency",
    max_batches=None,
    method="ig",                       # NEW
    steps=8,
    baseline="blur",
    blur_kernel=(1, 11, 11),
    blur_sigma=3,
    chunk_size=1,
    smooth_n_samples=25,               # NEW
    smooth_noise_level=0.10,           # NEW
    normalize="per_video",
    ref_idx=0,
    temporal_sigma=1.0,
    stabilize=True,
    auto_check_flow_sign=True,          # NEW
    warp_sign="minus",                  # NEW: used if auto_check_flow_sign=False
    save_flow_sign_debug=True,          # NEW
 ):
    model.eval()

    saliency_out_dir = Path(saliency_out_dir)
    heatmap_dir      = saliency_out_dir / "heatmaps"
    overlay_dir      = saliency_out_dir / "overlays"
    flow_debug_dir   = saliency_out_dir / "flow_sign_debug"
    flow_color_dir = saliency_out_dir / "flow_color"
    frame_flow_dir = saliency_out_dir / "frame_flow"
    baseline_dir = saliency_out_dir / "baselines"
    baseline_dir.mkdir(parents=True, exist_ok=True)

    flow_color_dir.mkdir(parents=True, exist_ok=True)
    frame_flow_dir.mkdir(parents=True, exist_ok=True)

    if save_flow_sign_debug:
        flow_debug_dir.mkdir(parents=True, exist_ok=True)

    for batch_idx, batch in enumerate(dataloader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        inputs, targets, meta = to_device_batch(batch, device, model.use_video, model.use_tabular)
        video   = inputs["video"]
        tabular = inputs["tabular"] if model.use_tabular else None

        # ---------- IG per sample ----------
        sal_list = []
        baseline_list = []
        for b in range(video.shape[0]):
            vid_single = video[b:b+1]
            tab_single = tabular[b:b+1] if tabular is not None else None

            if method == "ig":
                print("ig mood")
                sal_single , baseline_single = integrated_gradients_video(
                    model,
                    vid_single,
                    tabular=tab_single,
                    task_name=task_name,
                    steps=steps,
                    baseline=baseline,
                    blur_kernel=blur_kernel,
                    blur_sigma=blur_sigma,
                    chunk_size=chunk_size,
                    normalize=normalize,
                    return_baseline=True,
                )
                baseline_list.append(baseline_single)

            elif method == "smoothgrad":
                print("smoothgrad mood")
                sal_single = smooth_grad_video(
                    model,
                    vid_single,
                    tabular=tab_single,
                    task_name=task_name,
                    n_samples=smooth_n_samples,
                    noise_level=smooth_noise_level,
                    normalize=normalize,
                )
                baseline_list.append(None)
            
            elif method == "smooth_ig":
                print("smoothgrad ig mood")
                sal_single = smooth_integrated_gradients_video(
                    model,
                    vid_single,
                    tabular=tab_single,
                    task_name=task_name,
                    steps=steps,
                    n_samples=smooth_n_samples,
                    noise_level=smooth_noise_level,
                    baseline=baseline,
                    blur_kernel=blur_kernel,
                    blur_sigma=blur_sigma,
                    chunk_size=chunk_size,
                    normalize=normalize,
                    return_baseline=True,
                )
                baseline_list.append(baseline_single)

            else:
                raise ValueError("method must be one of: 'ig', 'smoothgrad', 'smooth_ig'")
            sal_list.append(sal_single)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        sal = torch.cat(sal_list, dim=0)   # (B,T,H,W)
        video_np = video.detach().cpu().numpy()   # (B,C,T,H,W)
        has_baseline = all(b is not None for b in baseline_list)

        if has_baseline:
            baseline_tensor = torch.cat(baseline_list, dim=0)  # (B,C,T,H,W)
            baseline_np = baseline_tensor.detach().cpu().numpy()
        else:
            baseline_np = None

        # ---------- Per sample post-processing ----------
        for i in range(sal.shape[0]):
            vid_name = meta["video_name"][i]

            if baseline_np is not None:
                base_frames = baseline_np[i]  # (C,T,H,W)

                if base_frames.shape[0] == 1:
                    base_frames = np.repeat(base_frames, 3, axis=0)

                base_frames = np.transpose(base_frames, (1, 2, 3, 0))  # (T,H,W,3)

                base_uint8 = (
                    (base_frames - base_frames.min()) /
                    (base_frames.max() - base_frames.min() + 1e-6) * 255.0
                ).clip(0, 255).astype(np.uint8)

                video_baseline_dir = baseline_dir / vid_name
                video_baseline_dir.mkdir(parents=True, exist_ok=True)

                for t in range(base_uint8.shape[0]):
                    out_path = video_baseline_dir / f"{vid_name}_baseline_{baseline}_t{t}.png"
                    cv2.imwrite(
                        str(out_path),
                        cv2.cvtColor(base_uint8[t], cv2.COLOR_RGB2BGR)
                    )

            frames = video_np[i]  # (C,T,H,W)
            if frames.shape[0] == 1:
                frames = np.repeat(frames, 3, axis=0)

            frames = np.transpose(frames, (1, 2, 3, 0))  # (T,H,W,3)

            # normalize to uint8 for optical flow and overlay
            frames_uint8 = (
                (frames - frames.min()) /
                (frames.max() - frames.min() + 1e-6) * 255.0
            ).clip(0, 255).astype(np.uint8)

            sal_np = sal[i].detach().cpu().numpy().astype(np.float32)  # (T,H,W)

            if stabilize:
                # flows = compute_optical_flow_to_reference(frames_uint8, ref_idx=ref_idx)
                flows = compute_optical_flow_to_reference_raft(frames_uint8, ref_idx=ref_idx)


                save_flow_color_images(
                    flows=flows,
                    out_dir=(flow_color_dir / vid_name),
                    video_name=vid_name,
                    clip_flow=None,
                )

                # Save frame | flow side-by-side
                save_frame_and_flow_side_by_side(
                    frames_uint8=frames_uint8,
                    flows=flows,
                    out_dir=(frame_flow_dir / vid_name),
                    video_name=vid_name,
                    clip_flow=None,
                )

                # ---------- check which sign is better ----------
                if auto_check_flow_sign:
                    _, _, best_sign = compare_flow_signs_on_video(
                        frames_uint8=frames_uint8,
                        flows=flows,
                        ref_idx=ref_idx,
                        out_dir=(flow_debug_dir / vid_name) if save_flow_sign_debug else None,
                        video_name=vid_name,
                    )
                else:
                    best_sign = warp_sign

                sal_stabilized = warp_and_stabilize_saliency_sign(
                    sal_frames=sal_np,
                    flows=flows,
                    sign=best_sign,
                    temporal_sigma=temporal_sigma,
                )
            else:
                best_sign = None
                sal_stabilized = sal_np

            # ---------- Save heatmaps ----------
            # create per-video folders
            video_heatmap_dir = heatmap_dir / vid_name
            video_overlay_dir = overlay_dir / vid_name
            video_heatmap_dir.mkdir(parents=True, exist_ok=True)
            video_overlay_dir.mkdir(parents=True, exist_ok=True)

            T_len = sal.shape[1]
            for t in range(T_len):
                heatmap = saliency_to_heatmap(sal_stabilized[t])
                heatmap_path = video_heatmap_dir / f"{vid_name}_t{t}.png"
                cv2.imwrite(str(heatmap_path), cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR))

                overlay = overlay_heatmap(frames_uint8[t], heatmap, alpha=0.50)
                overlay_path = video_overlay_dir / f"{vid_name}_t{t}.png"
                cv2.imwrite(str(overlay_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

        print(f"[Saliency] batch {batch_idx + 1} done — {sal.shape[0]} samples saved.")