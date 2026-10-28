from pathlib import Path
import torch
import torch.nn.functional as F
import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d

from captum.attr import visualization as viz
from captum.attr import (
    IntegratedGradients,
    Saliency,
    GradientShap,
    DeepLift,
    GuidedBackprop,
    NoiseTunnel,
)

from saliency.saliency_enhancements import (
    bilateral_temporal_smooth,
    ensemble_normalize_saliency,
    saliency_to_heatmap_perceptual,
    generate_contour_map,
    flow_weighted_saliency,
    overlay_heatmap_advanced
)

from saliency.saliency import compute_optical_flow_to_reference, warp_and_stabilize_saliency
from saliency.debug_sal import (
    save_flow_color_images,
    save_frame_and_flow_side_by_side,
    compare_flow_signs_on_video,
    warp_and_stabilize_saliency_sign,
    # saliency_to_heatmap
)

# def overlay_heatmap(frame, heatmap, alpha=0.4):
#     """
#     frame: (H,W,3) uint8
#     heatmap: (H,W,3) uint8
#     """
#     return np.clip(
#         (1 - alpha) * frame + alpha * heatmap,
#         0, 255
#     ).astype(np.uint8)

def overlay_saliency_on_frame(
    frame,
    saliency,
    colormap=cv2.COLORMAP_TURBO,
    alpha=0.55,
    threshold=0.15,
    blur=True,
):
    """
    frame:    (H,W,3) uint8 original frame
    saliency: (H,W) float, normalized to [0,1]

    Returns:
        overlay: (H,W,3) uint8
        heatmap_rgb: (H,W,3) uint8
    """

    H, W = frame.shape[:2]

    # Ensure saliency has same size as frame
    if saliency.shape[:2] != (H, W):
        saliency = cv2.resize(
            saliency,
            (W, H),
            interpolation=cv2.INTER_LINEAR,
        )

    saliency = np.nan_to_num(saliency)
    saliency = np.clip(saliency, 0, 1).astype(np.float32)

    # Optional smoothing for nicer visualization
    if blur:
        saliency = cv2.GaussianBlur(saliency, (5, 5), 0)

    # Remove weak background saliency
    mask = saliency.copy()
    mask[mask < threshold] = 0.0

    # Normalize remaining visible saliency
    if mask.max() > 0:
        mask = mask / (mask.max() + 1e-6)

    # Create heatmap
    heat_uint8 = (saliency * 255).astype(np.uint8)
    heatmap_bgr = cv2.applyColorMap(heat_uint8, colormap)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

    # Use saliency mask as alpha, not the full heatmap background
    alpha_map = (alpha * mask)[..., None]  # (H,W,1)

    overlay = (
        frame.astype(np.float32) * (1 - alpha_map)
        + heatmap_rgb.astype(np.float32) * alpha_map
    )

    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    return overlay, heatmap_rgb

def saliency_to_heatmap(sal_norm, colormap=cv2.COLORMAP_TURBO):
    """
    sal_norm: (H,W) already in [0,1]
    """
    sal_uint8 = (sal_norm * 255).clip(0, 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(sal_uint8, colormap)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    return heatmap

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

def get_blur_baseline(video, blur_kernel=(1, 11, 11)):
    """
    video: (B,C,T,H,W)
    Simple spatial blur baseline.
    """
    # Approximate blur with avg_pool3d.
    kt, kh, kw = blur_kernel
    pad = (kw // 2, kh // 2, kt // 2)

    baseline = F.avg_pool3d(
        video,
        kernel_size=(kt, kh, kw),
        stride=1,
        padding=(kt // 2, kh // 2, kw // 2),
    )
    return baseline

def make_baseline(video, baseline="zeros", blur_kernel=(1, 11, 11)):
    if baseline == "zeros":
        return torch.zeros_like(video)

    elif baseline == "mean":
        # one mean value per sample/channel
        mean = video.mean(dim=(2, 3, 4), keepdim=True)
        return mean.expand_as(video)

    elif baseline == "blur":
        return get_blur_baseline(video, blur_kernel)

    else:
        raise ValueError("baseline must be one of: zeros, mean, blur")

# def normalize_saliency(sal, normalize="per_video", eps=1e-6):
#     """
#     sal: (T,H,W)
#     """
#     if normalize == "none":
#         return sal

#     if normalize == "per_video":
#         vmin = sal.min()
#         vmax = sal.max()
#         return (sal - vmin) / (vmax - vmin + eps)

#     if normalize == "per_frame":
#         minv = sal.min(axis=(1, 2), keepdims=True)
#         maxv = sal.max(axis=(1, 2), keepdims=True)
#         return (sal - minv) / (maxv - minv + eps)

#     raise ValueError("normalize must be one of: none, per_video, per_frame")

def normalize_saliency(
    sal,
    normalize="per_video",
    eps=1e-6,
    q_low=1,
    q_high=99,
 ):
    """
    sal: (T,H,W)
    """

    if normalize == "none":
        return sal

    if normalize == "per_video":
        vmin = np.percentile(sal, q_low)
        vmax = np.percentile(sal, q_high)

        sal = np.clip(sal, vmin, vmax)

        return (sal - vmin) / (vmax - vmin + eps)

    if normalize == "per_frame":
        out = np.zeros_like(sal)

        for t in range(sal.shape[0]):
            frame = sal[t]

            vmin = np.percentile(frame, q_low)
            vmax = np.percentile(frame, q_high)

            frame = np.clip(frame, vmin, vmax)

            out[t] = (frame - vmin) / (vmax - vmin + eps)

        return out

    raise ValueError("normalize must be one of: none, per_video, per_frame")

def occlusion_sensitivity_video(
    forward_func,
    video,                 # (1,C,T,H,W)
    tabular=None,
    target=None,
    explain_negative=False,
    patch_size=(1, 16, 16),     # (t,h,w)
    stride=(1, 8, 8),
    occlusion_value=0.0,
 ):
    """
    Returns occlusion saliency with shape (1,1,T,H,W).
    Higher value = stronger drop in prediction when region is occluded.
    """

    forward_args = (tabular, explain_negative)

    with torch.no_grad():
        base_out = forward_func(video, *forward_args)

        if base_out.shape[1] == 1:
            base_score = base_out[0, 0]
        else:
            base_score = base_out[0, target]

    _, C, T, H, W = video.shape
    pt, ph, pw = patch_size
    st, sh, sw = stride

    saliency = torch.zeros((1, 1, T, H, W), device=video.device)
    counts = torch.zeros((1, 1, T, H, W), device=video.device)

    for t in range(0, T, st):
        t1 = min(t + pt, T)

        for y in range(0, H, sh):
            y1 = min(y + ph, H)

            for x in range(0, W, sw):
                x1 = min(x + pw, W)

                occluded = video.clone()
                occluded[:, :, t:t1, y:y1, x:x1] = occlusion_value

                with torch.no_grad():
                    out = forward_func(occluded, *forward_args)

                    if out.shape[1] == 1:
                        occ_score = out[0, 0]
                    else:
                        occ_score = out[0, target]

                drop = base_score - occ_score

                # only keep positive evidence
                drop = torch.clamp(drop, min=0.0)

                saliency[:, :, t:t1, y:y1, x:x1] += drop
                counts[:, :, t:t1, y:y1, x:x1] += 1.0

    saliency = saliency / (counts + 1e-6)

    return saliency


def generate_saliency_captum(
    model,
    dataloader,
    device,
    task_name="Rythm_M24",
    saliency_out_dir="runs/exp/saliency",
    max_batches=None,
    method="IntegratedGradients",
    steps=32,
    baseline="blur",
    blur_kernel=(1, 11, 11),
    smooth_n_samples=25,
    smooth_noise_level=0.10,
    normalize="per_video",
    ref_idx=0,
    temporal_sigma=1.0,
    threshold=None,
    stabilize=False,
 ):
    saliency_out_dir = Path(saliency_out_dir)
    heatmap_dir = saliency_out_dir / "heatmaps"
    contour_dir = saliency_out_dir / "contours"
    baseline_dir = saliency_out_dir / "baselines"
    raw_dir = saliency_out_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    heatmap_dir.mkdir(parents=True, exist_ok=True)
    baseline_dir.mkdir(parents=True, exist_ok=True)
    contour_dir.mkdir(parents=True, exist_ok=True)

    overlay_dir      = saliency_out_dir / "overlays"
    flow_color_dir = saliency_out_dir / "flow_color"
    frame_flow_dir = saliency_out_dir / "frame_flow"
    baseline_dir = saliency_out_dir / "baselines"
    baseline_dir.mkdir(parents=True, exist_ok=True)
 
    flow_color_dir.mkdir(parents=True, exist_ok=True)
    frame_flow_dir.mkdir(parents=True, exist_ok=True)
    model.to(device)
    model.eval()

    # Important: wrapper returns only the task logit/probability tensor.
    # def forward_func(video, tabular=None):
    #     if model.use_tabular:
    #         out = model(video, tabular)
    #     else:
    #         out = model(video)

    #     # Case 1: model returns dict
    #     if isinstance(out, dict):
    #         out = out[task_name]

    #     # Expected shapes:
    #     # binary: (B,), (B,1)
    #     # multi-class: (B,num_classes)
    #     if out.ndim == 1:
    #         out = out.unsqueeze(1)

    #     return out

    def forward_func(video, tabular=None, explain_negative=False):
        if model.use_tabular:
            out = model(video, tabular)
        else:
            out = model(video)

        if isinstance(out, dict):
            out = out[task_name]

        if out.ndim == 1:
            out = out.unsqueeze(1)

        if explain_negative:
            out = -out

        return out



    for batch_idx, batch in enumerate(dataloader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        inputs, targets, meta = to_device_batch(
            batch, device, model.use_video, model.use_tabular
        )

        video = inputs["video"].detach().to(device)          # (B,C,T,H,W)
        tabular = inputs["tabular"].detach().to(device) if model.use_tabular else None

        B = video.shape[0]

        # Forward once on the real video to get prediction target.
        with torch.no_grad():
            logits = forward_func(video, tabular)

            if logits.shape[1] == 1:
                probs = torch.sigmoid(logits)
                if threshold is None:
                    threshold = 0.5
                pred_class = (probs >= threshold).long().view(-1)
                print(f"threshold for binary classification: {threshold}")
                captum_target = None
            else:
                probs = torch.softmax(logits, dim=1)
                pred_class = probs.argmax(dim=1)
                captum_target = pred_class


        # Per-sample attribution is safer for variable targets and memory.
        for b in range(B):
            vid_single = video[b:b+1].clone().detach().requires_grad_(True)
            tab_single = (
                tabular[b:b+1].clone().detach()
                if tabular is not None
                else None
            )

            base_single = make_baseline(
                vid_single,
                baseline=baseline,
                blur_kernel=blur_kernel,
            )

            # target_single = None if captum_target is None else pred_class[b].item()

            if logits.shape[1] == 1:
                explain_negative = bool(pred_class[b].item() == 0)
                target_single = None
            else:
                explain_negative = False
                target_single = int(pred_class[b].item())

            # Attribute only video.
            # Tabular is passed as additional_forward_args, so Captum does not
            # compute heatmaps for tabular features.
            if method == "IntegratedGradients":
                print("Using Integrated Gradients...")
                attr_method = IntegratedGradients(forward_func)
                attr = attr_method.attribute(
                    inputs=vid_single,
                    baselines=base_single,
                    # additional_forward_args=(tab_single,),
                    additional_forward_args=(tab_single, explain_negative),
                    target=target_single,
                    n_steps=steps,
                    internal_batch_size=1,
                )
                print(f"attr shape: {attr.shape}")

            elif method == "SmoothGrad":
                print("Using SmoothGrad...")
                attr_method = Saliency(forward_func)
                nt = NoiseTunnel(attr_method)
                attr = nt.attribute(
                    inputs=vid_single,
                    additional_forward_args=(tab_single, explain_negative),
                    target=target_single,
                    nt_type="smoothgrad",
                    nt_samples=smooth_n_samples,
                    stdevs=smooth_noise_level,
                    internal_batch_size=1,
                )

            elif method == "SmoothIG":
                print("Using SmoothIG...")
                attr_method = IntegratedGradients(forward_func)
                nt = NoiseTunnel(attr_method)
                attr = nt.attribute(
                    inputs=vid_single,
                    baselines=base_single,
                    additional_forward_args=(tab_single, explain_negative),
                    target=target_single,
                    nt_type="smoothgrad",
                    nt_samples=smooth_n_samples,
                    stdevs=smooth_noise_level,
                    n_steps=steps,
                    internal_batch_size=1,
                )

            elif method == "GradientShap":
                print("Using GradientShap...")
                attr_method = GradientShap(forward_func)
                attr = attr_method.attribute(
                    inputs=vid_single,
                    baselines=base_single,
                    additional_forward_args=(tab_single, explain_negative),
                    target=target_single,
                    n_samples=smooth_n_samples,
                    stdevs=smooth_noise_level,
                    internal_batch_size=1,
                )

            elif method == "Saliency":
                print("Using Saliency...")
                attr_method = Saliency(forward_func)
                attr = attr_method.attribute(
                    inputs=vid_single,
                    additional_forward_args=(tab_single, explain_negative),
                    target=target_single,
                )

            elif method == "DeepLift":
                print("Using DeepLift...")
                attr_method = DeepLift(forward_func)
                attr = attr_method.attribute(
                    inputs=vid_single,
                    baselines=base_single,
                    additional_forward_args=(tab_single, explain_negative),
                    target=target_single,
                    internal_batch_size=1,
                )

            elif method == "GuidedBackprop":
                print("Using GuidedBackprop...")
                attr_method = GuidedBackprop(wrapped_model)
                attr = attr_method.attribute(
                    inputs=vid_single,
                    additional_forward_args=(tab_single, explain_negative),
                    target=target_single,

                )
            
            elif method == "Occlusion":
                print("Using Occlusion Sensitivity...")

                attr = occlusion_sensitivity_video(
                    forward_func=forward_func,
                    video=vid_single,
                    tabular=tab_single,
                    target=target_single,
                    explain_negative=explain_negative,
                    patch_size=(1, 8, 8),
                    stride=(1, 4, 4),
                    occlusion_value=0.0,
                )

            else:
                raise ValueError(
                    "method must be one of: IntegratedGradients, SmoothGrad, "
                    "SmoothIG, GradientShap, Saliency, DeepLift, GuidedBackprop"
                )

            # after attr is computed
            sal_single = attr.detach().abs().sum(dim=1)  # (1,T,H,W)
            sal_frames = sal_single[0].detach().cpu().numpy().astype(np.float32)

            vid_name = meta["video_id"][b]
            vid_name = str(vid_name).replace("/", "_").replace("\\", "_")

            np.save(raw_dir / f"{vid_name}_raw_saliency.npy", sal_frames)

            # Prepare original frames
            frames = video[b].detach().cpu().numpy()  # (C,T,H,W)

            if frames.shape[0] == 1:
                frames = np.repeat(frames, 3, axis=0)

            frames = np.transpose(frames, (1, 2, 3, 0))  # (T,H,W,3)

            frames_uint8 = (
                (frames - frames.min()) /
                (frames.max() - frames.min() + 1e-6) * 255.0
            ).clip(0, 255).astype(np.uint8)

            # Optional stabilization
            if stabilize:
                flows = compute_optical_flow_to_reference(
                    frames_uint8,
                    ref_idx=ref_idx
                )

                save_flow_color_images(
                    flows=flows,
                    out_dir=(flow_color_dir / vid_name),
                    video_name=vid_name,
                    clip_flow=None,
                )

                save_frame_and_flow_side_by_side(
                    frames_uint8=frames_uint8,
                    flows=flows,
                    out_dir=(frame_flow_dir / vid_name),
                    video_name=vid_name,
                    clip_flow=None,
                )

                sal_stabilized = warp_and_stabilize_saliency(
                    sal_frames=sal_frames,
                    flows=flows,
                    ref_idx=ref_idx,
                    temporal_sigma=temporal_sigma,
                )
                
                # new
                # sal_smooth = bilateral_temporal_smooth(
                #     sal_stabilized,
                #     temporal_sigma=1.0,
                #     flows=flows,  # Pass your flow magnitude
                #     preserve_peaks=True
                # )
            else:
                sal_stabilized = sal_frames

            np.save(raw_dir / f"{vid_name}_stabilized_saliency.npy", sal_stabilized)

            sal_vis = normalize_saliency(sal_stabilized, normalize=normalize)
            # new
            sal_vis = np.power(sal_vis, 1.3)

            # new
            # sal_vis = ensemble_normalize_saliency(
            #     sal_smooth,
            #     percentile=98,
            #     gamma=1.5,
            #     method="per_video"
            # )


            pred_label = int(pred_class[b].item())

            if logits.shape[1] == 1:
                prob_value = float(probs[b].item())
            else:
                prob_value = float(probs[b, pred_label].item())

            true_label = int(targets[task_name][b].item())

            with open(raw_dir / f"{vid_name}_metadata.txt", "w") as f:
                f.write(f"true_label: {true_label}\n")
                f.write(f"pred_label: {pred_label}\n")
                f.write(f"probability: {prob_value}\n")
                f.write(
                    f"explained_class: "
                    f"{'non-severe/sinus' if pred_label == 0 else 'severe/AF'}\n"
                )

            
            video_heatmap_dir = heatmap_dir / vid_name
            video_overlay_dir = overlay_dir / vid_name
            video_heatmap_dir.mkdir(parents=True, exist_ok=True)
            video_overlay_dir.mkdir(parents=True, exist_ok=True)

            for t in range(sal_vis.shape[0]):
                # heatmap = saliency_to_heatmap(sal_vis[t])

                # heatmap_path = video_heatmap_dir / f"{vid_name}_t{t:03d}.png"
                # cv2.imwrite(
                #     str(heatmap_path),
                #     cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR)
                # )

                # overlay, _ = overlay_saliency_on_frame(
                #         frame=frames_uint8[t],
                #         saliency=sal_vis[t],
                #         alpha=0.70,
                #         threshold=0.15,
                #         blur=True,
                #     )

                # overlay_path = video_overlay_dir / f"{vid_name}_t{t:03d}.png"
                # cv2.imwrite(
                #     str(overlay_path),
                #     cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
                # )
                overlay, heatmap = overlay_saliency_on_frame(
                    frame=frames_uint8[t],
                    saliency=sal_vis[t],
                    alpha=0.60,
                    threshold=0.15,
                    blur=True,
                )

                heatmap_path = video_heatmap_dir / f"{vid_name}_t{t:03d}.png"
                overlay_path = video_overlay_dir / f"{vid_name}_t{t:03d}.png"

                cv2.imwrite(
                    str(heatmap_path),
                    cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR)
                )

                cv2.imwrite(
                    str(overlay_path),
                    cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
                )

            # new

            # for t in range(sal_vis.shape[0]):
            #     # Perceptually uniform heatmap (replaces your saliency_to_heatmap)
            #     heatmap = saliency_to_heatmap_perceptual(sal_vis[t], colormap="turbo")
                
            #     # Add contour map for localization
            #     contour = generate_contour_map(sal_vis[t], thresholds=[0.3, 0.6, 0.85])
                
            #     # Enhanced overlay with contrast boost
            #     overlay = overlay_heatmap_advanced(
            #         frames_uint8[t],
            #         heatmap,
            #         alpha=0.4,
            #         enhance_contrast=True
            #     )
                
            #     # Save all three
            #     cv2.imwrite(str(heatmap_dir / f"{vid_name}_t{t:03d}.png"), cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR))
            #     cv2.imwrite(str(contour_dir / f"{vid_name}_t{t:03d}.png"), cv2.cvtColor(contour, cv2.COLOR_RGB2BGR))
            #     cv2.imwrite(str(overlay_dir / f"{vid_name}_t{t:03d}.png"), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))


            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        print(f"[Saliency] batch {batch_idx + 1} done — {B} samples saved.")