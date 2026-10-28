import numpy as np
import cv2
from scipy.ndimage import gaussian_filter, maximum_filter
from scipy.signal import savgol_filter


def bilateral_temporal_smooth(sal_frames, temporal_sigma=1.0, flows=None, preserve_peaks=True):
    """
    Smooth saliency across time while preserving frame-local peaks.
    
    Args:
        sal_frames: (T, H, W) saliency map
        temporal_sigma: Gaussian sigma for temporal smoothing
        flows: (T, H, W, 2) optical flow vectors, or (T, H, W) flow magnitude
        preserve_peaks: If True, only smooth in low-gradient regions
    
    Returns:
        Temporally smoothed saliency (T, H, W)
    """
    from scipy.ndimage import gaussian_filter
    import numpy as np
    
    T, H, W = sal_frames.shape
    
    # Standard temporal smoothing
    sal_smooth = gaussian_filter(sal_frames, sigma=(temporal_sigma, 0.5, 0.5))
    
    if preserve_peaks and flows is not None:
        # Compute flow magnitude if 2D flow vectors provided
        if flows.ndim == 4 and flows.shape[-1] == 2:
            flow_mag = np.sqrt(flows[:, :, :, 0]**2 + flows[:, :, :, 1]**2)
        else:
            flow_mag = flows
        
        # Edge-aware: reduce smoothing near motion boundaries
        flow_norm = (flow_mag - flow_mag.min()) / (flow_mag.max() - flow_mag.min() + 1e-6)
        edge_mask = 1.0 - np.clip(flow_norm, 0, 1)  # Low motion = smooth more
        sal_smooth = sal_frames * (1 - edge_mask * 0.5) + sal_smooth * (edge_mask * 0.5)
    
    return sal_smooth


def ensemble_normalize_saliency(sal_vis, percentile=98, gamma=1.5, method="per_video"):
    """
    Normalize saliency for visualization with contrast and gamma correction.
    
    Args:
        sal_vis: (T, H, W) saliency map
        percentile: Contrast stretching percentile (98 clips bright outliers)
        gamma: Gamma correction factor (>1 brightens, <1 darkens)
        method: "per_video" or "per_frame" normalization
    
    Returns:
        Normalized saliency in [0, 1] with gamma applied
    """
    if method == "per_video":
        v_min, v_max = sal_vis.min(), np.percentile(sal_vis, percentile)
    elif method == "per_frame":
        # Normalize each frame independently
        v_min = sal_vis.min(axis=(1, 2), keepdims=True)
        v_max = np.percentile(sal_vis, percentile, axis=(1, 2), keepdims=True)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Contrast stretch
    sal_norm = (sal_vis - v_min) / (v_max - v_min + 1e-6)
    sal_norm = np.clip(sal_norm, 0, 1)
    
    # Gamma correction (perceptual enhancement)
    sal_norm = np.power(sal_norm, 1.0 / gamma)
    
    return sal_norm


def saliency_to_heatmap_perceptual(sal_frame, colormap="turbo"):
    """
    Convert saliency to perceptually uniform heatmap.
    
    Args:
        sal_frame: (H, W) saliency map in [0, 1]
        colormap: "turbo", "viridis", "plasma", "magma", "hot"
    
    Returns:
        RGB heatmap (H, W, 3)
    """
    # Clip and convert to uint8
    sal_uint8 = (np.clip(sal_frame, 0, 1) * 255).astype(np.uint8)
    
    # OpenCV colormap options
    cmap_dict = {
        "turbo": cv2.COLORMAP_TURBO,  # Perceptually uniform (best for saliency)
        "viridis": cv2.COLORMAP_VIRIDIS,
        "plasma": cv2.COLORMAP_PLASMA,
        "magma": cv2.COLORMAP_MAGMA,
        "hot": cv2.COLORMAP_HOT,
    }
    
    cmap = cmap_dict.get(colormap, cv2.COLORMAP_TURBO)
    heatmap_bgr = cv2.applyColorMap(sal_uint8, cmap)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)
    
    return heatmap_rgb


def generate_contour_map(sal_frame, thresholds=None, linewidth=2):
    """
    Generate contour lines showing saliency regions.
    
    Args:
        sal_frame: (H, W) saliency map in [0, 1]
        thresholds: List of threshold levels (default: [0.3, 0.6, 0.85])
        linewidth: OpenCV line thickness
    
    Returns:
        RGB contour image (H, W, 3)
    """
    if thresholds is None:
        thresholds = [0.3, 0.6, 0.85]
    
    H, W = sal_frame.shape
    sal_uint8 = (np.clip(sal_frame, 0, 1) * 255).astype(np.uint8)
    
    contour_img = np.ones((H, W, 3), dtype=np.uint8) * 240  # Light gray background
    
    colors = [(200, 100, 50), (100, 150, 200), (50, 180, 100)]  # BGR
    
    for thresh, color in zip(thresholds, colors):
        thresh_val = int(thresh * 255)
        _, binary = cv2.threshold(sal_uint8, thresh_val, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(contour_img, contours, -1, color, linewidth)
    
    return cv2.cvtColor(contour_img, cv2.COLOR_BGR2RGB)


def flow_weighted_saliency(sal_frames, flows, blur_sigma=1.0):
    """
    Weight saliency by optical flow magnitude to highlight moving attended regions.
    
    Args:
        sal_frames: (T, H, W) saliency map
        flows: (T, H, W, 2) optical flow from optical flow computation
        blur_sigma: Gaussian blur on flow mag to reduce noise
    
    Returns:
        Flow-weighted saliency (T, H, W)
    """
    T = sal_frames.shape[0]
    sal_weighted = sal_frames.copy()
    
    for t in range(T):
        flow_mag = np.sqrt(flows[t, :, :, 0]**2 + flows[t, :, :, 1]**2)
        flow_mag = gaussian_filter(flow_mag, sigma=blur_sigma)
        flow_mag_norm = (flow_mag - flow_mag.min()) / (flow_mag.max() - flow_mag.min() + 1e-6)
        
        # Emphasize saliency in high-motion regions
        sal_weighted[t] = sal_frames[t] * (0.7 + 0.3 * flow_mag_norm)
    
    return sal_weighted


def overlay_heatmap_advanced(frame, heatmap, alpha=0.4, enhance_contrast=True):
    """
    Overlay heatmap on frame with optional contrast enhancement.
    
    Args:
        frame: (H, W, 3) uint8 frame
        heatmap: (H, W, 3) uint8 heatmap
        alpha: Blending weight [0, 1]
        enhance_contrast: Boost frame contrast before blending
    
    Returns:
        Overlaid frame (H, W, 3)
    """
    frame_float = frame.astype(np.float32)
    heatmap_float = heatmap.astype(np.float32)
    
    if enhance_contrast:
        # Mild CLAHE on frame for better visibility
        frame_lab = cv2.cvtColor(frame, cv2.COLOR_RGB2LAB)
        l_chan = frame_lab[:, :, 0]
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_enhanced = clahe.apply(l_chan)
        frame_lab[:, :, 0] = l_enhanced
        frame_float = cv2.cvtColor(frame_lab, cv2.COLOR_LAB2RGB).astype(np.float32)
    
    # Blend
    overlay = frame_float * (1 - alpha) + heatmap_float * alpha
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    
    return overlay


# Example usage in your pipeline:
# Instead of your current heatmap loop, do:
"""
sal_norm = ensemble_normalize_saliency(sal_vis, percentile=98, gamma=1.5)

for t in range(sal_norm.shape[0]):
    # Multi-modal outputs
    heatmap = saliency_to_heatmap_perceptual(sal_norm[t], colormap="turbo")
    contour = generate_contour_map(sal_norm[t], thresholds=[0.3, 0.6, 0.85])
    overlay = overlay_heatmap_advanced(frames_uint8[t], heatmap, alpha=0.4, enhance_contrast=True)
    
    # Save all variants
    cv2.imwrite(str(heatmap_dir / f"{vid_name}_t{t:03d}_heatmap.png"), cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(contour_dir / f"{vid_name}_t{t:03d}_contour.png"), cv2.cvtColor(contour, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(overlay_dir / f"{vid_name}_t{t:03d}_overlay.png"), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
"""
