"""
Integration layer for ensemble saliency generation
Fits seamlessly into your existing eval() pipeline
"""

from pathlib import Path
import torch
import torch.nn as nn
import numpy as np
import cv2
import json
from typing import Dict, Optional, Tuple
from dataclasses import asdict

from saliency.ensemble_saliency import (
    SaliencyGeneratorEnsemble,
    SaliencyConfig,
    saliency_to_heatmap,
    overlay_heatmap,
)


def to_device_batch(batch: dict, device: torch.device, use_video: bool = True, use_tabular: bool = True):
    """Extract and move batch to device (your existing function)"""
    inputs = {}
    if use_video:
        inputs["video"] = batch["frames"].to(device, non_blocking=True)
    else:
        inputs["video"] = None
    if use_tabular:
        tab = batch.get("tabular", None)
        if tab is not None:
            tab = tab.to(device, non_blocking=True)
        inputs["tabular"] = tab
    else:
        inputs["tabular"] = None
    targets = {k: v.to(device) for k, v in batch["targets"].items()}
    meta = {
        "video_id": batch.get("video_id", None),
        "video_name": batch.get("video_name", None),
        "fps": batch.get("fps", None),
        "num_frames": batch.get("num_frames", None),
    }
    return inputs, targets, meta


def generate_saliency_ensemble(
    model: nn.Module,
    dataloader,
    device: torch.device,
    task_name: str = "Rythm_M24",
    saliency_out_dir: str = "runs/exp/saliency",
    max_batches: Optional[int] = None,
    # Ensemble config
    methods: list = None,
    ensemble_weights: Dict[str, float] = None,
    # IG parameters
    ig_steps: int = 50,
    ig_baseline: str = "blur",
    ig_blur_kernel: Tuple = (1, 11, 11),
    # Occlusion parameters
    occlusion_patch_size: Tuple = (1, 8, 8),
    occlusion_stride: Tuple = (1, 4, 4),
    # Normalization & smoothing
    normalize: str = "per_video",
    temporal_sigma: float = 1.0,
    apply_stabilization: bool = True,
    # Visualization
    heatmap_alpha: float = 0.4,
    colormap: int = cv2.COLORMAP_TURBO,
    save_individual_maps: bool = True,
    save_ensemble_only: bool = False,
    threshold: float = 0.5,
):
    """
    Generate saliency maps using ensemble of methods
    
    Can be called directly from eval() to replace generate_saliency_captum()
    
    Usage in eval():
        generate_saliency_ensemble(
            model=model,
            dataloader=loader,
            device=device,
            task_name=cfg["saliency"].get("saliency_task", "Rythm_M24"),
            saliency_out_dir=out_dir / "saliency" / split,
            max_batches=cfg["saliency"].get("saliency_batches", None),
            methods=cfg["saliency"].get("methods", ["ig", "occlusion", "tabular_aware"]),
            ensemble_weights=cfg["saliency"].get("ensemble_weights", None),
            ig_steps=cfg["saliency"].get("ig_steps", 50),
            threshold=cfg.get("eval", {}).get("threshold", 0.5),
        )
    """
    
    saliency_out_dir = Path(saliency_out_dir)
    saliency_out_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories for each method
    raw_dir = saliency_out_dir / "raw"
    heatmap_dir = saliency_out_dir / "heatmaps"
    overlay_dir = saliency_out_dir / "overlays"
    metadata_dir = saliency_out_dir / "metadata"
    
    raw_dir.mkdir(parents=True, exist_ok=True)
    heatmap_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    
    if save_individual_maps and not save_ensemble_only:
        for method in (methods or ["ig", "occlusion", "tabular_aware"]):
            (heatmap_dir / method).mkdir(parents=True, exist_ok=True)
            (overlay_dir / method).mkdir(parents=True, exist_ok=True)
            (raw_dir / method).mkdir(parents=True, exist_ok=True)

    
    # Always create ensemble folders
    (heatmap_dir / "ensemble").mkdir(parents=True, exist_ok=True)
    (overlay_dir / "ensemble").mkdir(parents=True, exist_ok=True)
    (raw_dir / "ensemble").mkdir(parents=True, exist_ok=True)


    
        
    # Create ensemble config
    config = SaliencyConfig(
        device=str(device),
        task_name=task_name,
        methods=methods or ["ig", "occlusion", "tabular_aware"],
        ig_steps=ig_steps,
        ig_baseline=ig_baseline,
        ig_blur_kernel=ig_blur_kernel,
        occlusion_patch_size=occlusion_patch_size,
        occlusion_stride=occlusion_stride,
        normalize_method=normalize,
        temporal_smoothing_sigma=temporal_sigma,
        apply_stabilization=apply_stabilization,
    )
    
    if ensemble_weights is not None:
        config.ensemble_weights = ensemble_weights
    else:
        # Equal weight by default
        n_methods = len(config.methods)
        config.ensemble_weights = {m: 1.0 / n_methods for m in config.methods}
    
    print(f"\n[Ensemble Saliency] Configuration:")
    print(f"  Methods: {config.methods}")
    print(f"  Ensemble weights: {config.ensemble_weights}")
    print(f"  IG steps: {ig_steps}, baseline: {ig_baseline}")
    print(f"  Occlusion patch: {occlusion_patch_size}, stride: {occlusion_stride}")
    print(f"  Normalization: {normalize}, temporal smoothing: {temporal_sigma}")
    
    # Initialize ensemble generator
    generator = SaliencyGeneratorEnsemble(model, config)
    
    model.eval()
    
    # Process batches
    for batch_idx, batch in enumerate(dataloader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        
        inputs, targets, meta = to_device_batch(
            batch, device, model.use_video, model.use_tabular
        )
        
        video = inputs["video"]  # (B, C, T, H, W)
        tabular = inputs["tabular"]  # (B, K) or None
        B = video.shape[0]
        
        # Get model predictions
        with torch.no_grad():
            logits = generator._forward_func(video, tabular)
            
            if logits.shape[1] == 1:
                probs = torch.sigmoid(logits)
                pred_class = (probs >= threshold).long().view(-1)
            else:
                probs = torch.softmax(logits, dim=1)
                pred_class = probs.argmax(dim=1)
        
        # Process each sample
        for b in range(B):
            vid_single = video[b:b+1].detach()
            tab_single = tabular[b:b+1].detach() if tabular is not None else None
            
            class_b = int(pred_class[b].item())
            
            if logits.shape[1] == 1:
                prob_b = float(probs[b].item())
            else:
                prob_b = float(probs[b, class_b].item())
            
            true_label_b = int(targets[task_name][b].item())
            
            # Generate ensemble saliency
            print(f"\n[Batch {batch_idx+1}] Sample {b+1}/{B}: Generating saliency...")
            saliency_maps, metadata = generator.generate_ensemble_saliency(
                video=vid_single,
                tabular=tab_single,
                pred_class=class_b,
                prob_value=prob_b,
            )
            
            # Prepare frame data
            frames = vid_single[0].detach().cpu().numpy()  # (C, T, H, W)
            if frames.shape[0] == 1:
                frames = np.repeat(frames, 3, axis=0)
            frames = np.transpose(frames, (1, 2, 3, 0))  # (T, H, W, 3)
            frames_uint8 = (
                (frames - frames.min()) /
                (frames.max() - frames.min() + 1e-6) * 255.0
            ).clip(0, 255).astype(np.uint8)
            
            # Generate video name
            vid_name = str(meta["video_name"][b]).replace("/", "_").replace("\\", "_")
            
            # Save raw saliency maps
            for method_name, sal_map in saliency_maps.items():
                np.save(
                    raw_dir / method_name / f"{vid_name}_raw_{method_name}.npy",
                    sal_map
                )
            
            # Create visualizations
            for t in range(frames_uint8.shape[0]):
                for method_name, sal_map in saliency_maps.items():
                    if method_name not in config.methods and method_name != 'ensemble':
                        continue
                    
                    if save_ensemble_only and method_name != 'ensemble':
                        continue
                    
                    # Create heatmap
                    heatmap = saliency_to_heatmap(sal_map[t], colormap)

                    (raw_dir / method_name).mkdir(parents=True, exist_ok=True)
                    (heatmap_dir / method_name).mkdir(parents=True, exist_ok=True)
                    (overlay_dir / method_name).mkdir(parents=True, exist_ok=True)
                                    
                    # Save heatmap
                    heatmap_path = heatmap_dir / method_name / f"{vid_name}_t{t:03d}.png"
                    cv2.imwrite(
                        str(heatmap_path),
                        cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR)
                    )
                    
                    # Save overlay
                    overlay = overlay_heatmap(frames_uint8[t], heatmap, alpha=heatmap_alpha)
                    overlay_path = overlay_dir / method_name / f"{vid_name}_t{t:03d}.png"
                    cv2.imwrite(
                        str(overlay_path),
                        cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
                    )
            
            # Save metadata
            metadata.update({
                'video_name': vid_name,
                'true_label': true_label_b,
                'pred_label': class_b,
                'probability': prob_b,
                'explained_class': 'non-AF' if class_b == 0 else 'AF',
                'frames': int(frames_uint8.shape[0]),
                'height': int(frames_uint8.shape[1]),
                'width': int(frames_uint8.shape[2]),
            })
            
            metadata_path = metadata_dir / f"{vid_name}_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            print(f"  ✓ Saved to: {saliency_out_dir}")
            print(f"    - True label: {true_label_b}, Pred label: {class_b}, Prob: {prob_b:.3f}")
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        print(f"[Batch {batch_idx + 1}] Done — {B} samples processed.")
    
    print(f"\n[Ensemble Saliency] ✓ Complete! Outputs saved to: {saliency_out_dir}")



