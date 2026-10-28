"""
Ensemble Saliency Module for AF-Echo-Tab Model
=====================================================
Combines four robust saliency methods:
1. Integrated Gradients (IG)
2. Attention-Weighted Saliency (AW)
3. Perturbation-Based (Occlusion)
4. Tabular-Aware Gradient Saliency (TAG)

Each method captures different aspects of model decisions.
Ensemble combines all perspectives for robust interpretability.
"""

from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
from typing import Dict, Tuple, Optional, Callable
from dataclasses import dataclass
import json
from scipy.ndimage import gaussian_filter1d
import warnings

from captum.attr import IntegratedGradients, Saliency, GradientShap, NoiseTunnel
from captum.attr import visualization as viz


@dataclass
class SaliencyConfig:
    """Configuration for ensemble saliency generation"""
    # General
    device: str = "cuda"
    task_name: str = "Rythm_M24"
    
    # Method selection
    methods: list = None  # ["ig", "attention", "occlusion", "tabular_aware"]
    
    # IG parameters
    ig_steps: int = 50
    ig_baseline: str = "blur"  # "zeros", "mean", "blur"
    ig_blur_kernel: Tuple = (1, 11, 11)
    
    # SmoothIG parameters
    smooth_n_samples: int = 25
    smooth_noise_level: float = 0.10
    
    # Occlusion parameters
    occlusion_patch_size: Tuple = (1, 8, 8)
    occlusion_stride: Tuple = (1, 4, 4)
    
    # Tabular-aware parameters
    tabular_weight: float = 0.5  # Weight of tabular modulation gradient
    
    # Normalization & post-processing
    normalize_method: str = "per_video"  # "per_video", "per_frame", "none"
    temporal_smoothing_sigma: float = 1.0
    apply_stabilization: bool = True
    ref_idx: int = 0
    
    # Ensemble
    ensemble_weights: Dict[str, float] = None  # e.g., {"ig": 0.3, "attention": 0.3, "occlusion": 0.2, "tabular_aware": 0.2}
    
    def __post_init__(self):
        if self.methods is None:
            self.methods = ["ig", "occlusion", "tabular_aware"]
        if self.ensemble_weights is None:
            # Equal weight by default
            n_methods = len(self.methods)
            self.ensemble_weights = {m: 1.0 / n_methods for m in self.methods}


class SaliencyGeneratorEnsemble:
    """
    Unified ensemble saliency generation for AF-Echo-Tab model
    """
    
    def __init__(self, model: nn.Module, config: SaliencyConfig):
        self.model = model
        self.config = config
        self.device = torch.device(config.device)
        self.model.to(self.device)
        self.model.eval()
        
    #     # Store attention hooks
    #     self.attention_weights = {}
    #     self._register_hooks()
    
    # def _register_hooks(self):
    #     self.attention_weights = {}
    #     def get_attention_hook(name):
    #         def hook(module, input, output):
    #             # For MultiheadAttention, output is usually:
    #             # (attn_output, attn_weights)
    #             if isinstance(output, tuple) and len(output) > 1 and output[1] is not None:
    #                 self.attention_weights[name] = output[1].detach()
    #             # else:
    #             #     print(f"[AW] Hooked {name}, but no attention weights returned.")
    #         return hook

    #     for name, module in self.model.named_modules():
    #         if name.endswith("self_attn"):
    #             print(f"[AW] Registering hook on: {name}")
    #             module.register_forward_hook(get_attention_hook(name))
        
    def _forward_func(
        self,
        video: torch.Tensor,
        tabular: Optional[torch.Tensor] = None,
        explain_negative: bool = False
     ) -> torch.Tensor:
        """
        Wrapper for model forward that returns task-specific logits
        """
        with torch.no_grad():
            if self.model.use_tabular and tabular is not None:
                out = self.model(video, tabular)
            else:
                out = self.model(video)
            
            # Extract task output
            if isinstance(out, dict):
                out = out[self.config.task_name]
            
            # Ensure (B, C) shape
            if out.ndim == 1:
                out = out.unsqueeze(1)
            
            # Optionally negate for explaining negative class
            if explain_negative:
                out = -out
            
            return out
    
    def _forward_func_attr(
        self,
        video: torch.Tensor,
        tabular: Optional[torch.Tensor] = None,
        explain_negative: bool = False
     ) -> torch.Tensor:
        """Forward function for attribution (needs gradients)"""
        if self.model.use_tabular and tabular is not None:
            out = self.model(video, tabular)
        else:
            out = self.model(video)
        
        if isinstance(out, dict):
            out = out[self.config.task_name]
        
        if out.ndim == 1:
            out = out.unsqueeze(1)
        
        if explain_negative:
            out = -out
        
        return out
    
    # ========== Method 1: Integrated Gradients ==========
    def _generate_integrated_gradients(
        self,
        video: torch.Tensor,
        tabular: Optional[torch.Tensor],
        baseline_type: str = "blur",
        target: Optional[int] = None,
        explain_negative: bool = False
     ) -> torch.Tensor:
        """
        Integrated Gradients attribution
        Returns: (1, T, H, W)
        """
        print("[IG] Computing Integrated Gradients...")

        target = None
        
        # Create baseline
        baseline = self._make_baseline(video, baseline_type)
        
        # Create Captum attribution
        attr_method = IntegratedGradients(self._forward_func_attr)
        
        attr = attr_method.attribute(
            inputs=video,
            baselines=baseline,
            additional_forward_args=(tabular, explain_negative),
            target=target,
            n_steps=self.config.ig_steps,
            internal_batch_size=1,
        )
        
        # Aggregate to (B, T, H, W)
        sal = attr.abs().sum(dim=1)
        
        return sal
    
    # # ========== Method 2: Attention-Weighted Saliency ==========
    # def _generate_attention_saliency(
    #         self,
    #         video: torch.Tensor,
    #         tabular: Optional[torch.Tensor]
    #     ) -> torch.Tensor:

    #         print("[AW] Computing Attention-Weighted Saliency...")

    #         B, C, T, H, W = video.shape
    #         self.attention_weights.clear()

    #         with torch.no_grad():
    #             _ = self._forward_func(video, tabular)

    #         if len(self.attention_weights) == 0:
    #             print("[AW] WARNING: No attention weights captured. Returning zeros.")
    #             return torch.zeros(B, T, H, W, device=video.device)

    #         attn_list = list(self.attention_weights.values())

    #         processed = []
    #         for attn in attn_list:
    #             if attn is None:
    #                 continue

    #             if attn.ndim == 4:
    #                 attn = attn.mean(dim=1)  # average heads: (B, L, L)

    #             processed.append(attn)

    #         if len(processed) == 0:
    #             print("[AW] WARNING: Attention hooks fired, but no valid weights found.")
    #             return torch.zeros(B, T, H, W, device=video.device)

    #         attn = torch.stack(processed).mean(dim=0)  # (B, L, L)

    #         # Token importance: how much attention each token receives
    #         token_saliency = attn.sum(dim=1)  # (B, L)

    #         # Global attention strength per sample
    #         global_score = token_saliency.mean(dim=1)  # (B,)

    #         sal = global_score.view(B, 1, 1, 1).expand(B, T, H, W)

    #         return sal
    
    # ========== Method 3: Occlusion/Perturbation ==========
    def _generate_occlusion_saliency(
        self,
        video: torch.Tensor,
        tabular: Optional[torch.Tensor],
        target: Optional[int] = None,
        explain_negative: bool = False
    ) -> torch.Tensor:
        """
        Occlusion-based saliency via patch masking
        Returns: (1, T, H, W)
        """
        print("[Occlusion] Computing Occlusion Sensitivity...")
        
        B, C, T, H, W = video.shape
        patch_t, patch_h, patch_w = self.config.occlusion_patch_size
        stride_t, stride_h, stride_w = self.config.occlusion_stride
        
        # Get baseline prediction
        with torch.no_grad():
            base_out = self._forward_func(video, tabular, explain_negative)
            if base_out.shape[1] == 1:
                base_score = base_out[0, 0]
            else:
                base_score = base_out[0, target] if target is not None else base_out[0].max()
        
        saliency = torch.zeros((B, T, H, W), device=video.device)
        counts = torch.zeros((B, T, H, W), device=video.device)
        
        # Iterate over patches
        for t in range(0, T, stride_t):
            t1 = min(t + patch_t, T)
            for y in range(0, H, stride_h):
                y1 = min(y + patch_h, H)
                for x in range(0, W, stride_w):
                    x1 = min(x + patch_w, W)
                    
                    # Create occluded video
                    occluded = video.clone()
                    occluded[:, :, t:t1, y:y1, x:x1] = 0.0
                    
                    # Get occluded prediction
                    with torch.no_grad():
                        occ_out = self._forward_func(occluded, tabular, explain_negative)
                        if occ_out.shape[1] == 1:
                            occ_score = occ_out[0, 0]
                        else:
                            occ_score = occ_out[0, target] if target is not None else occ_out[0].max()
                    
                    # Drop in prediction = importance
                    drop = (base_score - occ_score).clamp(min=0.0)
                    
                    # Accumulate
                    saliency[:, t:t1, y:y1, x:x1] += drop.detach()
                    counts[:, t:t1, y:y1, x:x1] += 1.0
        
        # Average
        saliency = saliency / (counts + 1e-6)
        
        return saliency
    
    # ========== Method 4: Tabular-Aware Gradient Saliency ==========
    def _generate_tabular_aware_saliency(
        self,
        video: torch.Tensor,
        tabular: Optional[torch.Tensor],
        baseline_type: str = "blur",
        target: Optional[int] = None,
        explain_negative: bool = False
    ) -> torch.Tensor:
        """
        Gradient saliency that factors in affine modulation from tabular branch
        Captures how tabular features reshape video feature importance
        Returns: (1, T, H, W)
        """
        print("[TAG] Computing Tabular-Aware Gradient Saliency...")
        
        if tabular is None:
            print("[TAG] WARNING: No tabular input provided. Falling back to standard IG.")
            return self._generate_integrated_gradients(video, tabular, baseline_type, target, explain_negative)
        
        baseline = self._make_baseline(video, baseline_type)
        video_diff = video - baseline
        
        accumulated_grads = torch.zeros_like(video)
        
        for step in range(self.config.ig_steps):
            # Interpolate
            interpolated = baseline + (step / self.config.ig_steps) * video_diff
            interpolated.requires_grad_(True)
            
            # Forward pass
            if self.model.use_tabular:
                # Hook into intermediate layers to get modulation
                out = self.model(interpolated, tabular)
            else:
                out = self.model(interpolated)
            
            if isinstance(out, dict):
                out = out[self.config.task_name]
            
            if out.ndim == 1:
                out = out.unsqueeze(1)
            
            if explain_negative:
                out = -out
            
            # Backward
            if out.shape[1] == 1:
                loss = out[0, 0]
            else:
                loss = out[0, target] if target is not None else out[0].max()
            
            loss.backward()
            
            accumulated_grads += interpolated.grad.detach() / self.config.ig_steps
        
        # IG: gradient * input difference
        attr = accumulated_grads * video_diff
        
        # Aggregate to spatial saliency
        sal = attr.abs().sum(dim=1)
        
        return sal
    
    # ========== Utility Methods ==========
    def _make_baseline(self, video: torch.Tensor, baseline_type: str = "blur") -> torch.Tensor:
        """Create baseline for IG"""
        if baseline_type == "zeros":
            return torch.zeros_like(video)
        elif baseline_type == "mean":
            mean = video.mean(dim=(2, 3, 4), keepdim=True)
            return mean.expand_as(video)
        elif baseline_type == "blur":
            kt, kh, kw = self.config.ig_blur_kernel
            baseline = F.avg_pool3d(
                video,
                kernel_size=(kt, kh, kw),
                stride=1,
                padding=(kt // 2, kh // 2, kw // 2),
            )
            return baseline
        else:
            raise ValueError(f"Unknown baseline type: {baseline_type}")
    
    def _normalize_saliency(self, sal: torch.Tensor) -> torch.Tensor:
        """Normalize saliency map"""
        if self.config.normalize_method == "none":
            return sal
        elif self.config.normalize_method == "per_video":
            vmin = sal.min()
            vmax = sal.max()
            return (sal - vmin) / (vmax - vmin + 1e-6)
        elif self.config.normalize_method == "per_frame":
            minv = sal.min(dim=(1, 2), keepdim=True)[0]
            maxv = sal.max(dim=(1, 2), keepdim=True)[0]
            return (sal - minv) / (maxv - minv + 1e-6)
        else:
            raise ValueError(f"Unknown normalization: {self.config.normalize_method}")
    
    def _smooth_temporal(self, sal: np.ndarray, sigma: float = 1.0) -> np.ndarray:
        """Apply Gaussian smoothing along temporal axis"""
        if sigma == 0:
            return sal
        
        sal_smooth = np.zeros_like(sal)
        for h in range(sal.shape[1]):
            for w in range(sal.shape[2]):
                sal_smooth[:, h, w] = gaussian_filter1d(sal[:, h, w], sigma=sigma)
        return sal_smooth
    
    # ========== Main Generation Method ==========
    def generate_ensemble_saliency(
        self,
        video: torch.Tensor,
        tabular: Optional[torch.Tensor] = None,
        pred_class: Optional[int] = None,
        prob_value: Optional[float] = None
    ) -> Dict[str, np.ndarray]:
        """
        Generate all saliency maps and return as dict
        
        Args:
            video: (1, C, T, H, W)
            tabular: (1, K) or None
            pred_class: Predicted class for attribution
            prob_value: Probability for metadata
        
        Returns:
            {
                'ig': (T, H, W),
                'occlusion': (T, H, W),
                'tabular_aware': (T, H, W),
                'ensemble': (T, H, W),
                'metadata': {...}
            }
        """
        video = video.to(self.device)
        if tabular is not None:
            tabular = tabular.to(self.device)
        
        results = {}
        saliency_maps = {}
        
        # Determine target class for attribution
        explain_negative = False
        if pred_class == 0:
            explain_negative = True
        target = None if explain_negative else pred_class
        
        # Generate each method's saliency
        try:
            if "ig" in self.config.methods:
                sal_ig = self._generate_integrated_gradients(
                    video, tabular, self.config.ig_baseline, target, explain_negative
                )
                sal_ig = self._normalize_saliency(sal_ig)
                saliency_maps['ig'] = sal_ig.detach().cpu().numpy()[0]  # (T, H, W)
        except Exception as e:
            print(f"[IG] Error: {e}")
            saliency_maps['ig'] = np.zeros((video.shape[2], video.shape[3], video.shape[4]))
        
        # try:
        #     if "attention" in self.config.methods:
        #         sal_attn = self._generate_attention_saliency(video, tabular)
        #         sal_attn = self._normalize_saliency(sal_attn)
        #         saliency_maps['attention'] = sal_attn.detach().cpu().numpy()[0]  # (T, H, W)
        # except Exception as e:
        #     print(f"[Attention] Error: {e}")
        #     saliency_maps['attention'] = np.zeros((video.shape[2], video.shape[3], video.shape[4]))
        
        try:
            if "occlusion" in self.config.methods:
                sal_occ = self._generate_occlusion_saliency(video, tabular, target, explain_negative)
                sal_occ = self._normalize_saliency(sal_occ)
                saliency_maps['occlusion'] = sal_occ.detach().cpu().numpy()[0]  # (T, H, W)
        except Exception as e:
            print(f"[Occlusion] Error: {e}")
            saliency_maps['occlusion'] = np.zeros((video.shape[2], video.shape[3], video.shape[4]))
        
        try:
            if "tabular_aware" in self.config.methods:
                sal_tag = self._generate_tabular_aware_saliency(
                    video, tabular, self.config.ig_baseline, target, explain_negative
                )
                sal_tag = self._normalize_saliency(sal_tag)
                saliency_maps['tabular_aware'] = sal_tag.detach().cpu().numpy()[0]  # (T, H, W)
        except Exception as e:
            print(f"[Tabular-Aware] Error: {e}")
            saliency_maps['tabular_aware'] = np.zeros((video.shape[2], video.shape[3], video.shape[4]))
        
        # Ensemble: weighted average
        ensemble_sal = np.zeros_like(saliency_maps[list(saliency_maps.keys())[0]])
        for method_name, method_sal in saliency_maps.items():
            weight = self.config.ensemble_weights.get(method_name, 0.0)
            ensemble_sal += weight * method_sal
        
        saliency_maps['ensemble'] = ensemble_sal
        
        # Apply temporal smoothing if enabled
        if self.config.apply_stabilization:
            for method_name in saliency_maps.keys():
                saliency_maps[method_name] = self._smooth_temporal(
                    saliency_maps[method_name],
                    sigma=self.config.temporal_smoothing_sigma
                )
        
        # Metadata
        metadata = {
            'methods_used': self.config.methods,
            'ensemble_weights': self.config.ensemble_weights,
            'pred_class': int(pred_class) if pred_class is not None else None,
            'prob_value': float(prob_value) if prob_value is not None else None,
            'task_name': self.config.task_name,
            'config': {
                'ig_steps': self.config.ig_steps,
                'ig_baseline': self.config.ig_baseline,
                'normalize_method': self.config.normalize_method,
                'temporal_smoothing_sigma': self.config.temporal_smoothing_sigma,
            }
        }
        
        return saliency_maps, metadata


def saliency_to_heatmap(sal_norm: np.ndarray, colormap=cv2.COLORMAP_TURBO) -> np.ndarray:
    """Convert normalized saliency to RGB heatmap"""
    sal_uint8 = (sal_norm * 255).clip(0, 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(sal_uint8, colormap)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    return heatmap


def overlay_heatmap(frame: np.ndarray, heatmap: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    """Overlay heatmap on frame"""
    return np.clip(
        (1 - alpha) * frame + alpha * heatmap,
        0, 255
    ).astype(np.uint8)
