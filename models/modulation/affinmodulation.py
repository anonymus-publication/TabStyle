import torch
import torch.nn as nn


import torch
import torch.nn as nn


class AffineModulation(nn.Module):
    """
    Channel-wise affine modulation for 3D video features.

    Inputs:
        feat  : (B, C, T, H, W)
        gamma : (B, C)
        beta  : (B, C)

    Output:
        out   : (B, C, T, H, W)

    Modes:
        - direct:
            out = gamma * feat + beta
            or out = (1 + gamma) * feat + beta

        - instance_norm:
            feat_norm = instance_norm(feat)
            out = gamma * feat_norm + beta
            or out = (1 + gamma) * feat_norm + beta
    """

    def __init__(self, cfg):
        super().__init__()
        self.residual_gamma = cfg.get("residual_gamma", True)
        self.debug = cfg.get("debug_modulation", False)
        self._printed = False
        self.last_debug = None

        # new options
        self.normalization_mode = cfg.get("normalization_mode", "none").lower()
        self.eps = cfg.get("instance_norm_eps", 1e-5)

        valid_modes = ["none", "instance_norm"]
        if self.normalization_mode not in valid_modes:
            raise ValueError(
                f"Unknown normalization_mode='{self.normalization_mode}'. "
                f"Expected one of {valid_modes}"
            )

    def _instance_normalize(self, feat):
        """
        Instance normalize over (T, H, W) for each sample and channel.
        feat: (B, C, T, H, W)
        """
        mean = feat.mean(dim=(2, 3, 4), keepdim=True)
        std = feat.std(dim=(2, 3, 4), keepdim=True, unbiased=False)
        feat_norm = (feat - mean) / (std + self.eps)
        return feat_norm, mean, std

    def forward(self, feat, gamma, beta):
        if feat.dim() != 5:
            raise ValueError(f"Expected feat shape (B,C,T,H,W), got {feat.shape}")

        if gamma.dim() != 2:
            raise ValueError(f"Expected gamma shape (B,C), got {gamma.shape}")

        if beta.dim() != 2:
            raise ValueError(f"Expected beta shape (B,C), got {beta.shape}")

        B, C, T, H, W = feat.shape

        if gamma.shape[0] != B or gamma.shape[1] != C:
            raise ValueError(
                f"Gamma shape {gamma.shape} incompatible with feature shape {feat.shape}"
            )

        if beta.shape[0] != B or beta.shape[1] != C:
            raise ValueError(
                f"Beta shape {beta.shape} incompatible with feature shape {feat.shape}"
            )

        gamma_2d = gamma
        beta_2d = beta

        gamma = gamma.view(B, C, 1, 1, 1)
        beta = beta.view(B, C, 1, 1, 1)

        # choose what gamma/beta act on
        if self.normalization_mode == "none":
            mod_input = feat
            feat_norm = None
            feat_mean = None
            feat_std = None

        elif self.normalization_mode == "instance_norm":
            mod_input, feat_mean, feat_std = self._instance_normalize(feat)
            feat_norm = mod_input

        # affine modulation
        if self.residual_gamma:
            out = (1.0 + gamma) * mod_input + beta
        else:
            out = gamma * mod_input + beta

        # store debug info
        with torch.no_grad():
            self.last_debug = {
                "mode": self.normalization_mode,
                "feat_shape": tuple(feat.shape),
                "gamma_shape_before": tuple(gamma_2d.shape),
                "beta_shape_before": tuple(beta_2d.shape),
                "gamma_shape_after": tuple(gamma.shape),
                "beta_shape_after": tuple(beta.shape),
                "out_shape": tuple(out.shape),
                "feat_mean": feat.mean().item(),
                "feat_std": feat.std().item(),
                "gamma_mean": gamma_2d.mean().item(),
                "gamma_std": gamma_2d.std().item(),
                "beta_mean": beta_2d.mean().item(),
                "beta_std": beta_2d.std().item(),
                "out_mean": out.mean().item(),
                "out_std": out.std().item(),
                "mean_abs_change": (out - feat).abs().mean().item(),
            }

            if feat_norm is not None:
                self.last_debug["norm_feat_mean"] = feat_norm.mean().item()
                self.last_debug["norm_feat_std"] = feat_norm.std().item()

        if self.debug and not self._printed:
            print("=============== AFFINE MODULATION DEBUG START ===============")
            print(f"Mode                       : {self.normalization_mode}")
            print(f"Feature input              : {feat.shape}")
            print(f"Gamma before reshape       : {gamma_2d.shape}")
            print(f"Beta before reshape        : {beta_2d.shape}")
            print(f"Gamma after reshape        : {gamma.shape}")
            print(f"Beta after reshape         : {beta.shape}")
            print("-------------------------------------------------------------")
            print(f"Feature mean/std           : {feat.mean().item():.4f} / {feat.std().item():.4f}")

            if feat_norm is not None:
                print(f"Norm feature mean/std      : {feat_norm.mean().item():.4f} / {feat_norm.std().item():.4f}")

            print(f"Gamma mean/std             : {gamma_2d.mean().item():.4f} / {gamma_2d.std().item():.4f}")
            print(f"Beta mean/std              : {beta_2d.mean().item():.4f} / {beta_2d.std().item():.4f}")
            print(f"Output mean/std            : {out.mean().item():.4f} / {out.std().item():.4f}")
            print(f"Mean absolute change       : {(out - feat).abs().mean().item():.6f}")
            print(f"Residual gamma             : {self.residual_gamma}")
            print("=============== AFFINE MODULATION DEBUG END ===============")
            self._printed = True

        return out