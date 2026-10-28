import torch
import torch.nn as nn
import torch.nn.functional as F

class DePatchify(nn.Module):
    """
    Reverse of VideoTokenizer:
    tokens -> reconstruct (B, C, T, H, W)

    Args:
        out_channels (int): C
        patch (tuple): (t_p, h_p, w_p)
    """
    def __init__(self, out_channels, patch):
        super().__init__()
        t_p, h_p, w_p = patch
        self.patch = patch
        self.out_channels = out_channels
        self.patch_volume = out_channels * t_p * h_p * w_p

    def forward(self, tokens, meta):
        """
        tokens: (B, N, patch_volume)
        meta: dict returned by VideoTokenizer
        Returns:
            x: (B, C, T, H, W)  -> reconstructed 3D feature map
        """
        B, N, V = tokens.shape
        t_bins, h_bins, w_bins = meta["bins"]
        t_p, h_p, w_p = meta["patch"]
        pad_t, pad_h, pad_w = meta["pad"]
        T_orig, H_orig, W_orig = meta["orig_THW"]
        C = meta["C"]

        assert N == t_bins * h_bins * w_bins, (
        f"Token count mismatch: N={N} but expected {t_bins*h_bins*w_bins}. "
        "Patchifying and depatchifying inconsistent (check patch_size or padding).")

        # 1) reshape tokens into patch grid
        # (B, N, V) → (B, t_bins, h_bins, w_bins, C, t_p, h_p, w_p)
        x = tokens.view(
            B,
            t_bins, h_bins, w_bins,
            C, t_p, h_p, w_p
        )

        # 2) reorder dimensions into full video volume
        # from (B, t_bins, h_bins, w_bins, C, t_p, h_p, w_p)
        # to   (B, C, t_bins*t_p, h_bins*h_p, w_bins*w_p)
        x = x.permute(0, 4, 1, 5, 2, 6, 3, 7)
        x = x.contiguous().view(
            B,
            C,
            t_bins * t_p,
            h_bins * h_p,
            w_bins * w_p
        )

        # 3) remove padding
        T2 = t_bins * t_p
        H2 = h_bins * h_p
        W2 = w_bins * w_p

        T = T2 - pad_t
        H = H2 - pad_h
        W = W2 - pad_w

        x = x[:, :, :T, :H, :W]  # return to original THW

        return x
