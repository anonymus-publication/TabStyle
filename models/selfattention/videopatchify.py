import torch
import torch.nn as nn
import torch.nn.functional as F

class VideoTokenizer(nn.Module):
    """
    Tokenize a 3D feature map (B, C, T, H, W) into non-overlapping patches.

    Args:
        in_ch (int): number of input channels
        patch (tuple): (t_p, h_p, w_p) patch size
        pad (tuple): (t_pad, h_pad, w_pad) padding values (0 = no pad)
        add_pos_emb (bool): whether to add learnable positional embeddings per token
    """
    def __init__(self, in_ch, patch=(2,16,16), pad=(0,0,0), add_pos_emb=False):
        super().__init__()
        self.in_ch = in_ch
        self.patch = patch
        self.pad = pad
        self.add_pos_emb = add_pos_emb

        # Optional learnable positional embedding
        if add_pos_emb:
            self.pos = nn.Parameter(torch.zeros(1, 1, in_ch * patch[0] * patch[1] * patch[2]))
            nn.init.trunc_normal_(self.pos, std=0.02)
        else:
            self.pos = None

    def forward(self, x):
        """
        x: (B, C, T, H, W)
        Returns:
            tokens: (B, N, C*t_p*h_p*w_p)
            meta:   dict for depatchify
        """
        B, C, T, H, W = x.shape
        t_p, h_p, w_p = self.patch
        pad_t, pad_h, pad_w = self.pad

        # pad manually if specified
        if pad_t or pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h, 0, pad_t))  # pad W,H,T

        # ensure non-overlapping stride = patch size
        T2, H2, W2 = x.shape[2], x.shape[3], x.shape[4]

        assert T2 % t_p == 0 and H2 % h_p == 0 and W2 % w_p == 0, \
        "Padded input not divisible by patch size. Increase pad_size."

        t_bins, h_bins, w_bins = T2 // t_p, H2 // h_p, W2 // w_p

        # create non-overlapping patches
        x = x.unfold(2, t_p, t_p).unfold(3, h_p, h_p).unfold(4, w_p, w_p)
        # shape: (B, C, t_bins, h_bins, w_bins, t_p, h_p, w_p)
        x = x.contiguous().permute(0,2,3,4,1,5,6,7)   # (B, t_bins, h_bins, w_bins, C, t_p, h_p, w_p)
        tokens = x.reshape(B, t_bins * h_bins * w_bins, C * t_p * h_p * w_p)  # (B, N, patch_vol)

        if self.pos is not None:
            tokens = tokens + self.pos.expand(B, tokens.size(1), -1)

        meta = {
            "bins": (t_bins, h_bins, w_bins),
            "patch": (t_p, h_p, w_p),
            "pad": (pad_t, pad_h, pad_w),
            "orig_THW": (T, H, W),
            "C": C,
        }
        return tokens, meta
