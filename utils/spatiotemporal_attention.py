
import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class LearnablePositionalEncoding(nn.Module):
    """Learnable 1D positional encodings for a sequence of tokens.
    Adds (B, N, D) + (1, N, D).
    """
    def __init__(self, num_positions: int, dim: int):
        super().__init__()
        self.pe = nn.Parameter(torch.zeros(1, num_positions, dim))
        nn.init.trunc_normal_(self.pe, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, D)
        if self.pe.size(1) != x.size(1):
            # If N changed (e.g., due to different video size), interpolate along sequence
            pe = F.interpolate(self.pe.transpose(1, 2), size=x.size(1), mode="linear", align_corners=False)
            pe = pe.transpose(1, 2)
            return x + pe
        return x + self.pe


class AttentionPooling(nn.Module):
    """Single-head attention pooling over tokens.
    Computes weights alpha_i = softmax(w^T tanh(Wx_i)) and returns sum alpha_i x_i.
    """
    def __init__(self, dim: int, hidden: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, D)
        a = torch.tanh(self.fc1(x))           # (B, N, H)
        a = self.fc2(a).squeeze(-1)           # (B, N)
        w = torch.softmax(a, dim=1)           # (B, N)
        pooled = torch.einsum("bn,bnd->bd", w, x)  # (B, D)
        return pooled, w


class SpatioTemporalAttention(nn.Module):
    """3D patch-based spatiotemporal self-attention.

    Steps:
      1) Extract (possibly overlapping) 3D patches of size (t_p, h_p, w_p) with strides (t_s, h_s, w_s).
      2) Flatten each patch (C * t_p * h_p * w_p) -> project to embed_dim.
      3) Multi-head self-attention over all tokens (patches).
      4) Return either token sequence (B, N, embed_dim) or pooled vector (B, embed_dim).

    Args:
        in_channels: C from backbone output (B, C, T, H, W).
        patch_size: (t_p, h_p, w_p)
        patch_stride: (t_s, h_s, w_s) — default equals patch_size (non-overlapping).
        embed_dim: token embedding size for attention.
        num_heads: number of attention heads.
        add_positional_encoding: add learnable positional encodings over tokens.
        use_attention_pooling: if True, pool tokens with a small attention head; else mean pool.
        dropout: dropout applied after attention projection.
    """
    def __init__(
        self,
        in_channels: int,
        patch_size: Tuple[int, int, int] = (2, 8, 8),
        patch_stride: Optional[Tuple[int, int, int]] = None,
        embed_dim: int = 512,
        num_heads: int = 8,
        add_positional_encoding: bool = True,
        use_attention_pooling: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.pt, self.ph, self.pw = patch_size
        if patch_stride is None:
            patch_stride = patch_size
        self.st, self.sh, self.sw = patch_stride

        self.patch_dim = in_channels * self.pt * self.ph * self.pw
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        self.patch_proj = nn.Linear(self.patch_dim, embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        self.attn_norm = nn.LayerNorm(embed_dim)
        # can be also replaced with simple mean or attention pooling
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * embed_dim, embed_dim),
        )
        self.mlp_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

        self.add_pos = add_positional_encoding
        # We don't know N (num tokens) at init time; create a dummy and re-init on first forward.
        self.pos_enc: Optional[LearnablePositionalEncoding] = None

        self.use_attention_pooling = use_attention_pooling
        self.pool = AttentionPooling(embed_dim) if use_attention_pooling else None

    @staticmethod
    def _pad_to_divisible(x: torch.Tensor, pt: int, ph: int, pw: int) -> Tuple[torch.Tensor, Tuple[int,int,int,int,int,int]]:
        """Pad (T, H, W) dims so they are divisible by (pt, ph, pw). Returns padded x and pad tuple for unpad.
        Padding order for F.pad on 5D (B,C,T,H,W) is (W_left, W_right, H_left, H_right, T_left, T_right).
        We'll use right-only padding.
        """
        B, C, T, H, W = x.shape
        t_pad = (pt - (T % pt)) % pt
        h_pad = (ph - (H % ph)) % ph
        w_pad = (pw - (W % pw)) % pw
        pad = (0, w_pad, 0, h_pad, 0, t_pad)
        if any(pad):
            x = F.pad(x, pad)
        return x, pad

    @staticmethod
    def _unfold3d(x: torch.Tensor, pt: int, ph: int, pw: int, st: int, sh: int, sw: int) -> torch.Tensor:
        """Extract 3D patches via unfold on T, H, W dims. Returns (B, C, Nt, Nh, Nw, pt, ph, pw)."""
        # x: (B, C, T, H, W)
        x = x.unfold(dimension=2, size=pt, step=st)  # T -> Nt, pt
        x = x.unfold(dimension=3, size=ph, step=sh)  # H -> Nh, ph
        x = x.unfold(dimension=4, size=pw, step=sw)  # W -> Nw, pw
        return x  # (B, C, Nt, Nh, Nw, pt, ph, pw)

    def _ensure_posenc(self, num_tokens: int, device: torch.device):
        if not self.add_pos:
            return
        if self.pos_enc is None or self.pos_enc.pe.size(1) != num_tokens or self.pos_enc.pe.device != device:
            self.pos_enc = LearnablePositionalEncoding(num_tokens, self.embed_dim).to(device)

    def forward(
        self,
        x: torch.Tensor,
        return_tokens: bool = False,
        return_attn: bool = False,
    ):
        """Forward pass.
        Args:
            x: (B, C, T, H, W)
            return_tokens: if True, returns token sequence (B, N, D) instead of pooled vector.
            return_attn: if True, also returns the last attention weights (B, heads, N, N).
        Returns:
            If return_tokens:
                tokens: (B, N, D) and optionally attn
            Else:
                pooled: (B, D) and optionally attn
        """
        B, C, T, H, W = x.shape

        # 1) Pad to divisible by patch sizes
        x, pad = self._pad_to_divisible(x, self.pt, self.ph, self.pw)
        _, _, T2, H2, W2 = x.shape

        # 2) Unfold into 3D patches
        x = self._unfold3d(x, self.pt, self.ph, self.pw, self.st, self.sh, self.sw)
        # x: (B, C, Nt, Nh, Nw, pt, ph, pw)
        B, C, Nt, Nh, Nw, pt, ph, pw = x.shape
        N = Nt * Nh * Nw

        # 3) Flatten patches -> tokens
        x = x.contiguous().view(B, C, N, pt * ph * pw)     # (B, C, N, pt*ph*pw)
        x = x.permute(0, 2, 1, 3).contiguous()             # (B, N, C, pt*ph*pw)
        x = x.view(B, N, C * pt * ph * pw)                 # (B, N, patch_dim)

        # 4) Project to embed_dim
        tokens = self.patch_proj(x)                        # (B, N, D)

        # 5) Optional positional encoding
        if self.add_pos:
            self._ensure_posenc(num_tokens=N, device=tokens.device)
            tokens = self.pos_enc(tokens)                  # (B, N, D)

        # 6) Multi-head self-attention block with residual + MLP (Transformer encoder layer)
        # LayerNorm pre-norm
        y = self.attn_norm(tokens)
        attn_out, attn_weights = self.attn(y, y, y, need_weights=True, average_attn_weights=False)  # (B, N, D), (B, heads, N, N)
        tokens = tokens + self.dropout(attn_out)           # residual
        z = self.mlp_norm(tokens)
        z = self.mlp(z)
        tokens = tokens + self.dropout(z)                  # residual
        # tokens: (B, N, D)

        if return_tokens and not self.use_attention_pooling:
            return (tokens, attn_weights) if return_attn else tokens

        # 7) Pool tokens to a single vector per video
        if self.use_attention_pooling:
            pooled, pool_weights = self.pool(tokens)       # (B, D), (B, N)
        else:
            pooled = tokens.mean(dim=1)                    # (B, D)
            pool_weights = None

        if return_attn:
            # Return both MHSA attention weights and pooling weights (if available)
            return pooled, {"mhsa": attn_weights, "pool": pool_weights, "num_tokens": N, "grid": (Nt, Nh, Nw), "padded_size": (T2, H2, W2)}
        return pooled
