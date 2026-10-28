import os
import torch
import torch.nn as nn
from utils.spatiotemporal_attention import SpatioTemporalAttention  # add this import at the top

class AttentionPooling(nn.Module):
    def __init__(self, feat_dim, attn_cfg: dict):
        super().__init__()
        self.type = attn_cfg.get("type", "avg")  # now supports: avg | temporal | spatial | spatiotemporal
        heads = attn_cfg.get("num_heads", 4)

        valid = {"avg", "temporal", "spatial", "spatiotemporal"}
        assert self.type in valid, f"attention.type must be one of {valid}, got {self.type!r}"

        if self.type == "temporal":
            self.attn = nn.MultiheadAttention(embed_dim=feat_dim, num_heads=heads, batch_first=True)
        elif self.type == "spatial":
            self.attn = nn.MultiheadAttention(embed_dim=feat_dim, num_heads=heads, batch_first=True)
        elif self.type == "spatiotemporal":
            self.attn = SpatioTemporalAttention(
                in_channels=feat_dim,
                patch_size=tuple(attn_cfg.get("patch_size", [2, 8, 8])),
                patch_stride=tuple(attn_cfg.get("patch_stride", [2, 8, 8])),
                embed_dim=attn_cfg.get("embed_dim", feat_dim),
                num_heads=attn_cfg.get("num_heads", heads),
                add_positional_encoding=attn_cfg.get("add_positional_encoding", True),
                use_attention_pooling=attn_cfg.get("use_attention_pooling", True),
                dropout=attn_cfg.get("dropout", 0.1),
            )
        else:
            self.attn = None
            self.pool = nn.AdaptiveAvgPool3d((1, 1, 1))

    def forward(self, x):
        """
        Args:
            x: (B, C, T, H, W)
        Returns:
            (B, C)
        """
        if self.type == "avg":
            return self.pool(x).flatten(1)  # (B, C)

        elif self.type == "temporal":
            # reduce spatial dims
            x = x.mean([-2, -1])          # (B, C, T)
            x = x.permute(0, 2, 1)        # (B, T, C)
            out, _ = self.attn(x, x, x)   # (B, T, C)
            return out.mean(1)            # (B, C)

        elif self.type == "spatial":
            x = x.mean(2)  # (B, C, H, W)
            x = x.flatten(2).permute(0, 2, 1)
            out, _ = self.attn(x, x, x)
            return out.mean(1)

        elif self.type == "spatiotemporal":
            # directly call our module (returns pooled vector)
            out = self.attn(x)  # (B, C)
            # print(f"output shape from attention", out.shape)
            return out
