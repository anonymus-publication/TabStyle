import torch.nn as nn


class SpatioTemporalSelfAttention(nn.Module):
    """
    Self-attention over patch tokens.

    tokens: (B, N, D) with batch_first=True
    """
    def __init__(self, token_dim, num_heads=4, num_layers=2, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=token_dim,
            nhead=num_heads,
            dim_feedforward=int(token_dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, tokens):
        # tokens: (B, N, D)
        return self.encoder(tokens)  # (B, N, D)
