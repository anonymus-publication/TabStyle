import torch.nn as nn
from models.backbone.convblock import Conv2Plus1DBlock

class BasicBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        mid_channels,
        out_channels,
        stride=(1, 1, 1),
        padding=(1, 1, 1),
        downsample=None,
        use_temporal_conv=True,
        spatial_kernel_size=(1, 3, 3),
        temporal_kernel_size=(3, 1, 1),
        dropout: float = 0.0
    ):
        super().__init__()

        # First (1+2)D conv block (can include temporal conv)
        self.conv1 = Conv2Plus1DBlock(
            in_channels,
            mid_channels,
            out_channels,
            stride=stride,
            padding=padding,
            use_temporal_conv=use_temporal_conv,
            spatial_kernel_size=spatial_kernel_size,
            temporal_kernel_size=temporal_kernel_size
        )

        # Second block (usually no downsampling, always stride 1)
        self.conv2 = Conv2Plus1DBlock(
            out_channels,
            mid_channels,
            out_channels,
            stride=(1, 1, 1),
            padding=padding,
            use_temporal_conv=use_temporal_conv,
            spatial_kernel_size=spatial_kernel_size,
            temporal_kernel_size=temporal_kernel_size
        )

        self.downsample = downsample
        # self.relu = nn.ReLU(inplace=False)
        self.relu = nn.LeakyReLU(inplace=False)
        self.dropout = nn.Dropout3d(p=dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.conv2(out)
        out = self.dropout(out) 

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity   
        out = self.relu(out)   
        return out
