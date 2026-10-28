import torch.nn as nn

class Conv2Plus1DBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        mid_channels,
        out_channels,
        stride=(1, 1, 1),
        padding=(1, 1, 1),
        use_temporal_conv=True,
        spatial_kernel_size=(1, 3, 3),
        temporal_kernel_size=(3, 1, 1),
    ):
        super().__init__()

        layers = []

        if use_temporal_conv:
            # Spatial first (applies spatial stride)
            layers.extend([
                nn.Conv3d(
                    in_channels,
                    mid_channels,
                    kernel_size=spatial_kernel_size,
                    stride=(1, stride[1], stride[2]),
                    padding=(0, padding[1], padding[2]),
                    bias=False,
                ),
                # nn.InstanceNorm3d(mid_channels, affine=True),
                nn.GroupNorm(num_groups=8, num_channels=mid_channels),
                # nn.ReLU(inplace=False),
                nn.LeakyReLU(inplace=False),
            ])

            # Temporal next (applies temporal stride)
            layers.extend([
                nn.Conv3d(
                    mid_channels,
                    out_channels,
                    kernel_size=temporal_kernel_size,
                    stride=(stride[0], 1, 1),
                    padding=(padding[0], 0, 0),
                    bias=False,
                ),
                # nn.InstanceNorm3d(out_channels, affine=True),
                nn.GroupNorm(num_groups=8, num_channels=out_channels),
                # nn.ReLU(inplace=False),
                # nn.LeakyReLU(inplace=False),
            ])
        else:
            layers.extend([
                nn.Conv3d(
                    in_channels,
                    out_channels,
                    kernel_size=spatial_kernel_size,
                    stride=stride,
                    padding=padding,
                    bias=False,
                ),
                nn.InstanceNorm3d(out_channels, affine=True),
                nn.ReLU(inplace=False),
            ])

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)
