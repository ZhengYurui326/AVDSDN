import torch.nn as nn
from torch import Tensor


class DW_Conv(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int | tuple,
            stride: int | tuple = 1,
            padding: int | tuple = 0,
            bias: bool = False
    ) -> None:
        super(DW_Conv, self).__init__()

        self.depthwise = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels,
            bias=bias
        )

        self.pointwise = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x
