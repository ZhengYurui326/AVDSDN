import math

import torch
import torch.nn as nn
from torch import Tensor

from model.Conv_Stage.DW_Conv import DW_Conv


class MF_Extraction(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=False, Norm_type='Batch_norm'):
        super(MF_Extraction, self).__init__()

        self.Norm_type = Norm_type

        self.IQ_or_AP_Conv = DW_Conv(
            in_channels=in_channels // 2,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=True
        )
        self.IQ_and_AP_Conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=True
        )

        self.conv2 = DW_Conv(
            in_channels=out_channels + out_channels // 2,
            out_channels=out_channels + out_channels // 2,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=True
        )
        self.conv3 = DW_Conv(
            in_channels=out_channels * 3,
            out_channels=out_channels * 3,
            kernel_size=(1, 5),
            stride=1,
            padding=(0, 2),
            bias=bias
        )
        self.leaky_relu = nn.LeakyReLU()

        self.dropout = nn.Dropout(0.2)

        self.BN_Relu_Drop = nn.Sequential(
            nn.BatchNorm2d(out_channels * 3),
            nn.LeakyReLU(),
            nn.Dropout(0.2)
        )
        self.LN_Relu_Drop = nn.Sequential(
            nn.LayerNorm(normalized_shape=[out_channels * 3, 2, 128], elementwise_affine=True),
            nn.LeakyReLU(),
            nn.Dropout(0.2)
        )

        self.ECA = ECANet(channels=out_channels * 3)

    def forward(self, x):
        IQ_Data, AP_data = torch.split(x, [1, 1], dim=1)

        IQ_Data = self.IQ_or_AP_Conv(IQ_Data)
        AP_data = self.IQ_or_AP_Conv(AP_data)
        AP_IQ_data = self.IQ_and_AP_Conv(x)

        AP_IQ_data_1, AP_IQ_data_2 = torch.chunk(AP_IQ_data, 2, dim=1)

        AP_IQ_data_1 = torch.cat([AP_IQ_data_1, IQ_Data], dim=1)
        AP_IQ_data_2 = torch.cat([AP_IQ_data_2, AP_data], dim=1)

        AP_IQ_data_1 = self.conv2(AP_IQ_data_1)
        AP_IQ_data_2 = self.conv2(AP_IQ_data_2)
        AP_IQ_data_1 = self.leaky_relu(self.dropout(AP_IQ_data_1))
        AP_IQ_data_2 = self.leaky_relu(self.dropout(AP_IQ_data_2))

        out = torch.cat([AP_IQ_data_1, AP_IQ_data_2], dim=1)
        out = self.conv3(out)

        if self.Norm_type == 'Batch_norm':
            out = self.BN_Relu_Drop(out)
        else:
            out = self.LN_Relu_Drop(out)

        out = self.ECA(out)

        return out


class ECANet(nn.Module):
    def __init__(self, channels, gamma=2, b=1):
        super(ECANet, self).__init__()

        t = int(abs((math.log2(channels) + b) / gamma))
        k = t if t % 2 else t + 1
        self.kernel_size = k

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = y.squeeze(-1).transpose(-1, -2)

        y = self.conv(y)
        y = y.transpose(-1, -2).unsqueeze(-1)

        y = self.sigmoid(y)

        return x * y


class ECANet_Shrink(nn.Module):
    def __init__(self, channels, gamma=2, b=1) -> None:
        super(ECANet_Shrink, self).__init__()

        t = int(abs((math.log2(channels) + b) / gamma))
        k = t if t % 2 else t + 1
        self.kernel_size = k

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

        self.relu = nn.ReLU()
        self.Batch_norm = nn.BatchNorm2d(channels)

    def forward(self, x):
        x_init = x
        x_abs = torch.abs(x)

        y_avg = self.avg_pool(x)
        y = y_avg.squeeze(-1).transpose(-1, -2)

        y = self.conv(y)
        y = y.transpose(-1, -2).unsqueeze(-1)
        y = self.Batch_norm(y)

        y = self.sigmoid(y)

        y = y_avg * y

        sub = x_abs - y
        zeros = sub - sub
        n_sub = torch.max(sub, zeros)
        x = torch.mul(torch.sign(x_init), n_sub)

        return x


class SEAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(SEAttention, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(in_channels // reduction_ratio, in_channels)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.fc(self.avg_pool(x).view(x.size(0), -1))
        out = self.sigmoid(out).view(x.size(0), x.size(1), 1, 1)
        return x * out


class Channel_Shuffle(nn.Module):
    def __init__(self, groups):
        super(Channel_Shuffle, self).__init__()

        self.groups = groups

    def forward(self, x):
        batchsize, num_channels, height, width = x.data.size()
        assert (num_channels % self.groups == 0)
        channels_per_group = num_channels // self.groups
        x = x.view(batchsize, self.groups, channels_per_group, height, width)
        x = torch.transpose(x, 1, 2).contiguous()
        x = x.view(batchsize, -1, height, width)

        return x


class Bottleneck(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int | tuple,
            stride: int,
            padding: int | tuple,
            groups: int
    ) -> None:
        super(Bottleneck, self).__init__()
        self.stride = stride

        mid_channels = out_channels // 2

        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=mid_channels, kernel_size=1, stride=1, groups=groups),
            nn.BatchNorm2d(mid_channels),
            nn.LeakyReLU(),

            Channel_Shuffle(groups),

            DW_Conv(in_channels=mid_channels,
                    out_channels=mid_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding
            ),
            nn.BatchNorm2d(mid_channels),

            nn.Conv2d(in_channels=mid_channels, out_channels=out_channels, kernel_size=1, stride=1, groups=groups),
            nn.BatchNorm2d(out_channels),
        )

        if self.stride > 1:
            self.shortcut = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, groups=groups)

        self.relu = nn.LeakyReLU()

    def forward(self, x) -> Tensor:
        out = self.bottleneck(x)

        if self.stride > 1:
            x = self.shortcut(x)
            out = out + x
        else:
            out += x

        return self.relu(out)


class Conv_Backbone(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 groups,
                 norm_type='Batch_norm',
                 enable_downsample=True,
                 specified_out_channels=None
                 ):
        super(Conv_Backbone, self).__init__()

        self.enable_downsample = enable_downsample

        self.actual_out_channels = specified_out_channels if \
            (enable_downsample and specified_out_channels is not None) else out_channels

        self.Norm_type = norm_type
        mid_channels = in_channels // 4

        self.Conv1_1 = nn.Identity()
        self.Conv1_3 = DW_Conv(in_channels=mid_channels, out_channels=mid_channels, kernel_size=(1, 3), padding=(0, 1))
        self.Conv1_9 = DW_Conv(in_channels=mid_channels, out_channels=mid_channels, kernel_size=(1, 9), padding=(0, 4))
        self.Conv1_15 = DW_Conv(in_channels=mid_channels, out_channels=mid_channels, kernel_size=(1, 15), padding=(0, 7))

        self.BN_Relu = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU()
        )
        self.LN_Relu = nn.Sequential(
            nn.LayerNorm(normalized_shape=[in_channels, 2, 124], elementwise_affine=True),
            nn.LeakyReLU()
        )

        self.ECA = ECANet(channels=mid_channels)
        self.C_S = Channel_Shuffle(groups=groups)

        self.dropout = nn.Dropout(p=0.2)

        self.bottleneck1 = Bottleneck(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=(1, 3),
                stride=1,
                padding=(0, 1),
                groups=groups
        )
        self.bottleneck2 = Bottleneck(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=(1, 3),
                stride=1,
                padding=(0, 1),
                groups=groups
        )

        if self.enable_downsample:
            self.dw_conv = DW_Conv(
                in_channels=out_channels,
                out_channels=self.actual_out_channels,
                kernel_size=(1, 3),
                stride=(1, 2),
                padding=(0, 1),
                bias=False
            )
        else:
            self.dw_conv = nn.Identity()

        self.eca_shrink = ECANet_Shrink(channels=out_channels)

    def forward(self, x):
        x1, x2, x3, x4 = torch.chunk(x, chunks=4, dim=1)

        x1 = self.Conv1_1(x1)
        x2 = self.Conv1_3(x2)
        x3 = self.Conv1_9(x3)
        x4 = self.Conv1_15(x4)

        x1 = self.ECA(x1)
        x2 = self.ECA(x2)
        x3 = self.ECA(x3)
        x4 = self.ECA(x4)

        out = torch.cat([x1, x2, x3, x4], dim=1)
        out = self.C_S(out)

        if self.Norm_type == 'Batch_norm':
            out = self.BN_Relu(out)
        else:
            out = self.LN_Relu(out)

        out = self.dropout(out)

        out = out + x

        out = self.bottleneck1(out)
        out = self.bottleneck2(out)

        out = self.eca_shrink(out)

        out = self.dw_conv(out)

        return out


class Conv_Stage(nn.Module):
    def __init__(self,
                 in_channels=2,
                 mid_channels=32,
                 backbone_num=2,
                 bottle_groups=4,
                 downsample_flags=None,
                 output_channels_list=None
                 ):
        super(Conv_Stage, self).__init__()

        downsample_flags = downsample_flags or [True] * backbone_num

        self.mf_extra = MF_Extraction(
            in_channels=in_channels,
            out_channels=mid_channels,
            kernel_size=(1, 3),
            stride=1,
            padding=(0, 1),
            bias=False
        )

        self.conv_backbones = nn.ModuleList()
        current_channels = mid_channels * 3

        if output_channels_list is None:
            output_channels_list = []
            temp_channels = current_channels
            for i in range(backbone_num):
                if downsample_flags[i]:
                    temp_channels = temp_channels // 2
                output_channels_list.append(temp_channels)

        assert len(output_channels_list) == backbone_num, "输出通道数列表长度必须与backbone数量相同"

        for i in range(backbone_num):
            self.conv_backbones.append(
                Conv_Backbone(
                    in_channels=current_channels,
                    out_channels=current_channels,
                    groups=bottle_groups,
                    norm_type='Batch_norm',
                    enable_downsample=downsample_flags[i],
                    specified_out_channels=output_channels_list[i]
                )
            )
            if downsample_flags[i]:
                current_channels = output_channels_list[i]

        self.out_channels = current_channels

    def forward(self, x):
        out = self.mf_extra(x)
        for backbone in self.conv_backbones:
            out = backbone(out)
        return out
