import torch
import torch.nn as nn

from model.Signal_Augment.VMD_SA import SignalAugmentNetwork
from model.Conv_Stage.Conv_Stage import Conv_Stage
from model.Transformer_Stage.Transformer_Stage import Transformer_Stage
from model.Feature_Fusion.FF_Stage2 import FFstage2


class AVDSDN(nn.Module):
    def __init__(self):
        super(AVDSDN, self).__init__()

        self.signal_augment = SignalAugmentNetwork(kaiming=False)

        self.conv_stage = Conv_Stage(
            in_channels=2,
            mid_channels=32,
            backbone_num=3,
            bottle_groups=4,
            downsample_flags=[True, True, True],
            output_channels_list=[96, 128, 32]
        )

        self.transformer_stage = Transformer_Stage(
            in_ch=2,
            out_ch=32,
            kernel_size=(1, 3),
            conv_stride=(1, 2),
            padding=(0, 1),

            seq_num=64,
            dim=32 * 2,
            n_head=8,
            n_head_channels=8,
            n_groups=2,
            stride=2,
            ksizes=3,
            use_pe=True,
            num_encoders=3,
            lastconv=False
        )

        self.ff_stage = FFstage2(
            in_ch=2,
            expansion=16
        )

        self.classifier = Classify_Head()

    def forward(self, x):
        out = self.signal_augment(x)
        ap = to_amp_phase(out)
        out = torch.cat([ap, out], dim=1)

        c_out = self.conv_stage(out)
        t_out = self.transformer_stage(out)

        out = self.ff_stage(t_out, c_out)
        out = self.classifier(out)

        return out


class Classify_Head(nn.Module):
    def __init__(self):
        super(Classify_Head, self).__init__()

        self.gap = nn.AdaptiveAvgPool2d((8, 8))

        self.cl_head = nn.Sequential(
            nn.Linear(in_features=128, out_features=256, bias=False),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=256, out_features=11),
        )

    def forward(self, x):
        out = self.gap(x)
        out = torch.flatten(out, start_dim=1)
        out = self.cl_head(out)

        return out


def to_amp_phase(x, nsamples=128):
    x = x if isinstance(x, torch.Tensor) else torch.as_tensor(x)

    B, _, _, _ = x.shape
    I = x[:, 0, 0, :]
    Q = x[:, 0, 1, :]

    amp = torch.sqrt(I ** 2 + Q ** 2)
    phase = torch.atan2(Q, I)

    amp = amp.view(B, 1, 1, nsamples)
    phase = phase.view(B, 1, 1, nsamples)

    x = torch.cat([amp, phase], dim=2)
    return x
