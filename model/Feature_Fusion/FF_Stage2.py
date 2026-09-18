import torch
import torch.nn as nn
from einops import rearrange

from model.Conv_Stage.DW_Conv import DW_Conv


class FFstage2(nn.Module):
    def __init__(self, in_ch=2, expansion=16):
        super(FFstage2, self).__init__()

        mid_ch = in_ch * expansion
        self.expansion = expansion

        self.Channel = nn.Sequential(
            nn.Conv2d(
                in_channels=in_ch,
                out_channels=mid_ch,
                kernel_size=1,
                bias=False
            ),
            nn.BatchNorm2d(mid_ch),
            DW_Conv(
                in_channels=mid_ch,
                out_channels=mid_ch,
                kernel_size=5,
                stride=2,
                padding=0,
            ),
            nn.GELU(),
            nn.Conv2d(
                in_channels=mid_ch,
                out_channels=in_ch,
                kernel_size=1
            ),
            nn.Hardswish(),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.q_conv = nn.Conv2d(in_channels=2, out_channels=expansion, kernel_size=1)
        self.k_conv = nn.Conv2d(in_channels=2, out_channels=expansion, kernel_size=1)
        self.v_conv = nn.Conv2d(in_channels=4, out_channels=expansion, kernel_size=1)
        self.out_conv = nn.Conv2d(in_channels=expansion, out_channels=2, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.avg_pool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)

        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, t, c):
        c = rearrange(c, 'b c h w -> b (c h) w').unsqueeze(1)
        t = t.unsqueeze(1)

        x = torch.cat([t, c], dim=1)

        ch_out = self.Channel(x)
        out = x * ch_out

        b, _, height, width = x.size()

        q_out = self.q_conv(x)
        k_out = self.k_conv(x)

        q_out = rearrange(q_out, 'b c h w -> b c (h w)').permute(0, 2, 1)
        k_out = rearrange(k_out, 'b c h w -> b c (h w)')

        energy = torch.bmm(q_out, k_out)
        attention = self.softmax(energy)

        pool_out = torch.cat([self.max_pool(x), self.avg_pool(x)], dim=1)
        v_out = self.v_conv(pool_out)
        pool_out = rearrange(v_out, 'b c h w -> b c (h w)')

        sp_out = torch.bmm(pool_out, attention.permute(0, 2, 1))
        sp_out = sp_out.view(b, self.expansion, height, width)
        sp_out = self.out_conv(sp_out)

        return self.gamma * out + sp_out
