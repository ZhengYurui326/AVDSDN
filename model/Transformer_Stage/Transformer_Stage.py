import torch
import torch.nn as nn
from einops import rearrange

from model.Transformer_Stage.DAttention_V2 import DAttention


class Conv_embedding(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride, padding):
        super(Conv_embedding, self).__init__()

        self.conv_embed_1 = nn.Sequential(
            nn.Conv2d(
                in_channels=in_ch,
                out_channels=out_ch * 2,
                kernel_size=1,
                stride=1
            ),
            nn.Conv2d(
                in_channels=out_ch * 2,
                out_channels=out_ch,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding
            )
        )

    def forward(self, x):
        out = self.conv_embed_1(x)
        return out


class GluFFN(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super(GluFFN, self).__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(dim, hidden_dim)
        self.gelu = nn.ReLU()
        self.fc3 = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim)
        )

    def forward(self, x):
        x1 = self.fc1(x)
        x2 = self.fc2(x)
        x = self.gelu(x1) * x2
        f = self.fc3(x)
        return f


class ConvFFN(nn.Module):
    def __init__(self, seq_len, dim, kernel_size, stride, padding, expansion):
        super(ConvFFN, self).__init__()

        mid_ch = int(seq_len * expansion)

        self.layernorm = nn.LayerNorm(normalized_shape=dim)

        self.conv1 = nn.Conv1d(
            in_channels=seq_len,
            out_channels=mid_ch,
            kernel_size=1,
            stride=1,
            padding=0
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(
                in_channels=mid_ch,
                out_channels=mid_ch,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=mid_ch
            ),
            nn.Conv1d(
                in_channels=mid_ch,
                out_channels=mid_ch,
                kernel_size=1,
                stride=1,
                padding=0
            ),
        )
        self.conv3 = nn.Conv1d(
            in_channels=mid_ch,
            out_channels=seq_len,
            kernel_size=1,
            stride=1,
            padding=0
        )

        self.gleu = nn.GELU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        out1 = self.conv1(x)
        out2 = self.conv2(out1)
        out2 = out2 + out1
        out2 = self.gleu(out2)
        out2 = self.dropout(out2)
        out = self.conv3(out2)

        return out


class TransformerEncoder(nn.Module):
    def __init__(self, dim, seq_num, n_head, n_head_channels, n_groups, stride, ksize, use_pe):
        super(TransformerEncoder, self).__init__()

        kv_size = seq_num // stride

        attn_drop = 0.1
        proj_drop = 0.1
        offset_range_factor = 1.0

        expansion = 1.2

        self.attention = DAttention(
            q_size=seq_num,
            kv_size=kv_size,
            n_heads=n_head,
            n_head_channels=n_head_channels,
            n_groups=n_groups,
            stride=stride,
            ksize=ksize,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            offset_range_factor=offset_range_factor,
            use_pe=use_pe
        )
        self.layernorm1 = nn.LayerNorm(seq_num)
        self.layernorm2 = nn.LayerNorm(seq_num)
        self.ffn = ConvFFN(
            seq_len=dim,
            dim=seq_num,
            kernel_size=3,
            stride=1,
            padding=1,
            expansion=expansion
        )
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        out1 = self.layernorm1(self.dropout(self.attention(x)) + x)
        out2 = self.layernorm2(self.ffn(out1) + out1)
        return out2


class TransposeLayer(nn.Module):
    def forward(self, x):
        return x.transpose(1, 2)


class Transformer_Stage(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, conv_stride, padding,
                 dim, seq_num, n_head, n_head_channels, n_groups, stride, ksizes, use_pe,
                 num_encoders=1, lastconv=True):
        super(Transformer_Stage, self).__init__()

        self.pos_eb = nn.Parameter(torch.randn(1, seq_num, dim))
        self.lastconv = lastconv

        self.conv_eb = Conv_embedding(
            in_ch=in_ch,
            out_ch=out_ch,
            kernel_size=kernel_size,
            stride=conv_stride,
            padding=padding
        )

        self.encoders = nn.ModuleList()
        self.downsamples = nn.ModuleList()

        for i in range(num_encoders):
            current_seq = seq_num // (2 ** i)

            self.encoders.append(
                TransformerEncoder(
                    dim=dim,
                    seq_num=current_seq,
                    n_head=n_head,
                    n_head_channels=n_head_channels,
                    n_groups=n_groups,
                    stride=stride,
                    ksize=ksizes[i] if isinstance(ksizes, (list, tuple)) else ksizes,
                    use_pe=use_pe
                )
            )

            if (i < num_encoders - 1) or (i == num_encoders - 1 and lastconv):
                self.downsamples.append(
                    nn.Sequential(
                        nn.Conv1d(
                            in_channels=dim,
                            out_channels=dim,
                            kernel_size=3,
                            stride=2,
                            padding=1
                        ),
                        TransposeLayer(),
                        nn.LayerNorm(dim),
                        TransposeLayer()
                    )
                )

    def forward(self, x):
        x = self.conv_eb(x)

        x = rearrange(x, 'b c h w -> b (c h) w')

        x = x.permute(0, 2, 1)

        x = x + self.pos_eb[:, :x.size(1), :]

        x = rearrange(x, 'b s d -> b d s')

        for i in range(len(self.encoders)):
            x = self.encoders[i](x)

            if i < len(self.downsamples):
                x = self.downsamples[i](x)

        return x
