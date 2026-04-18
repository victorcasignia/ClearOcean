import torch
import torch.nn as nn


class CALayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


class CALayerV2(nn.Module):
    def __init__(self, channel, reduction=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x + x * y


class SHALayer(nn.Module):
    def __init__(self, in_channel, reduction=4):
        super().__init__()
        self.conv_ks1 = nn.Conv2d(in_channel, in_channel // reduction, 1, 1, 0)
        self.conv_ks3 = nn.Conv2d(in_channel // reduction, in_channel, 3, 1, 1)
        self.relu6 = nn.ReLU6()
        self.sigmoid = nn.Sigmoid()

    def forward(self, fea):
        b, c, h, w = fea.shape

        h_avg = torch.mean(fea, dim=3, keepdim=True)
        h_max, _ = torch.max(fea, dim=3, keepdim=True)
        h_enc = (h_avg + h_max).view(b, c, h, 1)

        v_avg = torch.mean(fea, dim=2, keepdim=True)
        v_max, _ = torch.max(fea, dim=2, keepdim=True)
        v_enc = (v_avg + v_max).view(b, c, w, 1)

        enc = torch.cat([h_enc, v_enc], dim=2)
        enc = self.relu6(self.conv_ks1(enc))

        reduce_c = enc.shape[1]
        h_branch, v_branch = torch.split(enc, [h, w], dim=2)
        h_branch = self.conv_ks3(h_branch.view(b, reduce_c, h, 1))
        v_branch = self.conv_ks3(v_branch.view(b, reduce_c, 1, w))

        attn_mask = self.sigmoid(h_branch * v_branch)
        return attn_mask * fea


class SHALayerV2(nn.Module):
    def __init__(self, in_channel, reduction=4):
        super().__init__()
        self.conv_ks1 = nn.Conv2d(in_channel, in_channel // reduction, 1, 1, 0)
        self.conv_ks3 = nn.Conv2d(in_channel // reduction, in_channel, 3, 1, 1)
        self.relu6 = nn.ReLU6()
        self.sigmoid = nn.Sigmoid()

    def forward(self, fea):
        b, c, h, w = fea.shape

        h_avg = torch.mean(fea, dim=3, keepdim=True)
        h_max, _ = torch.max(fea, dim=3, keepdim=True)
        h_enc = (h_avg + h_max).view(b, c, h, 1)

        v_avg = torch.mean(fea, dim=2, keepdim=True)
        v_max, _ = torch.max(fea, dim=2, keepdim=True)
        v_enc = (v_avg + v_max).view(b, c, w, 1)

        enc = torch.cat([h_enc, v_enc], dim=2)
        enc = self.relu6(self.conv_ks1(enc))

        reduce_c = enc.shape[1]
        h_branch, v_branch = torch.split(enc, [h, w], dim=2)
        h_branch = self.conv_ks3(h_branch.view(b, reduce_c, h, 1))
        v_branch = self.conv_ks3(v_branch.view(b, reduce_c, 1, w))

        attn_mask = self.sigmoid(h_branch * v_branch)
        return attn_mask * fea + fea
