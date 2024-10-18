import torch
from torch import nn
from kornia.color import rgb_to_hls, hls_to_rgb
from basicsr.utils.registry import ARCH_REGISTRY
import math

class PositionalEncoding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim  # 64

        self.to_log = -math.log(1e4)

    def forward(self, noise_level):
        count = self.dim // 2
        step = torch.arange(count, dtype=noise_level.dtype,
                            device=noise_level.device) / count
        encoding = noise_level.unsqueeze(
            1) * torch.exp(self.to_log * step.unsqueeze(0))
        encoding = torch.cat(
            [torch.sin(encoding), torch.cos(encoding)], dim=-1)
        return encoding


class FeatureWiseAffine(nn.Module):
    def __init__(self, in_channels, out_channels, use_affine_level=False):
        super(FeatureWiseAffine, self).__init__()
        self.use_affine_level = use_affine_level
        self.noise_func = nn.Sequential(
            nn.Linear(in_channels, out_channels*(1+self.use_affine_level))
        )

    def forward(self, x, noise_embed):
        batch = x.shape[0]
        if self.use_affine_level:
            gamma, beta = self.noise_func(noise_embed).view(
                batch, -1, 1, 1).chunk(2, dim=1)
            x = (1 + gamma) * x + beta
        else:
            x = x + self.noise_func(noise_embed).view(batch, -1, 1, 1)
        return x


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        conv1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        torch.nn.init.xavier_uniform_(conv1.weight)
        conv2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)
        torch.nn.init.xavier_uniform_(conv2.weight)
           
        self.fc = nn.Sequential(conv1,
                               nn.ReLU(),
                               conv2)


        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        
        torch.nn.init.xavier_uniform_(self.conv1.weight)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class RDB(nn.Module):
    def __init__(self, nf=64, gc=32, kernel=3, stride=3, padding=1, relu=nn.ReLU(inplace=True), noise_level_emb_dim=None, use_affine_level=False):
        super(RDB, self).__init__()
        self.conv = nn.Conv2d(nf, gc, kernel_size=kernel, stride=stride, padding=padding)
        # self.bn = nn.BatchNorm2d(gc)
        self.relu = relu
        
        torch.nn.init.kaiming_normal_(self.conv.weight, a=0.02, mode='fan_in', nonlinearity='leaky_relu')
        
        if noise_level_emb_dim is not None:
            self.noise_func = FeatureWiseAffine(noise_level_emb_dim, gc, use_affine_level)

    def forward(self, x, time_emb = None):
        x = self.conv(x)
        # x = self.bn(x)
        x = self.relu(x)

        if time_emb is not None:
            x = self.noise_func(x, time_emb)

        return x

class RRDB(nn.Module):
    def __init__(self, nf=64, gc=32, kernel=3, stride=3, relu=nn.ReLU(inplace=True), rdb_blocks=3, use_channel=True, use_spatial=True, noise_level_emb_dim=None, use_affine_level=False):
        super(RRDB, self).__init__()

        self.rdb_blocks = nn.ModuleList([RDB(nf + (i*gc), gc, kernel, stride, 1, relu, noise_level_emb_dim, use_affine_level) for i in range(rdb_blocks)])
        self.rdb_tail = RDB(nf + rdb_blocks * gc, nf, kernel, stride, 1, relu)

        self.use_channel = use_channel
        self.use_spatial = use_spatial

        if use_channel:
            self.ca_blocks = nn.ModuleList([ChannelAttention(gc) for i in range(rdb_blocks)])
            self.ca_tail = ChannelAttention(nf)

        if use_spatial:
            self.sa = SpatialAttention()
            
    def forward(self, x, time_emb = None):
        xs = [x]
        for i, l in enumerate(self.rdb_blocks):
            x_new = l(torch.cat(xs, 1), time_emb)
            if self.use_channel:
                x_new = self.ca_blocks[i](x_new) * x_new
            xs.append(x_new)
        
        x0 = self.rdb_tail(torch.cat(xs, 1))
        if self.use_channel:
            x0 = self.ca_tail(x0) * x0

        if self.use_spatial:
            x0 = self.sa(x0) * x0
        return x + x0

@ARCH_REGISTRY.register()
class RRDBUnet(nn.Module):
    # llfe_blocks = number of low-level feature extraction blocks
    # rrdb_blocks = number of RRDB
    # gff_blocks = number of global feature extraction blocks
    def __init__(self, 
                 in_channel=3,
                 llfe_blocks=1, 
                 rrdb_blocks = 4, 
                 gff_blocks = 1, 
                 relu = nn.LeakyReLU(0.02), 
                 fe_kernel=64, 
                 rrdb_kernel=32, 
                 kaiming_init=True, 
                 rdb_blocks=3, 
                 use_channel=True, 
                 use_spatial=True, 
                 color_space=None, 
                 color_blocks=3, 
                 use_affine_level=False):
        super(RRDBUnet, self).__init__()
        
        noise_level_channel = rrdb_kernel
        self.relu = relu

        self.llfe_head = nn.Conv2d(in_channel, fe_kernel, 3, 1, 1)
        self.llfe_blocks = nn.ModuleList([nn.Conv2d(fe_kernel, fe_kernel, 3, 1, 1) for i in range(llfe_blocks)])

        #residual dense blocks
        self.rrdb_blocks = nn.ModuleList([RRDB(fe_kernel, rrdb_kernel, 3, 1, self.relu, rdb_blocks=rdb_blocks, use_channel=use_channel, use_spatial=use_spatial, noise_level_emb_dim=noise_level_channel, use_affine_level=use_affine_level) for i in range(rrdb_blocks)])

        #global feature fusion
        self.gff_head = nn.Conv2d(fe_kernel*(rrdb_blocks+1), fe_kernel, 3, 1, 1)
        self.gff_blocks = nn.ModuleList([nn.Conv2d(fe_kernel, fe_kernel, 3, 1, 1) for i in range(gff_blocks)])
        self.gff_tail = nn.Conv2d(fe_kernel, 3, 3, 1, 1)

        self.color_space = color_space
        if not self.color_space == 'rgb':
            self.color_space_head = nn.Conv2d(3, fe_kernel, 3, 1, 1)
            self.color_space_blocks = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(fe_kernel, fe_kernel, 3, 1, 1), 
                    # nn.BatchNorm2d(fe_kernel),
                    self.relu,
                    nn.MaxPool2d(2)
                )
                for i in range(color_blocks)
            ])
            self.color_space_tail = nn.Sequential(
                nn.Conv2d(fe_kernel, 3, 3, 1, 1), 
                # nn.BatchNorm2d(3),
                self.relu,
                nn.AdaptiveAvgPool2d((1, 44)),
                nn.Linear(44, 1)
            )

       
        if kaiming_init:
            for l in self.llfe_blocks:
                torch.nn.init.kaiming_normal_(l.weight, a=0.02, mode='fan_in', nonlinearity='leaky_relu')
            for l in self.gff_blocks:
                torch.nn.init.kaiming_normal_(l.weight, a=0.02, mode='fan_in', nonlinearity='leaky_relu')

        self.noise_level_mlp = nn.Sequential(
            PositionalEncoding(rrdb_kernel),
            nn.Linear(rrdb_kernel, rrdb_kernel * 4),
            Swish(),
            nn.Linear(rrdb_kernel * 4, rrdb_kernel)
        )

    def forward(self, x, time = None):
        t = self.noise_level_mlp(time) 
        x = x0 = self.llfe_head(x)
        for l in self.llfe_blocks:
            x = l(x)
            
        rrdb_res = []

        f = x
        rrdb_res.append(f)
        for l in self.rrdb_blocks:
            x_rrdb = l(f, t)
            f = x_rrdb
            rrdb_res.append(f)

        x = self.relu(self.gff_head(torch.cat(rrdb_res, 1)))
        for l in self.gff_blocks:
            x = self.relu(l(x))

        x = x + x0
        x = self.relu(self.gff_tail(x))

        if not self.color_space == 'rgb' and self.color_space is not None:
            if self.color_space == 'hls':
                x_c0 = rgb_to_hls(x)
            else:
                x_c0 = x
                
            x_cs = self.color_space_head(x_c0)
            for l in self.color_space_blocks:
                x_cs = l(x_cs)
            x_cs = self.color_space_tail(x_cs)

            if self.color_space == 'hls':
                x_cs = hls_to_rgb(x_cs)

            x_cs = x_cs * x_c0

            x = x_cs + x
        return x
    

if __name__ == '__main__':
    model = RRDBUnet(in_channel=13)
    x = torch.randn(8, 13, 256, 256)
    time = torch.randn(8, 1)
    y = model(x, time)