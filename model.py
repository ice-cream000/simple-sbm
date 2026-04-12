import torch
import torch.nn as nn
import math


#-------------------------------CIFAR-10 32x32ピクセル用のUNetモデル-------------------------------
def sinusoidal_embedding(x, L=16):
    frequencies = torch.exp(
        torch.linspace(
            math.log(1.0),
            math.log(1000.0),
            L // 2
        )
    ).to(x.device)
    angular_speeds = 2.0 * math.pi * frequencies
    embeddings = torch.cat(
        [torch.sin(angular_speeds * x), torch.cos(angular_speeds * x)],
        dim=-1
    )
    return embeddings

class DownBlock(nn.Module):
    def __init__(self, emb_dim, in_channels, out_channels):
        super(DownBlock, self).__init__()
        self.res1 = ResnetBlock(emb_dim, in_channels, out_channels)
        self.res2 = ResnetBlock(emb_dim, out_channels, out_channels)
        self.pool = nn.AvgPool2d(2)

    def forward(self, x, emb):
        skips = []
        # 1段目のResNetブロック
        x = self.res1(x, emb)
        skips.append(x)
        # 2段目のResNetブロック
        x = self.res2(x, emb)
        skips.append(x)
        # ダウンサンプリング
        x = self.pool(x)
        return x, skips

class UpBlock(nn.Module):
    def __init__(self, emb_dim, in_channels, out_channels, skip_channels):
        super(UpBlock, self).__init__()
        self.res2 = ResnetBlock(emb_dim, in_channels + skip_channels, out_channels)
        self.res1 = ResnetBlock(emb_dim, out_channels + skip_channels, out_channels)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x, emb, skip_connections):
        # アップサンプリング
        x = self.upsample(x)
        # スキップ接続
        skip_x = skip_connections.pop()
        x = torch.cat([x, skip_x], dim=1)
        # 2段目のResNetブロック
        x = self.res2(x, emb)
        # スキップ接続
        skip_x = skip_connections.pop()
        x = torch.cat([x, skip_x], dim=1)
        # 1段目のResNetブロック
        x = self.res1(x, emb)
        return x


class ResnetBlock(nn.Module):
    def __init__(self, time_embed_dim, in_channels, out_channels, kernel_size=3, stride=1, padding=1, num_groups=8):
        super(ResnetBlock, self).__init__()

        self.mlp = nn.Sequential(
            nn.Linear(time_embed_dim, out_channels),
            nn.SiLU(),
            nn.Linear(out_channels, out_channels)
        )

        self.block1 = nn.Sequential(
            nn.GroupNorm(num_groups, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        )

        self.block2 = nn.Sequential(
            nn.GroupNorm(num_groups, out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding)
        )

        self.shortcut = nn.Identity()
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x, emb):
        h = self.block1(x)
        h += self.mlp(emb).view(h.size(0), h.size(1), 1, 1)
        h = self.block2(h)
        y = h + self.shortcut(x)
        return y

class UNet(nn.Module):
    def __init__(self, in_channels=1):
        super(UNet, self).__init__()

        self.emb_dim = 128
        base_channels = 64 

        ch_levels = [base_channels, 2*base_channels, 3*base_channels, 4*base_channels]

        self.embedding = nn.Embedding(10, self.emb_dim)
        
        self.input_layer = nn.Conv2d(in_channels, base_channels, kernel_size=1, stride=1, padding=0)

        self.down1 = DownBlock(self.emb_dim, ch_levels[0], ch_levels[1])
        self.down2 = DownBlock(self.emb_dim, ch_levels[1], ch_levels[2]) 
        self.down3 = DownBlock(self.emb_dim, ch_levels[2], ch_levels[3])  

        self.bot1 = ResnetBlock(self.emb_dim, ch_levels[3], ch_levels[3])
        self.bot2 = ResnetBlock(self.emb_dim, ch_levels[3], ch_levels[3])

        self.up3 = UpBlock(self.emb_dim, ch_levels[3], ch_levels[2], skip_channels=ch_levels[3])
        self.up2 = UpBlock(self.emb_dim, ch_levels[2], ch_levels[1], skip_channels=ch_levels[2])
        self.up1 = UpBlock(self.emb_dim, ch_levels[1], ch_levels[0], skip_channels=ch_levels[1])

        self.out = nn.Conv2d(base_channels, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x, sigma, labels=None):
        if labels is None:
            labels_emb = torch.zeros(x.size(0), self.emb_dim, device=x.device)
        else:
            labels_emb = self.embedding(labels)
        
        sigma_val = torch.log(sigma.view(x.size(0), 1))
        sigma_emb = sinusoidal_embedding(sigma_val, L=self.emb_dim)
        emb = labels_emb + sigma_emb
        x = self.input_layer(x)
        
        # Down 
        x, skips1 = self.down1(x, emb)
        x, skips2 = self.down2(x, emb)
        x, skips3 = self.down3(x, emb)

        # Bottleneck
        x = self.bot1(x, emb)
        x = self.bot2(x, emb)

        # Up
        x = self.up3(x, emb, skips3)
        x = self.up2(x, emb, skips2)
        x = self.up1(x, emb, skips1)

        y = self.out(x)
        return y / sigma.view(-1, 1, 1, 1)