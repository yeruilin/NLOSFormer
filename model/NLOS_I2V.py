import torch
import torch.nn as nn
import torch.nn.functional as F

## This is a self-implementation of "Long-Wave Infrared Non-Line-of-Sight Imaging with Visible Conversion",
## because the authors do not provide their codes.

# --- Basic Convolution Block ---
def conv_block(in_ch, out_ch, kernel_size, padding=0, stride=1):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding),
        nn.ReLU(inplace=True)
    )

# --- Encoder / Decoder ---
class NLOSEncoder(nn.Module):
    def __init__(self, in_ch=1, base_ch=64):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, base_ch, kernel_size=7, padding=3),
            nn.ReLU(inplace=True),
            conv_block(base_ch, base_ch, 3, padding=1),
            conv_block(base_ch, base_ch, 3, padding=1),
            conv_block(base_ch, base_ch, 3, padding=1),
        )

    def forward(self, x):
        return self.block(x)

class NLOSDecoder(nn.Module):
    def __init__(self, out_ch=1, base_ch=64):
        super().__init__()
        self.block = nn.Sequential(
            conv_block(base_ch, base_ch, 3, padding=1),
            conv_block(base_ch, base_ch, 3, padding=1),
            conv_block(base_ch, base_ch, 3, padding=1),
            nn.Conv2d(base_ch, out_ch, kernel_size=7, padding=3)
        )

    def forward(self, x):
        return self.block(x)

# --- Enhancer Block ---
class EnhancingBlock(nn.Module):
    def __init__(self, in_ch=1, base_ch=32):
        super().__init__()
        self.pre = nn.Sequential(
            nn.Conv2d(in_ch, base_ch, 3, padding=1),
            nn.LeakyReLU(inplace=True)
        )

        self.pool_scales = [4, 8, 16, 32]
        self.pools = nn.ModuleList()
        for scale in self.pool_scales:
            self.pools.append(
                nn.Sequential(
                    nn.AvgPool2d(scale, stride=scale),
                    nn.Conv2d(base_ch, base_ch, kernel_size=1),
                )
            )
        self.final = nn.Conv2d(base_ch, in_ch, kernel_size=3, padding=1)

    def forward(self, x):
        x0 = self.pre(x)
        upsampled = []

        for i, scale in enumerate(self.pool_scales):
            out = self.pools[i](F.pad(x0, (0, 0, 0, 0)))  # Padding maybe need
            up = F.interpolate(out, size=x.shape[-2:], mode='bilinear', align_corners=False)
            upsampled.append(up)

        out = sum(upsampled)
        out = self.final(out)
        return out

# --- Full Enhancer with two EnhancingBlocks and skip connections ---
class Enhancer(nn.Module):
    def __init__(self, in_ch=1):
        super().__init__()
        self.block1 = EnhancingBlock(in_ch)
        self.block2 = EnhancingBlock(in_ch)

    def forward(self, x, skip):
        out1 = self.block1(x + skip)
        out2 = self.block2(out1 + skip)
        return out2

# --- Full Network ---
class NLOS_I2V(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = NLOSEncoder()
        self.decoder = NLOSDecoder()
        self.enhancer = Enhancer()

    def forward(self, x):
        feat = self.encoder(x)
        mid = self.decoder(feat)
        enhanced = self.enhancer(mid, x)  # two-stage residual enhancement
        return enhanced

if __name__=="__main__":
    device="cuda:0"
    a=torch.ones([8,1,128,128]).float().to(device)
    model=NLOS_I2V().to(device)
    output=model(a)
    print(output.shape)