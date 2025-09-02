import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Encoder E1 / E2 ---
class Encoder(nn.Module):
    def __init__(self, input_nc=1,input_size = 128, z_dim=512,channels = [64, 128, 256, 512, 512, 512]):
        super().__init__()
        layers = []
        in_ch = input_nc
        for out_ch in channels:
            layers.append(nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1))
            layers.append(nn.BatchNorm2d(out_ch))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            in_ch = out_ch

        self.conv = nn.Sequential(*layers)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(512 * (input_size // 2**6)**2, z_dim)  # 2^6 = 64

    def forward(self, x):
        x = self.conv(x)          # -> [B, 512, 2, 2]
        x = self.flatten(x)       # -> [B, 2048]
        return self.fc(x)         # -> [B, 512]


# --- Decoder D1 ---
class Decoder(nn.Module):
    def __init__(self, z_dim=512, output_nc=1):
        super().__init__()
        self.fc = nn.Linear(z_dim, 512 * 2 * 2)

        self.deconv = nn.Sequential(
            nn.Unflatten(1, (512, 2, 2)),
            nn.ConvTranspose2d(512, 512, 4, 2, 1),  # -> 4x4
            nn.BatchNorm2d(512),
            nn.ReLU(),

            nn.ConvTranspose2d(512, 512, 4, 2, 1),  # -> 8x8
            nn.BatchNorm2d(512),
            nn.ReLU(),

            nn.ConvTranspose2d(512, 256, 4, 2, 1),  # -> 16x16
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.ConvTranspose2d(256, 128, 4, 2, 1),  # -> 32x32
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.ConvTranspose2d(128, 64, 4, 2, 1),   # -> 64x64
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.ConvTranspose2d(64, output_nc, 4, 2, 1),  # -> 128x128
            nn.Tanh(),  # match [-1, 1]
        )

    def forward(self, z):
        x = self.fc(z)
        return self.deconv(x)

class NLOS_OT(nn.Module):
    """
    input:  [B, 1, 128, 128]
    output:  [B, 1, 128, 128]
    """
    def __init__(self,
                 input_nc: int = 1,
                 output_nc: int = 1,
                 input_size: int = 128,
                 z_dim: int = 512):
        super().__init__()
        self.encoder = Encoder(input_nc, input_size, z_dim)
        self.decoder = Decoder(z_dim, output_nc)

    def load(self,encoder_pt_path="/data/NLOSFormer/Pth/nlos_ot/49encoder_E2.pth",
             decoder_pt_path="/data/NLOSFormer/Pth/nlos_ot/4decoder_D1.pth",
             map_location=None):
        
        state_dict1 = torch.load(encoder_pt_path, map_location=map_location)
        self.encoder.load_state_dict(state_dict1)

        state_dict2 = torch.load(decoder_pt_path, map_location=map_location)
        self.decoder.load_state_dict(state_dict2)

    def forward(self, x):
        z = self.encoder(x)      # [B, z_dim]
        out = self.decoder(z)    # [B, 1, 128, 128]
        return out

if __name__=="__main__":
    model=NLOS_OT()
    model.load()
    torch.save(model.state_dict(), "/data/NLOSFormer/Pth/nlos_ot/model-base.pt")