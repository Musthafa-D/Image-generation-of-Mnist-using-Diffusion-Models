import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, residual=False):
        super().__init__()
        self.residual = residual
        if not mid_channels:
            mid_channels = out_channels
        self.convblock = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, out_channels),
        )

    def forward(self, x):
        if self.residual:
            return F.gelu(x + self.convblock(x))
        else:
            return self.convblock(x)


class Encoder_block(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()
        self.down = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBlock(in_channels, in_channels, residual=True),
            ConvBlock(in_channels, out_channels),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim,
                out_channels
            ),
        )

    def forward(self, x, t):
        x = self.down(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb


class Decoder_block(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = nn.Sequential(
            ConvBlock(in_channels, in_channels, residual=True),
            ConvBlock(in_channels, out_channels, in_channels // 2),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim,
                out_channels
            ),
        )

    def forward(self, x, skip_x, t):
        x = self.up(x)
        x = torch.cat([skip_x, x], dim=1)
        x = self.conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb


class UNet(nn.Module):
    def __init__(self, initial_in_channels=3, final_out_channels=3, time_dim=256, hidden_channels=16, device="cuda"):
        super().__init__()
        self.device = device
        self.time_dim = time_dim
        # Downsample
        self.inc = ConvBlock(initial_in_channels, hidden_channels)
        self.encoder1 = Encoder_block(hidden_channels, hidden_channels*2)
        self.encoder2 = Encoder_block(hidden_channels*2, hidden_channels*4)
        self.encoder3 = Encoder_block(hidden_channels*4, hidden_channels*4)

        # bottleneck
        self.bmid1 = ConvBlock(hidden_channels*4, hidden_channels*8)
        self.bmid2 = ConvBlock(hidden_channels*8, hidden_channels*8)
        self.bmid3 = ConvBlock(hidden_channels*8, hidden_channels*4)

        # upsample
        self.decoder1 = Decoder_block(hidden_channels*8, hidden_channels*2)
        self.decoder2 = Decoder_block(hidden_channels*4, hidden_channels)
        self.decoder3 = Decoder_block(hidden_channels*2, hidden_channels)
        self.outc = nn.Conv2d(hidden_channels, final_out_channels, kernel_size=1)

    def pos_encoding(self, t, channels):
        inv_freq = 1.0 / (
                10000
                ** (torch.arange(0, channels, 2, device=self.device).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=1)
        return pos_enc

    def noise_prediction(self, ins, t):
        #print(f"ins: {ins.shape}")
        #print(f"t: {t.shape}")
        t = t.unsqueeze(-1).type(torch.float)
        #print(f"t: {t.shape}")
        t = self.pos_encoding(t, self.time_dim)
        #print(f"t: {t.shape}")

        x1 = self.inc(ins)
        #print(f"x1: {x1.shape}")
        x2 = self.encoder1(x1, t)
        #print(f"x2: {x2.shape}")
        x3 = self.encoder2(x2, t)
        #print(f"x3: {x3.shape}")
        x4 = self.encoder3(x3, t)
        #print(f"x4: {x4.shape}")

        x4 = self.bmid1(x4)
        #print(f"x4: {x4.shape}")
        x4 = self.bmid2(x4)
        #print(f"x4: {x4.shape}")
        x4 = self.bmid3(x4)
        #print(f"x4: {x4.shape}")

        x = self.decoder1(x4, x3, t)
        #print(f"x: {x.shape}")
        x = self.decoder2(x, x2, t)
        #print(f"x: {x.shape}")
        x = self.decoder3(x, x1, t)
        #print(f"x: {x.shape}")
        output = self.outc(x)
        #print(f"out: {output.shape}")
        return output
    
    def forward(self, ins, t):
        return self.noise_prediction(ins, t)


class UNet_conditional(nn.Module):
    def __init__(self, initial_in_channels=3, final_out_channels=3, time_dim=256, num_classes=None, hidden_channels=16, device="cuda"):
        super().__init__()
        self.device = device
        self.time_dim = time_dim
        # Downsample
        self.inc = ConvBlock(initial_in_channels, hidden_channels)
        self.encoder1 = Encoder_block(hidden_channels, hidden_channels*2)
        self.encoder2 = Encoder_block(hidden_channels*2, hidden_channels*4)
        self.encoder3 = Encoder_block(hidden_channels*4, hidden_channels*4)

        # bottleneck
        self.bmid1 = ConvBlock(hidden_channels*4, hidden_channels*8)
        self.bmid2 = ConvBlock(hidden_channels*8, hidden_channels*8)
        self.bmid3 = ConvBlock(hidden_channels*8, hidden_channels*4)

        # upsample
        self.decoder1 = Decoder_block(hidden_channels*8, hidden_channels*2)
        self.decoder2 = Decoder_block(hidden_channels*4, hidden_channels)
        self.decoder3 = Decoder_block(hidden_channels*2, hidden_channels)
        self.outc = nn.Conv2d(hidden_channels, final_out_channels, kernel_size=1)

        if num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_dim)

    def pos_encoding(self, t, channels):
        inv_freq = 1.0 / (
                10000
                ** (torch.arange(0, channels, 2, device=self.device).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=1)
        return pos_enc

    def noise_prediction(self, ins, t, c):
        #print(f"ins: {ins.shape}")
        #print(f"t: {t.shape}")
        t = t.unsqueeze(-1).type(torch.float)
        #print(f"t: {t.shape}")
        t = self.pos_encoding(t, self.time_dim)
        #print(f"t: {t.shape}")

        if c is not None:
            #print(f"c: {c.shape}")
            #print(f"c: {self.label_emb(c).shape}")
            t += self.label_emb(c)
            #print(f"t: {t.shape}")

        x1 = self.inc(ins)
        #print(f"x1: {x1.shape}")
        x2 = self.encoder1(x1, t)
        #print(f"x2: {x2.shape}")
        x3 = self.encoder2(x2, t)
        #print(f"x3: {x3.shape}")
        x4 = self.encoder3(x3, t)
        #print(f"x4: {x4.shape}")

        x4 = self.bmid1(x4)
        #print(f"x4: {x4.shape}")
        x4 = self.bmid2(x4)
        #print(f"x4: {x4.shape}")
        x4 = self.bmid3(x4)
        #print(f"x4: {x4.shape}")

        x = self.decoder1(x4, x3, t)
        #print(f"x: {x.shape}")
        x = self.decoder2(x, x2, t)
        #print(f"x: {x.shape}")
        x = self.decoder3(x, x1, t)
        #print(f"x: {x.shape}")
        output = self.outc(x)
        #print(f"out: {output.shape}")
        return output
    
    def forward(self, ins, t, c):
        return self.noise_prediction(ins, t, c)


if __name__ == '__main__':
    net = UNet(device="cpu")
    # net = UNet_conditional(num_classes=10, device="cpu")
    print(sum([p.numel() for p in net.parameters()]))
    x = torch.randn(3, 3, 64, 64)
    t = x.new_tensor([500] * x.shape[0]).long()
    y = x.new_tensor([1] * x.shape[0]).long()
    print(net(x, t).shape)
    print(net)
    # print(net(x, t, y).shape)
