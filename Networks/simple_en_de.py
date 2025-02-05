import torch
from ccbdl.network.base import BaseNetwork


class ConvBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ConvBlock, self).__init__()
        self.sequence = torch.nn.Sequential(torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding),
                                            torch.nn.BatchNorm2d(out_channels),
                                            torch.nn.LeakyReLU(0.2, inplace=True))
    def forward(self, ins):
        return self.sequence(ins)


class ConvTransposeBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ConvTransposeBlock, self).__init__()
        self.sequence = torch.nn.Sequential(torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding=padding),
                                            torch.nn.BatchNorm2d(out_channels),
                                            torch.nn.LeakyReLU(0.2, inplace=True))
    def forward(self, ins):
        return self.sequence(ins)


class Simple_EN_DE(BaseNetwork):
    def __init__(self, name, in_channels, hidden_channels, in_channels_de):
       super().__init__(name)
       
       self.encoder = torch.nn.Sequential(
           ConvBlock(in_channels, hidden_channels, 3, 1, 1), #32
           ConvBlock(hidden_channels, hidden_channels*2, 4, 2, 1), #16
           ConvBlock(hidden_channels*2, in_channels_de, 4, 2, 1) #8
       )

       self.decoder = torch.nn.Sequential(
           ConvTransposeBlock(in_channels_de, hidden_channels*2, 3, 1, 1), #8
           ConvTransposeBlock(hidden_channels*2, hidden_channels, 4, 2, 1), #16
           ConvTransposeBlock(hidden_channels, in_channels, 4, 2, 1), #32
       )
    
    def forward(self, ins):
        out = ins
        out = self.encoder(out)
        out = self.decoder(out)
        return out