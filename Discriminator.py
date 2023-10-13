import torch
import torch.nn as nn
from Generator import Generator

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels = 1, out_channels = 32, kernel_size = 4, stride = 2, padding = 1),
            nn.LeakyReLU(0.1, inplace = True)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 4, stride = 2, padding = 1),
            nn.LeakyReLU(0.1, inplace = True)
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 4, stride = 2, padding = 1),
            nn.LeakyReLU(0.1, inplace = True)
        )

        self.layer4 = nn.Sequential(
            nn.Conv2d(128, 1, kernel_size = 4, stride = 2, padding = 1),
        )
    
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        feature = out.view(out.size()[0], -1)
        out = self.layer4(out)

        return out, feature