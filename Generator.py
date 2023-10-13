import torch
import torch.nn as nn
import matplotlib.pyplot as plt

class Generator(nn.Module):

    def __init__(self, noise_dim):
        super(Generator, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Linear(noise_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace = True)
        )

        self.layer2 = nn.Sequential(
            nn.Linear(1024, 7 * 7 * 128),
            nn.BatchNorm1d(7 * 7 * 128),
            nn.ReLU(inplace = True)
        )

        self.layer3 = nn.Sequential(
            nn.ConvTranspose2d(in_channels = 128, out_channels = 64, kernel_size = 4, stride = 2, padding = 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace = True)
        )

        self.layer4 = nn.Sequential(
            nn.ConvTranspose2d(in_channels = 64, out_channels = 1, kernel_size = 4, stride = 2, padding = 1),
            nn.Tanh()
        )

    def forward(self, z):
        out = self.layer1(z)
        out = self.layer2(out)
        out = out.view(z.shape[0], 128, 7, 7)
        out = self.layer3(out)
        out = self.layer4(out)

        return out