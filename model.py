import torch
import torch.nn as nn

'''Model Definition'''
class UNet(nn.Module):
    def __init__(self, in_channels=2):  # Specify the number of input channels
        super(UNet, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Conv1d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose1d(64, 1, kernel_size=2, stride=2, padding=0),
            nn.Linear(640,1280)
        )

    def forward(self, x):
        # Encoder
        x1 = self.encoder(x)

        # Bottleneck
        x2 = self.bottleneck(x1)

        # Decoder
        x3 = self.decoder(x2)

        # Reshape the output to match the target size
        x3 = x3[:, :, :x.shape[2]]  # Trim or pad if necessary
        return x3
