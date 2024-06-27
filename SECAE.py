import torch
import torch.nn as nn
import torch.nn.functional as F

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=2):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv1d(in_channels, in_channels, kernel_size=kernel_size,
                                   padding=padding, groups=in_channels)
        self.pointwise = nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.init = nn.Conv1d(130, 16, kernel_size=1)
        self.conv1 = DepthwiseSeparableConv(16, 32, kernel_size=3)
        self.conv2 = DepthwiseSeparableConv(32, 64, kernel_size=3)
        self.conv3 = DepthwiseSeparableConv(64, 128, kernel_size=3)
        # self.conv4 = DepthwiseSeparableConv(128, 256, kernel_size=3)
        flatten_dim = 149760
        latent_dim = 64
        self.fc = nn.Linear(flatten_dim, latent_dim)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = F.relu(self.init(x))
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        # x = F.relu(self.conv4(x))
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        
        return x

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        latent_dim = 64
        self.fc = nn.Linear(latent_dim, 149760)
        # self.deconv0 = DepthwiseSeparableConv(256, 128, kernel_size=3)
        self.deconv1 = DepthwiseSeparableConv(128, 64, kernel_size=3)
        self.deconv2 = DepthwiseSeparableConv(64, 32, kernel_size=3)
        self.deconv3 = DepthwiseSeparableConv(32, 16, kernel_size=3)
        self.definal = nn.Conv1d(16, 130, kernel_size=1)

    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1, 128, 1170)
        # x = F.relu(self.deconv0(x))
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = F.relu(self.deconv3(x))
        x = F.sigmoid(self.definal(x))
        x = x[:, :, :1164]
        x = x.transpose(1, 2)
        return x

class SECAutoEncoder(nn.Module):
    def __init__(self):
        super(SECAutoEncoder, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
