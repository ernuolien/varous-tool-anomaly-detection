import torch
import torch.nn as nn
import torch.nn.functional as F



class Encoder(nn.Module):
    def __init__(self, in_channels=1164, hidden_channels=220, latent_dim=200):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, hidden_channels, kernel_size=7, stride=2, padding=3)
        self.conv2 = nn.Conv1d(hidden_channels, hidden_channels, kernel_size=7, stride=2, padding=3)
        self.conv3 = nn.Conv1d(hidden_channels, hidden_channels, kernel_size=7, stride=2, padding=3)
        # 估算最后一个卷积层输出的维度。这将取决于输入数据的长度。
        # 假设输入长度为L，则输出长度大约为 L / 2^3
        self.fc = nn.Linear(hidden_channels * 17, latent_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        # print("After conv1:", x.shape)  # 打印第一层卷积后的尺寸
        x = F.relu(self.conv2(x))
        # print("After conv2:", x.shape)  # 打印第二层卷积后的尺寸
        x = F.relu(self.conv3(x))
        # print("After conv3:", x.shape)  # 打印第三层卷积后的尺寸
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        # print("After fc:", x.shape)  # 打印全连接层后的尺寸
        return x
    
class Decoder(nn.Module):
    def __init__(self, out_channels=1164, hidden_channels=220, latent_dim=200):
        super(Decoder, self).__init__()
        # 估算与Encoder中相对应的扩展维度
        self.fc = nn.Linear(latent_dim, hidden_channels * 17)
        self.deconv1 = nn.ConvTranspose1d(hidden_channels, hidden_channels, kernel_size=7, stride=2, padding=3, output_padding=0)
        self.deconv2 = nn.ConvTranspose1d(hidden_channels, hidden_channels, kernel_size=7, stride=2, padding=3, output_padding=1)
        self.deconv3 = nn.ConvTranspose1d(hidden_channels, out_channels, kernel_size=7, stride=2, padding=3, output_padding=0)

    def forward(self, x):
        x = self.fc(x)
        # print("After fc in decoder:", x.shape)  # 打印进入解码器的全连接层后的尺寸
        x = x.view(-1, 220, 17)  # 根据实际输入调整
        # print("After view:", x.shape)  # 打印调整尺寸后的形状
        x = F.relu(self.deconv1(x))
        # print("After deconv1:", x.shape)  # 打印第一层转置卷积后的尺寸
        x = F.relu(self.deconv2(x))
        # print("After deconv2:", x.shape)  # 打印第二层转置卷积后的尺寸
        x = F.relu(self.deconv3(x))
        # print("After deconv3:", x.shape)  # 打印第三层转置卷积后的尺寸
        x = x[:, :, :130]
        return x



class ConvAutoencoder(nn.Module):
    def __init__(self, in_channels=1164, hidden_channels=220, latent_dim=200, out_channels=1164):
        super(ConvAutoencoder, self).__init__()
        self.encoder = Encoder(in_channels, hidden_channels, latent_dim)
        self.decoder = Decoder(out_channels, hidden_channels, latent_dim)
        
    def forward(self, x):
        # Encoder返回潜在空间的表示
        x = self.encoder(x)
        # Decoder使用潜在表示恢复数据
        x = self.decoder(x)
        return x

