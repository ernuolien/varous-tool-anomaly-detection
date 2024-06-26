import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(SelfAttention, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, encoder_outputs):
        energy = self.projection(encoder_outputs)
        weights = F.softmax(energy.squeeze(-1), dim=1)
        outputs = (encoder_outputs * weights.unsqueeze(-1)).sum(dim=1)
        return outputs

class Encoder(nn.Module):
    def __init__(self, in_channels=130, hidden_channels=220, latent_dim=200):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, hidden_channels, kernel_size=7, stride=2, padding=3)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)  # 添加第一个池化层
        
        self.conv2 = nn.Conv1d(hidden_channels, hidden_channels, kernel_size=7, stride=2, padding=3)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)  # 添加第二个池化层
        
        self.conv3 = nn.Conv1d(hidden_channels, hidden_channels, kernel_size=7, stride=2, padding=3)
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)  # 添加第三个池化层

        # 自注意力层
        self.self_attention = SelfAttention(hidden_channels, hidden_channels)

        # 计算最后一个卷积层输出的长度
        # out_length1 = (1164 + 2 * 3 - 7) // 2 + 1
        # out_length2 = (out_length1 + 2 * 3 - 7) // 2 + 1
        # out_length3 = (out_length2 + 2 * 3 - 7) // 2 + 1

        # 计算展平后的线性层输入维度
        flatten_dim = 3960
        self.fc = nn.Linear(flatten_dim, latent_dim)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = F.relu(self.conv1(x))
        x = self.pool1(x)  # 应用第一个池化层
        x = F.relu(self.conv2(x))
        x = self.pool2(x)  # 应用第二个池化层
        x = F.relu(self.conv3(x))
        x = self.pool3(x)  # 应用第三个池化层

        # 应用自注意力层
        x = x.transpose(1, 2)
        x = self.self_attention(x.unsqueeze(1)).squeeze(1)
        x = x.transpose(1, 2)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        return x

class Decoder(nn.Module):
    def __init__(self, out_channels=130, hidden_channels=220, latent_dim=200):
        super(Decoder, self).__init__()
        self.fc = nn.Linear(latent_dim, hidden_channels * 145)
        self.deconv1 = nn.ConvTranspose1d(hidden_channels, hidden_channels, kernel_size=7, stride=2, padding=3, output_padding=1)
        self.unpool1 = nn.ConvTranspose1d(hidden_channels, hidden_channels, kernel_size=2, stride=2)  # 添加第一个反池化层
        
        self.deconv2 = nn.ConvTranspose1d(hidden_channels, hidden_channels, kernel_size=7, stride=2, padding=2, output_padding=1)
        self.unpool2 = nn.ConvTranspose1d(hidden_channels, hidden_channels, kernel_size=2, stride=2)  # 添加第二个反池化层
        
        self.deconv3 = nn.ConvTranspose1d(hidden_channels, out_channels, kernel_size=7, stride=2, padding=2, output_padding=1)

    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1, 220, 145)
        x = F.relu(self.deconv1(x))
        x = self.unpool1(x)  # 应用第一个反池化层
        x = F.relu(self.deconv2(x))
        x = self.unpool2(x)  # 应用第二个反池化层
        x = F.relu(self.deconv3(x))
        x = x[:, :, :1164]
        x = x.transpose(1, 2)
        return x

class ConvAutoencoder(nn.Module):
    def __init__(self, in_channels=130, hidden_channels=220, latent_dim=200, out_channels=130):
        super(ConvAutoencoder, self).__init__()
        self.encoder = Encoder(in_channels, hidden_channels, latent_dim)
        self.decoder = Decoder(out_channels, hidden_channels, latent_dim)
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
