
import torch
import torch.nn as nn
import torch.nn.functional as F
import math



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
        energy = energy.transpose(1,3)
        weights = F.softmax(energy.squeeze(-1), dim=1)
        outputs = (encoder_outputs * weights.unsqueeze(-1)).sum(dim=1)
        return outputs
    


class AutoEncoderWithAttention(nn.Module):
    def __init__(self, input_size=130, attention_hidden_dim=32):
        super(AutoEncoderWithAttention, self).__init__()
        self.input_size = input_size
        self.attention_hidden_dim = attention_hidden_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU()
        )
        self.attention = SelfAttention(8, attention_hidden_dim)  # 注意力機制在編碼器的最後一層
        # self.attention = SelfAttention(attention_hidden_dim, num_attention_heads=8)
        self.decoder = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, input_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        # if labels["category"][0] == "AXIS_FRICTION":
        # attended = self.attention(encoded)
        # else:
        attended = self.attention(encoded.unsqueeze(1)).squeeze(1)
        # attended = self.attention(encoded.unsqueeze(1)).squeeze(1)
        # print(attended.shape)
        decoded = self.decoder(attended)
        return decoded
