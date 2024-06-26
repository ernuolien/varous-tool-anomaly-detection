
import torch
import torch.nn as nn
import torch.nn.functional as F

# 定義 Autoencoder 模型
# class AutoEncoder(nn.Module):
#     def __init__(self, input_size=130):
#         super(AutoEncoder, self).__init__()
#         self.encoder = nn.Sequential(
#             nn.Linear(input_size, 64),
#             nn.ReLU(),
#             nn.Linear(64, 32),
#             nn.ReLU(),
#             nn.Linear(32, 16),
#             nn.ReLU()
#         )
#         self.decoder = nn.Sequential(
#             nn.Linear(16, 32),
#             nn.ReLU(),
#             nn.Linear(32, 64),
#             nn.ReLU(),
#             nn.Linear(64, input_size),
#             nn.Sigmoid()  # Sigmoid 用於將輸出限制在 [0, 1] 範圍內，這在數據重建時有用
#         )

#     def forward(self, x):
#         # print('input',x)
#         x = self.encoder(x)
#         x = self.decoder(x)
        
#         # print('output',x)
#         return x

# class AutoEncoder(nn.Module):
#     def __init__(self, input_size=130):
#         super(AutoEncoder, self).__init__()
#         self.encoder = nn.Sequential(
#             nn.Linear(input_size, 64),
#             nn.ReLU(),
#             nn.Linear(64, 32),
#             nn.ReLU(),
#             nn.Linear(32, 16),
#             nn.ReLU(),
#             nn.Linear(16, 8),
#             nn.ReLU()
#         )
#         self.decoder = nn.Sequential(
#             nn.Linear(8, 16),
#             nn.ReLU(),
#             nn.Linear(16, 32),
#             nn.ReLU(),
#             nn.Linear(32, 64),
#             nn.ReLU(),
#             nn.Linear(64, input_size),
#             nn.Sigmoid()  # Sigmoid 用於將輸出限制在 [0, 1] 範圍內，這在數據重建時有用
#         )

#     def forward(self, x):
#         # print('input',x)
#         x = self.encoder(x)
#         x = self.decoder(x)
        
#         # print('output',x)
#         return x


class AutoEncoder(nn.Module):
    def __init__(self, input_size=130):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, input_size),
            nn.Sigmoid()  # Sigmoid 用於將輸出限制在 [0, 1] 範圍內，這在數據重建時有用
        )

    def forward(self, x):
        # print('input',x)
        x = self.encoder(x)
        x = self.decoder(x)
        
        # print('output',x)
        return x