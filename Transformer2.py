import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class PositionalEncoding(nn.Module):
    def __init__(self, d_model,dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        # pe[:, 0::2] = torch.sin(position * div_term)
        # pe[:, 1::2] = torch.cos(position * div_term)
        pe += torch.sin(position * div_term)
        pe += torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x

    
class TransformerModel(nn.Module):
    def __init__(self, input_dim, model_dim, num_heads, num_layers, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.model_dim = model_dim
        self.embedding = nn.Linear(input_dim, model_dim)
        self.pos_encoder = PositionalEncoding(model_dim)
        encoder_layers = nn.TransformerEncoderLayer(model_dim, num_heads, dim_feedforward=128, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        decoder_layers = nn.TransformerDecoderLayer(model_dim, num_heads, dim_feedforward=128, dropout=dropout)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layers, num_layers)
        self.fc = nn.Linear(model_dim, input_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, src, ):
        src = self.embedding(src) * torch.sqrt(torch.tensor(self.model_dim, dtype=torch.float32))
        src = self.pos_encoder(src)
        memory = self.transformer_encoder(src)
        
        output = self.transformer_decoder(src, memory)
        output = self.fc(output)
        output = self.sigmoid(output)
        return output

# input_dim = 130
# model_dim = 256
# num_heads = 8
# num_layers = 4
# batch_size = 32
# seq_length = 1164