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
    
class LSTMEncoderWithAttention(nn.Module):
    def __init__(self, seq_len, n_features, embedding_dim, attention_hidden_dim):
        super(LSTMEncoderWithAttention, self).__init__()
        self.seq_len = seq_len
        self.embedding_dim = embedding_dim
        self.hidden_dim2 = 2 * embedding_dim
        self.hidden_dim = 2 * self.hidden_dim2
        self.n_features = n_features
        
        # Neural Network Layers
        self.lstm1 = nn.LSTM(self.n_features, self.hidden_dim, num_layers=1, batch_first=True)
        self.lstm2 = nn.LSTM(self.hidden_dim, self.hidden_dim2, num_layers=1, batch_first=True)
        self.lstm3 = nn.LSTM(self.hidden_dim2, self.embedding_dim, num_layers=1, batch_first=True)
        
        self.attention = SelfAttention(self.embedding_dim, attention_hidden_dim)

    def forward(self, i):
        i, _ = self.lstm1(i)  # from (batch, seq_len, n_features) to (batch, seq_len, hidden_dim)
        i, _ = self.lstm2(i)  # from (batch, seq_len, hidden_dim) to (batch, seq_len, hidden_dim2)
        i, (hidden_n, _) = self.lstm3(i)  # from (batch, seq_len, hidden_dim2) to (batch, seq_len, embedding_dim)
        
        # Apply self-attention
        attention_output = self.attention(i)
        
        return attention_output

class LSTMDecoder(nn.Module):
    def __init__(self, seq_len, embedding_dim, n_features):
        super(LSTMDecoder, self).__init__()
        self.seq_len = seq_len
        self.embedding_dim = embedding_dim
        self.hidden_dim2 = 2 * embedding_dim
        self.hidden_dim = 2 * self.hidden_dim2
        self.n_features = n_features
        
        # Neural Network Layers
        self.lstm1 = nn.LSTM(self.embedding_dim, self.hidden_dim2, num_layers=1, batch_first=True)
        self.lstm2 = nn.LSTM(self.hidden_dim2, self.hidden_dim, num_layers=1, batch_first=True)
        self.output_layer = nn.Linear(self.hidden_dim, self.n_features)
    
    def forward(self, i):
        # Do padding
        i = i.repeat(self.seq_len, 1, 1)  # repeat (1, embedding_dim) to (seq_len, embedding_dim)
        i = i.reshape((-1, self.seq_len, self.embedding_dim))  # reshape to (batch, seq_len, embedding_dim)
        
        # Traverse neural layers
        i, _ = self.lstm1(i)  # from (batch, seq_len, embedding_dim) to (batch, seq_len, hidden_dim2)
        i, _ = self.lstm2(i)  # from (batch, seq_len, hidden_dim2) to (batch, seq_len, hidden_dim)
        i = self.output_layer(i)  # from (batch, seq_len, hidden_dim) to (batch, seq_len, n_features)
        
        return i

class LSTMAutoencoderWithAttention(nn.Module):
    def __init__(self, seq_len, n_features, embedding_dim, attention_hidden_dim):
        super(LSTMAutoencoderWithAttention, self).__init__()
        self.encoder = LSTMEncoderWithAttention(seq_len, n_features, embedding_dim, attention_hidden_dim)
        self.decoder = LSTMDecoder(seq_len, embedding_dim, n_features)
    
    def forward(self, i):
        encoded = self.encoder(i)
        decoded = self.decoder(encoded)
        return decoded
