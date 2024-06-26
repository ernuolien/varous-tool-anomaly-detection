
# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class LSTMEncoder(nn.Module):
    
#     def __init__(self, seq_len, n_features, embedding_dim):
#         super(LSTMEncoder, self).__init__()
        
#         # Parameters
#         self.seq_len = seq_len
#         self.n_features = n_features
#         self.embedding_dim = embedding_dim
#         self.hidden_dim = 2*embedding_dim
        
#         # Neural Network Layers
#         self.lstm1 = nn.LSTM(self.n_features, self.hidden_dim, num_layers=1, batch_first=True)
#         self.lstm2 = nn.LSTM(self.hidden_dim, self.embedding_dim, num_layers=1, batch_first=True)
    
#     def forward(self, i): 
#         i, _ = self.lstm1(i)               # from (batch, seq_len, n_features) to (batch, seq_len, hidden_dim)
#         i, (hidden_n, _) = self.lstm2(i)   # from (batch, seq_len, hidden_dim) to (batch, seq_len, embedding_dim)
#         return hidden_n                    # hidden_n shape: (num_layers*num_directions, batch, embedding_dim)


# class LSTMDecoder(nn.Module):

#     def __init__(self, seq_len, embedding_dim, n_features=1):
#         super(LSTMDecoder, self).__init__()

#         # Parameters
#         self.seq_len = seq_len
#         self.embedding_dim = embedding_dim
#         self.hidden_dim = 2*embedding_dim
#         self.n_features = n_features
        
#         # Neural Network Layers
#         self.lstm1 = nn.LSTM(self.embedding_dim, self.embedding_dim, num_layers=1, batch_first=True)
#         self.lstm2 = nn.LSTM(self.embedding_dim, self.hidden_dim, num_layers=1, batch_first=True)
#         self.output_layer = nn.Linear(self.hidden_dim, n_features)
        
#     def forward(self, i):
#         # Do padding
#         i = i.repeat(self.seq_len, 1, 1)                       # repeat (1, embedding_dim) to (seq_len, embedding_dim)
#         i = i.reshape((-1, self.seq_len, self.embedding_dim))  # reshape to (batch, seq_len, embedding_dim)
        
#         # Traverse neural layers
#         i, _ = self.lstm1(i)      # from (batch, seq_len, embedding_dim) to (batch, seq_len, embedding_dim)
#         i, _ = self.lstm2(i)      # from (batch, seq_len, embedding_dim) to (batch, seq_len, hidden_dim)
#         i = self.output_layer(i)  # from (batch, seq_len, hidden_dim) to (batch, seq_len, n_features)
        
#         return i



# class LSTMAutoencoder(nn.Module):
#     def __init__(self, seq_len, n_features, embedding_dim):
#         super(LSTMAutoencoder, self).__init__()
#         self.encoder = LSTMEncoder(seq_len, n_features, embedding_dim)
#         self.decoder = LSTMDecoder(seq_len, embedding_dim, n_features)
        
#     def forward(self, i):
#         i = self.encoder(i)
#         i = self.decoder(i)
#         return i
    

import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTMEncoder(nn.Module):
    
    def __init__(self, seq_len, n_features, embedding_dim):
        super(LSTMEncoder, self).__init__()
        
        # Parameters
        self.seq_len = seq_len
        self.embedding_dim = embedding_dim
        self.hidden_dim2 = 2*embedding_dim
        self.hidden_dim = 2*self.hidden_dim2
        self.n_features = n_features
        
        # Neural Network Layers
        self.lstm1 = nn.LSTM(self.n_features, self.hidden_dim, num_layers=1, batch_first=True)
        self.lstm2 = nn.LSTM(self.hidden_dim, self.hidden_dim2, num_layers=1, batch_first=True)
        self.lstm3 = nn.LSTM(self.hidden_dim2, self.embedding_dim, num_layers=1, batch_first=True)
    
    def forward(self, i): 
        i, _ = self.lstm1(i)               # from (batch, seq_len, n_features) to (batch, seq_len, hidden_dim)
        i, (hidden_n, _) = self.lstm2(i)   # from (batch, seq_len, hidden_dim) to (batch, seq_len, embedding_dim)
        i, (hidden_n, _) = self.lstm3(i)   # from (batch, seq_len, embedding_dim) to (batch, seq_len, embedding_dim)
        return hidden_n                    # hidden_n shape: (num_layers*num_directions, batch, embedding_dim)


class LSTMDecoder(nn.Module):

    def __init__(self, seq_len, embedding_dim, n_features):
        super(LSTMDecoder, self).__init__()

        # Parameters
        self.seq_len = seq_len
        self.embedding_dim = embedding_dim
        self.hidden_dim2 = 2*embedding_dim
        self.hidden_dim = 2*self.hidden_dim2
        self.n_features = n_features
        
        # Neural Network Layers
        self.lstm1 = nn.LSTM(self.embedding_dim, self.hidden_dim2, num_layers=1, batch_first=True)
        self.lstm2 = nn.LSTM(self.hidden_dim2, self.hidden_dim, num_layers=1, batch_first=True)
        self.output_layer = nn.Linear(self.hidden_dim, n_features)
        
    def forward(self, i):
        # Do padding
        i = i.repeat(self.seq_len, 1, 1)                       # repeat (1, embedding_dim) to (seq_len, embedding_dim)
        i = i.reshape((-1, self.seq_len, self.embedding_dim))  # reshape to (batch, seq_len, embedding_dim)
        
        # Traverse neural layers
        i, _ = self.lstm1(i)      # from (batch, seq_len, embedding_dim) to (batch, seq_len, embedding_dim)
        i, _ = self.lstm2(i)      # from (batch, seq_len, embedding_dim) to (batch, seq_len, hidden_dim)
        i = self.output_layer(i)  # from (batch, seq_len, hidden_dim) to (batch, seq_len, n_features)
        
        return i


class LSTMAutoencoder(nn.Module):
    def __init__(self, seq_len, n_features, embedding_dim):
        super(LSTMAutoencoder, self).__init__()
        self.encoder = LSTMEncoder(seq_len, n_features, embedding_dim)
        self.decoder = LSTMDecoder(seq_len, embedding_dim, n_features)
        
    def forward(self, i):
        i = self.encoder(i)
        i = self.decoder(i)
        return i