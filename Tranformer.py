import torch.nn as nn
import torch.optim as optim

# 定義Transformer模型
class Transformer(nn.Module):
    def __init__(self, latent_size=8, num_heads=4, num_layers=2):
        super(Transformer, self).__init__()
        self.pos_encoder = nn.Sequential(
            nn.Linear(latent_size, latent_size),
            nn.ReLU()
        )
        encoder_layer = nn.TransformerEncoderLayer(d_model=latent_size, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pooling = nn.AvgPool1d(kernel_size=2, stride=2)
        self.decoder = nn.Sequential(
            nn.Linear(latent_size//2, latent_size),
            nn.ReLU(),
            nn.Linear(latent_size, latent_size)
        )

    def forward(self, x):
        x = self.pos_encoder(x)
        # x = x.unsqueeze(0)
        output = self.transformer_encoder(x)
        output = self.pooling(output.squeeze(0).transpose(0, 1)).transpose(0, 1)
        output = self.decoder(output)
        return output

# 訓練Transformer模型
def train_transformer(autoencoder, train_data, lr, num_epochs):
    model = Transformer(latent_size=64, num_heads=4, num_layers=2)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for epoch in range(num_epochs):
        for i in range(len(train_data)-1):
            optimizer.zero_grad()
            curr_encoding = autoencoder.encoder(train_data[i])
            next_encoding = autoencoder.encoder(train_data[i+1])
            output = model(curr_encoding)
            loss = criterion(output, next_encoding)
            loss.backward()
            optimizer.step()
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

    return model