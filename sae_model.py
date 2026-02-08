import torch
import torch.nn as nn

class SparseAutoencoder(nn.Module):
    def __init__(self, input_features, dict_features):
        super(SparseAutoencoder, self).__init__()
        self.encoder = nn.Linear(input_features, dict_features)
        self.decoder = nn.Linear(dict_features, input_features)

    def forward(self, x):
        encoded = torch.relu(self.encoder(x))
        decoded = self.decoder(encoded)
        return decoded, encoded
