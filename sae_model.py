import torch
import torch.nn as nn

class SparseAutoencoder(nn.Module):
    def __init__(self, input_features, dict_features):
        super(SparseAutoencoder, self).__init__()
        # The encoder is a linear transformation (y = Wx + b).
        # self.encoder.weight corresponds to the weight matrix W_enc.
        # self.encoder.bias corresponds to the bias vector b_enc.
        self.encoder = nn.Linear(input_features, dict_features)

        # The decoder is also a linear transformation.
        # self.decoder.weight corresponds to the weight matrix W_dec.
        # self.decoder.bias corresponds to the bias vector b_dec.
        self.decoder = nn.Linear(dict_features, input_features)

    def forward(self, x):
        encoded = torch.relu(self.encoder(x))
        decoded = self.decoder(encoded)
        return decoded, encoded
