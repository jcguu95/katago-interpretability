import torch
import torch.nn as nn

class SparseAutoencoder(nn.Module):
    def __init__(self, input_features, dict_features, activation='relu'):
        super(SparseAutoencoder, self).__init__()
        # The encoder is a linear transformation (y = Wx + b).
        # self.encoder.weight corresponds to the weight matrix W_enc.
        # self.encoder.bias corresponds to the bias vector b_enc.
        self.encoder = nn.Linear(input_features, dict_features)

        # The decoder is also a linear transformation.
        # self.decoder.weight corresponds to the weight matrix W_dec.
        # self.decoder.bias corresponds to the bias vector b_dec.
        self.decoder = nn.Linear(dict_features, input_features)

        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        else:
            raise ValueError(f"Unsupported activation function: {activation}")

    def forward(self, x):
        # The "nonlinear twist" is the activation function.
        # ReLU is the standard choice for SAEs because its property of setting
        # negative inputs to zero naturally encourages sparse activations.
        # Other functions could be used, but ReLU's properties align well with
        # the model's objective.
        encoded = self.activation(self.encoder(x))
        decoded = self.decoder(encoded)
        return decoded, encoded
