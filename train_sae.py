import argparse
import torch
import torch.nn as nn
import json
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import os
from tqdm import tqdm


class SparseAutoencoder(nn.Module):
    def __init__(self, input_features, dict_features):
        super(SparseAutoencoder, self).__init__()
        self.encoder = nn.Linear(input_features, dict_features)
        self.decoder = nn.Linear(dict_features, input_features)

    def forward(self, x):
        encoded = torch.relu(self.encoder(x))
        decoded = self.decoder(encoded)
        return decoded, encoded


def main():
    parser = argparse.ArgumentParser(description="Train a Sparse Autoencoder on KataGo activations.")
    parser.add_argument("--activations-file", required=True, help="Path to the .pt file with activations and metadata.")
    parser.add_argument("--output-model-file", required=True, help="Path to save the trained SAE model.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for training.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--l1-lambda", type=float, default=1e-3, help="Sparsity penalty coefficient.")
    parser.add_argument("--dict-size-factor", type=int, default=4, help="Factor to determine dictionary size relative to input features.")
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {device}")

    print(f"Loading data from {args.activations_file}...")
    data = torch.load(args.activations_file)
    
    metadata = data["metadata"]
    activations = data["activations"]
    
    print("\n--- Successfully loaded data ---")
    print("Metadata:")
    print(json.dumps(metadata, indent=2))
    
    print("\nActivations Tensor Info:")
    print(f"  - Shape: {activations.shape}")
    print(f"  - DType: {activations.dtype}")

    # Reshape activations for SAE training
    num_samples, C, H, W = activations.shape
    input_features = C
    # Treat each spatial location (pixel) as a sample, and channels as features.
    # From (N, C, H, W) to (N*H*W, C)
    activations = activations.permute(0, 2, 3, 1).contiguous()
    activations = activations.view(-1, C)
    print(f"  - Reshaped for SAE: {activations.shape}")

    dict_features = input_features * args.dict_size_factor

    print(f"\nInitializing SAE model...")
    print(f"  - Input features: {input_features}")
    print(f"  - Dictionary features: {dict_features}")
    model = SparseAutoencoder(input_features, dict_features)
    model.to(device)
    print(model)

    dataset = TensorDataset(activations)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    mse_loss = nn.MSELoss()

    print("\n--- Starting training ---")
    print(f"Epochs: {args.epochs}, Batch size: {args.batch_size}, LR: {args.lr}, L1 lambda: {args.l1_lambda}")

    for epoch in range(args.epochs):
        epoch_loss = 0.0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{args.epochs}", unit="batch")
        for batch in progress_bar:
            inputs = batch[0].to(device)
            optimizer.zero_grad()
            
            reconstructed, encoded = model(inputs)
            
            reconstruction_loss = mse_loss(reconstructed, inputs)
            l1_loss = torch.norm(encoded, 1, dim=1).mean()
            
            loss = reconstruction_loss + args.l1_lambda * l1_loss
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            progress_bar.set_postfix(loss=f"{loss.item():.6f}")
            
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch + 1}/{args.epochs} - Average Loss: {avg_loss:.6f}")

    print("\n--- Training complete ---")

    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.output_model_file), exist_ok=True)

    torch.save(model.state_dict(), args.output_model_file)
    print(f"Saved trained model to {args.output_model_file}")
    

if __name__ == "__main__":
    main()
