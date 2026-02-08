import argparse
import torch
import torch.nn as nn
import json
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import os
from tqdm import tqdm

from sae_model import SparseAutoencoder


def main():
    parser = argparse.ArgumentParser(description="Train a Sparse Autoencoder on KataGo activations.")
    parser.add_argument("--activations-file", required=True, help="Path to the .pt file with activations and metadata.")
    parser.add_argument("--output-model-file", required=True, help="Path to save the trained SAE model.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for training.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--sparsity-coeff", type=float, default=1e-3, help="Sparsity penalty coefficient.")
    parser.add_argument("--sparsity-type", type=str, default='l1', choices=['l1', 'lp'], help="Type of sparsity penalty.")
    parser.add_argument("--lp-norm-p", type=float, default=0.9, help="The p value for the Lp norm sparsity penalty, if used.")
    parser.add_argument("--dict-size-factor", type=int, default=8, help="Factor to determine dictionary size relative to input features (activation dimension).")
    parser.add_argument("--validation-split", type=float, default=0.2, help="Fraction of data to use for validation.")
    parser.add_argument("--spike-threshold", type=float, default=1e-6, help="Threshold for considering a feature activation a 'spike'.")
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

    # --- Reshaping Activations for SAE Training ---
    # The `trunkfinal` tensor for a single board position has the shape (C, H, W), 
    # which is (512, 19, 19). You are correct that this is a high-dimensional object.
    # For the SAE, we make a critical architectural choice: we treat each of the 19x19
    # spatial locations as an independent data point. The "features" for each data point
    # are the 512 channel values at that location.
    #
    # This transforms the problem from learning features on a 512x19x19 space to learning
    # features on a 512-dimensional space, but with many more samples (N * 19 * 19).
    # This is a standard approach for applying SAEs to convolutional activations.
    # The key trade-off is that this approach discards all spatial information; the SAE
    # cannot learn features that represent patterns across multiple board locations (e.g.
    # the shape of a group of stones). It can only learn features that exist at a single point.
    num_samples, C, H, W = activations.shape

    # The 'original space' is the channel dimension of the activations.
    # 'input_features' is a standard ML term for the dimensionality of the input vector.
    input_features = C
    
    # Reshape from (N, C, H, W) to (N*H*W, C)
    activations = activations.permute(0, 2, 3, 1).contiguous()
    activations = activations.view(-1, C)
    print(f"  - Reshaped for SAE: {activations.shape}")

    # The 'large space' is the dictionary feature space, which is intentionally overcomplete.
    # 'dict_features' refers to the size of the SAE's internal "dictionary" of features.
    # From a math perspective, this is the dimension of the higher-dimensional space we are embedding into.
    dict_features = input_features * args.dict_size_factor

    print(f"\nInitializing SAE model...")
    print(f"  - Activation dimension ('input_features'): {input_features}")
    print(f"  - SAE hidden dimension ('dict_features'): {dict_features}")
    model = SparseAutoencoder(input_features, dict_features)
    model.to(device)
    print(model)

    dataset = TensorDataset(activations)
    
    # Split data into training and validation sets
    num_samples = len(dataset)
    num_validation = int(num_samples * args.validation_split)
    num_train = num_samples - num_validation
    train_dataset, val_dataset = random_split(dataset, [num_train, num_validation])
    
    print(f"\nSplitting data into {len(train_dataset)} training and {len(val_dataset)} validation samples.")
    
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    mse_loss = nn.MSELoss()

    print("\n--- Starting training ---")
    print(f"Epochs: {args.epochs}, Batch size: {args.batch_size}, LR: {args.lr}, Validation split: {args.validation_split}")
    print(f"Sparsity: type={args.sparsity_type}, coeff={args.sparsity_coeff}" + (f", p={args.lp_norm_p}" if args.sparsity_type == 'lp' else ""))

    for epoch in range(args.epochs):
        # Training phase
        model.train()
        train_recon_loss = 0.0
        train_sparsity_loss = 0.0
        train_avg_spikes = 0.0
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{args.epochs} [Train]", unit="batch")
        for batch in progress_bar:
            inputs = batch[0].to(device)
            optimizer.zero_grad()
            
            reconstructed, encoded = model(inputs)
            
            reconstruction_loss = mse_loss(reconstructed, inputs)

            if args.sparsity_type == 'l1':
                sparsity_loss = torch.norm(encoded, 1, dim=1).mean()
            elif args.sparsity_type == 'lp':
                sparsity_loss = torch.norm(encoded, p=args.lp_norm_p, dim=1).mean()
            else:
                raise ValueError(f"Unknown sparsity type: {args.sparsity_type}")
            
            loss = reconstruction_loss + args.sparsity_coeff * sparsity_loss
            
            loss.backward()
            optimizer.step()
            
            train_recon_loss += reconstruction_loss.item()
            train_sparsity_loss += sparsity_loss.item()
            
            # The 'expected count of spikes' is the average number of features that are non-zero.
            # We calculate this by counting features with an absolute value above a small threshold.
            num_spikes = (encoded.abs() > args.spike_threshold).float().sum(dim=1)
            train_avg_spikes += num_spikes.mean().item()

            progress_bar.set_postfix(recon_loss=f"{reconstruction_loss.item():.6f}",
                                     sparsity_loss=f"{sparsity_loss.item():.6f}",
                                     avg_spikes=f"{num_spikes.mean().item():.2f}")
            
        avg_train_recon_loss = train_recon_loss / len(train_dataloader)
        avg_train_sparsity_loss = train_sparsity_loss / len(train_dataloader)
        avg_train_spikes = train_avg_spikes / len(train_dataloader)

        # Validation phase
        model.eval()
        val_recon_loss = 0.0
        val_sparsity_loss = 0.0
        val_avg_spikes = 0.0
        with torch.no_grad():
            for batch in val_dataloader:
                inputs = batch[0].to(device)
                reconstructed, encoded = model(inputs)
                reconstruction_loss = mse_loss(reconstructed, inputs)
                
                if args.sparsity_type == 'l1':
                    sparsity_loss = torch.norm(encoded, 1, dim=1).mean()
                elif args.sparsity_type == 'lp':
                    sparsity_loss = torch.norm(encoded, p=args.lp_norm_p, dim=1).mean()

                val_recon_loss += reconstruction_loss.item()
                val_sparsity_loss += sparsity_loss.item()

                num_spikes = (encoded.abs() > args.spike_threshold).float().sum(dim=1)
                val_avg_spikes += num_spikes.mean().item()

        avg_val_recon_loss = val_recon_loss / len(val_dataloader)
        avg_val_sparsity_loss = val_sparsity_loss / len(val_dataloader)
        avg_val_spikes = val_avg_spikes / len(val_dataloader)

        print(f"Epoch {epoch + 1}/{args.epochs} - "
              f"Train Recon: {avg_train_recon_loss:.6f}, Sparsity: {avg_train_sparsity_loss:.6f}, Spikes: {avg_train_spikes:.2f} | "
              f"Val Recon: {avg_val_recon_loss:.6f}, Sparsity: {avg_val_sparsity_loss:.6f}, Spikes: {avg_val_spikes:.2f}")

    print("\n--- Training complete ---")

    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.output_model_file), exist_ok=True)

    hyperparameters = vars(args)
    hyperparameters['input_features'] = input_features
    hyperparameters['dict_features'] = dict_features

    torch.save({
        'model_state_dict': model.state_dict(),
        'hyperparameters': hyperparameters
    }, args.output_model_file)
    print(f"Saved trained model and hyperparameters to {args.output_model_file}")
    

if __name__ == "__main__":
    main()
