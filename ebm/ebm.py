#!/usr/bin/env python3
"""
Train and test Conditional EBMs on multiple checkerboard datasets.
For each dataset (e.g., abc, xyz), it:
- Trains a model (or loads a saved one)
- Generates plots: loss curves, energy histograms, recovery accuracy
- Saves plots in ./plots/<dataset_name>/
"""

import json
import glob
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm

# ------------------------------
# Dataset loader (same as before)
# ------------------------------
class CheckerboardDataset(Dataset):
    def __init__(self, empty_json, solved_json, vocab_json):
        with open(empty_json, 'r') as f:
            self.empty_data = json.load(f)
        with open(solved_json, 'r') as f:
            self.solved_data = json.load(f)
        with open(vocab_json, 'r') as f:
            self.vocab = json.load(f)
        assert len(self.empty_data) == len(self.solved_data)
        self.vocab_size = len(self.vocab)
        self.pad_idx = self.vocab['<PAD>']
        self.mask_idx = self.vocab['<MASK>']
        self.grid_size = len(self.empty_data[0]['grid'])
        
    def __len__(self):
        return len(self.empty_data)
    
    def __getitem__(self, idx):
        empty_grid = torch.tensor(self.empty_data[idx]['grid'], dtype=torch.long)
        solved_grid = torch.tensor(self.solved_data[idx]['grid'], dtype=torch.long)
        return {'empty': empty_grid, 'solved': solved_grid, 'id': self.empty_data[idx]['id']}

# ------------------------------
# Model definition (same as before)
# ------------------------------
class CondCheckerboardEBM(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, n_filters=32):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.conv1 = nn.Conv2d(embed_dim*2, n_filters, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(n_filters)
        self.conv2 = nn.Conv2d(n_filters, n_filters*2, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(n_filters*2)
        self.conv3 = nn.Conv2d(n_filters*2, n_filters*2, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(n_filters*2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(n_filters*2, 1)
        self.apply(self._add_spectral_norm)
        
    def _add_spectral_norm(self, module):
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            nn.utils.spectral_norm(module)
    
    def forward(self, empty_grid, solved_grid):
        e_emb = self.embed(empty_grid).permute(0, 3, 1, 2)
        s_emb = self.embed(solved_grid).permute(0, 3, 1, 2)
        x = torch.cat([e_emb, s_emb], dim=1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x).view(x.size(0), -1)
        return self.fc(x)

# ------------------------------
# Training function (returns history)
# ------------------------------
def train_ebm(model, dataloader, epochs, lr=1e-4, n_steps=10, step_size=0.1, noise_std=0.05, device='cuda'):
    optimizer = Adam(model.parameters(), lr=lr)
    neg_buffer = None
    history = {'real_energy': [], 'fake_energy': [], 'loss': []}
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        total_real = 0.0
        total_fake = 0.0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch in pbar:
            empty = batch['empty'].to(device)
            solved = batch['solved'].to(device)
            B = empty.size(0)
            
            energy_real = model(empty, solved).mean()
            
            if neg_buffer is None:
                neg_buffer = solved.clone().detach()
            else:
                if neg_buffer.shape != solved.shape:
                    neg_buffer = solved.clone().detach()
            
            neg_buffer = neg_buffer.detach().requires_grad_(True)
            for _ in range(n_steps):
                energy_neg = model(empty, neg_buffer).sum()
                grad = torch.autograd.grad(energy_neg, neg_buffer, create_graph=False)[0]
                noise = torch.randn_like(neg_buffer) * noise_std
                neg_buffer = neg_buffer - step_size * grad + noise
                neg_buffer.data = neg_buffer.data.clamp(0, model.embed.num_embeddings - 1).long()
            neg_buffer = neg_buffer.detach()
            energy_fake = model(empty, neg_buffer).mean()
            
            loss = energy_real - energy_fake
            optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            total_real += energy_real.item()
            total_fake += energy_fake.item()
            pbar.set_postfix({'loss': loss.item(), 'E_real': energy_real.item(), 'E_fake': energy_fake.item()})
        
        avg_loss = total_loss / len(dataloader)
        avg_real = total_real / len(dataloader)
        avg_fake = total_fake / len(dataloader)
        history['loss'].append(avg_loss)
        history['real_energy'].append(avg_real)
        history['fake_energy'].append(avg_fake)
        print(f"Epoch {epoch+1}: Loss={avg_loss:.4f} Real={avg_real:.4f} Fake={avg_fake:.4f}")
    
    return history

# ------------------------------
# Fill masked cells (for testing)
# ------------------------------
def fill_masked_cells(model, empty_grid, mask_idx, vocab_size, n_steps=100, step_size=0.05, device='cuda'):
    model.eval()
    current = empty_grid.clone().detach()
    mask = (current == mask_idx).long()
    init_tokens = torch.randint(2, vocab_size, current.shape, device=device) * mask
    current = current * (1 - mask) + init_tokens
    current = current.detach().requires_grad_(True)
    for _ in range(n_steps):
        energy = model(empty_grid, current).sum()
        grad = torch.autograd.grad(energy, current)[0]
        grad = grad * mask
        current = current - step_size * grad
        current.data = current.data.clamp(0, vocab_size-1).long()
        current = current.detach().requires_grad_(True)
    return current.detach()

def compute_accuracy(pred_grid, true_grid, mask):
    pred_masked = pred_grid[mask.bool()]
    true_masked = true_grid[mask.bool()]
    if len(pred_masked) == 0:
        return 1.0
    return (pred_masked == true_masked).float().mean().item()

# ------------------------------
# Plotting functions
# ------------------------------
def plot_training_history(history, save_path):
    epochs = range(1, len(history['loss'])+1)
    plt.figure(figsize=(12,4))
    plt.subplot(1,3,1)
    plt.plot(epochs, history['loss'], 'b-', label='Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (E_real - E_fake)')
    plt.title('Training Loss')
    plt.grid(True)
    
    plt.subplot(1,3,2)
    plt.plot(epochs, history['real_energy'], 'g-', label='Real Energy')
    plt.plot(epochs, history['fake_energy'], 'r-', label='Fake Energy')
    plt.xlabel('Epoch')
    plt.ylabel('Energy')
    plt.title('Real vs Fake Energy')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1,3,3)
    plt.plot(epochs, np.array(history['fake_energy']) - np.array(history['real_energy']), 'm-')
    plt.xlabel('Epoch')
    plt.ylabel('Fake - Real Energy')
    plt.title('Energy Gap')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

def plot_energy_comparison(real_energies, random_energies, filled_energies, save_path):
    plt.figure(figsize=(10,5))
    # Histogram
    plt.subplot(1,2,1)
    plt.hist(real_energies, bins=30, alpha=0.5, label='Real', color='green')
    plt.hist(random_energies, bins=30, alpha=0.5, label='Random', color='red')
    if filled_energies:
        plt.hist(filled_energies, bins=30, alpha=0.5, label='Filled', color='blue')
    plt.xlabel('Energy')
    plt.ylabel('Frequency')
    plt.title('Energy Distribution')
    plt.legend()
    plt.grid(True)
    
    # Bar chart of means
    plt.subplot(1,2,2)
    means = [np.mean(real_energies), np.mean(random_energies)]
    labels = ['Real', 'Random']
    colors = ['green', 'red']
    if filled_energies:
        means.append(np.mean(filled_energies))
        labels.append('Filled')
        colors.append('blue')
    plt.bar(labels, means, color=colors)
    plt.ylabel('Average Energy')
    plt.title('Average Energy Comparison')
    plt.grid(True, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

def plot_recovery_accuracy(accuracies, save_path):
    plt.figure(figsize=(8,5))
    plt.bar(range(len(accuracies)), accuracies, color='skyblue')
    plt.axhline(y=np.mean(accuracies), color='r', linestyle='--', label=f'Mean = {np.mean(accuracies):.3f}')
    plt.xlabel('Test Sample Index')
    plt.ylabel('Masked Cell Accuracy')
    plt.title('Recovery Accuracy per Test Sample')
    plt.legend()
    plt.grid(True, axis='y')
    plt.savefig(save_path, dpi=150)
    plt.close()

# ------------------------------
# Process a single dataset
# ------------------------------
def process_dataset(prefix, args, device):
    print(f"\n{'='*60}\nProcessing dataset: {prefix}\n{'='*60}")
    
    # File paths
    empty_file = f"{prefix}_checkerboard_empty.json"
    solved_file = f"{prefix}_checkerboard_solved.json"
    vocab_file = f"{prefix}_vocab.json"
    
    # Create output directory for plots
    out_dir = os.path.join(args.plot_dir, prefix)
    os.makedirs(out_dir, exist_ok=True)
    
    # Load dataset
    dataset = CheckerboardDataset(empty_file, solved_file, vocab_file)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    vocab_size = dataset.vocab_size
    mask_idx = dataset.mask_idx
    print(f"Vocab size: {vocab_size}, Grid size: {dataset.grid_size}, Train: {train_size}, Test: {test_size}")
    
    # Model
    model = CondCheckerboardEBM(vocab_size, embed_dim=args.embed_dim, n_filters=args.n_filters)
    model.to(device)
    
    model_path = os.path.join(out_dir, "model.pt")
    history = None
    
    if args.mode in ["train", "both"]:
        print("Training...")
        history = train_ebm(model, train_loader, epochs=args.epochs, lr=args.lr,
                            n_steps=args.n_steps_langevin, step_size=0.1, noise_std=0.05,
                            device=device)
        torch.save(model.state_dict(), model_path)
        print(f"Model saved to {model_path}")
        # Plot training curves
        plot_training_history(history, os.path.join(out_dir, "training_curves.png"))
    else:
        # Load existing model
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=device))
            print(f"Loaded model from {model_path}")
        else:
            print(f"No model found at {model_path}, skipping testing.")
            return
    
    # ---- Testing ----
    print("Evaluating on test set...")
    model.eval()
    
    # Collect energies
    real_energies = []
    random_energies = []
    filled_energies = []  # one per test sample
    recover_accuracies = []
    
    with torch.no_grad():
        for batch in test_loader:
            empty = batch['empty'].to(device)
            solved = batch['solved'].to(device)
            e_real = model(empty, solved).cpu().numpy().flatten()
            real_energies.extend(e_real)
            
            random_solved = torch.randint(0, vocab_size, solved.shape, device=device)
            e_rand = model(empty, random_solved).cpu().numpy().flatten()
            random_energies.extend(e_rand)
    
    # For recovery accuracy, take first N test samples (limit to avoid too many)
    n_recover = min(args.n_recover, len(test_dataset))
    for i in range(n_recover):
        sample = test_dataset[i]
        empty_grid = sample['empty'].unsqueeze(0).to(device)
        true_grid = sample['solved'].unsqueeze(0).to(device)
        # Fill masked cells
        filled = fill_masked_cells(model, empty_grid, mask_idx, vocab_size,
                                   n_steps=args.n_steps_fill, step_size=0.05, device=device)
        # Energy of filled grid
        e_filled = model(empty_grid, filled).item()
        filled_energies.append(e_filled)
        # Accuracy
        mask = (empty_grid == mask_idx).long()
        acc = compute_accuracy(filled, true_grid, mask)
        recover_accuracies.append(acc)
    
    # Plot energy comparison
    plot_energy_comparison(real_energies, random_energies, filled_energies,
                           os.path.join(out_dir, "energy_comparison.png"))
    
    # Plot recovery accuracies
    plot_recovery_accuracy(recover_accuracies, os.path.join(out_dir, "recovery_accuracy.png"))
    
    # Print summary
    print(f"\nResults for {prefix}:")
    print(f"  Avg Real Energy: {np.mean(real_energies):.4f} ± {np.std(real_energies):.4f}")
    print(f"  Avg Random Energy: {np.mean(random_energies):.4f} ± {np.std(random_energies):.4f}")
    print(f"  Avg Filled Energy: {np.mean(filled_energies):.4f} ± {np.std(filled_energies):.4f}")
    print(f"  Avg Recovery Accuracy: {np.mean(recover_accuracies)*100:.2f}%")
    
    # Optionally save metrics to JSON
    metrics = {
        "real_energy_mean": float(np.mean(real_energies)),
        "real_energy_std": float(np.std(real_energies)),
        "random_energy_mean": float(np.mean(random_energies)),
        "random_energy_std": float(np.std(random_energies)),
        "filled_energy_mean": float(np.mean(filled_energies)),
        "filled_energy_std": float(np.std(filled_energies)),
        "recovery_accuracy_mean": float(np.mean(recover_accuracies)),
        "recovery_accuracy_std": float(np.std(recover_accuracies))
    }
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    
    print(f"Plots and metrics saved to {out_dir}")

# ------------------------------
# Main: find all datasets and process each
# ------------------------------
def main():
    parser = argparse.ArgumentParser(description="Train/test EBM on multiple checkerboard datasets, generate plots per dataset.")
    parser.add_argument("--data_dir", type=str, default=".", help="Directory containing dataset files")
    parser.add_argument("--plot_dir", type=str, default="./plots", help="Root directory to save plots")
    parser.add_argument("--mode", type=str, choices=["train", "test", "both"], default="both")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--embed_dim", type=int, default=64)
    parser.add_argument("--n_filters", type=int, default=32)
    parser.add_argument("--n_steps_langevin", type=int, default=10)
    parser.add_argument("--n_steps_fill", type=int, default=100)
    parser.add_argument("--n_recover", type=int, default=20, help="Number of test samples for recovery accuracy")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    
    # Find all dataset prefixes by looking for *_vocab.json files
    vocab_files = glob.glob(os.path.join(args.data_dir, "*_vocab.json"))
    if not vocab_files:
        print("No *_vocab.json files found. Please run conversion.py first.")
        return
    
    prefixes = sorted(set([os.path.basename(f).replace("_vocab.json", "") for f in vocab_files]))
    print(f"Found {len(prefixes)} datasets: {prefixes}")
    
    for prefix in prefixes:
        process_dataset(prefix, args, args.device)
    
    print("\nAll datasets processed. Plots saved in", args.plot_dir)

if __name__ == "__main__":
    main()
