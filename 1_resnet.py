import glob
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# ── Config ───────────────────────────────────────────────
BATCH_SIZE = 512
LR         = 1e-3
HIDDEN_DIM = 256
NUM_LAYERS = 4
DROPOUT    = 0 
WD         = 0
EPOCHS     = 500
EVAL_EVERY = 1
# ─────────────────────────────────────────────────────────

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Data ─────────────────────────────────────────────────
def parse_col(series: pd.Series) -> np.ndarray:
    strings = series.astype(str).str.strip()
    max_len = strings.str.len().max()
    rows = []
    for s in strings:
        rows.append([float(b) for b in s.zfill(max_len)])
    return np.array(rows, dtype=np.float32)

class BinaryCSVDataset(Dataset):
    def __init__(self, path):
        df = pd.read_csv(path, header=None, dtype=str)
        self.y = parse_col(df.iloc[:, 0])
        self.X = parse_col(df.iloc[:, 1])

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx]), torch.tensor(self.y[idx])

# ── Model ────────────────────────────────────────────────
class ResBlock(nn.Module):
    def __init__(self, dim, dropout):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
        )
        self.act = nn.ReLU()

    def forward(self, x):
        return self.act(x + self.block(x))

class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_layers, out_dim, dropout):
        super().__init__()
        self.input_proj = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.ReLU())
        self.blocks     = nn.Sequential(*[ResBlock(hidden_dim, dropout) for _ in range(num_layers)])
        self.head       = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        return self.head(self.blocks(self.input_proj(x)))

criterion = nn.BCEWithLogitsLoss()

# ── Auto-detect datasets ─────────────────────────────────
train_files = sorted(glob.glob("*_train.csv"))
results = []

os.makedirs("plots", exist_ok=True)

# ── Loop through all dataset pairs ───────────────────────
for train_file in train_files:
    base = train_file.replace("_train.csv", "")
    test_file = f"{base}_test.csv"

    if not os.path.exists(test_file):
        print(f"Skipping {base}: no test file")
        continue

    print(f"\n🚀 Running dataset: {base}")

    # Load data
    train_data = BinaryCSVDataset(train_file)
    test_data  = BinaryCSVDataset(test_file)

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    test_loader  = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

    in_dim  = train_data.X.shape[1]
    out_dim = train_data.y.shape[1]

    # Reset model each time
    model = MLP(in_dim, HIDDEN_DIM, NUM_LAYERS, out_dim, DROPOUT).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)

    train_losses = []
    test_losses = []
    epochs_axis = []

    best_test_loss = float("inf")

    # ── Training loop ────────────────────────────────────
    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_train_loss = 0.0

        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)

            logits = model(X_b)
            loss   = criterion(logits, y_b)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item() * len(X_b)

        # ── Evaluation ───────────────────────────────────
        if epoch % EVAL_EVERY == 0:
            model.eval()
            total_test_loss = 0.0
            avg_train_loss = running_train_loss / len(train_data)

            with torch.no_grad():
                for X_b, y_b in test_loader:
                    X_b, y_b = X_b.to(device), y_b.to(device)
                    logits = model(X_b)
                    loss   = criterion(logits, y_b)
                    total_test_loss += loss.item() * len(X_b)

            avg_test_loss = total_test_loss / len(test_data)

            train_losses.append(avg_train_loss)
            test_losses.append(avg_test_loss)
            epochs_axis.append(epoch)

            print(f"{base} | Epoch {epoch} | Train: {avg_train_loss:.4f} | Test: {avg_test_loss:.4f}")

            if avg_test_loss < best_test_loss:
                best_test_loss = avg_test_loss

    # ── Save plot ────────────────────────────────────────
    plt.figure(figsize=(10, 6))
    plt.plot(epochs_axis, train_losses, label="Train Loss")
    plt.plot(epochs_axis, test_losses, label="Test Loss")

    plt.title(f"{base} - Training vs Test Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    plot_path = f"plots/{base}_loss.png"
    plt.savefig(plot_path)
    plt.close()

    print(f"📊 Saved: {plot_path}")

    results.append((base, best_test_loss))


# ── Final ranking ────────────────────────────────────────
print("\n🏆 FINAL RESULTS:")
results.sort(key=lambda x: x[1])

for name, loss in results:
    print(f"{name}: {loss:.6f}")
