import glob
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# ── Config ───────────────────────────────────────────────
BATCH_SIZE = 128
LR         = 1e-3
EPOCHS     = 200

FOURIER_DIM = 64
FOURIER_SCALE = 3.0   # 🔥 important

HIDDEN_DIM = 128
ENC_DEPTH  = 3

BOTTLENECK = 64
MLP_DEPTH  = 3
MLP_WIDTH  = 128

DROPOUT    = 0.05
WD         = 1e-5

EVAL_EVERY = 1
NOISE_STD  = 0.01
CLIP_NORM  = 1.0
# ─────────────────────────────────────────────────────────

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Binary parsing ───────────────────────────────────────
def parse_binary(series: pd.Series) -> np.ndarray:
    strings = series.astype(str).str.strip()
    max_len = strings.str.len().max()

    rows = []
    for s in strings:
        s = s.zfill(max_len)
        rows.append([float(c) for c in s])

    return np.array(rows, dtype=np.float32)


# ── Dataset ──────────────────────────────────────────────
class BinaryCSVDataset(Dataset):
    def __init__(self, file):
        df = pd.read_csv(file, header=None, dtype=str)

        self.X = parse_binary(df.iloc[:, 0])
        self.y = parse_binary(df.iloc[:, 1])

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx]), torch.tensor(self.y[idx])


# ── Fourier Features ─────────────────────────────────────
class FourierFeatures(nn.Module):
    def __init__(self, in_dim, fourier_dim, scale=3.0):
        super().__init__()
        self.B = nn.Parameter(torch.randn(in_dim, fourier_dim) * scale)

    def forward(self, x):
        # normalize binary to [-1, 1]
        x = x * 2 - 1
        x_proj = 2 * np.pi * x @ self.B
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


# ── Encoder ──────────────────────────────────────────────
class EncoderBlock(nn.Module):
    def __init__(self, dim, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim * 2)
        self.fc2 = nn.Linear(dim * 2, dim)
        self.gate = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h = self.norm(x)
        h = torch.relu(self.fc1(h))
        h = self.dropout(h)
        h = self.fc2(h)

        g = torch.sigmoid(self.gate(x))
        return x + g * h


class Encoder(nn.Module):
    def __init__(self, in_dim, embed_dim, depth, dropout):
        super().__init__()
        self.input_proj = nn.Linear(in_dim, embed_dim)
        self.blocks = nn.ModuleList([
            EncoderBlock(embed_dim, dropout)
            for _ in range(depth)
        ])

    def forward(self, x):
        x = self.input_proj(x)
        for blk in self.blocks:
            x = blk(x)
        return x


# ── Bottleneck + MLP ─────────────────────────────────────
class MLPHead(nn.Module):
    def __init__(self, in_dim, bottleneck, width, depth, out_dim, dropout):
        super().__init__()

        self.bottleneck = nn.Sequential(
            nn.Linear(in_dim, bottleneck),
            nn.GELU(),
            nn.LayerNorm(bottleneck)
        )

        layers = []
        dim = bottleneck

        for _ in range(depth):
            layers.append(nn.Linear(dim, width))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            dim = width

        layers.append(nn.Linear(dim, out_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        x = self.bottleneck(x)
        return self.mlp(x)


# ── Full Model ───────────────────────────────────────────
class Model(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()

        self.fourier = FourierFeatures(in_dim, FOURIER_DIM, FOURIER_SCALE)
        self.encoder = Encoder(FOURIER_DIM * 2, HIDDEN_DIM, ENC_DEPTH, DROPOUT)
        self.head    = MLPHead(HIDDEN_DIM, BOTTLENECK, MLP_WIDTH, MLP_DEPTH, out_dim, DROPOUT)

    def forward(self, x):
        if self.training:
            x = x + NOISE_STD * torch.randn_like(x)

        x = self.fourier(x)
        x = self.encoder(x)
        return self.head(x)


criterion = nn.BCEWithLogitsLoss()


# ── Auto-detect datasets ─────────────────────────────────
train_files = sorted(glob.glob("*_train.csv"))
results = []

os.makedirs("plots", exist_ok=True)


# ── Loop datasets ────────────────────────────────────────
for train_file in train_files:
    base = train_file.replace("_train.csv", "")
    test_file = f"{base}_test.csv"

    if not os.path.exists(test_file):
        print(f"Skipping {base}: no test file")
        continue

    print(f"\n🚀 Running dataset: {base}")

    train_data = BinaryCSVDataset(train_file)
    test_data  = BinaryCSVDataset(test_file)

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    test_loader  = DataLoader(test_data, batch_size=BATCH_SIZE)

    in_dim  = train_data.X.shape[1]
    out_dim = train_data.y.shape[1]

    model = Model(in_dim, out_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)

    train_losses = []
    test_losses  = []
    epochs_axis  = []

    best_test_loss = float("inf")

    # ── Training ────────────────────────────────────────
    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_train_loss = 0.0

        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)

            logits = model(X_b)
            loss   = criterion(logits, y_b)

            optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)
            optimizer.step()

            running_train_loss += loss.item() * len(X_b)

        # ── Eval ────────────────────────────────────────
        if epoch % EVAL_EVERY == 0:
            model.eval()
            total_test_loss = 0.0

            with torch.no_grad():
                for X_b, y_b in test_loader:
                    X_b, y_b = X_b.to(device), y_b.to(device)
                    logits = model(X_b)
                    loss   = criterion(logits, y_b)
                    total_test_loss += loss.item() * len(X_b)

            avg_train_loss = running_train_loss / len(train_data)
            avg_test_loss  = total_test_loss / len(test_data)

            train_losses.append(avg_train_loss)
            test_losses.append(avg_test_loss)
            epochs_axis.append(epoch)

            print(f"{base} | Epoch {epoch} | Train: {avg_train_loss:.4f} | Test: {avg_test_loss:.4f}")

            if avg_test_loss < best_test_loss:
                best_test_loss = avg_test_loss

    # ── Plot ────────────────────────────────────────────
    plt.figure(figsize=(10, 6))
    plt.plot(epochs_axis, train_losses, label="Train")
    plt.plot(epochs_axis, test_losses, label="Test")

    plt.title(f"{base} Loss")
    plt.xlabel("Epoch")
    plt.ylabel("BCE Loss")
    plt.legend()
    plt.grid(True)

    path = f"plots/{base}_loss.png"
    plt.savefig(path)
    plt.close()

    print(f"📊 Saved: {path}")

    results.append((base, best_test_loss))


# ── Final ranking ────────────────────────────────────────
print("\n🏆 FINAL RESULTS:")
results.sort(key=lambda x: x[1])

for name, loss in results:
    print(f"{name}: {loss:.6f}")
