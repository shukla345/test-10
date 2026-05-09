import glob
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import math

# ── Config ───────────────────────────────────────────────
BATCH_SIZE = 256
LR         = 5e-4
EPOCHS     = 80

EMBED_DIM  = 32
NUM_HEADS  = 2
NUM_LAYERS = 2
FF_DIM     = 64

DROPOUT    = 0.1
WD         = 1e-5

CLIP_NORM  = 1.0
WARMUP_STEPS = 1000

PAD = 0
BOS = 2
EOS = 3
VOCAB_SIZE = 4
# ─────────────────────────────────────────────────────────

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Dataset ──────────────────────────────────────────────
def parse_seq(s):
    return [int(c) for c in s.strip()]

class BinarySeqDataset(Dataset):
    def __init__(self, file):
        df = pd.read_csv(file, header=None, dtype=str)
        self.X = [parse_seq(x) for x in df.iloc[:, 0]]
        self.y = [parse_seq(y) for y in df.iloc[:, 1]]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def collate_fn(batch):
    Xs, ys = zip(*batch)

    max_x = max(len(x) for x in Xs)
    max_y = max(len(y) for y in ys) + 2

    X_pad, y_in, y_out = [], [], []

    for x, y in zip(Xs, ys):
        x_pad = x + [PAD]*(max_x - len(x))

        y_seq = [BOS] + y + [EOS]
        y_pad = y_seq + [PAD]*(max_y - len(y_seq))

        y_in.append(y_pad[:-1])
        y_out.append(y_pad[1:])
        X_pad.append(x_pad)

    return torch.tensor(X_pad), torch.tensor(y_in), torch.tensor(y_out)


# ── Positional Encoding ──────────────────────────────────
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=2048):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1)

        div = torch.exp(torch.arange(0, d_model, 2)*(-math.log(10000)/d_model))
        pe[:, 0::2] = torch.sin(pos*div)
        pe[:, 1::2] = torch.cos(pos*div)

        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


# ── Model ────────────────────────────────────────────────
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOCAB_SIZE, EMBED_DIM)
        self.pos   = PositionalEncoding(EMBED_DIM)

        self.tf = nn.Transformer(
            d_model=EMBED_DIM,
            nhead=NUM_HEADS,
            num_encoder_layers=NUM_LAYERS,
            num_decoder_layers=NUM_LAYERS,
            dim_feedforward=FF_DIM,
            dropout=DROPOUT,
            batch_first=True
        )

        self.head = nn.Linear(EMBED_DIM, VOCAB_SIZE)

    def forward(self, src, tgt):
        src_mask = (src == PAD)
        tgt_mask_pad = (tgt == PAD)

        causal_mask = torch.triu(
            torch.ones(tgt.size(1), tgt.size(1), dtype=torch.bool, device=tgt.device),
            diagonal=1
        )

        src = self.pos(self.embed(src))
        tgt = self.pos(self.embed(tgt))

        out = self.tf(
            src, tgt,
            tgt_mask=causal_mask,
            src_key_padding_mask=src_mask,
            tgt_key_padding_mask=tgt_mask_pad
        )

        return self.head(out)


# ── Loss ─────────────────────────────────────────────────
criterion = nn.CrossEntropyLoss(ignore_index=PAD)


# ── LR Scheduler ─────────────────────────────────────────
def get_lr(step):
    return (EMBED_DIM ** -0.5) * min(
        step ** -0.5,
        step * (WARMUP_STEPS ** -1.5)
    )


# ── Training Loop ────────────────────────────────────────
train_files = sorted(glob.glob("*_train.csv"))
results = []

os.makedirs("plots", exist_ok=True)

for train_file in train_files:
    base = train_file.replace("_train.csv", "")
    test_file = f"{base}_test.csv"

    if not os.path.exists(test_file):
        continue

    print(f"\n🚀 {base}")

    train_data = BinarySeqDataset(train_file)
    test_data  = BinarySeqDataset(test_file)

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    test_loader  = DataLoader(test_data, batch_size=BATCH_SIZE, collate_fn=collate_fn)

    model = Model().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)

    step = 1

    train_losses = []
    test_losses  = []
    epochs_axis  = []

    # ── Training ────────────────────────────────────────
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_train = 0

        for src, tgt_in, tgt_out in train_loader:
            src, tgt_in, tgt_out = src.to(device), tgt_in.to(device), tgt_out.to(device)

            logits = model(src, tgt_in)

            loss = criterion(
                logits.view(-1, VOCAB_SIZE),
                tgt_out.view(-1)
            )

            optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)

            for g in optimizer.param_groups:
                g['lr'] = get_lr(step)

            optimizer.step()

            total_train += loss.item() * len(src)
            step += 1

        # ── Evaluation ──────────────────────────────────
        model.eval()
        total_test = 0

        with torch.no_grad():
            for src, tgt_in, tgt_out in test_loader:
                src, tgt_in, tgt_out = src.to(device), tgt_in.to(device), tgt_out.to(device)

                logits = model(src, tgt_in)

                loss = criterion(
                    logits.view(-1, VOCAB_SIZE),
                    tgt_out.view(-1)
                )

                total_test += loss.item() * len(src)

        avg_train = total_train / len(train_data)
        avg_test  = total_test / len(test_data)

        train_losses.append(avg_train)
        test_losses.append(avg_test)
        epochs_axis.append(epoch)

        print(f"{base} | Epoch {epoch} | Train: {avg_train:.4f} | Test: {avg_test:.4f}")

    # ── Plot ────────────────────────────────────────────
    plt.figure(figsize=(10, 6))
    plt.plot(epochs_axis, train_losses, label="Train")
    plt.plot(epochs_axis, test_losses, label="Test")

    plt.title(f"{base} Transformer Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    path = f"plots/{base}_transformer_loss.png"
    plt.savefig(path)
    plt.close()

    print(f"📊 Saved: {path}")

    results.append((base, avg_test))


# ── Final Results ────────────────────────────────────────
print("\n🏆 FINAL RESULTS:")
results.sort(key=lambda x: x[1])

for name, loss in results:
    print(f"{name}: {loss:.6f}")
