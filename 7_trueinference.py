import glob
import os
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

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
EVAL_EVERY = 1  # Evaluate every N epochs to save time
PAD = 2
BOS = 3
EOS = 4
VOCAB_SIZE = 5
MAX_GEN_LEN = 50  # Max tokens to generate during inference
MAX_EVAL_SAMPLES = 200

# ── Hardware Optimization ────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True

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
    max_y = max(len(y) for y in ys) + 2  # +BOS +EOS

    X_pad, y_in, y_out = [], [], []
    for x, y in zip(Xs, ys):
        x_pad = x + [PAD] * (max_x - len(x))
        y_seq = [BOS] + y + [EOS]
        y_pad = y_seq + [PAD] * (max_y - len(y_seq))
        y_in.append(y_pad[:-1])
        y_out.append(y_pad[1:])
        X_pad.append(x_pad)

    return (
        torch.tensor(X_pad, dtype=torch.long),
        torch.tensor(y_in, dtype=torch.long),
        torch.tensor(y_out, dtype=torch.long)
    )

# ── Positional Encoding ──────────────────────────────────
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=2048):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
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
        
        # Precompute causal mask dynamically (fast for small seqs)
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

# ── Loss & LR Scheduler ────────────────────────────────
criterion = nn.CrossEntropyLoss(ignore_index=PAD)

def get_lr(step):
    return (EMBED_DIM ** -0.5) * min(step ** -0.5, step * (WARMUP_STEPS ** -1.5))

# ── Batched Greedy Decode (FAST) ─────────────────────────
def batched_greedy_decode(model, src_batch, max_len=MAX_GEN_LEN):
    model.eval()
    src_batch = src_batch.to(device)
    batch_size = src_batch.size(0)
    tgt = torch.full((batch_size, 1), BOS, dtype=torch.long, device=device)
    
    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
    
    for _ in range(max_len):
        with torch.no_grad():
            logits = model(src_batch, tgt)  # [B, L, V]
            next_token = logits[:, -1].argmax(-1).unsqueeze(1)  # [B, 1]
            tgt = torch.cat([tgt, next_token], dim=1)
            finished |= (next_token.squeeze(1) == EOS)
            if finished.all():
                break
    return tgt.tolist()

# ── True Inference Evaluation (Optimized) ────────────────
def evaluate_true_inference(model, dataset, max_samples=MAX_EVAL_SAMPLES):
    model.eval()
    samples = min(len(dataset), max_samples)
    
    # Collect & pad in one go
    src_list, true_list = [], []
    for i in range(samples):
        src, true = dataset[i]
        src_list.append(torch.tensor(src, dtype=torch.long))
        true_list.append(true)
        
    max_x = max(len(s) for s in src_list)
    src_pad = [s.tolist() + [PAD]*(max_x - len(s)) for s in src_list]
    src_tensor = torch.tensor(src_pad, dtype=torch.long, device=device)
    
    # Batched generation
    pred_seqs = batched_greedy_decode(model, src_tensor)
    
    seq_correct = 0
    token_correct = 0
    token_total = 0
    
    for i, true in enumerate(true_list):
        pred = pred_seqs[i][1:]  # remove BOS
        if EOS in pred:
            pred = pred[:pred.index(EOS)]
            
        if pred == true:
            seq_correct += 1
            
        min_len = min(len(pred), len(true))
        token_correct += sum(p == t for p, t in zip(pred[:min_len], true[:min_len]))
        token_total += len(true)
        
    return seq_correct / samples, token_correct / token_total

# ── Training Loop ────────────────────────────────────────
train_files = sorted(glob.glob("*_train.csv"))
os.makedirs("plots", exist_ok=True)

for train_file in train_files:
    base = train_file.replace("_train.csv", "")
    test_file = f"{base}_test.csv"
    if not os.path.exists(test_file):
        continue

    print(f"\n🚀 Training: {base} | Device: {device}")
    
    train_data = BinarySeqDataset(train_file)
    test_data  = BinarySeqDataset(test_file)
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=0)

    model = Model().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    
    step = 1
    train_losses, seq_accs, tok_accs, epochs_axis = [], [], [], []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_train = 0
        
        for src, tgt_in, tgt_out in train_loader:
            src, tgt_in, tgt_out = src.to(device), tgt_in.to(device), tgt_out.to(device)
            
            logits = model(src, tgt_in)
            loss = criterion(logits.view(-1, VOCAB_SIZE), tgt_out.view(-1))
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)
            
            # LR warmup & decay
            for g in optimizer.param_groups:
                g['lr'] = get_lr(step)
            optimizer.step()
            
            total_train += loss.item() * len(src)
            step += 1
            
        avg_train = total_train / len(train_data)
        train_losses.append(avg_train)
        epochs_axis.append(epoch)
        
        # ── Periodic Evaluation ──────────────────────────
        if epoch % EVAL_EVERY == 0 or epoch == EPOCHS:
            seq_acc, tok_acc = evaluate_true_inference(model, test_data)
            seq_accs.append(seq_acc)
            tok_accs.append(tok_acc)
            print(f"📊 {base} | Epoch {epoch:3d} | Train Loss: {avg_train:.4f} | SeqAcc: {seq_acc:.3f} | TokAcc: {tok_acc:.3f}")
        else:
            # Interpolate accuracy for plotting (keep arrays aligned)
            if seq_accs:
                seq_accs.append(seq_accs[-1])
                tok_accs.append(tok_accs[-1])
            else:
                seq_accs.append(0.0)
                tok_accs.append(0.0)

    # ── Plot TRUE performance ────────────────────────────
    plt.figure(figsize=(10, 6))
    plt.plot(epochs_axis, seq_accs, label="Sequence Accuracy", marker='o', markersize=4)
    plt.plot(epochs_axis, tok_accs, label="Token Accuracy", marker='s', markersize=4)
    plt.title(f"{base} Transformer TRUE Inference")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    path = f"plots/{base}_true_inference.png"
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"💾 Saved plot: {path}")
