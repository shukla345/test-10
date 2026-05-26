import glob
import os
import math
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# ── Config ───────────────────────────────────────────────
BATCH_SIZE = 64
LR         = 3e-4
EPOCHS     = 100
EMBED_DIM  = 128
NUM_HEADS  = 4
NUM_LAYERS = 3
MEM_SLOTS  = 32
DROPOUT    = 0.1
WD         = 1e-5
CLIP_NORM  = 1.0

# Your token configuration
PAD = 2
BOS = 3
EOS = 4
VOCAB_SIZE = 5

EVAL_EVERY       = 1  # True inference every N epochs
MAX_EVAL_SAMPLES = 50

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
        self.X = [parse_seq(x) for x in df.iloc[:, 1]]
        self.y = [parse_seq(y) for y in df.iloc[:, 0]]

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

# ── Memory Module ────────────────────────────────────────
class Memory(nn.Module):
    def __init__(self):
        super().__init__()
        self.memory = nn.Parameter(torch.randn(MEM_SLOTS, EMBED_DIM))

    def forward(self, batch_size):
        return self.memory.unsqueeze(0).repeat(batch_size, 1, 1)

# ── Model ────────────────────────────────────────────────
class MemoryTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOCAB_SIZE, EMBED_DIM)
        self.pos   = PositionalEncoding(EMBED_DIM)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=EMBED_DIM, nhead=NUM_HEADS, batch_first=True, dropout=DROPOUT
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, NUM_LAYERS)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=EMBED_DIM, nhead=NUM_HEADS, batch_first=True, dropout=DROPOUT
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, NUM_LAYERS)

        self.memory = Memory()
        self.head = nn.Linear(EMBED_DIM, VOCAB_SIZE)

    def forward(self, src, tgt):
        B = src.size(0)
        src_mask = (src == PAD)
        tgt_mask_pad = (tgt == PAD)
        
        # Fast causal mask (replaces deprecated generate_square_subsequent_mask)
        causal_mask = torch.triu(
            torch.ones(tgt.size(1), tgt.size(1), dtype=torch.bool, device=tgt.device),
            diagonal=1
        )

        src = self.pos(self.embed(src))
        tgt = self.pos(self.embed(tgt))

        enc_out = self.encoder(src, src_key_padding_mask=src_mask)
        mem = self.memory(B)
        combined = torch.cat([enc_out, mem], dim=1)

        out = self.decoder(
            tgt, combined,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=tgt_mask_pad
        )
        return self.head(out)

# ── Loss ─────────────────────────────────────────────────
criterion = nn.CrossEntropyLoss(ignore_index=PAD)

# ── Batched Greedy Decode (TRUE Inference) ───────────────
@torch.no_grad()
def batched_greedy_decode(model, src_batch, max_len=100):
    model.eval()
    src_batch = src_batch.to(device)
    batch_size = src_batch.size(0)
    tgt = torch.full((batch_size, 1), BOS, dtype=torch.long, device=device)
    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
    
    for _ in range(max_len):
        logits = model(src_batch, tgt)
        next_token = logits[:, -1].argmax(-1).unsqueeze(1)
        tgt = torch.cat([tgt, next_token], dim=1)
        finished |= (next_token.squeeze(1) == EOS)
        if finished.all():
            break
    return tgt.tolist()

# ── True Inference Evaluation ────────────────────────────
def evaluate_true_inference(model, dataset, max_samples=MAX_EVAL_SAMPLES):
    model.eval()
    samples = min(len(dataset), max_samples)
    seq_correct, token_correct, token_total = 0, 0, 0

    # Collect & pad sources
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

    print(f"\n🚀 Training: {base} (Memory Transformer | True Inference)")
    
    train_data = BinarySeqDataset(train_file)
    test_data  = BinarySeqDataset(test_file)
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=0)

    model = MemoryTransformer().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)

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
            optimizer.step()
            
            total_train += loss.item() * len(src)

        avg_train = total_train / len(train_data)
        train_losses.append(avg_train)
        epochs_axis.append(epoch)

        # ── Periodic TRUE Inference ──────────────────────
        if epoch % EVAL_EVERY == 0 or epoch == EPOCHS:
            seq_acc, tok_acc = evaluate_true_inference(model, test_data)
            seq_accs.append(seq_acc)
            tok_accs.append(tok_acc)
            print(f"📊 {base} | Epoch {epoch:3d} | Train: {avg_train:.4f} | SeqAcc: {seq_acc:.3f} | TokAcc: {tok_acc:.3f}")
        else:
            # Carry forward last metrics for smooth plotting
            seq_accs.append(seq_accs[-1] if seq_accs else 0.0)
            tok_accs.append(tok_accs[-1] if tok_accs else 0.0)

    # ── Plot ────────────────────────────────────────────
    plt.figure(figsize=(10, 6))
    plt.plot(epochs_axis, train_losses, label="Train Loss (Teacher Forced)", linestyle='-')
    plt.plot(epochs_axis, seq_accs, label="Sequence Accuracy (True)", marker='o')
    plt.plot(epochs_axis, tok_accs, label="Token Accuracy (True)", marker='s')
    plt.title(f"{base} Memory Transformer | TRUE Inference")
    plt.xlabel("Epoch")
    plt.ylabel("Loss / Accuracy")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    path = f"plots/{base}_memory_true_inference.png"
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"💾 Saved plot: {path}")

print("\n✅ All datasets completed.")
