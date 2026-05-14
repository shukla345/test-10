import glob, os, math
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# ── Config ───────────────────────────────────────────────
BATCH_SIZE   = 64
LR           = 5e-4
EPOCHS       = 80

EMBED_DIM    = 96
NUM_HEADS    = 4
NUM_LAYERS   = 2
MEM_SLOTS    = 16
DROPOUT      = 0.1

PAD, BOS, EOS = 2, 3, 4
VOCAB_SIZE    = 5

EVAL_EVERY = 1
MAX_EVAL_SAMPLES = 50
MAX_LEN = 50

# GPU setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True

# ── Dataset ──────────────────────────────────────────────
def parse_seq(s): return [int(c) for c in s.strip()]

class BinarySeqDataset(Dataset):
    def __init__(self, file):
        df = pd.read_csv(file, header=None, dtype=str)
        self.X = [parse_seq(x) for x in df.iloc[:, 0]]
        self.y = [parse_seq(y) for y in df.iloc[:, 1]]

    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]

def collate_fn(batch):
    Xs, ys = zip(*batch)
    max_x = max(len(x) for x in Xs)
    max_y = max(len(y) for y in ys) + 2

    X_pad, y_in, y_out = [], [], []
    for x, y in zip(Xs, ys):
        x_pad = x + [PAD]*(max_x-len(x))
        y_seq = [BOS] + y + [EOS]
        y_pad = y_seq + [PAD]*(max_y-len(y_seq))
        X_pad.append(x_pad)
        y_in.append(y_pad[:-1])
        y_out.append(y_pad[1:])

    return (
        torch.tensor(X_pad, dtype=torch.long),
        torch.tensor(y_in, dtype=torch.long),
        torch.tensor(y_out, dtype=torch.long)
    )

# ── Positional Encoding ──────────────────────────────────
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1024):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000)/d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

# ── Model ────────────────────────────────────────────────
class Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOCAB_SIZE, EMBED_DIM)
        self.pos   = PositionalEncoding(EMBED_DIM)

        enc_layer = nn.TransformerEncoderLayer(
            EMBED_DIM, NUM_HEADS, batch_first=True, dropout=DROPOUT
        )
        self.encoder = nn.TransformerEncoder(enc_layer, NUM_LAYERS)

        dec_layer = nn.TransformerDecoderLayer(
            EMBED_DIM, NUM_HEADS, batch_first=True, dropout=DROPOUT
        )
        self.decoder = nn.TransformerDecoder(dec_layer, NUM_LAYERS)

        self.memory = nn.Parameter(torch.randn(MEM_SLOTS, EMBED_DIM))
        self.head   = nn.Linear(EMBED_DIM, VOCAB_SIZE)

    def encode(self, src):
        src_mask = (src == PAD)
        src = self.pos(self.embed(src))
        enc = self.encoder(src, src_key_padding_mask=src_mask)
        mem = self.memory.unsqueeze(0).expand(src.size(0), -1, -1)
        return torch.cat([enc, mem], dim=1)

    def decode(self, tgt, memory):
        tgt_mask = torch.triu(
            torch.ones(tgt.size(1), tgt.size(1), device=tgt.device),
            diagonal=1
        ).bool()

        tgt = self.pos(self.embed(tgt))
        out = self.decoder(tgt, memory, tgt_mask=tgt_mask)
        return self.head(out)

    def forward(self, src, tgt):
        memory = self.encode(src)
        return self.decode(tgt, memory)

# ── Loss ─────────────────────────────────────────────────
criterion = nn.CrossEntropyLoss(ignore_index=PAD, label_smoothing=0.1)

# ── AMP (mixed precision) ────────────────────────────────
scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

# ── Decode ───────────────────────────────────────────────
@torch.no_grad()
def fast_decode(model, src):
    memory = model.encode(src)
    B = src.size(0)
    tgt = torch.full((B, 1), BOS, dtype=torch.long, device=src.device)

    for _ in range(MAX_LEN):
        logits = model.decode(tgt, memory)
        next_tok = logits[:, -1].argmax(-1, keepdim=True)
        tgt = torch.cat([tgt, next_tok], dim=1)
        if (next_tok == EOS).all():
            break
    return tgt

# ── Eval ─────────────────────────────────────────────────
def evaluate(model, dataset):
    model.eval()
    n = min(len(dataset), MAX_EVAL_SAMPLES)

    srcs, trues = [], []
    for i in range(n):
        s, t = dataset[i]
        srcs.append(torch.tensor(s, device=device))
        trues.append(t)

    max_x = max(len(s) for s in srcs)
    src_pad = [s.tolist() + [PAD]*(max_x-len(s)) for s in srcs]
    src = torch.tensor(src_pad, device=device)

    preds = fast_decode(model, src).cpu().tolist()

    seq_ok = tok_ok = tok_total = 0
    for p, t in zip(preds, trues):
        p = p[1:]
        if EOS in p: p = p[:p.index(EOS)]
        if p == t: seq_ok += 1
        m = min(len(p), len(t))
        tok_ok += sum(p[i]==t[i] for i in range(m))
        tok_total += len(t)

    return seq_ok/n, tok_ok/tok_total

# ── Training ─────────────────────────────────────────────
train_files = sorted(glob.glob("*_train.csv"))
os.makedirs("plots", exist_ok=True)

for train_file in train_files:
    base = train_file.replace("_train.csv", "")
    test_file = f"{base}_test.csv"
    if not os.path.exists(test_file): continue

    print(f"\n🚀 Training: {base} (GPU Optimized)")

    train_data = BinarySeqDataset(train_file)
    test_data  = BinarySeqDataset(test_file)

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE,
                              shuffle=True, collate_fn=collate_fn,
                              num_workers=2, pin_memory=True)

    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE,
                             collate_fn=collate_fn,
                             num_workers=2, pin_memory=True)

    model = Transformer().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    train_losses, test_losses = [], []
    seq_accs, tok_accs = [], []
    epochs_axis = []

    for epoch in range(1, EPOCHS+1):
        model.train()
        total_train = 0

        for src, tgt_in, tgt_out in train_loader:
            src = src.to(device, non_blocking=True)
            tgt_in = tgt_in.to(device, non_blocking=True)
            tgt_out = tgt_out.to(device, non_blocking=True)

            optimizer.zero_grad()

            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                logits = model(src, tgt_in)
                loss = criterion(logits.view(-1, VOCAB_SIZE),
                                 tgt_out.view(-1))

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_train += loss.item() * len(src)

        train_loss = total_train / len(train_data)

        # test loss
        model.eval()
        total_test = 0
        with torch.no_grad():
            for src, tgt_in, tgt_out in test_loader:
                src = src.to(device, non_blocking=True)
                tgt_in = tgt_in.to(device, non_blocking=True)
                tgt_out = tgt_out.to(device, non_blocking=True)

                logits = model(src, tgt_in)
                loss = criterion(logits.view(-1, VOCAB_SIZE),
                                 tgt_out.view(-1))
                total_test += loss.item() * len(src)

        test_loss = total_test / len(test_data)

        train_losses.append(train_loss)
        test_losses.append(test_loss)
        epochs_axis.append(epoch)

        if epoch % EVAL_EVERY == 0 or epoch == EPOCHS:
            seq_acc, tok_acc = evaluate(model, test_data)
            print(f"Epoch {epoch} | Train {train_loss:.4f} | Test {test_loss:.4f} | SeqAcc {seq_acc:.3f}")
        else:
            seq_acc = seq_accs[-1] if seq_accs else 0
            tok_acc = tok_accs[-1] if tok_accs else 0

        seq_accs.append(seq_acc)
        tok_accs.append(tok_acc)

    # ── Plot ─────────────────────────────────────────────
    plt.figure(figsize=(10, 6))
    plt.plot(epochs_axis, train_losses, label="Train Loss")
    plt.plot(epochs_axis, test_losses, label="Test Loss")
    plt.plot(epochs_axis, seq_accs, label="Sequence Acc", linestyle='--')
    plt.plot(epochs_axis, tok_accs, label="Token Acc", linestyle='--')

    plt.legend()
    plt.title(f"{base} GPU Transformer")
    plt.xlabel("Epoch")
    plt.ylabel("Loss / Accuracy")
    plt.grid(True, alpha=0.3)

    plt.savefig(f"plots/{base}_gpu.png", dpi=150, bbox_inches='tight')
    plt.close()

print("\n✅ Training complete.")
