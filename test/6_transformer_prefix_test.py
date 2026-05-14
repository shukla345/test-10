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

CLIP_NORM   = 1.0
WARMUP_STEPS = 1000

# ── Token IDs ────────────────────────────────────────────
PAD   = 0

TOK_0 = 1
TOK_1 = 2

BOS   = 3
EOS   = 4

VOCAB_SIZE = 5
# ─────────────────────────────────────────────────────────

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Dataset ──────────────────────────────────────────────
def parse_seq(s):

    out = []

    for c in s.strip():

        if c == '0':
            out.append(TOK_0)

        elif c == '1':
            out.append(TOK_1)

    return out


ID_TO_CHAR = {
    TOK_0: '0',
    TOK_1: '1'
}


def decode_seq(seq):

    chars = []

    for t in seq:

        if t in ID_TO_CHAR:
            chars.append(ID_TO_CHAR[t])

    return ''.join(chars)


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

    X_pad = []
    y_in  = []
    y_out = []

    for x, y in zip(Xs, ys):

        # Encoder input
        x_pad = x + [PAD] * (max_x - len(x))

        # Decoder sequence
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

        div = torch.exp(
            torch.arange(0, d_model, 2) *
            (-math.log(10000) / d_model)
        )

        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)

        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):

        return x + self.pe[:, :x.size(1)]


# ── Transformer Model ────────────────────────────────────
class Model(nn.Module):

    def __init__(self):

        super().__init__()

        self.embed = nn.Embedding(
            VOCAB_SIZE,
            EMBED_DIM
        )

        self.pos = PositionalEncoding(
            EMBED_DIM
        )

        self.tf = nn.Transformer(
            d_model=EMBED_DIM,
            nhead=NUM_HEADS,
            num_encoder_layers=NUM_LAYERS,
            num_decoder_layers=NUM_LAYERS,
            dim_feedforward=FF_DIM,
            dropout=DROPOUT,
            batch_first=True
        )

        self.head = nn.Linear(
            EMBED_DIM,
            VOCAB_SIZE
        )

    def forward(self, src, tgt):

        src_mask = (src == PAD)

        tgt_mask_pad = (tgt == PAD)

        causal_mask = torch.triu(
            torch.ones(
                tgt.size(1),
                tgt.size(1),
                dtype=torch.bool,
                device=tgt.device
            ),
            diagonal=1
        )

        src = self.pos(
            self.embed(src)
        )

        tgt = self.pos(
            self.embed(tgt)
        )

        out = self.tf(
            src,
            tgt,
            tgt_mask=causal_mask,
            src_key_padding_mask=src_mask,
            tgt_key_padding_mask=tgt_mask_pad
        )

        return self.head(out)


# ── Loss ─────────────────────────────────────────────────
criterion = nn.CrossEntropyLoss(
    ignore_index=PAD
)


# ── LR Scheduler ─────────────────────────────────────────
def get_lr(step):

    return (EMBED_DIM ** -0.5) * min(
        step ** -0.5,
        step * (WARMUP_STEPS ** -1.5)
    )


# ===================== TOKEN ACCURACY =====================
@torch.no_grad()
def evaluate_accuracy(model, loader):

    model.eval()

    total_correct = 0
    total_tokens  = 0

    for src, tgt_in, tgt_out in loader:

        src = src.to(device)
        tgt_in = tgt_in.to(device)
        tgt_out = tgt_out.to(device)

        logits = model(src, tgt_in)

        preds = logits.argmax(dim=-1)

        mask = (tgt_out != PAD)

        correct = (
            (preds == tgt_out) & mask
        ).sum().item()

        total = mask.sum().item()

        total_correct += correct
        total_tokens += total

    return total_correct / total_tokens


# ===================== PREFIX ACCURACY =====================
@torch.no_grad()
def evaluate_prefix_accuracy(
    model,
    test_loader,
    max_prefix_len=None
):

    model.eval()

    all_accs = {}

    for src, tgt_in, tgt_out in test_loader:

        src = src.to(device)
        tgt_in = tgt_in.to(device)
        tgt_out = tgt_out.to(device)

        B, T = tgt_out.shape

        non_pad_mask = (tgt_out != PAD)

        sample_lens = (
            non_pad_mask.sum(dim=1)
            .cpu()
            .numpy()
        )

        max_k = (
            max_prefix_len
            if max_prefix_len is not None
            else max(sample_lens)
        )

        max_k = min(max_k, T - 1)

        for k in range(0, max_k + 1):

            dec_input = tgt_in.clone()

            if k + 1 < T:
                dec_input[:, k + 1:] = PAD

            logits = model(src, dec_input)

            preds = logits.argmax(dim=-1)

            pos_mask = (
                torch.arange(T, device=device)
                .unsqueeze(0) > k
            )

            eval_mask = (
                non_pad_mask & pos_mask
            )

            total = eval_mask.sum().item()

            if total == 0:
                continue

            correct = (
                (preds == tgt_out) & eval_mask
            ).sum().item()

            acc = correct / total

            if k not in all_accs:
                all_accs[k] = []

            all_accs[k].append(acc)

    prefix_lengths = sorted(all_accs.keys())

    avg_accs = [
        np.mean(all_accs[k])
        for k in prefix_lengths
    ]

    return prefix_lengths, avg_accs


# ===================== TRAINING LOOP =====================
def train_model(train_file, test_file):

    print(f"\n🚀 Training on {train_file}")

    train_data = BinarySeqDataset(train_file)
    test_data  = BinarySeqDataset(test_file)

    train_loader = DataLoader(
        train_data,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn
    )

    test_loader = DataLoader(
        test_data,
        batch_size=BATCH_SIZE,
        collate_fn=collate_fn
    )

    model = Model().to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WD
    )

    step = 1

    train_losses = []
    test_losses  = []

    train_accs = []
    test_accs  = []

    epochs_axis = []

    # ── Training ────────────────────────────────────────
    for epoch in range(1, EPOCHS + 1):

        model.train()

        total_train = 0

        for src, tgt_in, tgt_out in train_loader:

            src = src.to(device)
            tgt_in = tgt_in.to(device)
            tgt_out = tgt_out.to(device)

            logits = model(src, tgt_in)

            loss = criterion(
                logits.view(-1, VOCAB_SIZE),
                tgt_out.view(-1)
            )

            optimizer.zero_grad()

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                CLIP_NORM
            )

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

                src = src.to(device)
                tgt_in = tgt_in.to(device)
                tgt_out = tgt_out.to(device)

                logits = model(src, tgt_in)

                loss = criterion(
                    logits.view(-1, VOCAB_SIZE),
                    tgt_out.view(-1)
                )

                total_test += loss.item() * len(src)

        avg_train = total_train / len(train_data)
        avg_test  = total_test / len(test_data)

        train_acc = evaluate_accuracy(
            model,
            train_loader
        )

        test_acc = evaluate_accuracy(
            model,
            test_loader
        )

        train_losses.append(avg_train)
        test_losses.append(avg_test)

        train_accs.append(train_acc)
        test_accs.append(test_acc)

        epochs_axis.append(epoch)

        print(
            f"Epoch {epoch:3d} | "
            f"Train Loss: {avg_train:.4f} | "
            f"Test Loss: {avg_test:.4f} | "
            f"Train Acc: {train_acc:.4f} | "
            f"Test Acc: {test_acc:.4f}"
        )

    # ================= LOSS PLOT =================
    plt.figure(figsize=(10, 5))

    plt.plot(
        epochs_axis,
        train_losses,
        label='Train Loss'
    )

    plt.plot(
        epochs_axis,
        test_losses,
        label='Test Loss'
    )

    plt.xlabel('Epoch')
    plt.ylabel('Cross-Entropy Loss')

    plt.title(
        f'Loss Curves – {os.path.basename(train_file)}'
    )

    plt.legend()
    plt.grid(True)

    plt.savefig(
        f'plots/{os.path.basename(train_file)}_loss_curve.png'
    )

    plt.close()

    # ================= ACCURACY PLOT =================
    plt.figure(figsize=(10, 5))

    plt.plot(
        epochs_axis,
        train_accs,
        label='Train Accuracy'
    )

    plt.plot(
        epochs_axis,
        test_accs,
        label='Test Accuracy'
    )

    plt.xlabel('Epoch')
    plt.ylabel('Token Accuracy')

    plt.title(
        f'Accuracy Curves – {os.path.basename(train_file)}'
    )

    plt.legend()
    plt.grid(True)

    plt.savefig(
        f'plots/{os.path.basename(train_file)}_accuracy_curve.png'
    )

    plt.close()

    return model, test_loader


# ===================== MAIN =====================
def main():

    os.makedirs("plots", exist_ok=True)

    train_files = sorted(
        glob.glob("*_train.csv")
    )

    if not train_files:

        print("❌ No *_train.csv found!")

        return

    for train_file in train_files:

        base = train_file.replace(
            "_train.csv",
            ""
        )

        test_file = f"{base}_test.csv"

        if not os.path.exists(test_file):

            print(
                f"⚠️ Skipping {base} "
                f"(no test file)"
            )

            continue

        # ── Train ───────────────────────────────────────
        model, test_loader = train_model(
            train_file,
            test_file
        )

        # ── Prefix Accuracy Evaluation ──────────────────
        print("\n📊 Evaluating prefix accuracy...")

        prefix_lens, avg_accs = (
            evaluate_prefix_accuracy(
                model,
                test_loader,
                max_prefix_len=30
            )
        )

        # ================= PREFIX ACCURACY PLOT =================
        plt.figure(figsize=(8, 5))

        plt.plot(
            prefix_lens,
            avg_accs,
            marker='o',
            linestyle='-'
        )

        plt.xlabel(
            'Number of correct prefix tokens given (k)'
        )

        plt.ylabel(
            'Continuation Token Accuracy'
        )

        plt.title(
            f'{base} – Accuracy vs Prefix Length'
        )

        plt.grid(True, alpha=0.3)

        plt.xticks(prefix_lens)

        plt.tight_layout()

        plot_path = (
            f'plots/{base}_prefix_accuracy.png'
        )

        plt.savefig(plot_path, dpi=150)

        plt.close()

        print(f"💾 Saved: {plot_path}")

        # ── Numeric Output ──────────────────────────────
        print("\nPrefix Length vs Accuracy:")

        for k, acc in zip(
            prefix_lens,
            avg_accs
        ):

            print(
                f"  k = {k:2d} : "
                f"{acc:.4f}"
            )

        print("=" * 60)


if __name__ == "__main__":
    main()
