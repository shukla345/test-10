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
BATCH_SIZE = 128
LR         = 1e-3
EPOCHS     = 1000

EMBED_DIM  = 16
NUM_HEADS  = 2
NUM_LAYERS = 1
FF_DIM     = 32

DROPOUT    = 0.0
WD         = 0.0

CLIP_NORM   = 1.0
WARMUP_STEPS = 200

# ── Token IDs ────────────────────────────────────────────
PAD   = 0

TOK_0 = 1
TOK_1 = 2

BOS   = 3
EOS   = 4

VOCAB_SIZE = 5
# ─────────────────────────────────────────────────────────

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

print(f"Using device: {device}")


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

        df = pd.read_csv(
            file,
            header=None,
            dtype=str
        )

        self.X = [
            parse_seq(x)
            for x in df.iloc[:, 0]
        ]

        self.y = [
            parse_seq(y)
            for y in df.iloc[:, 1]
        ]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):

        return (
            self.X[idx],
            self.y[idx]
        )


def collate_fn(batch):

    Xs, ys = zip(*batch)

    max_x = max(len(x) for x in Xs)

    max_y = (
        max(len(y) for y in ys) + 2
    )

    X_pad = []
    y_in  = []
    y_out = []

    for x, y in zip(Xs, ys):

        # encoder input
        x_pad = (
            x + [PAD] * (max_x - len(x))
        )

        # decoder sequence
        y_seq = [BOS] + y + [EOS]

        y_pad = (
            y_seq +
            [PAD] * (max_y - len(y_seq))
        )

        y_in.append(y_pad[:-1])

        y_out.append(y_pad[1:])

        X_pad.append(x_pad)

    return (
        torch.tensor(
            X_pad,
            dtype=torch.long
        ),

        torch.tensor(
            y_in,
            dtype=torch.long
        ),

        torch.tensor(
            y_out,
            dtype=torch.long
        )
    )


# ── Positional Encoding ──────────────────────────────────
class PositionalEncoding(nn.Module):

    def __init__(
        self,
        d_model,
        max_len=2048
    ):

        super().__init__()

        pe = torch.zeros(
            max_len,
            d_model
        )

        pos = torch.arange(
            0,
            max_len
        ).unsqueeze(1)

        div = torch.exp(
            torch.arange(
                0,
                d_model,
                2
            ) *
            (
                -math.log(10000)
                / d_model
            )
        )

        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)

        self.register_buffer(
            "pe",
            pe.unsqueeze(0)
        )

    def forward(self, x):

        return (
            x +
            self.pe[:, :x.size(1)]
        )


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

    return (
        EMBED_DIM ** -0.5
    ) * min(
        step ** -0.5,
        step * (
            WARMUP_STEPS ** -1.5
        )
    )


# ===================== TRUE AUTOREGRESSIVE ACCURACY =====================
@torch.no_grad()
def evaluate_true_accuracy(
    model,
    loader
):

    model.eval()

    total_correct = 0
    total_tokens  = 0

    for src, _, tgt_out in loader:

        src = src.to(device)

        tgt_out = tgt_out.to(device)

        B = src.size(0)

        T = tgt_out.size(1)

        # start with BOS only
        generated = torch.full(
            (B, 1),
            BOS,
            dtype=torch.long,
            device=device
        )

        finished = torch.zeros(
            B,
            dtype=torch.bool,
            device=device
        )

        all_preds = []

        # autoregressive generation
        for step in range(T):

            logits = model(
                src,
                generated
            )

            next_token = (
                logits[:, -1]
                .argmax(dim=-1)
            )

            all_preds.append(
                next_token
            )

            generated = torch.cat(
                [
                    generated,
                    next_token.unsqueeze(1)
                ],
                dim=1
            )

            finished |= (
                next_token == EOS
            )

            if finished.all():
                break

        preds = torch.stack(
            all_preds,
            dim=1
        )

        pred_len = preds.size(1)

        target = tgt_out[:, :pred_len]

        mask = (target != PAD)

        correct = (
            (preds == target) & mask
        ).sum().item()

        total = mask.sum().item()

        total_correct += correct
        total_tokens += total

    return (
        total_correct / total_tokens
    )


# ===================== TRAINING LOOP =====================
def train_model(
    train_file,
    test_file
):

    print(
        f"\n🚀 Training on "
        f"{train_file}"
    )

    train_data = BinarySeqDataset(
        train_file
    )

    test_data = BinarySeqDataset(
        test_file
    )

    train_loader = DataLoader(
        train_data,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn
    )

    test_loader = DataLoader(
        test_data,
        batch_size=BATCH_SIZE,
        shuffle=False,
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

    true_accs = []

    epochs_axis = []

    # ── Training ────────────────────────────────────────
    for epoch in range(
        1,
        EPOCHS + 1
    ):

        model.train()

        total_train = 0

        for src, tgt_in, tgt_out in train_loader:

            src = src.to(device)

            tgt_in = tgt_in.to(device)

            tgt_out = tgt_out.to(device)

            logits = model(
                src,
                tgt_in
            )

            loss = criterion(
                logits.view(
                    -1,
                    VOCAB_SIZE
                ),
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

            total_train += (
                loss.item() * len(src)
            )

            step += 1

        avg_train = (
            total_train /
            len(train_data)
        )

        # TRUE AUTOREGRESSIVE EVALUATION
        true_acc = evaluate_true_accuracy(
            model,
            test_loader
        )

        train_losses.append(avg_train)

        true_accs.append(true_acc)

        epochs_axis.append(epoch)

        print(
            f"Epoch {epoch:3d} | "
            f"Train Loss: "
            f"{avg_train:.4f} | "
            f"True Test Acc: "
            f"{true_acc:.4f}"
        )

    # ================= TRUE ACCURACY PLOT =================
    plt.figure(figsize=(24, 10))

    plt.plot(
        epochs_axis,
        true_accs,
        marker='o'
    )

    plt.xlabel('Epoch')

    plt.ylabel(
        'True Autoregressive '
        'Token Accuracy'
    )

    plt.title(
        f'True Inference Accuracy – '
        f'{os.path.basename(train_file)}'
    )

    plt.grid(True)

    plt.tight_layout()

    plot_path = (
        f'plots/'
        f'{os.path.basename(train_file)}'
        f'_true_accuracy.png'
    )

    plt.savefig(
        plot_path,
        dpi=600
    )

    plt.close()

    print(f"💾 Saved: {plot_path}")

    return model


# ===================== MAIN =====================
def main():

    os.makedirs(
        "plots",
        exist_ok=True
    )

    train_files = sorted(
        glob.glob("*_train.csv")
    )

    if not train_files:

        print(
            "❌ No *_train.csv found!"
        )

        return

    for train_file in train_files:

        base = train_file.replace(
            "_train.csv",
            ""
        )

        test_file = (
            f"{base}_test.csv"
        )

        if not os.path.exists(
            test_file
        ):

            print(
                f"⚠️ Skipping {base} "
                f"(no test file)"
            )

            continue

        train_model(
            train_file,
            test_file
        )

        print("=" * 60)


if __name__ == "__main__":
    main()
