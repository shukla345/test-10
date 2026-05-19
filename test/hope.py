import glob
import os
import math
import gc

import numpy as np
import pandas as pd

import torch
import torch.nn as nn

from torch.utils.data import (
    Dataset,
    DataLoader
)

import matplotlib.pyplot as plt


# ────────────────────────────────────────────────────────
# Config (A100 + 10k rows)
# ────────────────────────────────────────────────────────
BATCH_SIZE = 256

LR         = 2e-4
EPOCHS     = 200

EMBED_DIM  = 16
NUM_HEADS  = 2
NUM_LAYERS = 2
FF_DIM     = 64

DROPOUT    = 0
WD         = 0

CLIP_NORM  = 1.0

NUM_WORKERS = 2


# ────────────────────────────────────────────────────────
# Token IDs
# ────────────────────────────────────────────────────────
PAD   = 0

TOK_0 = 1
TOK_1 = 2

BOS   = 3
EOS   = 4

VOCAB_SIZE = 5


# ────────────────────────────────────────────────────────
# Device
# ────────────────────────────────────────────────────────
device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print(f"Using device: {device}")

if torch.cuda.is_available():

    print(
        torch.cuda.get_device_name(0)
    )

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    torch.set_float32_matmul_precision('high')


# ────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────
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

            chars.append(
                ID_TO_CHAR[t]
            )

    return ''.join(chars)


# ────────────────────────────────────────────────────────
# Dataset
# ────────────────────────────────────────────────────────
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


# ────────────────────────────────────────────────────────
# Collate
# ────────────────────────────────────────────────────────
def collate_fn(batch):

    Xs, ys = zip(*batch)

    max_x = max(
        len(x)
        for x in Xs
    )

    max_y = (
        max(
            len(y)
            for y in ys
        ) + 2
    )

    X_pad = []
    y_in  = []
    y_out = []

    for x, y in zip(Xs, ys):

        x_pad = (
            x +
            [PAD] * (
                max_x - len(x)
            )
        )

        y_seq = (
            [BOS] +
            y +
            [EOS]
        )

        y_pad = (
            y_seq +
            [PAD] * (
                max_y - len(y_seq)
            )
        )

        y_in.append(
            y_pad[:-1]
        )

        y_out.append(
            y_pad[1:]
        )

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


# ────────────────────────────────────────────────────────
# Dynamic Positional Encoding
# ────────────────────────────────────────────────────────
class PositionalEncoding(nn.Module):

    def __init__(
        self,
        d_model,
        max_len=256
    ):

        super().__init__()

        self.d_model = d_model

        pe = self.build_pe(max_len)

        self.register_buffer(
            "pe",
            pe,
            persistent=False
        )

    def build_pe(
        self,
        max_len
    ):

        pe = torch.zeros(
            max_len,
            self.d_model
        )

        pos = torch.arange(
            0,
            max_len,
            dtype=torch.float
        ).unsqueeze(1)

        div = torch.exp(

            torch.arange(
                0,
                self.d_model,
                2,
                dtype=torch.float
            ) *

            (
                -math.log(10000)
                / self.d_model
            )
        )

        pe[:, 0::2] = torch.sin(
            pos * div
        )

        pe[:, 1::2] = torch.cos(
            pos * div
        )

        return pe.unsqueeze(0)

    def forward(self, x):

        seq_len = x.size(1)

        if seq_len > self.pe.size(1):

            new_pe = self.build_pe(
                seq_len * 2
            ).to(x.device)

            self.register_buffer(
                "pe",
                new_pe,
                persistent=False
            )

        return (
            x +
            self.pe[:, :seq_len]
        )


# ────────────────────────────────────────────────────────
# Transformer Model
# ────────────────────────────────────────────────────────
class Model(nn.Module):

    def __init__(self):

        super().__init__()

        self.embed = nn.Embedding(
            VOCAB_SIZE,
            EMBED_DIM,
            padding_idx=PAD
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

        self.mask_cache = {}

    def get_causal_mask(
        self,
        T,
        device
    ):

        key = (T, device)

        if key not in self.mask_cache:

            self.mask_cache[key] = torch.triu(

                torch.ones(
                    T,
                    T,
                    dtype=torch.bool,
                    device=device
                ),

                diagonal=1
            )

        return self.mask_cache[key]

    def forward(
        self,
        src,
        tgt
    ):

        src_padding_mask = (
            src == PAD
        )

        tgt_padding_mask = (
            tgt == PAD
        )

        causal_mask = self.get_causal_mask(
            tgt.size(1),
            tgt.device
        )

        src = self.embed(src)
        tgt = self.embed(tgt)

        src = self.pos(src)
        tgt = self.pos(tgt)

        out = self.tf(

            src,
            tgt,

            tgt_mask=causal_mask,

            src_key_padding_mask=src_padding_mask,

            tgt_key_padding_mask=tgt_padding_mask,

            memory_key_padding_mask=src_padding_mask
        )

        return self.head(out)


# ────────────────────────────────────────────────────────
# Loss
# ────────────────────────────────────────────────────────
criterion = nn.CrossEntropyLoss(
    ignore_index=PAD
)


# ────────────────────────────────────────────────────────
# True Autoregressive Accuracy
# ────────────────────────────────────────────────────────
@torch.inference_mode()
def evaluate_true_accuracy(
    model,
    loader
):

    model.eval()

    total_correct = 0
    total_tokens  = 0

    for src, _, tgt_out in loader:

        src = src.to(
            device,
            non_blocking=True
        )

        tgt_out = tgt_out.to(
            device,
            non_blocking=True
        )

        B = src.size(0)

        T = tgt_out.size(1)

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

        preds = torch.empty(

            (B, T),

            dtype=torch.long,

            device=device
        )

        for step in range(T):

            logits = model(
                src,
                generated
            )

            next_token = (
                logits[:, -1]
                .argmax(dim=-1)
            )

            next_token = torch.where(
                finished,
                torch.full_like(
                    next_token,
                    EOS
                ),
                next_token
            )

            preds[:, step] = next_token

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

        pred_len = step + 1

        target = tgt_out[:, :pred_len]

        preds = preds[:, :pred_len]

        mask = (
            target != PAD
        )

        correct = (
            (preds == target) & mask
        ).sum().item()

        total = (
            mask.sum().item()
        )

        total_correct += correct
        total_tokens += total

    gc.collect()
    torch.cuda.empty_cache()

    return (
        total_correct /
        max(total_tokens, 1)
    )


# ────────────────────────────────────────────────────────
# Training
# ────────────────────────────────────────────────────────
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

        collate_fn=collate_fn,

        num_workers=NUM_WORKERS,

        pin_memory=True,

        persistent_workers=True
    )

    test_loader = DataLoader(

        test_data,

        batch_size=BATCH_SIZE,

        shuffle=False,

        collate_fn=collate_fn,

        num_workers=NUM_WORKERS,

        pin_memory=True,

        persistent_workers=True
    )

    model = Model().to(device)

    model = torch.compile(model)

    optimizer = torch.optim.AdamW(

        model.parameters(),

        lr=LR,

        weight_decay=WD
    )

    scaler = torch.amp.GradScaler(
        'cuda'
    )

    train_losses = []
    true_accs    = []
    epochs_axis  = []

    for epoch in range(
        1,
        EPOCHS + 1
    ):

        model.train()

        total_train = 0

        for src, tgt_in, tgt_out in train_loader:

            src = src.to(
                device,
                non_blocking=True
            )

            tgt_in = tgt_in.to(
                device,
                non_blocking=True
            )

            tgt_out = tgt_out.to(
                device,
                non_blocking=True
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            with torch.amp.autocast('cuda'):

                logits = model(
                    src,
                    tgt_in
                )

                loss = criterion(

                    logits.reshape(
                        -1,
                        VOCAB_SIZE
                    ),

                    tgt_out.reshape(-1)
                )

            scaler.scale(
                loss
            ).backward()

            scaler.unscale_(
                optimizer
            )

            torch.nn.utils.clip_grad_norm_(

                model.parameters(),

                CLIP_NORM
            )

            scaler.step(optimizer)

            scaler.update()

            total_train += (
                loss.item() * len(src)
            )

        avg_train = (
            total_train /
            len(train_data)
        )

        true_acc = evaluate_true_accuracy(
            model,
            test_loader
        )

        train_losses.append(
            avg_train
        )

        true_accs.append(
            true_acc
        )

        epochs_axis.append(
            epoch
        )

        print(

            f"Epoch {epoch:3d} | "

            f"Train Loss: "
            f"{avg_train:.4f} | "

            f"True Test Acc: "
            f"{true_acc:.4f}"
        )

        gc.collect()
        torch.cuda.empty_cache()

    # ────────────────────────────────────────────────
    # Plot
    # ────────────────────────────────────────────────
    plt.figure(
        figsize=(24, 10)
    )

    plt.plot(
        epochs_axis,
        true_accs,
        linewidth=1.5,
        marker='o',
        markersize=3,
        markevery=10
    )

    plt.xlabel("Epoch")

    plt.ylabel(
        "True Autoregressive "
        "Token Accuracy"
    )

    plt.title(
        f"True Accuracy - "
        f"{os.path.basename(train_file)}"
    )

    plt.grid(True)

    plt.tight_layout()

    plot_path = (

        "/kaggle/working/plots/" +

        os.path.basename(train_file) +

        "_true_accuracy.png"
    )

    plt.savefig(
        plot_path,
        dpi=600
    )

    plt.close()

    print(
        f"💾 Saved: {plot_path}"
    )

    gc.collect()
    torch.cuda.empty_cache()

    return model


# ────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────
def main():

    os.makedirs(
        "/kaggle/working/plots",
        exist_ok=True
    )

    train_files = sorted(
        glob.glob("/kaggle/input/datasets/classstudents/test38/*_train.csv")
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

        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":

    main()
