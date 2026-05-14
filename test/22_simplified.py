#!/usr/bin/env python3

import os
import glob
import math
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader

# =========================================================
# HARDWARE
# =========================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {device}")

if torch.cuda.is_available():

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    # Faster transformer kernels
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_math_sdp(True)

# =========================================================
# OUTPUTS
# =========================================================

OUT_DIR = "/content/"
CKPT_DIR = os.path.join(OUT_DIR, "checkpoints")
PLOT_DIR = os.path.join(OUT_DIR, "plots")

os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

# =========================================================
# TOKENS
# =========================================================

PAD = 2
BOS = 3
EOS = 4
VOCAB = 5

# =========================================================
# GLOBAL SETTINGS
# =========================================================

SEEDS = [42, 123, 999]

PREFIX_BITS = [0, 4, 8, 16, 32]

COMMON = {
    "BATCH": 64,
    "EPOCHS": 200,
    "LR": 3e-4,
    "WD": 0,
    "CLIP": 1.0,
    "DROP": 0,
}

# =========================================================
# UTILS
# =========================================================

def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_binary_vector(series):

    strings = series.astype(str).str.strip()

    max_len = strings.str.len().max()

    return np.array([
        [float(b) for b in s.zfill(max_len)]
        for s in strings
    ], dtype=np.float32)


def parse_seq(s):

    return [int(c) for c in s.strip()]

# =========================================================
# DATASETS
# =========================================================

class BinaryVectorDataset(Dataset):

    def __init__(self, path):

        df = pd.read_csv(path, header=None, dtype=str)

        self.y = parse_binary_vector(df.iloc[:, 0])
        self.X = parse_binary_vector(df.iloc[:, 1])

    def __len__(self):

        return len(self.X)

    def __getitem__(self, idx):

        return (
            torch.tensor(self.X[idx]),
            torch.tensor(self.y[idx])
        )


class BinarySeqDataset(Dataset):

    def __init__(self, path):

        df = pd.read_csv(path, header=None, dtype=str)

        self.X = [parse_seq(x) for x in df.iloc[:, 0]]
        self.y = [parse_seq(y) for y in df.iloc[:, 1]]

    def __len__(self):

        return len(self.X)

    def __getitem__(self, idx):

        return self.X[idx], self.y[idx]

# =========================================================
# COLLATE
# =========================================================

def collate_fn(batch):

    Xs, ys = zip(*batch)

    max_x = max(len(x) for x in Xs)
    max_y = max(len(y) for y in ys) + 2

    X_pad = []
    y_in = []
    y_out = []

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

# =========================================================
# RESNET BASELINE
# =========================================================

class ResBlock(nn.Module):

    def __init__(self, dim, drop):

        super().__init__()

        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(dim, dim)
        )

        self.act = nn.ReLU()

    def forward(self, x):

        return self.act(x + self.block(x))


class ResNetMLP(nn.Module):

    def __init__(self,
                 in_dim,
                 hidden_dim,
                 layers,
                 out_dim,
                 drop):

        super().__init__()

        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU()
        )

        self.blocks = nn.Sequential(*[
            ResBlock(hidden_dim, drop)
            for _ in range(layers)
        ])

        self.head = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):

        return self.head(
            self.blocks(
                self.input_proj(x)
            )
        )

# =========================================================
# TRANSFORMER
# =========================================================

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=2048):

        super().__init__()

        pe = torch.zeros(max_len, d_model)

        pos = torch.arange(0, max_len).unsqueeze(1)

        div = torch.exp(
            torch.arange(0, d_model, 2)
            * (-math.log(10000) / d_model)
        )

        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)

        self.register_buffer(
            "pe",
            pe.unsqueeze(0)
        )

    def forward(self, x):

        return x + self.pe[:, :x.size(1)]


class TransformerModel(nn.Module):

    def __init__(self):

        super().__init__()

        d_model = 64
        nhead = 2
        layers = 2
        ff = 128

        self.embed = nn.Embedding(VOCAB, d_model)

        self.pos = PositionalEncoding(d_model)

        self.tf = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=layers,
            num_decoder_layers=layers,
            dim_feedforward=ff,
            dropout=0.1,
            batch_first=True
        )

        self.head = nn.Linear(d_model, VOCAB)

    def forward(self, src, tgt):

        src_mask = (src == PAD)
        tgt_pad = (tgt == PAD)

        causal = torch.triu(
            torch.ones(
                tgt.size(1),
                tgt.size(1),
                device=tgt.device,
                dtype=torch.bool
            ),
            diagonal=1
        )

        out = self.tf(
            self.pos(self.embed(src)),
            self.pos(self.embed(tgt)),
            tgt_mask=causal,
            src_key_padding_mask=src_mask,
            tgt_key_padding_mask=tgt_pad
        )

        return self.head(out)

# =========================================================
# CHECKPOINTS
# =========================================================

def save_checkpoint(path,
                    model,
                    optimizer,
                    scaler,
                    epoch,
                    metrics):

    model_state = (
        model.module.state_dict()
        if hasattr(model, "module")
        else model.state_dict()
    )

    tmp_path = path + ".tmp"

    torch.save({
        "model": model_state,
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "epoch": epoch,
        "metrics": metrics,
    }, tmp_path)

    os.replace(tmp_path, path)


def load_checkpoint(path,
                    model,
                    optimizer,
                    scaler):

    if not os.path.exists(path):
        return None

    try:

        print(f"Loading checkpoint: {path}")

        ckpt = torch.load(
            path,
            map_location=device
        )

        model.load_state_dict(
            ckpt["model"]
        )

        try:
            optimizer.load_state_dict(
                ckpt["optimizer"]
            )
        except Exception as e:
            print(f"Optimizer load skipped: {e}")

        try:
            scaler.load_state_dict(
                ckpt["scaler"]
            )
        except Exception as e:
            print(f"Scaler load skipped: {e}")

        print(
            f"Resumed from epoch "
            f"{ckpt['epoch']}"
        )

        return ckpt

    except Exception as e:

        print(
            f"Checkpoint corrupted/incompatible: {e}"
        )

        print("Starting fresh training.")

        return None

# =========================================================
# METRICS
# =========================================================

@torch.no_grad()
def evaluate_vector(model, loader):

    model.eval()

    exact_correct = 0
    total = 0

    bit_correct = 0
    bit_total = 0

    for X, y in loader:

        X = X.to(device)
        y = y.to(device)

        logits = model(X)

        pred = (
            torch.sigmoid(logits) > 0.5
        ).float()

        exact_correct += (
            (pred == y)
            .all(dim=1)
            .sum()
            .item()
        )

        bit_correct += (
            pred == y
        ).sum().item()

        bit_total += y.numel()

        total += len(X)

    return {
        "seq_acc": exact_correct / total,
        "bit_acc": bit_correct / bit_total,
    }


@torch.no_grad()
def greedy_decode(model,
                  src,
                  max_len):

    B = src.size(0)

    tgt = torch.full(
        (B, max_len),
        PAD,
        dtype=torch.long,
        device=device
    )

    tgt[:, 0] = BOS

    for step in range(1, max_len):

        logits = model(
            src,
            tgt[:, :step]
        )

        next_token = (
            logits[:, -1]
            .argmax(dim=-1)
        )

        tgt[:, step] = next_token

        if (next_token == EOS).all():
            break

    return tgt


@torch.no_grad()
def prefix_decode(model,
                  src,
                  target,
                  prefix_len,
                  max_len):

    B = src.size(0)

    tgt = torch.full(
        (B, max_len),
        PAD,
        dtype=torch.long,
        device=device
    )

    tgt[:, 0] = BOS

    usable = min(
        prefix_len,
        target.size(1)
    )

    if usable > 0:
        tgt[:, 1:usable+1] = target[:, :usable]

    start = usable + 1

    for step in range(start, max_len):

        logits = model(
            src,
            tgt[:, :step]
        )

        next_token = (
            logits[:, -1]
            .argmax(dim=-1)
        )

        tgt[:, step] = next_token

        if (next_token == EOS).all():
            break

    return tgt


@torch.no_grad()
def evaluate_seq(model,
                 loader,
                 prefix_len=0,
                 use_prefix=False):

    model.eval()

    exact_correct = 0
    total = 0

    bit_correct = 0
    bit_total = 0

    for src, y_in, y_out in loader:

        src = src.to(device)

        y_out = y_out.to(device)

        target = y_out.clone()

        max_len = target.size(1) + 2

        if use_prefix:

            pred = prefix_decode(
                model,
                src,
                target,
                prefix_len,
                max_len=max_len
            )

        else:

            pred = greedy_decode(
                model,
                src,
                max_len=max_len
            )

        pred = pred[:, 1:]

        for p, t in zip(pred, target):

            p = p.tolist()
            t = t.tolist()

            if EOS in p:
                p = p[:p.index(EOS)]

            if EOS in t:
                t = t[:t.index(EOS)]

            p = [
                x for x in p
                if x != PAD
            ]

            t = [
                x for x in t
                if x != PAD
            ]

            exact_correct += (p == t)

            m = min(len(p), len(t))

            bit_correct += sum(
                pp == tt
                for pp, tt
                in zip(p[:m], t[:m])
            )

            bit_total += len(t)

            total += 1

    return {
        "seq_acc": exact_correct / total,
        "bit_acc": bit_correct / bit_total,
    }

# =========================================================
# TRAIN VECTOR
# =========================================================

def train_vector(train_file,
                 test_file,
                 dataset_name,
                 seed):

    set_seed(seed)

    train_ds = BinaryVectorDataset(train_file)
    test_ds = BinaryVectorDataset(test_file)

    tr_loader = DataLoader(
        train_ds,
        batch_size=COMMON["BATCH"],
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )

    te_loader = DataLoader(
        test_ds,
        batch_size=COMMON["BATCH"],
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    model = ResNetMLP(
        train_ds.X.shape[1],
        256,
        6,
        train_ds.y.shape[1],
        0.1
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=COMMON["LR"],
        weight_decay=COMMON["WD"]
    )

    scaler = torch.amp.GradScaler("cuda")

    criterion = nn.BCEWithLogitsLoss()

    ckpt_path = os.path.join(
        CKPT_DIR,
        f"{dataset_name}_resnet_seed{seed}.pt"
    )

    metrics = {
        "train_seq": [],
        "train_bit": [],
        "test_seq": [],
        "test_bit": [],
    }

    start_epoch = 1

    resume = load_checkpoint(
        ckpt_path,
        model,
        optimizer,
        scaler
    )

    if resume is not None:
        metrics = resume["metrics"]
        start_epoch = resume["epoch"] + 1

    print("Starting ResNet training...")

    for ep in range(start_epoch,
                    COMMON["EPOCHS"] + 1):

        model.train()

        for batch_idx, (X, y) in enumerate(tr_loader):

            X = X.to(device)
            y = y.to(device)

            optimizer.zero_grad()

            with torch.autocast(
                device_type="cuda",
                enabled=False
            ):

                logits = model(X)

                loss = criterion(
                    logits,
                    y
                )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                COMMON["CLIP"]
            )

            optimizer.step()

        train_eval = evaluate_vector(
            model,
            tr_loader
        )

        test_eval = evaluate_vector(
            model,
            te_loader
        )

        metrics["train_seq"].append(
            train_eval["seq_acc"]
        )

        metrics["train_bit"].append(
            train_eval["bit_acc"]
        )

        metrics["test_seq"].append(
            test_eval["seq_acc"]
        )

        metrics["test_bit"].append(
            test_eval["bit_acc"]
        )

        save_checkpoint(
            ckpt_path,
            model,
            optimizer,
            scaler,
            ep,
            metrics
        )

        print(
            f"ResNet | Ep {ep} | "
            f"TrainSeq={train_eval['seq_acc']:.4f} | "
            f"TestSeq={test_eval['seq_acc']:.4f}"
        )

        torch.cuda.empty_cache()

    return metrics

# =========================================================
# TRAIN TRANSFORMER
# =========================================================

def train_transformer(train_file,
                      test_file,
                      dataset_name,
                      seed,
                      method_name,
                      scheduled_sampling=False):

    set_seed(seed)

    train_ds = BinarySeqDataset(train_file)
    test_ds = BinarySeqDataset(test_file)

    tr_loader = DataLoader(
        train_ds,
        batch_size=COMMON["BATCH"],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=True,
    )

    te_loader = DataLoader(
        test_ds,
        batch_size=COMMON["BATCH"],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=True,
    )

    model = TransformerModel().to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=COMMON["LR"],
        weight_decay=COMMON["WD"]
    )

    scaler = torch.amp.GradScaler("cuda")

    criterion = nn.CrossEntropyLoss(
        ignore_index=PAD
    )

    ckpt_path = os.path.join(
        CKPT_DIR,
        f"{dataset_name}_{method_name}_seed{seed}.pt"
    )

    metrics = {
        "train_seq": [],
        "train_bit": [],
        "test_seq": [],
        "test_bit": [],
    }

    start_epoch = 1

    resume = load_checkpoint(
        ckpt_path,
        model,
        optimizer,
        scaler
    )

    if resume is not None:
        metrics = resume["metrics"]
        start_epoch = resume["epoch"] + 1

    print(f"Starting {method_name} training...")

    for ep in range(start_epoch,
                    COMMON["EPOCHS"] + 1):

        model.train()

        for batch_idx, (
            src,
            y_in,
            y_out
        ) in enumerate(tr_loader):

            src = src.to(device)
            y_in = y_in.to(device)
            y_out = y_out.to(device)

            optimizer.zero_grad()

            decoder_in = y_in.clone()

            # scheduled sampling disabled
            if scheduled_sampling and ep > 999999:

                with torch.no_grad():

                    pred_logits = model(
                        src,
                        y_in
                    )

                    pred_tokens = (
                        pred_logits
                        .argmax(dim=-1)
                    )

                mask = (
                    torch.rand_like(
                        decoder_in.float()
                    ) > 0.5
                )

                mask &= (
                    decoder_in != PAD
                )

                decoder_in[mask] = (
                    pred_tokens[mask]
                )

            with torch.autocast(
                device_type="cuda",
                enabled=False
            ):

                logits = model(
                    src,
                    decoder_in
                )

                loss = criterion(
                    logits.view(-1, VOCAB),
                    y_out.view(-1)
                )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                COMMON["CLIP"]
            )

            optimizer.step()

        # expensive inference evaluation
        # only every 5 epochs

        if ep % 5 == 0 or ep == 1:

            train_eval = {
                "seq_acc": 0.0,
                "bit_acc": 0.0
            }

            test_eval = evaluate_seq(
                model,
                te_loader
            )

            metrics["train_seq"].append(
                train_eval["seq_acc"]
            )

            metrics["train_bit"].append(
                train_eval["bit_acc"]
            )

            metrics["test_seq"].append(
                test_eval["seq_acc"]
            )

            metrics["test_bit"].append(
                test_eval["bit_acc"]
            )

            save_checkpoint(
                ckpt_path,
                model,
                optimizer,
                scaler,
                ep,
                metrics
            )

            print(
                f"{method_name} | "
                f"Ep {ep} | "
                f"TestSeq={test_eval['seq_acc']:.4f} | "
                f"TestBit={test_eval['bit_acc']:.4f}"
            )

            torch.cuda.empty_cache()

    return model, metrics

# =========================================================
# MAIN
# =========================================================

def main():

    train_files = sorted(
        glob.glob("/content/*_train.csv")
    )

    for train_file in train_files:

        base = train_file.replace(
            "_train.csv",
            ""
        )

        test_file = f"{base}_test.csv"

        if not os.path.exists(test_file):
            continue

        dataset_name = os.path.basename(base)

        print("=" * 80)
        print(f"DATASET: {dataset_name}")
        print("=" * 80)

        for seed in SEEDS:

            print(f"\nSEED: {seed}\n")

            train_vector(
                train_file,
                test_file,
                dataset_name,
                seed
            )

            train_transformer(
                train_file,
                test_file,
                dataset_name,
                seed,
                method_name="Transformer_TF",
                scheduled_sampling=False
            )

            train_transformer(
                train_file,
                test_file,
                dataset_name,
                seed,
                method_name="Transformer_SS",
                scheduled_sampling=True
            )

    print("DONE")


if __name__ == "__main__":
    main()
