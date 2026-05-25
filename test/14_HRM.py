import glob
import os
import math
import gc

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader

# =========================================================
# CLEANUP
# =========================================================

gc.collect()

if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# =========================================================
# CONFIG
# =========================================================

BATCH_SIZE = 512

LR = 3e-4

EPOCHS = 300

EMBED_DIM = 32
NUM_HEADS = 4

ENC_LAYERS = 4
DEC_LAYERS = 4

FF_MULT = 8

DROPOUT = 0.0
WD = 0.0

MAX_LEN = 256

NUM_LATENTS = 4
REASONING_STEPS = 4

CLIP_NORM = 1.0

WARMUP_STEPS = 1000

EVAL_EVERY = 1

USE_COMPILE = True

NUM_WORKERS = 2

PIN_MEMORY = True

# =========================================================
# TOKEN IDS
# =========================================================

PAD = 0

TOK_0 = 1
TOK_1 = 2

BOS = 3
EOS = 4

VOCAB_SIZE = 5

# =========================================================
# DEVICE
# =========================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print(f"Using device: {device}")

if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))

# =========================================================
# DATASET
# =========================================================

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

    max_y = max(len(y) for y in ys) + 2

    X_pad = []
    y_in = []
    y_out = []

    for x, y in zip(Xs, ys):

        x_pad = x + [PAD] * (max_x - len(x))

        y_seq = [BOS] + y + [EOS]

        y_pad = (
            y_seq +
            [PAD] * (max_y - len(y_seq))
        )

        X_pad.append(x_pad)

        y_in.append(y_pad[:-1])

        y_out.append(y_pad[1:])

    return (
        torch.tensor(X_pad, dtype=torch.long),
        torch.tensor(y_in, dtype=torch.long),
        torch.tensor(y_out, dtype=torch.long)
    )

# =========================================================
# RMSNORM
# =========================================================

class RMSNorm(nn.Module):

    def __init__(
        self,
        dim,
        eps=1e-6
    ):

        super().__init__()

        self.eps = eps

        self.scale = nn.Parameter(
            torch.ones(dim)
        )

    def forward(self, x):

        norm = x.pow(2).mean(
            dim=-1,
            keepdim=True
        )

        x = x * torch.rsqrt(
            norm + self.eps
        )

        return x * self.scale

# =========================================================
# ROTARY
# =========================================================

class RotaryEmbedding(nn.Module):

    def __init__(
        self,
        dim,
        max_seq_len=4096
    ):

        super().__init__()

        inv_freq = 1.0 / (
            10000 ** (
                torch.arange(
                    0,
                    dim,
                    2
                ).float() / dim
            )
        )

        t = torch.arange(max_seq_len)

        freqs = torch.einsum(
            "i,j->ij",
            t,
            inv_freq
        )

        emb = torch.cat(
            [freqs, freqs],
            dim=-1
        )

        self.register_buffer(
            "cos_cached",
            emb.cos()[None, None],
            persistent=False
        )

        self.register_buffer(
            "sin_cached",
            emb.sin()[None, None],
            persistent=False
        )

    def forward(self, x):

        seq_len = x.shape[-2]

        cos = self.cos_cached[:, :, :seq_len]
        sin = self.sin_cached[:, :, :seq_len]

        x1 = x[..., ::2]
        x2 = x[..., 1::2]

        x_rot = torch.stack(
            (-x2, x1),
            dim=-1
        ).flatten(-2)

        return (
            x * cos
            +
            x_rot * sin
        )

# =========================================================
# GEGLU
# =========================================================

class GEGLU(nn.Module):

    def forward(self, x):

        x, gate = x.chunk(2, dim=-1)

        return x * F.gelu(gate)

# =========================================================
# FEEDFORWARD
# =========================================================

class FeedForward(nn.Module):

    def __init__(
        self,
        dim,
        mult=4
    ):

        super().__init__()

        inner_dim = dim * mult

        self.net = nn.Sequential(

            nn.Linear(
                dim,
                inner_dim * 2
            ),

            GEGLU(),

            nn.Linear(
                inner_dim,
                dim
            )
        )

    def forward(self, x):

        return self.net(x)

# =========================================================
# ATTENTION
# =========================================================

class Attention(nn.Module):

    def __init__(
        self,
        dim,
        heads=8,
        causal=False
    ):

        super().__init__()

        self.heads = heads

        self.head_dim = dim // heads

        self.causal = causal

        self.norm = RMSNorm(dim)

        self.to_qkv = nn.Linear(
            dim,
            dim * 3,
            bias=False
        )

        self.to_out = nn.Linear(
            dim,
            dim,
            bias=False
        )

        self.rotary = RotaryEmbedding(
            self.head_dim
        )

    def forward(
        self,
        x,
        context=None,
        kv_cache=None
    ):

        x = self.norm(x)

        if context is None:
            context = x

        B, T, D = x.shape

        H = self.heads

        if context is x:

            qkv = self.to_qkv(x)

            q, k, v = qkv.chunk(3, dim=-1)

        else:

            q = self.to_qkv(x)[..., :D]

            kv = self.to_qkv(context)

            _, k, v = kv.chunk(3, dim=-1)

        q = q.view(
            B,
            -1,
            H,
            self.head_dim
        ).transpose(1, 2)

        k = k.view(
            B,
            -1,
            H,
            self.head_dim
        ).transpose(1, 2)

        v = v.view(
            B,
            -1,
            H,
            self.head_dim
        ).transpose(1, 2)

        q = self.rotary(q)
        k = self.rotary(k)

        # =========================================
        # KV CACHE
        # =========================================

        if kv_cache is not None:

            if "k" in kv_cache:

                k = torch.cat(
                    [kv_cache["k"], k],
                    dim=2
                )

                v = torch.cat(
                    [kv_cache["v"], v],
                    dim=2
                )

            kv_cache["k"] = k
            kv_cache["v"] = v

        # =========================================
        # FLASH ATTENTION / SDPA
        # =========================================

        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=(
                self.causal
                and
                kv_cache is None
            )
        )

        out = out.transpose(
            1,
            2
        ).contiguous()

        out = out.view(
            B,
            T,
            D
        )

        return self.to_out(out)

# =========================================================
# TRANSFORMER BLOCK
# =========================================================

class TransformerBlock(nn.Module):

    def __init__(
        self,
        dim,
        heads,
        causal=False
    ):

        super().__init__()

        self.attn = Attention(
            dim,
            heads,
            causal=causal
        )

        self.ff = FeedForward(dim)

        self.ff_norm = RMSNorm(dim)

    def forward(
        self,
        x,
        context=None,
        kv_cache=None
    ):

        x = x + self.attn(
            x,
            context=context,
            kv_cache=kv_cache
        )

        x = x + self.ff(
            self.ff_norm(x)
        )

        return x

# =========================================================
# MODEL
# =========================================================

class HierarchicalReasoningModel(nn.Module):

    def __init__(self):

        super().__init__()

        self.token_emb = nn.Embedding(
            VOCAB_SIZE,
            EMBED_DIM
        )

        self.encoder = nn.ModuleList([

            TransformerBlock(
                EMBED_DIM,
                NUM_HEADS,
                causal=False
            )

            for _ in range(ENC_LAYERS)

        ])

        self.latents = nn.Parameter(
            torch.randn(
                NUM_LATENTS,
                EMBED_DIM
            )
        )

        self.latent_attn = Attention(
            EMBED_DIM,
            NUM_HEADS,
            causal=False
        )

        self.reasoning_blocks = nn.ModuleList([

            TransformerBlock(
                EMBED_DIM,
                NUM_HEADS,
                causal=False
            )

            for _ in range(REASONING_STEPS)

        ])

        self.decoder = nn.ModuleList([

            TransformerBlock(
                EMBED_DIM,
                NUM_HEADS,
                causal=True
            )

            for _ in range(DEC_LAYERS)

        ])

        self.cross_attn = nn.ModuleList([

            Attention(
                EMBED_DIM,
                NUM_HEADS,
                causal=False
            )

            for _ in range(DEC_LAYERS)

        ])

        self.final_norm = RMSNorm(
            EMBED_DIM
        )

        self.to_logits = nn.Linear(
            EMBED_DIM,
            VOCAB_SIZE,
            bias=False
        )

        self.to_logits.weight = (
            self.token_emb.weight
        )

    # =====================================================
    # ENCODER
    # =====================================================

    def encode_reason(
        self,
        src
    ):

        x = self.token_emb(src)

        for block in self.encoder:

            x = block(x)

        B = x.size(0)

        latents = self.latents.unsqueeze(0).expand(
            B,
            -1,
            -1
        )

        latents = latents + self.latent_attn(
            latents,
            context=x
        )

        for block in self.reasoning_blocks:

            latents = block(latents)

            latents = latents + self.latent_attn(
                latents,
                context=x
            )

        return x, latents

    # =====================================================
    # TRAINING
    # =====================================================

    def forward(
        self,
        src,
        tgt
    ):

        enc, latents = self.encode_reason(src)

        x = self.token_emb(tgt)

        for i, block in enumerate(self.decoder):

            x = block(x)

            x = x + self.cross_attn[i](
                x,
                context=latents
            )

            x = x + self.cross_attn[i](
                x,
                context=enc
            )

        x = self.final_norm(x)

        return self.to_logits(x)

    # =====================================================
    # FAST INFERENCE
    # =====================================================

    @torch.no_grad()
    def generate(
        self,
        src,
        max_new_tokens=512
    ):

        self.eval()

        enc, latents = self.encode_reason(src)

        B = src.size(0)

        generated = torch.full(
            (B, 1),
            BOS,
            dtype=torch.long,
            device=src.device
        )

        kv_caches = [
            {}
            for _ in range(len(self.decoder))
        ]

        for _ in range(max_new_tokens):

            # ONLY NEW TOKEN
            x = self.token_emb(
                generated[:, -1:]
            )

            for i, block in enumerate(self.decoder):

                x = block(
                    x,
                    kv_cache=kv_caches[i]
                )

                x = x + self.cross_attn[i](
                    x,
                    context=latents
                )

                x = x + self.cross_attn[i](
                    x,
                    context=enc
                )

            x = self.final_norm(x)

            logits = self.to_logits(x)

            next_token = logits[:, -1].argmax(
                dim=-1,
                keepdim=True
            )

            generated = torch.cat(
                [generated, next_token],
                dim=1
            )

            if (next_token == EOS).all():
                break

        return generated[:, 1:]

# =========================================================
# LOSS
# =========================================================

criterion = nn.CrossEntropyLoss(
    ignore_index=PAD
)

# =========================================================
# LR SCHEDULE
# =========================================================

def get_lr(step):

    return (
        EMBED_DIM ** -0.5
    ) * min(
        step ** -0.5,
        step * (
            WARMUP_STEPS ** -1.5
        )
    )

# =========================================================
# TRUE ACC
# =========================================================

@torch.no_grad()
def evaluate_true_accuracy(
    model,
    loader
):

    model.eval()

    total_correct = 0
    total_tokens = 0

    for src, _, tgt_out in loader:

        src = src.to(device)

        tgt_out = tgt_out.to(device)

        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16
        ):

            preds = model.generate(
                src,
                max_new_tokens=tgt_out.size(1)
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

    return total_correct / total_tokens

# =========================================================
# TRAINING
# =========================================================

def train_model(
    train_file,
    test_file
):

    print(f"\n🚀 Training on {train_file}")

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
        pin_memory=PIN_MEMORY,
        persistent_workers=True
    )

    test_loader = DataLoader(
        test_data,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        persistent_workers=True
    )

    model = HierarchicalReasoningModel().to(device)

    if USE_COMPILE:

        print("torch.compile enabled...")

        model = torch.compile(model)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WD,
        betas=(0.9, 0.95),
        fused=True
    )

    scaler = torch.amp.GradScaler(
        "cuda"
    )

    step = 1

    train_losses = []
    true_accs = []

    epochs_axis = []

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

            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16
            ):

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

            scaler.scale(loss).backward()

            scaler.unscale_(optimizer)

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                CLIP_NORM
            )

            for g in optimizer.param_groups:

                g['lr'] = get_lr(step)

            scaler.step(optimizer)

            scaler.update()

            total_train += (
                loss.item() * len(src)
            )

            step += 1

        avg_train = (
            total_train /
            len(train_data)
        )

        # =========================================
        # EVALUATE SOMETIMES
        # =========================================

        if (
            epoch % EVAL_EVERY == 0
            or epoch == 1
        ):

            true_acc = evaluate_true_accuracy(
                model,
                test_loader
            )

        else:

            true_acc = -1

        train_losses.append(avg_train)
        true_accs.append(true_acc)
        epochs_axis.append(epoch)

        if true_acc >= 0:

            print(
                f"Epoch {epoch:3d} | "
                f"Train Loss: {avg_train:.4f} | "
                f"True Test Acc: {true_acc:.4f}"
            )

        else:

            print(
                f"Epoch {epoch:3d} | "
                f"Train Loss: {avg_train:.4f}"
            )

    # =====================================================
    # PLOT
    # =====================================================

    plt.figure(figsize=(24, 10))

    valid_epochs = [
        e for e, a in zip(
            epochs_axis,
            true_accs
        )
        if a >= 0
    ]

    valid_accs = [
        a for a in true_accs
        if a >= 0
    ]

    plt.plot(
        valid_epochs,
        valid_accs,
        marker='o'
    )

    plt.xlabel("Epoch")

    plt.ylabel(
        "True Autoregressive Accuracy"
    )

    plt.title(
        f"Hierarchical Reasoning Accuracy "
        f"- {os.path.basename(train_file)}"
    )

    plt.grid(True)

    plt.tight_layout()

    os.makedirs(
        "/kaggle/working/plots",
        exist_ok=True
    )

    plot_path = (
        f"/kaggle/working/plots/"
        f"{os.path.basename(train_file)}"
        f"_hierarchical.png"
    )

    plt.savefig(
        plot_path,
        dpi=300
    )

    plt.close()

    print(f"💾 Saved: {plot_path}")

    return model

# =========================================================
# MAIN
# =========================================================

def main():

    train_files = sorted(
        glob.glob(
            "/kaggle/input/datasets/classstudents/test99/*_train.csv"
        )
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

        if not os.path.exists(test_file):

            print(
                f"⚠️ Skipping {base}"
            )

            continue

        train_model(
            train_file,
            test_file
        )

        print("=" * 80)

if __name__ == "__main__":

    main()
