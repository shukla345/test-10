#!/usr/bin/env python3
"""
UNIFIED MULTI-MODEL STRUCTURAL BENCHMARK FRAMEWORK
All 14 variants – Fixed Token Semantics (PAD=0, ZERO=1, ONE=2, BOS=3, EOS=4)

Includes:
- Vector models (ResNet, DenseNet, GatedEncoder, BottleNeck, Fourier)
- Transformer models (standard, memory, hierarchical reasoning)
- Full training/evaluation loops with exact-match and token-avg accuracy
- Prefix‑continuation analysis (variant 9)
- High‑DPI plots (loss curves, accuracy curves, prefix profile)
"""

import glob
import os
import math
import gc
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from typing import Dict, List, Tuple, Optional

# ═══════════════════════════════════════════════════════════════════════════
# GLOBAL HARDWARE & TOKEN CONFIGURATION (CLEAN SEMANTICS)
# ═══════════════════════════════════════════════════════════════════════════

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True
print(f"🚀 Execution Engine: {device}")
torch.set_float32_matmul_precision('high')
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# ─── Unified token semantics ───────────────────────────────────────────────
TOKEN_CFG = {
    "PAD": 0,
    "ZERO": 1,
    "ONE": 2,
    "BOS": 3,
    "EOS": 4,
    "V": 5
}
PAD_ID = TOKEN_CFG["PAD"]
ZERO_ID = TOKEN_CFG["ZERO"]
ONE_ID = TOKEN_CFG["ONE"]
BOS_ID = TOKEN_CFG["BOS"]
EOS_ID = TOKEN_CFG["EOS"]
VOCAB_SIZE = TOKEN_CFG["V"]


# ═══════════════════════════════════════════════════════════════════════════
# DATA UTILITIES & DATASETS
# ═══════════════════════════════════════════════════════════════════════════

def parse_binary_vector(series: pd.Series) -> np.ndarray:
    """Convert binary strings to float32 arrays of 0/1."""
    strings = series.astype(str).str.strip()
    max_len = strings.str.len().max()
    return np.array([[float(b) for b in s.zfill(max_len)] for s in strings], dtype=np.float32)

def parse_seq(s: str) -> List[int]:
    """Convert raw binary string to token ids (ZERO=1, ONE=2)."""
    out = []
    for c in s.strip():
        if c == '0':
            out.append(ZERO_ID)
        elif c == '1':
            out.append(ONE_ID)
        else:
            raise ValueError(f"Unexpected character: {c}")
    return out

class BinaryVectorDataset(Dataset):
    """For vector‑based models (ResNet, DenseNet, etc.)"""
    def __init__(self, path: str):
        df = pd.read_csv(path, header=None, dtype=str)
        self.y = parse_binary_vector(df.iloc[:, 0])
        self.X = parse_binary_vector(df.iloc[:, 1])
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return torch.tensor(self.X[idx]), torch.tensor(self.y[idx])

class BinarySeqDataset(Dataset):
    """For sequence models (Transformers, MemoryTransformer, HRM)"""
    def __init__(self, file: str):
        df = pd.read_csv(file, header=None, dtype=str)
        raw_y = df.iloc[:, 0]
        raw_x = df.iloc[:, 1]
        self.X = [parse_seq(x) for x in raw_x]
        self.y = [parse_seq(y) for y in raw_y]
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def collate_fn(batch, token_cfg=TOKEN_CFG):
    """Pad encoder/decoder sequences. Returns (src, y_in, y_out)."""
    PAD = token_cfg["PAD"]
    BOS = token_cfg["BOS"]
    EOS = token_cfg["EOS"]

    Xs, ys = zip(*batch)
    max_x = max(len(x) for x in Xs)
    max_y = max(len(y) for y in ys) + 2

    X_pad, y_in, y_out = [], [], []
    for x, y in zip(Xs, ys):
        X_pad.append(x + [PAD] * (max_x - len(x)))
        y_seq = [BOS] + y + [EOS]
        y_pad = y_seq + [PAD] * (max_y - len(y_seq))
        y_in.append(y_pad[:-1])
        y_out.append(y_pad[1:])

    return (
        torch.tensor(X_pad, dtype=torch.long),
        torch.tensor(y_in, dtype=torch.long),
        torch.tensor(y_out, dtype=torch.long)
    )


# ═══════════════════════════════════════════════════════════════════════════
# VECTOR MODELS (1 – 5)
# ═══════════════════════════════════════════════════════════════════════════

class ResBlock(nn.Module):
    def __init__(self, dim, dropout):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(dim, dim)
        )
        self.act = nn.ReLU()
    def forward(self, x):
        return self.act(x + self.block(x))

class ResNetMLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_layers, out_dim, dropout):
        super().__init__()
        self.input_proj = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.ReLU())
        self.blocks = nn.Sequential(*[ResBlock(hidden_dim, dropout) for _ in range(num_layers)])
        self.head = nn.Linear(hidden_dim, out_dim)
    def forward(self, x):
        return self.head(self.blocks(self.input_proj(x)))

class DenseLayer(nn.Module):
    def __init__(self, in_dim, growth_rate, dropout):
        super().__init__()
        self.layer = nn.Sequential(nn.Linear(in_dim, growth_rate), nn.ReLU(), nn.Dropout(dropout))
    def forward(self, x):
        return torch.cat([x, self.layer(x)], dim=1)

class DenseBlock(nn.Module):
    def __init__(self, in_dim, num_layers, growth_rate, dropout):
        super().__init__()
        self.layers = nn.ModuleList()
        current_dim = in_dim
        for _ in range(num_layers):
            self.layers.append(DenseLayer(current_dim, growth_rate, dropout))
            current_dim += growth_rate
        self.out_dim = current_dim
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class DenseNetMLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_layers, growth_rate, out_dim, dropout):
        super().__init__()
        self.input_proj = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.ReLU())
        self.block = DenseBlock(hidden_dim, num_layers, growth_rate, dropout)
        self.head = nn.Linear(self.block.out_dim, out_dim)
    def forward(self, x):
        return self.head(self.block(self.input_proj(x)))

class GatedEncoderBlock(nn.Module):
    def __init__(self, dim, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim * 2)
        self.fc2 = nn.Linear(dim * 2, dim)
        self.gate = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        h = self.dropout(torch.relu(self.fc1(self.norm(x))))
        h = self.fc2(h)
        return x + torch.sigmoid(self.gate(x)) * h

class SystemEncoder(nn.Module):
    def __init__(self, in_dim, embed_dim, depth, dropout):
        super().__init__()
        self.input_proj = nn.Linear(in_dim, embed_dim)
        self.blocks = nn.ModuleList([GatedEncoderBlock(embed_dim, dropout) for _ in range(depth)])
    def forward(self, x):
        x = self.input_proj(x)
        for blk in self.blocks:
            x = blk(x)
        return x

class EncoderDenseMLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, enc_depth, num_layers, growth_rate, out_dim, dropout, noise_std=0.01):
        super().__init__()
        self.encoder = SystemEncoder(in_dim, hidden_dim, enc_depth, dropout)
        self.block = DenseBlock(hidden_dim, num_layers, growth_rate, dropout)
        self.head = nn.Linear(self.block.out_dim, out_dim)
        self.noise_std = noise_std
    def forward(self, x):
        if self.training:
            x = x + self.noise_std * torch.randn_like(x)
        return self.head(self.block(self.encoder(x)))

class MLPHead(nn.Module):
    def __init__(self, in_dim, bottleneck, width, depth, out_dim, dropout):
        super().__init__()
        self.bottleneck = nn.Sequential(nn.Linear(in_dim, bottleneck), nn.GELU(), nn.LayerNorm(bottleneck))
        layers, dim = [], bottleneck
        for _ in range(depth):
            layers.extend([nn.Linear(dim, width), nn.GELU(), nn.Dropout(dropout)])
            dim = width
        layers.append(nn.Linear(dim, out_dim))
        self.mlp = nn.Sequential(*layers)
    def forward(self, x):
        return self.mlp(self.bottleneck(x))

class BottleNeckModel(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim, enc_depth, bottleneck, mlp_width, mlp_depth, dropout, noise_std=0.01):
        super().__init__()
        self.encoder = SystemEncoder(in_dim, hidden_dim, enc_depth, dropout)
        self.head = MLPHead(hidden_dim, bottleneck, mlp_width, mlp_depth, out_dim, dropout)
        self.noise_std = noise_std
    def forward(self, x):
        if self.training:
            x = x + self.noise_std * torch.randn_like(x)
        return self.head(self.encoder(x))

class FourierFeatures(nn.Module):
    def __init__(self, in_dim, fourier_dim, scale=3.0):
        super().__init__()
        self.B = nn.Parameter(torch.randn(in_dim, fourier_dim) * scale)
    def forward(self, x):
        x = x * 2 - 1
        return torch.cat([torch.sin(2 * np.pi * x @ self.B), torch.cos(2 * np.pi * x @ self.B)], dim=-1)

class FourierModel(nn.Module):
    def __init__(self, in_dim, out_dim, fourier_dim, fourier_scale, hidden_dim, enc_depth, bottleneck, mlp_width, mlp_depth, dropout, noise_std=0.01):
        super().__init__()
        self.fourier = FourierFeatures(in_dim, fourier_dim, fourier_scale)
        self.encoder = SystemEncoder(fourier_dim * 2, hidden_dim, enc_depth, dropout)
        self.head = MLPHead(hidden_dim, bottleneck, mlp_width, mlp_depth, out_dim, dropout)
        self.noise_std = noise_std
    def forward(self, x):
        if self.training:
            x = x + self.noise_std * torch.randn_like(x)
        return self.head(self.encoder(self.fourier(x)))


# ═══════════════════════════════════════════════════════════════════════════
# TRANSFORMER & ADVANCED SEQUENCE MODELS (6 – 14)
# ═══════════════════════════════════════════════════════════════════════════

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class TransformerModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers, ff_dim, dropout, pad_id):
        super().__init__()
        self.pad_id = pad_id
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.pos = PositionalEncoding(embed_dim)
        self.tf = nn.Transformer(
            d_model=embed_dim, nhead=num_heads,
            num_encoder_layers=num_layers, num_decoder_layers=num_layers,
            dim_feedforward=ff_dim, dropout=dropout, batch_first=True
        )
        self.head = nn.Linear(embed_dim, vocab_size)
    def forward(self, src, tgt):
        src_mask = (src == self.pad_id)
        tgt_pad_mask = (tgt == self.pad_id)
        T = tgt.size(1)
        causal_mask = torch.triu(torch.ones(T, T, device=tgt.device, dtype=torch.bool), diagonal=1)
        src_emb = self.pos(self.embed(src))
        tgt_emb = self.pos(self.embed(tgt))
        out = self.tf(src_emb, tgt_emb, tgt_mask=causal_mask,
                      src_key_padding_mask=src_mask, tgt_key_padding_mask=tgt_pad_mask)
        return self.head(out)

class Memory(nn.Module):
    def __init__(self, mem_slots, embed_dim):
        super().__init__()
        self.memory = nn.Parameter(torch.randn(mem_slots, embed_dim))
    def forward(self, batch_size):
        return self.memory.unsqueeze(0).repeat(batch_size, 1, 1)

class MemoryTransformer(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers, mem_slots, dropout, pad_id):
        super().__init__()
        self.pad_id = pad_id
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.pos = PositionalEncoding(embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True, dropout=dropout)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        decoder_layer = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True, dropout=dropout)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers)
        self.memory = Memory(mem_slots, embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size)
    def forward(self, src, tgt):
        B = src.size(0)
        causal_mask = torch.triu(torch.ones(tgt.size(1), tgt.size(1), device=tgt.device, dtype=torch.bool), diagonal=1)
        src_mask = (src == self.pad_id)
        tgt_pad_mask = (tgt == self.pad_id)
        enc_out = self.encoder(self.pos(self.embed(src)), src_key_padding_mask=src_mask)
        out = self.decoder(self.pos(self.embed(tgt)), torch.cat([enc_out, self.memory(B)], dim=1),
                           tgt_mask=causal_mask, tgt_key_padding_mask=tgt_pad_mask)
        return self.head(out)

# ----- Hierarchical Reasoning Model (HRM) -----
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))
    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps) * self.scale

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=4096):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_seq_len)
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None], persistent=False)
    def forward(self, x):
        seq_len = x.shape[-2]
        return x * self.cos_cached[:, :, :seq_len] + \
               torch.stack((-x[..., 1::2], x[..., ::2]), dim=-1).flatten(-2) * self.sin_cached[:, :, :seq_len]

class GEGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return x * F.gelu(gate)

class FeedForward(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2), GEGLU(), nn.Linear(dim * mult, dim)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads=8, causal=False):
        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads
        self.causal = causal
        self.norm = RMSNorm(dim)
        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.to_out = nn.Linear(dim, dim, bias=False)
        self.rotary = RotaryEmbedding(self.head_dim)
    def forward(self, x, context=None):
        x = self.norm(x)
        if context is None:
            context = x
        B, T, D = x.shape
        H = self.heads
        if context is x:
            q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        else:
            q = self.to_qkv(x)[..., :D]
            _, k, v = self.to_qkv(context).chunk(3, dim=-1)
        q = q.view(B, -1, H, self.head_dim).transpose(1, 2)
        k = k.view(B, -1, H, self.head_dim).transpose(1, 2)
        v = v.view(B, -1, H, self.head_dim).transpose(1, 2)
        q, k = self.rotary(q), self.rotary(k)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=self.causal)
        return self.to_out(out.transpose(1, 2).contiguous().view(B, T, D))

class TransformerBlock(nn.Module):
    def __init__(self, dim, heads, causal=False):
        super().__init__()
        self.attn = Attention(dim, heads, causal=causal)
        self.ff = FeedForward(dim)
        self.ff_norm = RMSNorm(dim)
    def forward(self, x, context=None):
        x = x + self.attn(x, context=context)
        return x + self.ff(self.ff_norm(x))

class HierarchicalReasoningModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=32, enc_layers=4, num_latents=4,
                 reasoning_steps=4, dec_layers=4, num_heads=4, pad_id=PAD_ID):
        super().__init__()
        self.pad_id = pad_id
        self.token_emb = nn.Embedding(vocab_size, embed_dim)
        self.encoder = nn.ModuleList([TransformerBlock(embed_dim, num_heads, causal=False) for _ in range(enc_layers)])
        self.latents = nn.Parameter(torch.randn(num_latents, embed_dim))
        self.latent_attn = Attention(embed_dim, num_heads, causal=False)
        self.reasoning_blocks = nn.ModuleList([TransformerBlock(embed_dim, num_heads, causal=False) for _ in range(reasoning_steps)])
        self.decoder = nn.ModuleList([TransformerBlock(embed_dim, num_heads, causal=True) for _ in range(dec_layers)])
        self.cross_attn = nn.ModuleList([Attention(embed_dim, num_heads, causal=False) for _ in range(dec_layers)])
        self.final_norm = RMSNorm(embed_dim)
        self.to_logits = nn.Linear(embed_dim, vocab_size, bias=False)
        self.to_logits.weight = self.token_emb.weight
    def forward(self, src, tgt):
        x = self.token_emb(src)
        for block in self.encoder:
            x = block(x)
        B = x.size(0)
        latents = self.latents.unsqueeze(0).expand(B, -1, -1)
        latents = latents + self.latent_attn(latents, context=x)
        for block in self.reasoning_blocks:
            latents = block(latents)
            latents = latents + self.latent_attn(latents, context=x)
        dec_out = self.token_emb(tgt)
        for i, block in enumerate(self.decoder):
            dec_out = block(dec_out)
            dec_out = dec_out + self.cross_attn[i](dec_out, context=latents)
            dec_out = dec_out + self.cross_attn[i](dec_out, context=x)
        return self.to_logits(self.final_norm(dec_out))


# ═══════════════════════════════════════════════════════════════════════════
# INFERENCE & EVALUATION UTILITIES
# ═══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def batched_greedy_decode(model, src_batch, bos_id, eos_id, max_len=45, pad_id=PAD_ID):
    """Autoregressive generation for sequence models."""
    model.eval()
    B = src_batch.size(0)
    tgt = torch.full((B, max_len), pad_id, dtype=torch.long, device=src_batch.device)
    tgt[:, 0] = bos_id
    finished = torch.zeros(B, dtype=torch.bool, device=src_batch.device)
    for step in range(1, max_len):
        logits = model(src_batch, tgt[:, :step])
        next_token = logits[:, -1].argmax(dim=-1)
        tgt[:, step] = next_token
        finished |= (next_token == eos_id)
        if finished.all():
            break
    return tgt

@torch.no_grad()
def evaluate_true_inference(model, dataset, bos_id, eos_id, pad_id, max_samples=40):
    """Exact‑match accuracy (full sequence correctness)."""
    model.eval()
    samples = min(len(dataset), max_samples)
    src_list, true_list = [], []
    for i in range(samples):
        src, true = dataset[i]
        src_list.append(torch.tensor(src, dtype=torch.long, device=device))
        true_list.append(true)
    max_x = max(len(s) for s in src_list)
    src_tensor = torch.full((samples, max_x), pad_id, dtype=torch.long, device=device)
    for i, s in enumerate(src_list):
        src_tensor[i, :len(s)] = s
    pred_tensor = batched_greedy_decode(model, src_tensor, bos_id, eos_id, max_len=45, pad_id=pad_id).cpu()
    seq_correct = 0
    for i, true in enumerate(true_list):
        pred = pred_tensor[i].tolist()
        if bos_id in pred:
            pred = pred[pred.index(bos_id)+1:]
        if eos_id in pred:
            pred = pred[:pred.index(eos_id)]
        pred = [p for p in pred if p != pad_id]
        if pred == true:
            seq_correct += 1
    return seq_correct / samples

@torch.no_grad()
def evaluate_token_accuracy(model, dataset, bos_id, eos_id, pad_id, max_samples=40):
    """Token‑level accuracy (average fraction of correctly predicted tokens)."""
    model.eval()
    samples = min(len(dataset), max_samples)
    src_list, true_list = [], []
    for i in range(samples):
        src, true = dataset[i]
        src_list.append(torch.tensor(src, dtype=torch.long, device=device))
        true_list.append(true)
    max_x = max(len(s) for s in src_list)
    src_tensor = torch.full((samples, max_x), pad_id, dtype=torch.long, device=device)
    for i, s in enumerate(src_list):
        src_tensor[i, :len(s)] = s
    pred_tensor = batched_greedy_decode(model, src_tensor, bos_id, eos_id, max_len=45, pad_id=pad_id).cpu()
    seq_token_accs = []
    for i, true in enumerate(true_list):
        pred = pred_tensor[i].tolist()
        if bos_id in pred:
            pred = pred[pred.index(bos_id)+1:]
        if eos_id in pred:
            pred = pred[:pred.index(eos_id)]
        pred = [p for p in pred if p != pad_id]
        if len(true) == 0:
            seq_token_accs.append(1.0 if len(pred) == 0 else 0.0)
            continue
        min_len = min(len(pred), len(true))
        matches = sum(p == t for p, t in zip(pred[:min_len], true[:min_len]))
        seq_token_accs.append(matches / len(true))
    return np.mean(seq_token_accs) if seq_token_accs else 0.0

@torch.no_grad()
def evaluate_prefix_accuracy(model, loader, max_prefix_len=30, pad_id=PAD_ID):
    """Prefix‑continuation accuracy for variant 9."""
    model.eval()
    prefix_lens = list(range(1, max_prefix_len + 1))
    correct_counts = {k: 0 for k in prefix_lens}
    total_counts = {k: 0 for k in prefix_lens}
    for src, y_in, y_out in loader:
        src, y_in, y_out = src.to(device), y_in.to(device), y_out.to(device)
        logits = model(src, y_in)
        preds = logits.argmax(dim=-1)
        for k in prefix_lens:
            if k >= y_in.size(1):
                continue
            mask = (y_out[:, k] != pad_id)
            if mask.sum() == 0:
                continue
            matches = (preds[:, k] == y_out[:, k]) & mask
            correct_counts[k] += matches.sum().item()
            total_counts[k] += mask.sum().item()
    accs = [correct_counts[k] / total_counts[k] if total_counts[k] > 0 else 0.0 for k in prefix_lens]
    return prefix_lens, accs


# ═══════════════════════════════════════════════════════════════════════════
# TRAINING LOOP & MODEL REGISTRY
# ═══════════════════════════════════════════════════════════════════════════

COMMON = {"BATCH": 256, "LR": 3e-4, "EPOCHS": 200, "WD": 0, "DROP": 0, "CLIP": 1.0}

METHODS = {
    # Vector models (binary classification with BCE)
    "1_ResNet_Vector": {
        "type": "vector",
        "build": lambda i, o: ResNetMLP(i, 256, 6, o, 0.1),
        "criterion": nn.BCEWithLogitsLoss()
    },
    "2_DenseNet_Vector": {
        "type": "vector",
        "build": lambda i, o: DenseNetMLP(i, 128, 6, 32, o, 0.1),
        "criterion": nn.BCEWithLogitsLoss()
    },
    "3_GatedEncoder_Vector": {
        "type": "vector",
        "build": lambda i, o: EncoderDenseMLP(i, 256, 3, 6, 32, o, 0.1, 0.01),
        "criterion": nn.BCEWithLogitsLoss()
    },
    "4_BottleNeckMLP_Vector": {
        "type": "vector",
        "build": lambda i, o: BottleNeckModel(i, o, 256, 3, 128, 256, 3, 0.1, 0.01),
        "criterion": nn.BCEWithLogitsLoss()
    },
    "5_FourierProjection_Vector": {
        "type": "vector",
        "build": lambda i, o: FourierModel(i, o, 64, 3.0, 256, 3, 128, 256, 3, 0.1, 0.01),
        "criterion": nn.BCEWithLogitsLoss()
    },
    # Sequence models (autoregressive, cross‑entropy)
    "6_Transformer_TeacherForced": {
        "type": "seq",
        "cfg": {"V": VOCAB_SIZE, "E": 128, "H": 4, "L": 3, "FF": 256},
        "build": lambda c: TransformerModel(c["V"], c["E"], c["H"], c["L"], c["FF"], 0.1, pad_id=PAD_ID),
        "metric": "loss"
    },
    "7_Transformer_TrueInference": {
        "type": "seq",
        "cfg": {"V": VOCAB_SIZE, "E": 128, "H": 4, "L": 3, "FF": 256},
        "build": lambda c: TransformerModel(c["V"], c["E"], c["H"], c["L"], c["FF"], 0.1, pad_id=PAD_ID),
        "metric": "accuracy"
    },
    "8_TrueInference_Worked_Mode": {
        "type": "seq",
        "cfg": {"V": VOCAB_SIZE, "E": 16, "H": 2, "L": 1, "FF": 32},
        "build": lambda c: TransformerModel(c["V"], c["E"], c["H"], c["L"], c["FF"], 0.0, pad_id=PAD_ID),
        "metric": "accuracy"
    },
    "9_Transformer_PrefixTest": {
        "type": "seq",
        "cfg": {"V": VOCAB_SIZE, "E": 32, "H": 2, "L": 2, "FF": 64},
        "build": lambda c: TransformerModel(c["V"], c["E"], c["H"], c["L"], c["FF"], 0.1, pad_id=PAD_ID),
        "metric": "prefix_isolated"
    },
    "10_MemoryTransformer_TF": {
        "type": "seq",
        "cfg": {"V": VOCAB_SIZE, "E": 128, "H": 4, "L": 3, "MEM": 32},
        "build": lambda c: MemoryTransformer(c["V"], c["E"], c["H"], c["L"], c["MEM"], 0.1, pad_id=PAD_ID),
        "metric": "loss"
    },
    "11_TrueInference_Memory": {
        "type": "seq",
        "cfg": {"V": VOCAB_SIZE, "E": 128, "H": 4, "L": 3, "MEM": 32},
        "build": lambda c: MemoryTransformer(c["V"], c["E"], c["H"], c["L"], c["MEM"], 0.1, pad_id=PAD_ID),
        "metric": "accuracy"
    },
    "12_Hope_Variant": {
        "type": "seq",
        "cfg": {"V": VOCAB_SIZE, "E": 16, "H": 2, "L": 2, "FF": 64},
        "build": lambda c: TransformerModel(c["V"], c["E"], c["H"], c["L"], c["FF"], 0.0, pad_id=PAD_ID),
        "metric": "accuracy"
    },
    "13_HierarchicalReasoning_HRM": {
        "type": "seq",
        "cfg": {"V": VOCAB_SIZE},
        "build": lambda c: HierarchicalReasoningModel(vocab_size=c["V"], pad_id=PAD_ID),
        "metric": "accuracy"
    }
}

def train_and_collect(base, train_file, test_file, m_name, m_cfg):
    typ = m_cfg["type"]
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        scaler = torch.amp.GradScaler('cuda')
    else:
        scaler = None

    if typ == "vector":
        train_ds = BinaryVectorDataset(train_file)
        test_ds = BinaryVectorDataset(test_file)
        tr_ld = DataLoader(train_ds, batch_size=COMMON["BATCH"], shuffle=True, num_workers=2, pin_memory=use_cuda)
        te_ld = DataLoader(test_ds, batch_size=COMMON["BATCH"], shuffle=False, num_workers=2, pin_memory=use_cuda)
        model = m_cfg["build"](train_ds.X.shape[1], train_ds.y.shape[1]).to(device)
        criterion = m_cfg["criterion"]
    else:   # sequence model
        train_ds = BinarySeqDataset(train_file)
        test_ds = BinarySeqDataset(test_file)
        coll = lambda b: collate_fn(b, TOKEN_CFG)
        tr_ld = DataLoader(train_ds, batch_size=COMMON["BATCH"], shuffle=True, collate_fn=coll, num_workers=2, pin_memory=use_cuda)
        te_ld = DataLoader(test_ds, batch_size=COMMON["BATCH"], shuffle=False, collate_fn=coll, num_workers=2, pin_memory=use_cuda)
        cfg = m_cfg["cfg"]
        model = m_cfg["build"](cfg).to(device)
        criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID)

    optimizer = torch.optim.AdamW(model.parameters(), lr=COMMON["LR"], weight_decay=COMMON["WD"])
    metrics = {"epochs": [], "test_loss": [], "seq_acc": [], "token_acc": []}
    if typ == "vector":
        metrics["bit_acc"] = []   # store bitwise accuracy per epoch

    for ep in range(1, COMMON["EPOCHS"] + 1):
        model.train()
        for batch_data in tr_ld:
            optimizer.zero_grad()
            if typ == "vector":
                X, y = batch_data[0].to(device), batch_data[1].to(device)
                if use_cuda:
                    with torch.amp.autocast('cuda'):
                        loss = criterion(model(X), y)
                else:
                    loss = criterion(model(X), y)
            else:
                src, y_in, y_out = [t.to(device) for t in batch_data]
                if use_cuda:
                    with torch.amp.autocast('cuda'):
                        logits = model(src, y_in)
                        loss = criterion(logits.view(-1, VOCAB_SIZE), y_out.view(-1))
                else:
                    logits = model(src, y_in)
                    loss = criterion(logits.view(-1, VOCAB_SIZE), y_out.view(-1))

            if use_cuda:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), COMMON["CLIP"])
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), COMMON["CLIP"])
                optimizer.step()

        # ---- validation ----
        model.eval()
        total_loss = 0.0
        total_samples = 0

        # For vector models, accumulate bitwise accuracy
        if typ == "vector":
            total_bits_correct = 0
            total_bits = 0

        with torch.no_grad():
            for batch_data in te_ld:
                if typ == "vector":
                    X, y = batch_data[0].to(device), batch_data[1].to(device)
                    logits = model(X)
                    loss = criterion(logits, y)
                    total_loss += loss.item() * X.size(0)
                    total_samples += X.size(0)

                    # Bitwise accuracy
                    probs = torch.sigmoid(logits)          # [B, out_dim]
                    preds = (probs > 0.5).float()          # threshold at 0.5
                    correct_bits = (preds == y).float()    # [B, out_dim]
                    total_bits_correct += correct_bits.sum().item()
                    total_bits += correct_bits.numel()

                else:
                    src, y_in, y_out = [t.to(device) for t in batch_data]
                    logits = model(src, y_in)
                    loss = criterion(logits.view(-1, VOCAB_SIZE), y_out.view(-1))
                    total_loss += loss.item() * src.size(0)
                    total_samples += src.size(0)

        metrics["epochs"].append(ep)
        metrics["test_loss"].append(total_loss / total_samples)

        if typ == "vector":
            bit_acc = total_bits_correct / total_bits if total_bits > 0 else 0.0
            metrics["bit_acc"].append(bit_acc)
            # For compatibility with plotting (seq models only), set seq_acc to 0
            metrics["seq_acc"].append(0.0)
            metrics["token_acc"].append(0.0)
        else:  # seq model
            if m_cfg.get("metric") == "accuracy":
                acc_exact = evaluate_true_inference(model, test_ds, BOS_ID, EOS_ID, PAD_ID)
                acc_token = evaluate_token_accuracy(model, test_ds, BOS_ID, EOS_ID, PAD_ID)
                metrics["seq_acc"].append(acc_exact)
                metrics["token_acc"].append(acc_token)
            else:
                metrics["seq_acc"].append(0.0)
                metrics["token_acc"].append(0.0)

    # special prefix analysis for variant 9
    if m_name == "9_Transformer_PrefixTest":
        print(f"  🔍 Isolated continuation analysis: {m_name}")
        p_lens, p_accs = evaluate_prefix_accuracy(model, te_ld, max_prefix_len=20, pad_id=PAD_ID)
        metrics["prefix_lens"] = p_lens
        metrics["prefix_accs"] = p_accs

    # Final print: show appropriate accuracy metric
    if typ == "vector":
        last_acc = metrics["bit_acc"][-1] if metrics["bit_acc"] else 0.0
        print(f"  {m_name:30s} Complete | Final Loss: {metrics['test_loss'][-1]:.4f} | Bit Acc: {last_acc:.3f}")
    else:
        last_acc = metrics["seq_acc"][-1] if metrics["seq_acc"] else 0.0
        print(f"  {m_name:30s} Complete | Final Loss: {metrics['test_loss'][-1]:.4f} | Exact Acc: {last_acc:.3f}")
    return metrics

# ═══════════════════════════════════════════════════════════════════════════
# MAIN EXECUTION & PLOTTING
# ═══════════════════════════════════════════════════════════════════════════

def main():
    train_files = sorted(glob.glob("/kaggle/input/datasets/classstudents/test52/*_train.csv"))
    if not train_files:
        print("⚠️  No training files found. Please adjust the glob pattern or provide a valid path.")
        return
    os.makedirs("/kaggle/working/plots", exist_ok=True)

    for t_file in train_files:
        base = os.path.basename(t_file).replace("_train.csv", "")
        te_file = t_file.replace("_train.csv", "_test.csv")
        if not os.path.exists(te_file):
            continue

        print(f"\n⚡ Benchmark set: {base}")
        all_metrics = {}
        for name, cfg in METHODS.items():
            all_metrics[name] = train_and_collect(base, t_file, te_file, name, cfg)

        # ----- main combined plot (loss + accuracy) -----
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 10))
        colors = cm.get_cmap('tab20', len(METHODS))

        for i, (name, met) in enumerate(all_metrics.items()):
            if name == "9_Transformer_PrefixTest":
                continue   # skip prefix model in main plot
            c = colors(i)
            clean_label = name.replace('_', ' ')
            
            # Loss curves (all models)
            ax1.plot(met["epochs"], met["test_loss"], label=clean_label, color=c, linewidth=2.5, alpha=0.85)
            
            # Accuracy curves for sequence models (exact-match)
            if METHODS[name]["type"] == "seq" and METHODS[name].get("metric") == "accuracy":
                ax2.plot(met["epochs"], met["seq_acc"], label=clean_label, color=c,
                         linewidth=2.5, marker='o', markevery=10, alpha=0.85)
            
            # Token-avg accuracy for sequence models (dashed)
            if "token_acc" in met and len(met["token_acc"]) == len(met["epochs"]) and sum(met["token_acc"]) > 0:
                ax2.plot(met["epochs"], met["token_acc"],
                         label=f"{clean_label} (token-avg)", color=c,
                         linewidth=1.5, linestyle='--', alpha=0.7)
            
            # Bit accuracy for vector models (dotted line)
            if METHODS[name]["type"] == "vector" and "bit_acc" in met:
                ax2.plot(met["epochs"], met["bit_acc"],
                         label=f"{clean_label} (bit acc)", color=c,
                         linewidth=2.0, linestyle=':', alpha=0.8)

        ax1.set_xlabel("Epoch", fontsize=14, fontweight='bold')
        ax1.set_ylabel("Cross Entropy Loss", fontsize=14, fontweight='bold')
        ax1.set_title("Test Loss Convergence", fontsize=16, fontweight='bold')
        ax1.grid(True, linestyle='--', alpha=0.6)
        ax1.legend(fontsize=10, loc='upper right', framealpha=0.9)

        ax2.set_xlabel("Epoch", fontsize=14, fontweight='bold')
        ax2.set_ylabel("Accuracy", fontsize=14, fontweight='bold')
        ax2.set_title("Model Accuracy (Exact / Token / Bitwise)", fontsize=16, fontweight='bold')
        ax2.grid(True, linestyle='--', alpha=0.6)
        ax2.set_ylim(-0.02, 1.02)
        ax2.text(0.02, 0.98, "─ exact match  │  -- token‑avg (seq)  │  : bit acc (vector)",
                 transform=ax2.transAxes, fontsize=9, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        ax2.legend(fontsize=9, loc='lower right', framealpha=0.9, ncol=2)

        plt.suptitle(f"Unified Benchmark – {base}", fontsize=20, fontweight='bold', y=0.98)
        plt.tight_layout()
        main_plot = f"/kaggle/working/plots/{base}_unified_evaluation.png"
        plt.savefig(main_plot, dpi=600, bbox_inches='tight')
        plt.close()
        print(f"💾 Saved: {main_plot}")

        # ----- separate prefix plot for variant 9 -----
        prefix_data = all_metrics.get("9_Transformer_PrefixTest")
        if prefix_data and "prefix_lens" in prefix_data:
            plt.figure(figsize=(24, 10))
            plt.plot(prefix_data["prefix_lens"], prefix_data["prefix_accs"],
                     marker='s', color='#2b5c8f', linewidth=2.5, markersize=8)
            plt.xlabel("Correct Prefix Tokens (k)", fontsize=12, fontweight='bold')
            plt.ylabel("Continuation Token Accuracy", fontsize=12, fontweight='bold')
            plt.title(f"Prefix Continuation Profile – {base}\n(Model: Transformer_PrefixTest)",
                      fontsize=14, fontweight='bold')
            plt.grid(True, linestyle=':', alpha=0.8)
            plt.xticks(prefix_data["prefix_lens"])
            plt.ylim(-0.05, 1.05)
            plt.tight_layout()
            prefix_plot = f"/kaggle/working/plots/{base}_prefix_accuracy.png"
            plt.savefig(prefix_plot, dpi=600)
            plt.close()
            print(f"💾 Saved: {prefix_plot}")

if __name__ == "__main__":
    main()
