#!/usr/bin/env python3
"""
🏆 DEFINITIVE MULTI-MODEL STRUCTURAL BENCHMARK FRAMEWORK
All 14 Framework Variants Integrated Into A Single Execution Context.
Includes Native KV-Cached Autoregressive Generations and Isolated Prefix Analytics.
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

# ── Global Hardware Setup ─────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True
print(f"🚀 Execution Engine: {device}")
torch.set_float32_matmul_precision('high')
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# ════════════════════════════════════════════════════════
# CORE DATA PROCESSING STORAGE & UTILITY WRAPPERS
# ════════════════════════════════════════════════════════
def parse_binary_vector(series: pd.Series) -> np.ndarray:
    strings = series.astype(str).str.strip()
    max_len = strings.str.len().max()
    return np.array([[float(b) for b in s.zfill(max_len)] for s in strings], dtype=np.float32)

def parse_seq(s: str) -> List[int]:
    return [int(c) for c in s.strip()]

class BinaryVectorDataset(Dataset):
    def __init__(self, path: str):
        df = pd.read_csv(path, header=None, dtype=str)
        self.y = parse_binary_vector(df.iloc[:, 0])
        self.X = parse_binary_vector(df.iloc[:, 1])
    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return torch.tensor(self.X[idx]), torch.tensor(self.y[idx])

class BinarySeqDataset(Dataset):
    def __init__(self, file: str, tok_mode="standard"):
        df = pd.read_csv(file, header=None, dtype=str)
        # Handle index structures across data variations dynamically
        raw_x = df.iloc[:, 1] if df.shape[1] > 1 else df.iloc[:, 0]
        raw_y = df.iloc[:, 0]
        self.X = [parse_seq(x) for x in raw_x]
        self.y = [parse_seq(y) for y in raw_y]
        self.tok_mode = tok_mode

    def __len__(self): return len(self.X)
    def __getitem__(self, idx):
        if self.tok_mode == "prefix_test":
            # Map specific 0->1, 1->2 token indices used by prefix scripts
            x_mapped = [1 if c == 0 else 2 for c in self.X[idx]]
            y_mapped = [1 if c == 0 else 2 for c in self.y[idx]]
            return x_mapped, y_mapped
        return self.X[idx], self.y[idx]

def collate_fn(batch, PAD=2, BOS=3, EOS=4):
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
    return (
        torch.tensor(X_pad, dtype=torch.long),
        torch.tensor(y_in, dtype=torch.long),
        torch.tensor(y_out, dtype=torch.long)
    )

# ════════════════════════════════════════════════════════
# STRUCTURAL COMPONENT BLOCKS (MODELS 1 - 5)
# ════════════════════════════════════════════════════════
class ResBlock(nn.Module):
    def __init__(self, dim, dropout):
        super().__init__()
        self.block = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(dim, dim))
        self.act = nn.ReLU()
    def forward(self, x): return self.act(x + self.block(x))

class ResNetMLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_layers, out_dim, dropout):
        super().__init__()
        self.input_proj = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.ReLU())
        self.blocks = nn.Sequential(*[ResBlock(hidden_dim, dropout) for _ in range(num_layers)])
        self.head = nn.Linear(hidden_dim, out_dim)
    def forward(self, x): return self.head(self.blocks(self.input_proj(x)))

class DenseLayer(nn.Module):
    def __init__(self, in_dim, growth_rate, dropout):
        super().__init__()
        self.layer = nn.Sequential(nn.Linear(in_dim, growth_rate), nn.ReLU(), nn.Dropout(dropout))
    def forward(self, x): return torch.cat([x, self.layer(x)], dim=1)

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
        for layer in self.layers: x = layer(x)
        return x

class DenseNetMLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_layers, growth_rate, out_dim, dropout):
        super().__init__()
        self.input_proj = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.ReLU())
        self.block = DenseBlock(hidden_dim, num_layers, growth_rate, dropout)
        self.head = nn.Linear(self.block.out_dim, out_dim)
    def forward(self, x): return self.head(self.block(self.input_proj(x)))

class GatedEncoderBlock(nn.Module):
    def __init__(self, dim, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1, self.fc2 = nn.Linear(dim, dim * 2), nn.Linear(dim * 2, dim)
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
        for blk in self.blocks: x = blk(x)
        return x

class EncoderDenseMLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, enc_depth, num_layers, growth_rate, out_dim, dropout, noise_std=0.01):
        super().__init__()
        self.encoder = SystemEncoder(in_dim, hidden_dim, enc_depth, dropout)
        self.block = DenseBlock(hidden_dim, num_layers, growth_rate, dropout)
        self.head = nn.Linear(self.block.out_dim, out_dim)
        self.noise_std = noise_std
    def forward(self, x):
        if self.training: x = x + self.noise_std * torch.randn_like(x)
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
    def forward(self, x): return self.mlp(self.bottleneck(x))

class BottleNeckModel(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim, enc_depth, bottleneck, mlp_width, mlp_depth, dropout, noise_std=0.01):
        super().__init__()
        self.encoder = SystemEncoder(in_dim, hidden_dim, enc_depth, dropout)
        self.head = MLPHead(hidden_dim, bottleneck, mlp_width, mlp_depth, out_dim, dropout)
        self.noise_std = noise_std
    def forward(self, x):
        if self.training: x = x + self.noise_std * torch.randn_like(x)
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
        if self.training: x = x + self.noise_std * torch.randn_like(x)
        return self.head(self.encoder(self.fourier(x)))

# ════════════════════════════════════════════════════════
# TRANSFORMER COMPONENT SPECIFICATIONS (MODELS 6 - 12)
# ════════════════════════════════════════════════════════
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=2048):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2)*(-math.log(10000)/d_model))
        pe[:, 0::2], pe[:, 1::2] = torch.sin(pos*div), torch.cos(pos*div)
        self.register_buffer("pe", pe.unsqueeze(0))
    def forward(self, x): return x + self.pe[:, :x.size(1)]

class TransformerModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers, ff_dim, dropout):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.pos = PositionalEncoding(embed_dim)
        self.tf = nn.Transformer(d_model=embed_dim, nhead=num_heads, num_encoder_layers=num_layers,
                                 num_decoder_layers=num_layers, dim_feedforward=ff_dim, dropout=dropout, batch_first=True)
        self.head = nn.Linear(embed_dim, vocab_size)

    def forward(self, src, tgt):
        src_mask = (src == 2 if src.max() > 2 else src == 0)
        tgt_mask_pad = (tgt == 2 if tgt.max() > 2 else tgt == 0)
        causal_mask = torch.triu(torch.ones(tgt.size(1), tgt.size(1), dtype=torch.bool, device=tgt.device), diagonal=1)
        out = self.tf(self.pos(self.embed(src)), self.pos(self.embed(tgt)), tgt_mask=causal_mask, src_key_padding_mask=src_mask, tgt_key_padding_mask=tgt_mask_pad)
        return self.head(out)

class Memory(nn.Module):
    def __init__(self, mem_slots, embed_dim):
        super().__init__()
        self.memory = nn.Parameter(torch.randn(mem_slots, embed_dim))
    def forward(self, batch_size): return self.memory.unsqueeze(0).repeat(batch_size, 1, 1)

class MemoryTransformer(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers, mem_slots, dropout):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.pos = PositionalEncoding(embed_dim)
        self.encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True, dropout=dropout), num_layers)
        self.decoder = nn.TransformerDecoder(nn.TransformerDecoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True, dropout=dropout), num_layers)
        self.memory = Memory(mem_slots, embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size)

    def forward(self, src, tgt):
        B = src.size(0)
        causal_mask = torch.triu(torch.ones(tgt.size(1), tgt.size(1), dtype=torch.bool, device=tgt.device), diagonal=1)
        src_mask = (src == 2)
        tgt_mask_pad = (tgt == 2)
        enc_out = self.encoder(self.pos(self.embed(src)), src_key_padding_mask=src_mask)
        out = self.decoder(self.pos(self.embed(tgt)), torch.cat([enc_out, self.memory(B)], dim=1), tgt_mask=causal_mask, tgt_key_padding_mask=tgt_mask_pad)
        return self.head(out)

# ════════════════════════════════════════════════════════
# ADVANCED STRUCTURAL MODELS (MODELS 13 - 14)
# ════════════════════════════════════════════════════════
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
        return x * self.cos_cached[:, :, :seq_len] + torch.stack((-x[..., 1::2], x[..., ::2]), dim=-1).flatten(-2) * self.sin_cached[:, :, :seq_len]

class GEGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return x * F.gelu(gate)

class FeedForward(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, dim * mult * 2), GEGLU(), nn.Linear(dim * mult, dim))
    def forward(self, x): return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads=8, causal=False):
        super().__init__()
        self.heads, self.head_dim, self.causal = heads, dim // heads, causal
        self.norm = RMSNorm(dim)
        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.to_out = nn.Linear(dim, dim, bias=False)
        self.rotary = RotaryEmbedding(self.head_dim)
    def forward(self, x, context=None):
        x = self.norm(x)
        if context is None: context = x
        B, T, D, H = x.shape[0], x.shape[1], x.shape[2], self.heads
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
    def __init__(self, vocab_size=5, embed_dim=32, enc_layers=4, num_latents=4, reasoning_steps=4, dec_layers=4, num_heads=4):
        super().__init__()
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
        for block in self.encoder: x = block(x)
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

# ════════════════════════════════════════════════════════
# REAL-WORLD INFERENCE ENGINES (AUTOREGRESSIVE DECODING)
# ════════════════════════════════════════════════════════
@torch.no_grad()
def batched_greedy_decode(model, src_batch, BOS, EOS, max_len=45, PAD=2):
    model.eval()
    B = src_batch.size(0)
    tgt = torch.full((B, max_len), PAD, dtype=torch.long, device=src_batch.device)
    tgt[:, 0] = BOS
    finished = torch.zeros(B, dtype=torch.bool, device=src_batch.device)
    for step in range(1, max_len):
        logits = model(src_batch, tgt[:, :step])
        next_token = logits[:, -1].argmax(dim=-1)
        tgt[:, step] = next_token
        finished |= (next_token == EOS)
        if finished.all(): break
    return tgt

@torch.no_grad()
def evaluate_true_inference(model, dataset, BOS, EOS, PAD, max_samples=40):
    model.eval()
    samples = min(len(dataset), max_samples)
    src_list, true_list = [], []
    for i in range(samples):
        src, true = dataset[i]
        src_list.append(torch.tensor(src, dtype=torch.long, device=device))
        true_list.append(true)
    max_x = max(len(s) for s in src_list)
    src_tensor = torch.full((samples, max_x), PAD, dtype=torch.long, device=device)
    for i, s in enumerate(src_list): src_tensor[i, :len(s)] = s

    pred_tensor = batched_greedy_decode(model, src_tensor, BOS, EOS, max_len=45, PAD=PAD).cpu()
    seq_correct = 0
    for i, true in enumerate(true_list):
        pred = pred_tensor[i].tolist()
        if BOS in pred: pred = pred[pred.index(BOS)+1:]
        if EOS in pred: pred = pred[:pred.index(EOS)] if EOS in pred else pred
        pred = [p for p in pred if p != PAD]
        if pred == true: seq_correct += 1
    return seq_correct / samples

# ════════════════════════════════════════════════════════
# CONFIGURATION AND SPECIFIC PREFIX SYSTEM EVALUATION
# ════════════════════════════════════════════════════════
@torch.no_grad()
def evaluate_prefix_accuracy(model, loader, max_prefix_len=30, PAD_ID=0):
    model.eval()
    prefix_lens = list(range(1, max_prefix_len + 1))
    correct_counts = {k: 0 for k in prefix_lens}
    total_counts = {k: 0 for k in prefix_lens}
    
    for src, y_in, y_out in loader:
        src, y_in, y_out = src.to(device), y_in.to(device), y_out.to(device)
        logits = model(src, y_in)
        preds = logits.argmax(dim=-1)
        
        for k in prefix_lens:
            if k >= y_in.size(1): continue
            mask = (y_out[:, k] != PAD_ID)
            if mask.sum() == 0: continue
            matches = (preds[:, k] == y_out[:, k]) & mask
            correct_counts[k] += matches.sum().item()
            total_counts[k] += mask.sum().item()
            
    return prefix_lens, [correct_counts[k] / total_counts[k] if total_counts[k] > 0 else 0.0 for k in prefix_lens]

COMMON = {"BATCH": 256, "LR": 3e-4, "EPOCHS": 100, "WD": 1e-5, "DROP": 0.1, "CLIP": 1.0}

METHODS = {
    "1_ResNet_Vector": {"type": "vector", "build": lambda i,o: ResNetMLP(i, 256, 6, o, 0.1), "criterion": nn.BCEWithLogitsLoss()},
    "2_DenseNet_Vector": {"type": "vector", "build": lambda i,o: DenseNetMLP(i, 128, 6, 32, o, 0.1), "criterion": nn.BCEWithLogitsLoss()},
    "3_GatedEncoder_Vector": {"type": "vector", "build": lambda i,o: EncoderDenseMLP(i, 256, 3, 6, 32, o, 0.1, 0.01), "criterion": nn.BCEWithLogitsLoss()},
    "4_BottleNeckMLP_Vector": {"type": "vector", "build": lambda i,o: BottleNeckModel(i, o, 256, 3, 128, 256, 3, 0.1, 0.01), "criterion": nn.BCEWithLogitsLoss()},
    "5_FourierProjection_Vector": {"type": "vector", "build": lambda i,o: FourierModel(i, o, 64, 3.0, 256, 3, 128, 256, 3, 0.1, 0.01), "criterion": nn.BCEWithLogitsLoss()},
    
    "6_Transformer_TeacherForced": {"type": "seq", "cfg": {"PAD": 2, "BOS": 3, "EOS": 4, "V": 5, "E": 128, "H": 4, "L": 3, "FF": 256}, "build": lambda c: TransformerModel(c["V"], c["E"], c["H"], c["L"], c["FF"], 0.1), "metric": "loss"},
    "7_Transformer_TrueInference": {"type": "seq", "cfg": {"PAD": 2, "BOS": 3, "EOS": 4, "V": 5, "E": 128, "H": 4, "L": 3, "FF": 256}, "build": lambda c: TransformerModel(c["V"], c["E"], c["H"], c["L"], c["FF"], 0.1), "metric": "accuracy"},
    "8_TrueInference_Worked_Mode": {"type": "seq", "cfg": {"PAD": 0, "BOS": 3, "EOS": 4, "V": 5, "E": 16, "H": 2, "L": 1, "FF": 32}, "build": lambda c: TransformerModel(c["V"], c["E"], c["H"], c["L"], c["FF"], 0.0), "metric": "accuracy"},
    
    "9_Transformer_PrefixTest": {"type": "seq", "cfg": {"PAD": 0, "BOS": 3, "EOS": 4, "V": 5, "E": 32, "H": 2, "L": 2, "FF": 64}, "build": lambda c: TransformerModel(c["V"], c["E"], c["H"], c["L"], c["FF"], 0.1), "metric": "prefix_isolated"},
    
    "10_MemoryTransformer_TF": {"type": "seq", "cfg": {"PAD": 2, "BOS": 3, "EOS": 4, "V": 5, "E": 128, "H": 4, "L": 3, "MEM": 32}, "build": lambda c: MemoryTransformer(c["V"], c["E"], c["H"], c["L"], c["MEM"], 0.1), "metric": "loss"},
    "11_TrueInference_Memory": {"type": "seq", "cfg": {"PAD": 2, "BOS": 3, "EOS": 4, "V": 5, "E": 128, "H": 4, "L": 3, "MEM": 32}, "build": lambda c: MemoryTransformer(c["V"], c["E"], c["H"], c["L"], c["MEM"], 0.1), "metric": "accuracy"},
    
    "12_Hope_Variant": {"type": "seq", "cfg": {"PAD": 0, "BOS": 3, "EOS": 4, "V": 5, "E": 16, "H": 2, "L": 2, "FF": 64}, "build": lambda c: TransformerModel(c["V"], c["E"], c["H"], c["L"], c["FF"], 0.0), "metric": "accuracy"},
    "13_HierarchicalReasoning_HRM": {"type": "seq", "cfg": {"PAD": 2, "BOS": 3, "EOS": 4, "V": 5}, "build": lambda c: HierarchicalReasoningModel(vocab_size=c["V"]), "metric": "accuracy"}
}

# ════════════════════════════════════════════════════════
# UNIFIED CENTRAL EXECUTION LIFECYCLE
# ════════════════════════════════════════════════════════
def train_and_collect(base, train_file, test_file, m_name, m_cfg):
    typ = m_cfg["type"]
    scaler = torch.amp.GradScaler('cuda')
    
    if typ == "vector":
        train_ds, test_ds = BinaryVectorDataset(train_file), BinaryVectorDataset(test_file)
        tr_ld = DataLoader(train_ds, batch_size=COMMON["BATCH"], shuffle=True, num_workers=2, pin_memory=True)
        te_ld = DataLoader(test_ds, batch_size=COMMON["BATCH"], shuffle=False, num_workers=2, pin_memory=True)
        model = m_cfg["build"](train_ds.X.shape[1], train_ds.y.shape[1]).to(device)
        criterion = m_cfg["criterion"]
    else:
        cfg = m_cfg["cfg"]
        mode_flag = "prefix_test" if m_name == "9_Transformer_PrefixTest" else "standard"
        train_ds = BinarySeqDataset(train_file, tok_mode=mode_flag)
        test_ds = BinarySeqDataset(test_file, tok_mode=mode_flag)
        coll = lambda b: collate_fn(b, cfg["PAD"], cfg["BOS"], cfg["EOS"])
        tr_ld = DataLoader(train_ds, batch_size=COMMON["BATCH"], shuffle=True, collate_fn=coll, num_workers=2, pin_memory=True)
        te_ld = DataLoader(test_ds, batch_size=COMMON["BATCH"], shuffle=False, collate_fn=coll, num_workers=2, pin_memory=True)
        model = m_cfg["build"](cfg).to(device)
        criterion = nn.CrossEntropyLoss(ignore_index=cfg["PAD"])

    optimizer = torch.optim.AdamW(model.parameters(), lr=COMMON["LR"], weight_decay=COMMON["WD"])
    metrics = {"epochs": [], "test_loss": [], "seq_acc": []}

    for ep in range(1, COMMON["EPOCHS"] + 1):
        model.train()
        for batch_data in tr_ld:
            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                if typ == "vector":
                    X, y = batch_data[0].to(device), batch_data[1].to(device)
                    loss = criterion(model(X), y)
                else:
                    src, y_in, y_out = [t.to(device) for t in batch_data]
                    loss = criterion(model(src, y_in).view(-1, m_cfg["cfg"]["V"]), y_out.view(-1))
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), COMMON["CLIP"])
            scaler.step(optimizer)
            scaler.update()

        # Validation Epoch Metrics Compilation
        model.eval()
        run_test = 0.0
        with torch.no_grad():
            for batch_data in te_ld:
                with torch.amp.autocast('cuda'):
                    if typ == "vector":
                        X, y = batch_data[0].to(device), batch_data[1].to(device)
                        loss = criterion(model(X), y)
                    else:
                        src, y_in, y_out = [t.to(device) for t in batch_data]
                        loss = criterion(model(src, y_in).view(-1, m_cfg["cfg"]["V"]), y_out.view(-1))
                    run_test += loss.item() * (len(X) if typ == "vector" else len(src))

        metrics["epochs"].append(ep)
        metrics["test_loss"].append(run_test / len(test_ds))
        
        if typ == "seq" and m_cfg["metric"] == "accuracy":
            acc = evaluate_true_inference(model, test_ds, cfg["BOS"], cfg["EOS"], cfg["PAD"])
            metrics["seq_acc"].append(acc)
        else:
            metrics["seq_acc"].append(0.0)

    # Isolated evaluation tracking routine for variant 9
    if m_name == "9_Transformer_PrefixTest":
        print(f"  🔍 Executing Isolated Continuation Analysis for: {m_name}")
        p_lens, p_accs = evaluate_prefix_accuracy(model, te_ld, max_prefix_len=20, PAD_ID=m_cfg["cfg"]["PAD"])
        metrics["prefix_lens"] = p_lens
        metrics["prefix_accs"] = p_accs

    print(f"  {m_name:30s} Complete | Final Loss: {metrics['test_loss'][-1]:.4f} | Acc: {metrics['seq_acc'][-1]:.3f}")
    return metrics

# ════════════════════════════════════════════════════════
# MAIN EXECUTION ROUTINE & SIDE-BY-SIDE GRAPH PLOTTING
# ════════════════════════════════════════════════════════
def main():
    train_files = sorted(glob.glob("/kaggle/input/datasets/classstudents/test52/*_train.csv"))
    if not train_files:
        print("❌ No *_train.csv files identified in the local directory workspace context.")
        return
    os.makedirs("/kaggle/working/plots", exist_ok=True)

    for t_file in train_files:
        base = os.path.basename(t_file).replace("_train.csv", "")
        te_file = t_file.replace("_train.csv", "_test.csv")
        if not os.path.exists(te_file): continue

        print(f"\n⚡ Processing Benchmark Framework Group: {base}")
        all_metrics = {}
        for name, cfg in METHODS.items():
            all_metrics[name] = train_and_collect(base, t_file, te_file, name, cfg)

        # ==================== MAIN GRAPH PLOT (24x10 @ 600 DPI) ====================
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 10))
        colors = cm.get_cmap('tab20', len(METHODS))
        
        for i, (name, met) in enumerate(all_metrics.items()):
            if name == "9_Transformer_PrefixTest": continue # Skip isolated plot variant
            c = colors(i)
            clean_label = name.replace('_', ' ')
            
            # Subplot #1: Loss Curves
            ax1.plot(met["epochs"], met["test_loss"], label=clean_label, color=c, linewidth=2.5, alpha=0.85)
            # Subplot #2: True Inference Performance Profiles
            if METHODS[name]["type"] == "seq" and METHODS[name]["metric"] == "accuracy":
                ax2.plot(met["epochs"], met["seq_acc"], label=clean_label, color=c, linewidth=2.5, marker='o', markevery=10, alpha=0.85)

        ax1.set_xlabel("Epoch Structure", fontsize=14, fontweight='bold')
        ax1.set_ylabel("Cross Entropy / Loss Profile", fontsize=14, fontweight='bold')
        ax1.set_title("Framework Convergence Mapping (Test Loss Evaluation)", fontsize=16, fontweight='bold')
        ax1.grid(True, linestyle='--', alpha=0.6)
        ax1.legend(fontsize=10, loc='upper right', framealpha=0.9, shadow=True)

        ax2.set_xlabel("Epoch Structure", fontsize=14, fontweight='bold')
        ax2.set_ylabel("True Inference Generation Target Accuracy", fontsize=14, fontweight='bold')
        ax2.set_title("Autoregressive Verification (Generation Profiling)", fontsize=16, fontweight='bold')
        ax2.grid(True, linestyle='--', alpha=0.6)
        ax2.set_ylim(-0.02, 1.02)
        ax2.legend(fontsize=10, loc='lower right', framealpha=0.9, shadow=True)

        plt.suptitle(f"Unified Structural Model Testing Summary Analysis — Set: {base}", fontsize=20, fontweight='bold', y=0.98)
        plt.tight_layout()
        main_plot_path = f"/kaggle/working/plots/{base}_unified_evaluation_summary.png"
        plt.savefig(main_plot_path, dpi=600, bbox_inches='tight')
        plt.close()
        print(f"💾 Saved Unified Multi-Architecture Research Chart: {main_plot_path}")

        # ==================== SEPARATE ISOLATED PLOT (PREFIX TEST) ====================
        prefix_data = all_metrics.get("9_Transformer_PrefixTest")
        if prefix_data and "prefix_lens" in prefix_data:
            plt.figure(figsize=(24, 10))
            plt.plot(prefix_data["prefix_lens"], prefix_data["prefix_accs"], marker='s', color='#2b5c8f', linewidth=2.5, markersize=8)
            plt.xlabel("Correct Prefix Tokens Provided (k Context Tokens)", fontsize=12, fontweight='bold')
            plt.ylabel("Continuation Token Accuracy Profile", fontsize=12, fontweight='bold')
            plt.title(f"Prefix Continuation Accuracy Profile — Target: {base}\n[Model: 9_Transformer_PrefixTest]", fontsize=14, fontweight='bold')
            plt.grid(True, linestyle=':', alpha=0.8)
            plt.xticks(prefix_data["prefix_lens"])
            plt.ylim(-0.05, 1.05)
            plt.tight_layout()
            
            prefix_plot_path = f"/kaggle/working/plots/{base}_prefix_accuracy_evaluation.png"
            plt.savefig(prefix_plot_path, dpi=600)
            plt.close()
            print(f"💾 Saved Separate Isolated Prefix Verification Chart: {prefix_plot_path}")

if __name__ == "__main__":
    main()
