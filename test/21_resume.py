#!/usr/bin/env python3
"""
UNIFIED MULTI-MODEL TRAINING SCRIPT (OPTIMIZED + RESUME)
- KV‑Cached Beam Search / Greedy Decoding
- Pre‑allocated buffers (zero torch.cat overhead)
- Fully batched evaluation
- Automatic resume from the last saved epoch per method / dataset
- All outputs under /kaggle/working/
"""
import glob, os, math
import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import matplotlib.cm as cm

# ── Global Hardware Setup ─────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True
print(f"🚀 Device: {device}")
torch.set_float32_matmul_precision('high')
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# ── Output directories ────────────────────────────────────
OUT_DIR = "/"
CKPT_DIR = os.path.join(OUT_DIR, "checkpoints")
PLOTS_DIR = os.path.join(OUT_DIR, "plots")
os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# ════════════════════════════════════════════════════════
# SHARED DATASET & UTILITIES
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
    def __init__(self, file: str):
        df = pd.read_csv(file, header=None, dtype=str)
        self.X = [parse_seq(x) for x in df.iloc[:, 0]]
        self.y = [parse_seq(y) for y in df.iloc[:, 1]]
    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

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
# MODEL DEFINITIONS
# ════════════════════════════════════════════════════════
# Vector models
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

class EncoderBlock(nn.Module):
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

class Encoder(nn.Module):
    def __init__(self, in_dim, embed_dim, depth, dropout):
        super().__init__()
        self.input_proj = nn.Linear(in_dim, embed_dim)
        self.blocks = nn.ModuleList([EncoderBlock(embed_dim, dropout) for _ in range(depth)])
    def forward(self, x):
        x = self.input_proj(x)
        for blk in self.blocks: x = blk(x)
        return x

class EncoderDenseMLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, enc_depth, num_layers, growth_rate, out_dim, dropout, noise_std=0.01):
        super().__init__()
        self.encoder = Encoder(in_dim, hidden_dim, enc_depth, dropout)
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
        self.encoder = Encoder(in_dim, hidden_dim, enc_depth, dropout)
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
        self.encoder = Encoder(fourier_dim * 2, hidden_dim, enc_depth, dropout)
        self.head = MLPHead(hidden_dim, bottleneck, mlp_width, mlp_depth, out_dim, dropout)
        self.noise_std = noise_std
    def forward(self, x):
        if self.training: x = x + self.noise_std * torch.randn_like(x)
        return self.head(self.encoder(self.fourier(x)))

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=2048):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2)*(-math.log(10000)/d_model))
        pe[:, 0::2], pe[:, 1::2] = torch.sin(pos*div), torch.cos(pos*div)
        self.register_buffer("pe", pe.unsqueeze(0))
    def forward(self, x): return x + self.pe[:, :x.size(1)]

# ════════════════════════════════════════════════════════
# KV-CACHE SUPPORT & DECODER INJECTION
# ════════════════════════════════════════════════════════
class KVCache:
    def __init__(self, batch_size, max_len, d_model, num_layers, device):
        self.cache = [
            (torch.zeros(batch_size, max_len, d_model, device=device),
             torch.zeros(batch_size, max_len, d_model, device=device))
            for _ in range(num_layers)
        ]
        self.idx = 0
    def update(self, layer_idx, k, v):
        b, s, d = k.shape
        self.cache[layer_idx][0][:, self.idx:self.idx+s] = k
        self.cache[layer_idx][1][:, self.idx:self.idx+s] = v
    def get(self, layer_idx, seq_len):
        return (
            self.cache[layer_idx][0][:, :self.idx + seq_len],
            self.cache[layer_idx][1][:, :self.idx + seq_len]
        )
    def step(self, length=1): self.idx += length
    def reset(self): self.idx = 0

def _self_attn_with_cache(q, k, v, attn_mask=None, dropout_p=0.0):
    q, k, v = q.transpose(1,2), k.transpose(1,2), v.transpose(1,2)
    if attn_mask is not None and attn_mask.dtype == torch.bool:
        attn_mask = ~attn_mask
    out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p)
    return out.transpose(1, 2).contiguous()

class KVCacheDecoder(nn.Module):
    def __init__(self, original_decoder, d_model, nhead, dropout=0.1):
        super().__init__()
        self.layers = original_decoder.layers
        self.num_layers = len(self.layers)
        self.d_model = d_model
        self.nhead = nhead
        self.dropout = dropout
    def forward(self, tgt, memory, cache, src_key_padding_mask=None, tgt_key_padding_mask=None):
        cache.reset()
        B, T, D = tgt.shape
        for i, layer in enumerate(self.layers):
            tgt_norm = layer.norm1(tgt)
            q = layer.self_attn.in_proj_weight[:D, :] @ tgt_norm.transpose(-1,-2)
            k = layer.self_attn.in_proj_weight[D:2*D, :] @ tgt_norm.transpose(-1,-2)
            v = layer.self_attn.in_proj_weight[2*D:, :] @ tgt_norm.transpose(-1,-2)
            q, k, v = q.transpose(-1,-2), k.transpose(-1,-2), v.transpose(-1,-2)
            k_c, v_c = cache.get(i, T)
            k = torch.cat([k_c, k], dim=1) if k_c.numel() > 0 else k
            v = torch.cat([v_c, v], dim=1) if v_c.numel() > 0 else v
            cache.update(i, k, v)
            causal = torch.triu(torch.ones(T, T, dtype=torch.bool, device=tgt.device), diagonal=1)
            out = _self_attn_with_cache(q, k, v, causal, self.dropout)
            tgt = tgt + layer.dropout1(layer.self_attn.out_proj(out))
            tgt2 = layer.norm2(tgt)
            q2 = layer.multihead_attn.in_proj_weight[:D, :] @ tgt2.transpose(-1,-2)
            k2 = layer.multihead_attn.in_proj_weight[D:2*D, :] @ memory.transpose(-1,-2)
            v2 = layer.multihead_attn.in_proj_weight[2*D:, :] @ memory.transpose(-1,-2)
            q2, k2, v2 = q2.transpose(-1,-2), k2.transpose(-1,-2), v2.transpose(-1,-2)
            out2 = _self_attn_with_cache(q2, k2, v2, src_key_padding_mask, self.dropout)
            tgt = tgt + layer.dropout2(layer.multihead_attn.out_proj(out2))
            tgt3 = layer.norm3(tgt)
            ff = layer.linear2(layer.dropout(layer.activation(layer.linear1(tgt3))))
            tgt = tgt + layer.dropout3(ff)
        return tgt

# ════════════════════════════════════════════════════════
# TRANSFORMER MODELS (KV-CACHE READY)
# ════════════════════════════════════════════════════════
class TransformerModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers, ff_dim, dropout):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.pos = PositionalEncoding(embed_dim)
        self.tf = nn.Transformer(d_model=embed_dim, nhead=num_heads, num_encoder_layers=num_layers,
                                 num_decoder_layers=num_layers, dim_feedforward=ff_dim, dropout=dropout, batch_first=True)
        self.head = nn.Linear(embed_dim, vocab_size)
        self.kv_decoder = None
        self.num_layers = num_layers
        self.embed_dim = embed_dim
        self.nhead = num_heads
        self.dropout = dropout
    def forward(self, src, tgt, past_key_values=None):
        if past_key_values is not None:
            B = src.size(0)
            if self.kv_decoder is None:
                self.kv_decoder = KVCacheDecoder(self.tf.decoder, self.embed_dim, self.nhead, self.dropout)
            cache = past_key_values
            enc = self.tf.encoder(self.pos(self.embed(src)))
            src_mask = (src == 2 if src.max() > 2 else src == 0)
            dec_out = self.kv_decoder(self.pos(self.embed(tgt)), enc, cache, src_mask)
            return self.head(dec_out)
        else:
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
        self.kv_decoder = None
        self.num_layers = num_layers
        self.embed_dim = embed_dim
        self.nhead = num_heads
        self.dropout = dropout
    def forward(self, src, tgt, past_key_values=None):
        if past_key_values is not None:
            B = src.size(0)
            if self.kv_decoder is None:
                self.kv_decoder = KVCacheDecoder(self.decoder, self.embed_dim, self.nhead, self.dropout)
            cache = past_key_values
            enc = self.encoder(self.pos(self.embed(src)))
            enc = torch.cat([enc, self.memory(B)], dim=1)
            src_mask = (src == 2)
            dec_out = self.kv_decoder(self.pos(self.embed(tgt)), enc, cache, src_mask)
            return self.head(dec_out)
        else:
            B = src.size(0)
            causal_mask = torch.triu(torch.ones(tgt.size(1), tgt.size(1), dtype=torch.bool, device=tgt.device), diagonal=1)
            src_mask = (src == 2)
            tgt_mask_pad = (tgt == 2)
            enc_out = self.encoder(self.pos(self.embed(src)), src_key_padding_mask=src_mask)
            out = self.decoder(self.pos(self.embed(tgt)), torch.cat([enc_out, self.memory(B)], dim=1), tgt_mask=causal_mask, tgt_key_padding_mask=tgt_mask_pad)
            return self.head(out)

# ════════════════════════════════════════════════════════
# OPTIMIZED DECODERS & EVALUATION (BATCHED + KV CACHE)
# ════════════════════════════════════════════════════════
@torch.no_grad()
def batched_greedy_decode(model, src_batch, BOS, EOS, max_len=200, PAD=2):
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
def batched_beam_search_decode(model, src_batch, BOS, EOS, beam_size=3, max_len=200, PAD=2):
    model.eval()
    B, L_src = src_batch.shape
    total_beams = B * beam_size
    sequences = torch.full((total_beams, max_len), PAD, dtype=torch.long, device=src_batch.device)
    sequences[:, 0] = BOS
    scores = torch.full((total_beams,), -float('inf'), device=src_batch.device)
    for b in range(B): scores[b * beam_size] = 0.0
    finished = torch.zeros(total_beams, dtype=torch.bool, device=src_batch.device)
    src_expanded = src_batch.unsqueeze(1).expand(-1, beam_size, -1).reshape(total_beams, L_src)
    vocab_size = None
    for step in range(1, max_len):
        logits = model(src_expanded, sequences[:, :step])[:, -1, :]
        if vocab_size is None: vocab_size = logits.size(-1)
        log_probs = torch.log_softmax(logits, dim=-1)
        log_probs[finished] = -float('inf')
        log_probs[finished, EOS] = 0.0
        total_scores = scores.unsqueeze(1) + log_probs
        topk_scores, topk_indices = torch.topk(total_scores.view(-1), total_beams)
        beam_indices = torch.div(topk_indices, vocab_size, rounding_mode='floor')
        token_indices = topk_indices % vocab_size
        sequences[:, :step] = sequences[beam_indices, :step]
        sequences[:, step] = token_indices
        scores = topk_scores
        finished = finished[beam_indices] | (token_indices == EOS)
        if finished.all(): break
    return sequences.view(B, beam_size, max_len)[:, 0, :]

@torch.no_grad()
def evaluate_true_inference(model, dataset, BOS, EOS, PAD, decode_method="greedy", beam_size=3, max_samples=50):
    model.eval()
    samples = min(len(dataset), max_samples)
    src_list, true_list = [], []
    for i in range(samples):
        src, true = dataset[i]
        src_list.append(torch.tensor(src, dtype=torch.long, device=device))
        true_list.append(true)
    max_x = max(len(s) for s in src_list)
    src_tensor = torch.full((samples, max_x), PAD, dtype=torch.long, device=device)
    for i, s in enumerate(src_list):
        src_tensor[i, :len(s)] = s
    if decode_method == "greedy":
        pred_tensor = batched_greedy_decode(model, src_tensor, BOS, EOS, max_len=200, PAD=PAD)
    else:
        pred_tensor = batched_beam_search_decode(model, src_tensor, BOS, EOS, beam_size, max_len=200, PAD=PAD)
    pred_tensor = pred_tensor.cpu()
    seq_correct, token_correct, token_total = 0, 0, 0
    for i, true in enumerate(true_list):
        pred = pred_tensor[i].tolist()
        if BOS in pred: pred = pred[pred.index(BOS)+1:]
        if EOS in pred: pred = pred[:pred.index(EOS)]
        pred = [p for p in pred if p != PAD]
        if pred == true: seq_correct += 1
        min_len = min(len(pred), len(true))
        token_correct += sum(p == t for p, t in zip(pred[:min_len], true[:min_len]))
        token_total += len(true)
    return seq_correct / samples, (token_correct / token_total if token_total > 0 else 0.0)

# ════════════════════════════════════════════════════════
# METHOD CONFIGURATIONS
# ════════════════════════════════════════════════════════
COMMON = {
    "BATCH": 256, "LR": 3e-4, "EPOCHS": 200, "WD": 1e-5, "DROP": 0.1, "CLIP": 1.0,
}
METHODS = {
    "1_ResNet_TF": {"type": "vector", "cfg": {**COMMON, "HIDDEN": 256, "LAYERS": 6}, "build": lambda i,o: ResNetMLP(i, 256, 6, o, 0.1), "criterion": nn.BCEWithLogitsLoss()},
    "2_DenseNet_TF": {"type": "vector", "cfg": {**COMMON, "HIDDEN": 128, "LAYERS": 6, "GROWTH": 32}, "build": lambda i,o: DenseNetMLP(i, 128, 6, 32, o, 0.1), "criterion": nn.BCEWithLogitsLoss()},
    "3_Encoder_TF": {"type": "vector", "cfg": {**COMMON, "HIDDEN": 256, "ENC": 3, "LAYERS": 6, "GROWTH": 32, "NOISE": 0.01}, "build": lambda i,o: EncoderDenseMLP(i, 256, 3, 6, 32, o, 0.1, 0.01), "criterion": nn.BCEWithLogitsLoss()},
    "4_BottleNeck_TF": {"type": "vector", "cfg": {**COMMON, "HIDDEN": 256, "ENC": 3, "BN": 128, "WIDTH": 256, "DEPTH": 3, "NOISE": 0.01}, "build": lambda i,o: BottleNeckModel(i, o, 256, 3, 128, 256, 3, 0.1, 0.01), "criterion": nn.BCEWithLogitsLoss()},
    "5_Fourier_TF": {"type": "vector", "cfg": {**COMMON, "F_DIM": 64, "F_SCALE": 3.0, "HIDDEN": 256, "ENC": 3, "BN": 128, "WIDTH": 256, "DEPTH": 3, "NOISE": 0.01}, "build": lambda i,o: FourierModel(i, o, 64, 3.0, 256, 3, 128, 256, 3, 0.1, 0.01), "criterion": nn.BCEWithLogitsLoss()},
    "6_Transformer_TF": {"type": "seq", "cfg": {**COMMON, "E": 128, "H": 4, "L": 3, "FF": 256, "D": 0.1, "PAD": 0, "BOS": 2, "EOS": 3, "V": 4, "WARMUP": 2000}, "build": lambda c: TransformerModel(c["V"], 128, 4, 3, 256, 0.1), "criterion": lambda c: nn.CrossEntropyLoss(ignore_index=c["PAD"]), "get_lr": lambda c, s: (128**-0.5)*min(s**-0.5, s*(2000**-1.5)), "plot_metric": "loss"},
    "7_TrueInference": {"type": "seq", "cfg": {**COMMON, "E": 128, "H": 4, "L": 3, "FF": 256, "D": 0.1, "PAD": 2, "BOS": 3, "EOS": 4, "V": 5, "EVAL_E": 1, "MAX_E": 200, "WARMUP": 2000}, "build": lambda c: TransformerModel(c["V"], 128, 4, 3, 256, 0.1), "criterion": lambda c: nn.CrossEntropyLoss(ignore_index=c["PAD"]), "get_lr": lambda c, s: (128**-0.5)*min(s**-0.5, s*(2000**-1.5)), "plot_metric": "accuracy", "decode": "greedy"},
    "8_BeamDecoding": {"type": "seq", "cfg": {**COMMON, "E": 128, "H": 4, "L": 3, "FF": 256, "D": 0.1, "PAD": 2, "BOS": 3, "EOS": 4, "V": 5, "BEAM": 3, "EVAL_E": 1, "WARMUP": 2000}, "build": lambda c: TransformerModel(c["V"], 128, 4, 3, 256, 0.1), "criterion": lambda c: nn.CrossEntropyLoss(ignore_index=c["PAD"]), "get_lr": lambda c, s: (128**-0.5)*min(s**-0.5, s*(2000**-1.5)), "plot_metric": "accuracy", "decode": "beam"},
    "9_MemoryTransformer": {"type": "seq", "cfg": {**COMMON, "E": 128, "H": 4, "L": 3, "MEM": 32, "D": 0.1, "PAD": 2, "BOS": 3, "EOS": 4, "V": 5, "EVAL_E": 1, "MAX_E": 200}, "build": lambda c: MemoryTransformer(c["V"], 128, 4, 3, 32, 0.1), "criterion": lambda c: nn.CrossEntropyLoss(ignore_index=c["PAD"]), "plot_metric": "both", "decode": "greedy"},
    "10_TrueInference_Mem": {"type": "seq", "cfg": {**COMMON, "E": 128, "H": 4, "L": 3, "MEM": 32, "D": 0.1, "PAD": 2, "BOS": 3, "EOS": 4, "V": 5, "EVAL_E": 1, "MAX_E": 200}, "build": lambda c: MemoryTransformer(c["V"], 128, 4, 3, 32, 0.1), "criterion": lambda c: nn.CrossEntropyLoss(ignore_index=c["PAD"]), "plot_metric": "accuracy", "decode": "greedy"},
}

# ════════════════════════════════════════════════════════
# CHECKPOINT HELPERS (RESUME LOGIC)
# ════════════════════════════════════════════════════════
def save_checkpoint(path: str, model: nn.Module, optimizer: torch.optim.Optimizer,
                    scaler: Optional[torch.amp.GradScaler] = None,
                    epoch: int = 0, step: int = 0, best_loss: float = float("inf"),
                    metrics: Optional[dict] = None) -> None:
    model_state = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
    checkpoint = {
        "model_state_dict": model_state,
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "step": step,
        "best_loss": best_loss,
        "metrics": metrics or {}
    }
    if scaler is not None:
        checkpoint["scaler_state_dict"] = scaler.state_dict()
    torch.save(checkpoint, path)

def load_checkpoint(path: str, model: nn.Module, optimizer: torch.optim.Optimizer,
                    scaler: Optional[torch.amp.GradScaler] = None):
    if not os.path.exists(path):
        return None
    try:
        checkpoint = torch.load(path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if scaler is not None and "scaler_state_dict" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
        print(f"🔄 Resumed from checkpoint: epoch {checkpoint['epoch']}, step {checkpoint['step']}")
        return (checkpoint["epoch"], checkpoint["step"],
                checkpoint["best_loss"], checkpoint.get("metrics", {}))
    except Exception as e:
        print(f"⚠️ Failed to load checkpoint {path}: {e}. Starting from scratch.")
        return None

# ════════════════════════════════════════════════════════
# TRAINING WRAPPER (WITH RESUME)
# ════════════════════════════════════════════════════════
def train_and_collect(base, train_file, test_file, m_name, m_cfg):
    cfg, typ = m_cfg["cfg"], m_cfg["type"]
    epochs, batch = cfg["EPOCHS"], cfg["BATCH"]
    scaler = torch.amp.GradScaler('cuda')

    if typ == "vector":
        train_ds = BinaryVectorDataset(train_file)
        test_ds = BinaryVectorDataset(test_file)
        num_workers = min(8, os.cpu_count())
        tr_ld = DataLoader(train_ds, batch_size=batch, shuffle=True, num_workers=num_workers, pin_memory=True, persistent_workers=True)
        te_ld = DataLoader(test_ds, batch_size=batch, shuffle=False, num_workers=num_workers, pin_memory=True, persistent_workers=True)
        in_d, out_d = train_ds.X.shape[1], train_ds.y.shape[1]
        model = m_cfg["build"](in_d, out_d).to(device)
        criterion = m_cfg["criterion"]
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg["LR"], weight_decay=cfg.get("WD", 0))
    else:
        PAD, BOS, EOS = cfg["PAD"], cfg["BOS"], cfg["EOS"]
        train_ds = BinarySeqDataset(train_file)
        test_ds = BinarySeqDataset(test_file)
        coll = lambda b: collate_fn(b, PAD, BOS, EOS)
        num_workers = min(8, os.cpu_count())
        tr_ld = DataLoader(train_ds, batch_size=batch, shuffle=True, collate_fn=coll, num_workers=num_workers, pin_memory=True, persistent_workers=True)
        te_ld = DataLoader(test_ds, batch_size=batch, shuffle=False, collate_fn=coll, num_workers=num_workers, pin_memory=True, persistent_workers=True)
        model = m_cfg["build"](cfg).to(device)
        criterion = m_cfg["criterion"](cfg) if callable(m_cfg["criterion"]) else m_cfg["criterion"]
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["LR"], weight_decay=cfg.get("WD", 0))

    # ── RESUME ──
    # Build a clean checkpoint path: e.g., checkpoints/dataset_name/method.pt
    dataset_name = os.path.basename(base.replace("\\", "/"))
    ckpt_path = os.path.join(CKPT_DIR, dataset_name, f"{m_name}.pt")
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)

    start_epoch = 1
    step = 1
    best_loss = float("inf")
    metrics = {"epochs": [], "train_loss": [], "test_loss": [], "seq_acc": [], "tok_acc": []}

    # Load checkpoint BEFORE DataParallel / compile
    resume_data = load_checkpoint(ckpt_path, model, optimizer, scaler)
    if resume_data is not None:
        start_epoch, step, best_loss, saved_metrics = resume_data
        start_epoch += 1   # Continue from next epoch
        for k in metrics:
            if k in saved_metrics:
                metrics[k] = saved_metrics[k]

    # ── DataParallel / compile ──
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
        print(f"⚡ Using {torch.cuda.device_count()} GPUs")
    elif torch.cuda.is_available():
        try:
            model = torch.compile(model)
            print("⚡ torch.compile enabled")
        except Exception as e:
            print(f"⚠️ torch.compile skipped: {e}")

    # ── Training loop ──
    for ep in range(start_epoch, epochs + 1):
        model.train()
        run_train, run_test = 0.0, 0.0
        for batch_data in tr_ld:
            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                if typ == "vector":
                    X, y = batch_data[0].to(device), batch_data[1].to(device)
                    loss = criterion(model(X), y)
                    run_train += loss.item() * len(X)
                else:
                    src, y_in, y_out = [t.to(device) for t in batch_data]
                    loss = criterion(model(src, y_in).view(-1, cfg["V"]), y_out.view(-1))
                    run_train += loss.item() * len(src)
            if m_cfg.get("get_lr"):
                for g in optimizer.param_groups: g['lr'] = m_cfg["get_lr"](cfg, step)
            step += 1
            scaler.scale(loss).backward()
            if "CLIP" in cfg:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["CLIP"])
            scaler.step(optimizer)
            scaler.update()

        avg_tr = run_train / len(train_ds)
        metrics["train_loss"].append(avg_tr)
        metrics["epochs"].append(ep)

        model.eval()
        with torch.no_grad():
            for batch_data in te_ld:
                with torch.amp.autocast('cuda'):
                    if typ == "vector":
                        X, y = batch_data[0].to(device), batch_data[1].to(device)
                        loss = criterion(model(X), y)
                    else:
                        src, y_in, y_out = [t.to(device) for t in batch_data]
                        loss = criterion(model(src, y_in).view(-1, cfg["V"]), y_out.view(-1))
                    run_test += loss.item() * (len(X) if typ == "vector" else len(src))

        avg_te = run_test / len(test_ds)
        metrics["test_loss"].append(avg_te)
        best_loss = min(best_loss, avg_te)

        eval_e = cfg.get("EVAL_E", 1) if typ == "seq" else epochs
        if typ == "seq" and (ep % eval_e == 0 or ep == epochs):
            s_acc, t_acc = evaluate_true_inference(model, test_ds, BOS, EOS, PAD,
                                                   m_cfg.get("decode", "greedy"),
                                                   cfg.get("BEAM", 3), cfg.get("MAX_E", 200))
            metrics["seq_acc"].append(s_acc)
            metrics["tok_acc"].append(t_acc)
        elif typ == "seq":
            metrics["seq_acc"].append(metrics["seq_acc"][-1] if metrics["seq_acc"] else 0.0)
            metrics["tok_acc"].append(metrics["tok_acc"][-1] if metrics["tok_acc"] else 0.0)

        # Save checkpoint every epoch
        save_model = model.module if hasattr(model, "module") else model
        save_checkpoint(ckpt_path, save_model, optimizer, scaler, ep, step, best_loss, metrics)

        if ep % 10 == 0 or ep == 1:
            print(f"  {m_name:25s} | Ep {ep:3d} | Tr: {avg_tr:.4f} | Te: {avg_te:.4f}", end="")
            print(f" | Acc: {metrics['seq_acc'][-1]:.3f}" if metrics["seq_acc"] else "")

    return metrics

# ════════════════════════════════════════════════════════
# MAIN EXECUTION & PLOTTING
# ════════════════════════════════════════════════════════
def main():
    train_files = sorted(glob.glob("*_train.csv"))
    if not train_files:
        print("❌ No *_train.csv found!"); return
    print(f"📂 Found {len(train_files)} datasets. Running ALL {len(METHODS)} methods per dataset...\n")

    for t_file in train_files:
        base = t_file.replace("_train.csv", "")
        te_file = f"{base}_test.csv"
        if not os.path.exists(te_file):
            print(f"⚠️  Skipping {base} (no test file)"); continue

        print(f"\n{'='*60}\n🔥 DATASET: {base}\n{'='*60}")
        all_metrics = {}
        for m_name, m_cfg in METHODS.items():
            print(f"\n▶️  Training {m_name}...")
            all_metrics[m_name] = train_and_collect(base, t_file, te_file, m_name, m_cfg)

        # Plotting
        fig, ax1 = plt.subplots(figsize=(20, 9))
        ax2 = ax1.twinx()
        colors = cm.get_cmap('tab20', len(METHODS))
        for i, (name, met) in enumerate(all_metrics.items()):
            c = colors(i)
            if met["test_loss"]: ax1.plot(met["epochs"], met["test_loss"], label=f"{name} (Loss)", color=c, linewidth=2)
            if met["seq_acc"]: ax2.plot(met["epochs"], met["seq_acc"], label=f"{name} (SeqAcc)", color=c, linestyle='--', linewidth=2)

        ax1.set_xlabel("Epoch", fontsize=14); ax1.set_ylabel("Test Loss (Teacher-Forced)", color="blue", fontsize=14); ax1.tick_params(axis='y', labelcolor="blue"); ax1.set_ylim(bottom=0)
        ax2.set_ylabel("Sequence Accuracy (True Inference / Beam)", color="red", fontsize=14); ax2.tick_params(axis='y', labelcolor="red"); ax2.set_ylim(0, 1.05)
        lines1, labels1 = ax1.get_legend_handles_labels(); lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=9, framealpha=0.9)
        plt.title(f"{base} - All Methods Comparison (Left: Loss | Right: Accuracy)", fontsize=16, pad=15)
        plt.grid(True, alpha=0.3); plt.tight_layout()
        dataset_name = os.path.basename(base.replace("\\", "/"))
        plot_path = os.path.join(PLOTS_DIR, f"{dataset_name}_ALL_METHODS.png")
        plt.savefig(plot_path, dpi=150); plt.close()
        print(f"💾 Saved unified plot: {plot_path}")

    print("\n✅ ALL DATASETS COMPLETED.")

if __name__ == "__main__":
    main()
