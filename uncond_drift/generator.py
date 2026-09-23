"""Text drifting generator: continuous Gaussian z -> continuous embedding seq.

One-step, unconditional, no timestep. Outputs continuous embeddings (B, T, 768)
that decode to tokens via cosine/L2-NN.

Backbone is a pre-LN transformer matching nn.TransformerEncoderLayer(norm_first=
True) EXACTLY (LayerNorm, ReLU FFN, learned pos-emb), with ONE addition: QK-norm
(RMSNorm on per-head q and k before attention). QK-norm keeps attention logits
bounded at depth, which is the first stabilizer needed to scale past ~6 layers
without the deep stack oversmoothing / collapsing.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import TextDriftConfig
from seq_drifting_common.layers import RMSNorm


class QKNormAttention(nn.Module):
    """Multi-head self-attention with QK-norm (RMSNorm on per-head q, k)."""

    def __init__(self, dim: int, nhead: int):
        super().__init__()
        assert dim % nhead == 0
        self.nhead = nhead
        self.hd = dim // nhead
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.q_norm = RMSNorm(self.hd)
        self.k_norm = RMSNorm(self.hd)

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.nhead, self.hd).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]            # (B, H, T, hd)
        q = self.q_norm(q)                          # QK-norm
        k = self.k_norm(k)
        attn = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.hd))
        attn = attn.softmax(dim=-1)
        out = attn @ v                              # (B, H, T, hd)
        out = out.transpose(1, 2).reshape(B, T, C)
        return self.proj(out)


class Block(nn.Module):
    """Pre-LN block + QK-norm + LayerScale (per-channel residual gate, tiny init).

    LayerScale (ls1, ls2 init ~1e-4) makes every block start as ~identity, so the
    input (z) flows through the whole deep stack un-smoothed at init -> deep stacks
    no longer collapse to a constant. The gates grow during training.
    """

    def __init__(self, dim: int, nhead: int, ffn: int, ls_init: float = 1e-4):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = QKNormAttention(dim, nhead)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, ffn), nn.ReLU(), nn.Linear(ffn, dim))
        self.ls1 = nn.Parameter(torch.full((dim,), ls_init))
        self.ls2 = nn.Parameter(torch.full((dim,), ls_init))

    def forward(self, x):
        x = x + self.ls1 * self.attn(self.norm1(x))
        x = x + self.ls2 * self.mlp(self.norm2(x))
        return x


class TextDriftGenerator(nn.Module):
    def __init__(self, cfg: TextDriftConfig, wte: torch.Tensor = None):
        super().__init__()
        self.T = cfg.seq_len
        self.D = cfg.d_model

        self.noise_proj = nn.Linear(cfg.noise_dim, self.D * self.T)
        self.pos_emb = nn.Parameter(torch.randn(1, self.T, self.D) * 0.02)
        self.blocks = nn.ModuleList(
            [Block(self.D, cfg.nhead, cfg.ffn_dim) for _ in range(cfg.num_layers)]
        )
        self.out_proj = nn.Linear(self.D, self.D)

        nn.init.xavier_uniform_(self.noise_proj.weight)
        nn.init.zeros_(self.noise_proj.bias)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z (B, noise_dim) -> continuous embeddings (B, T, D)."""
        B = z.shape[0]
        z_seq = self.noise_proj(z).view(B, self.T, self.D)
        h = self.pos_emb + z_seq
        for blk in self.blocks:
            h = blk(h)
        return self.out_proj(h)

    def sample_z(self, n: int, noise_dim: int, temp: float, device) -> torch.Tensor:
        return torch.randn(n, noise_dim, device=device) * temp
