"""Conditional generator: (query embeddings, z) -> response embeddings, ONE forward
pass, non-autoregressive.

Query tokens (their frozen Qwen input embeddings) are projected into the generator
width and concatenated with resp_len learned response slots (pos-emb + noise). Full
(non-causal) self-attention lets every response slot read the whole query and the
other slots at once; the response slots are read out and projected back to the Qwen
embedding dim. Left-padded query positions are masked out of attention.

Reuses the parent generator's stabilisers (QK-norm, LayerScale, pre-LN); the only
addition is a key-padding mask on attention.
"""
import math

import torch
import torch.nn as nn

from seq_drifting_common.layers import RMSNorm


class CondAttention(nn.Module):
    def __init__(self, dim: int, nhead: int):
        super().__init__()
        assert dim % nhead == 0
        self.nhead, self.hd = nhead, dim // nhead
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.q_norm = RMSNorm(self.hd)
        self.k_norm = RMSNorm(self.hd)

    def forward(self, x, key_mask):
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.nhead, self.hd).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]                       # (B, H, T, hd)
        q, k = self.q_norm(q), self.k_norm(k)
        attn = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.hd))  # (B, H, T, T)
        if key_mask is not None:                              # mask padded KEYS (query pads)
            attn = attn.masked_fill(~key_mask[:, None, None, :], float("-inf"))
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, T, C)
        return self.proj(out)


class QueryCrossAttention(nn.Module):
    """UNGATED cross-attention: response slots (Q) read the query tokens (K/V).

    Why it exists: in the main stack the query can only reach the response through
    self-attention, which is gated by LayerScale (ls1 ~ 1e-4). In the UNconditional
    generator that tiny gate is a fix -- z is injected DIRECTLY at layer 0, so
    near-identity blocks let z through un-smoothed. Here z is still direct but the
    QUERY is not, so the same gate suppresses the conditioning: at init the generator
    is a query-INDEPENDENT function and settles into "emit the per-position marginal
    of the gold responses" (every query producing the same 'The total number of ...').

    This layer keeps attention (each response slot computes its OWN weights over the
    query tokens and takes the weighted sum of their values, so token-level structure
    -- the specific numbers and entities -- survives, unlike a mean-pool) but writes
    back with NO LayerScale, so the query is effective from step 0.
    """

    def __init__(self, dim: int, nhead: int, h_in: int):
        super().__init__()
        assert dim % nhead == 0
        self.nhead, self.hd = nhead, dim // nhead
        self.norm_q = nn.LayerNorm(dim)
        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(h_in, dim)
        self.to_v = nn.Linear(h_in, dim)
        self.proj = nn.Linear(dim, dim)
        self.q_norm = RMSNorm(self.hd)
        self.k_norm = RMSNorm(self.hd)

    def forward(self, r, query_emb, query_mask):
        """r (B, Lr, D) response slots; query_emb (B, Lq, H); query_mask (B, Lq) bool."""
        B, Lr, D = r.shape
        Lq = query_emb.shape[1]
        q = self.to_q(self.norm_q(r)).view(B, Lr, self.nhead, self.hd).transpose(1, 2)
        k = self.to_k(query_emb).view(B, Lq, self.nhead, self.hd).transpose(1, 2)
        v = self.to_v(query_emb).view(B, Lq, self.nhead, self.hd).transpose(1, 2)
        q, k = self.q_norm(q), self.k_norm(k)
        attn = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.hd))     # (B, H, Lr, Lq)
        attn = attn.masked_fill(~query_mask[:, None, None, :], float("-inf"))
        attn = attn.softmax(dim=-1)                                       # weights
        # unconditional warmup: a sample with an ALL-masked query (query_mask all False) gives an
        # all -inf row -> softmax = nan; zero it so the query contributes nothing (pure z gen).
        attn = torch.nan_to_num(attn)
        out = (attn @ v).transpose(1, 2).reshape(B, Lr, D)               # weighted sum of V
        return self.proj(out)


class CondBlock(nn.Module):
    def __init__(self, dim, nhead, ffn, ls_init: float = 1e-4):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = CondAttention(dim, nhead)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, ffn), nn.ReLU(), nn.Linear(ffn, dim))
        self.ls1 = nn.Parameter(torch.full((dim,), ls_init))
        self.ls2 = nn.Parameter(torch.full((dim,), ls_init))

    def forward(self, x, key_mask):
        x = x + self.ls1 * self.attn(self.norm1(x), key_mask)
        x = x + self.ls2 * self.mlp(self.norm2(x))
        return x


class CondDriftGenerator(nn.Module):
    def __init__(self, cfg, embed_dim: int):
        super().__init__()
        self.Lq, self.Lr = cfg.query_len, cfg.resp_len
        self.D = cfg.d_model
        self.H = embed_dim

        self.in_query = nn.Linear(self.H, self.D)
        self.noise_proj = nn.Linear(cfg.noise_dim, self.D * self.Lr)
        self.q_pos = nn.Parameter(torch.randn(1, self.Lq, self.D) * 0.02)
        self.r_pos = nn.Parameter(torch.randn(1, self.Lr, self.D) * 0.02)
        self.seg = nn.Parameter(torch.randn(2, self.D) * 0.02)         # [query, response]
        self.query_cross = (QueryCrossAttention(self.D, cfg.nhead, self.H)
                            if cfg.query_cross_attn else None)
        self.blocks = nn.ModuleList(
            [CondBlock(self.D, cfg.nhead, cfg.ffn_dim, ls_init=cfg.ls_init)
             for _ in range(cfg.num_layers)])
        self.out_proj = nn.Linear(self.D, self.H)

        nn.init.xavier_uniform_(self.in_query.weight); nn.init.zeros_(self.in_query.bias)
        nn.init.xavier_uniform_(self.noise_proj.weight); nn.init.zeros_(self.noise_proj.bias)
        nn.init.xavier_uniform_(self.out_proj.weight); nn.init.zeros_(self.out_proj.bias)

    def forward(self, query_emb, query_mask, z):
        """query_emb (B, Lq, H), query_mask (B, Lq) bool, z (B, noise) -> (B, Lr, H)."""
        B = z.shape[0]
        query_mask = query_mask.to(query_emb.device)
        q = self.in_query(query_emb) + self.q_pos + self.seg[0]        # (B, Lq, D)
        r = (self.r_pos + self.noise_proj(z).view(B, self.Lr, self.D)
             + self.seg[1])                                            # (B, Lr, D)
        if self.query_cross is not None:                              # ungated: no LayerScale
            r = r + self.query_cross(r, query_emb, query_mask)
        h = torch.cat([q, r], dim=1)                                   # (B, Lq+Lr, D)
        key_mask = torch.cat(
            [query_mask.to(h.device), torch.ones(B, self.Lr, dtype=torch.bool, device=h.device)],
            dim=1)
        for blk in self.blocks:
            h = blk(h, key_mask)
        return self.out_proj(h[:, self.Lq:, :])                        # (B, Lr, H)

    def sample_z(self, n, noise_dim, temp, device):
        return torch.randn(n, noise_dim, device=device) * temp
