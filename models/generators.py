"""Unconditional, prefix-conditioned, and Qwen generators. Parameter names match released weights."""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from common.config import TextDriftConfig, ReasoningConfig

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


class CondAttention(nn.Module):
    def __init__(self, dim: int, nhead: int):
        super().__init__()
        assert dim % nhead == 0
        self.nhead, self.hd = nhead, dim // nhead
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.q_norm = RMSNorm(self.hd)
        self.k_norm = RMSNorm(self.hd)

    def forward(self, x, key_mask=None):
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.nhead, self.hd).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]                       # (B, H, T, hd)
        q, k = self.q_norm(q), self.k_norm(k)
        # SDPA selects Flash/efficient attention on supported GPUs. This matters for
        # XSum's 1024 condition tokens; explicitly materialising T x T per layer is costly.
        attn_mask = key_mask[:, None, None, :] if key_mask is not None else None
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=0.0, is_causal=False)
        out = out.transpose(1, 2).reshape(B, T, C)
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

    def forward(self, x, key_mask=None):
        x = x + self.ls1 * self.attn(self.norm1(x), key_mask)
        x = x + self.ls2 * self.mlp(self.norm2(x))
        return x


class CondDriftGenerator(nn.Module):
    def __init__(self, cfg, embed_dim: int):
        super().__init__()
        self.Lq, self.Lr = cfg.query_len, cfg.resp_len
        self.D = cfg.d_model
        self.H = embed_dim
        self.gradient_checkpointing = getattr(cfg, "gradient_checkpointing", False)
        self.query_positions = not isinstance(cfg, ReasoningConfig)

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

    def forward(self, query_emb, query_mask, z, prev_emb=None):
        """query_emb (B, Lq, H), query_mask (B, Lq) bool, z (B, noise) -> (B, Lr, H)."""
        B = z.shape[0]
        query_mask = query_mask.to(query_emb.device)
        q = self.in_query(query_emb)
        if self.query_positions:
            q = q + self.q_pos
        q = q + self.seg[0]
        r = (self.r_pos + self.noise_proj(z).view(B, self.Lr, self.D)
             + self.seg[1])                                            # (B, Lr, D)
        if prev_emb is not None:
            r = r + self.in_query(prev_emb.to(r.dtype))
        if self.query_cross is not None:                              # ungated: no LayerScale
            r = r + self.query_cross(r, query_emb, query_mask)
        h = torch.cat([q, r], dim=1)                                   # (B, Lq+Lr, D)
        key_mask = torch.cat(
            [query_mask.to(h.device), torch.ones(B, self.Lr, dtype=torch.bool, device=h.device)],
            dim=1)
        for blk in self.blocks:
            if self.gradient_checkpointing and self.training:
                h = checkpoint(blk, h, key_mask, use_reentrant=False)
            else:
                h = blk(h, key_mask)
        return self.out_proj(h[:, query_emb.shape[1]:, :])                        # (B, Lr, H)

    def sample_z(self, n, noise_dim, temp, device):
        return torch.randn(n, noise_dim, device=device) * temp


class TextDriftGenerator(nn.Module):
    def __init__(self, cfg: TextDriftConfig, wte: torch.Tensor = None):
        super().__init__()
        self.T = cfg.seq_len
        self.D = cfg.d_model

        self.noise_proj = nn.Linear(cfg.noise_dim, self.D * self.T)
        self.pos_emb = nn.Parameter(torch.randn(1, self.T, self.D) * 0.02)
        self.blocks = nn.ModuleList(
            [CondBlock(self.D, cfg.nhead, cfg.ffn_dim) for _ in range(cfg.num_layers)]
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


class QwenCondGenerator(nn.Module):
    """Non-autoregressive one-step generator whose BACKBONE is a pretrained Qwen2.5 model,
    initialised from its weights (for the reasoning migration).

    Input sequence = [query token embeddings (Lq)] ++ [response slots (Lr)], where each
    response slot i = r_pos[i] + noise_proj(z). The Qwen transformer runs over it with its
    native CAUSAL attention (kept, so the pretrained weights are used in-distribution), and
    the response positions' final hidden states are read out and mapped by out_proj to the
    manifold. out_proj is initialised to IDENTITY: since Qwen2.5 ties its lm_head to the input
    embeddings, a hidden state's cosine-NN to those embeddings is (up to the norm) Qwen's own
    argmax next-token -- so at INITIALISATION, before any drift training, the generator already
    decodes to Qwen's predictions rather than to noise.

    Interface matches CondDriftGenerator: forward(query_emb, query_mask, z) -> (B, Lr, H).
    """

    def __init__(self, cfg, embed_dim: int):
        super().__init__()
        from transformers import AutoModelForCausalLM
        self.Lq, self.Lr = cfg.query_len, cfg.resp_len
        self.H = embed_dim
        backbone = getattr(cfg, "backbone_model", "") or cfg.teacher_model
        # Keep trainable parameters and AdamW states in fp32. The training loop may run
        # the forward under bf16 autocast, which accelerates matmuls without making the
        # optimizer state itself bf16.
        print(f"[qwen-gen] loading trainable backbone {backbone} (fp32) ...")
        lm = AutoModelForCausalLM.from_pretrained(
            backbone, torch_dtype=torch.float32, trust_remote_code=True)
        assert lm.config.hidden_size == embed_dim, (
            f"backbone hidden {lm.config.hidden_size} != manifold embed_dim {embed_dim}; "
            f"the generator backbone and the teacher/manifold must be the same model family")
        self.body = lm.model                        # Qwen2Model transformer (trainable)
        # forward feeds inputs_embeds, so Qwen2Model NEVER calls self.embed_tokens -- that matrix
        # gets no gradient and DDP flags it as an unused parameter ("did not receive grad"). It is
        # genuinely dead weight here (the manifold/teacher hold their own frozen embedding copy), so
        # freeze it: DDP skips requires_grad=False params, no find_unused_parameters overhead.
        self.body.embed_tokens.weight.requires_grad_(False)

        # response slots: per-position learned base + a per-sample noise vector for diversity.
        self.r_pos = nn.Parameter(torch.randn(1, self.Lr, self.H) * 0.02)
        self.noise_proj = nn.Linear(cfg.noise_dim, self.H * self.Lr)
        nn.init.normal_(self.noise_proj.weight, std=0.02)
        nn.init.zeros_(self.noise_proj.bias)
        # map response hidden -> manifold; identity init keeps the meaningful starting point.
        self.out_proj = nn.Linear(self.H, self.H)
        nn.init.eye_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, query_emb, query_mask, z, prev_emb=None):
        """query_emb (B, Lq, H), query_mask (B, Lq) bool, z (B, noise) -> (B, Lr, H).
        query_emb may be longer than self.Lq (extra prefix tokens appended); the slice
        uses the actual query length so response slots are always read from the tail."""
        B = z.shape[0]
        Lq_actual = query_emb.shape[1]
        dt = next(self.body.parameters()).dtype
        query_emb = query_emb.to(dt)
        r = self.r_pos.expand(B, -1, -1) + self.noise_proj(z).view(B, self.Lr, self.H)
        if prev_emb is not None:
            r = r + prev_emb.to(r.dtype)
        inp = torch.cat([query_emb, r.to(dt)], dim=1)
        amask = torch.cat(
            [query_mask.to(inp.device),
             torch.ones(B, self.Lr, dtype=query_mask.dtype, device=inp.device)], dim=1)
        pos_ids = (amask.long().cumsum(dim=1) - 1).clamp_min(0)
        h = self.body(inputs_embeds=inp, attention_mask=amask,
                      position_ids=pos_ids).last_hidden_state
        resp = h[:, Lq_actual:, :].float()                              # (B, Lr, H)
        return self.out_proj(resp)

    def sample_z(self, n, noise_dim, temp, device):
        return torch.randn(n, noise_dim, device=device) * temp
