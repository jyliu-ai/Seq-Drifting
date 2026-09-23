"""Teacher generation manifold: its INPUT embedding matrix is the space the
generator outputs into and decodes from, and the SAME model is the teacher.

Loaded once and shared (lru_cache) between the embedder and teacher.
"""
from functools import lru_cache

import torch

# typographic chars that DO occur in clean English text (openwebtext), allowed on top of
# printable ASCII: curly quotes, en/em dash, ellipsis, bullet, angle quotes, dagger, degree.
_ALLOWED_EXTRA = set("‘’“”–—…• ‹›«»†‡°")


def _char_ok(c: str) -> bool:
    if c in "\t\n\r":
        return True
    o = ord(c)
    if 0x20 <= o <= 0x7e:                                    # printable ASCII
        return True
    return c in _ALLOWED_EXTRA


def _build_illegal_mask(tokenizer, vocab_size: int) -> torch.Tensor:
    """(V,) bool, True = a token that must never be decoded to / offered as a candidate.
    EOS is the one allowed special token because it is an explicit supervised target and is
    required to terminate generation. Every other legal token must decode to a non-empty string
    containing only printable ASCII / common whitespace / common typographic characters."""
    bad = torch.zeros(vocab_size, dtype=torch.bool)
    special = set(getattr(tokenizer, "all_special_ids", []) or [])
    eos_id = getattr(tokenizer, "eos_token_id", None)
    for i in range(vocab_size):
        if eos_id is not None and i == eos_id:
            continue
        if i in special:
            bad[i] = True
            continue
        s = tokenizer.decode([i])
        if s == "" or any(not _char_ok(c) for c in s):
            bad[i] = True
    return bad


@lru_cache(maxsize=1)
def load_qwen(model_name: str, device_str: str, dtype_str: str = "bf16"):
    """-> (model, tokenizer), cached. Frozen, eval, no grad."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"[teacher] loading {model_name} ({dtype_str}) ...")
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[dtype_str]
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=dtype, trust_remote_code=True).to(torch.device(device_str))
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    print(f"[teacher] loaded. hidden={model.config.hidden_size} vocab={model.config.vocab_size}")
    return model, tok


class QwenEmbedder:
    """Not an nn.Module -- a thin holder around the frozen teacher embeddings."""

    def __init__(self, cfg, tokenizer, device):
        model, _ = load_qwen(cfg.teacher_model, str(device),
                             "bf16" if cfg.use_bf16 else "fp32")
        self.device = device
        self.sphere = cfg.sphere_norm
        self.chunk = cfg.decode_chunk
        self.eos_id = tokenizer.eos_token_id
        self._emb = model.get_input_embeddings()             # nn.Embedding (V, H), frozen
        wte = self._emb.weight.detach().float()              # (V, H)
        self.H = wte.shape[1]
        self.V = wte.shape[0]
        if self.sphere:
            wte = wte / wte.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        self.wte = wte                                       # (V, H) manifold rows (grad-free)
        compute_dtype = (torch.bfloat16 if getattr(cfg, "use_bf16", False)
                         and device.type == "cuda" else torch.float32)
        self.compute_wte = wte.to(device=device, dtype=compute_dtype)
        # OFF-sphere only: argmin ||p - w||^2 == argmax (p.w - ||w||^2 / 2), since ||p||^2 is
        # the same for every candidate. Without the second term a raw dot product ranks rows
        # by projection alone and so favours long rows -- and these rows are far from equal
        # length (2.42 to 6.04), so plain dot-product NN and the squared-error the training
        # loss minimises disagree on most positions. Keep it in fp32: ||w||^2 reaches ~36,
        # where bf16 steps by ~0.25 and would swamp the margins between candidates.
        self.half_sq = None if self.sphere else (
            0.5 * wte.pow(2).sum(dim=-1)).to(device=device, dtype=torch.float32)

        # forbid decoding to / attracting toward junk tokens. mask_illegal (default) uses the
        # ALLOWLIST (printable ASCII + common typographic) so control chars, UTF-8 fragments
        # ('ÃÂ'), non-English scripts and other peripheral junk are never picked -- without this
        # the continuous drift's nearest neighbour lands on them and the batch cascades into
        # word salad. mask_special is the weaker special-tokens-only fallback.
        if getattr(cfg, "mask_illegal", True):
            print(f"  building illegal-token mask over {self.V} tokens ...")
            forbid = _build_illegal_mask(tokenizer, self.V)
            print(f"  illegal-token mask: {int(forbid.sum())}/{self.V} forbidden")
        else:
            forbid = torch.zeros(self.V, dtype=torch.bool)
            if getattr(cfg, "mask_special", True):
                sp = set(getattr(tokenizer, "all_special_ids", []) or [])
                if sp:
                    forbid[list(sp)] = True
        self.forbid_mask = forbid.to(device)

    def query_embeds(self, query_ids: torch.Tensor) -> torch.Tensor:
        """(B, Lq) ids -> (B, Lq, H) RAW input embeddings for generator conditioning."""
        with torch.no_grad():
            return self._emb(query_ids.to(self.device)).float()

    def lookup(self, token_ids: torch.Tensor) -> torch.Tensor:
        """gather manifold rows for candidate/own tokens. (...,) -> (..., H)."""
        return self.wte[token_ids.to(self.device)]

    @torch.no_grad()
    def decode(self, emb: torch.Tensor) -> torch.Tensor:
        """(B, T, H) manifold embeddings -> (B, T) nearest token ids, chunked over the vocab
        for memory, forbidden tokens excluded. On the sphere this is cosine-NN; off it, it is
        nearest-neighbour in SQUARED EUCLIDEAN distance -- the same quantity the training loss
        minimises -- rather than a raw dot product."""
        B, T, H = emb.shape
        e = emb.reshape(B * T, H).to(self.compute_wte.dtype)
        if self.sphere:
            e = e / e.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        best_sim = torch.full((B * T,), float("-inf"), device=e.device)
        best_idx = torch.zeros(B * T, dtype=torch.long, device=e.device)
        for s in range(0, self.V, self.chunk):
            w = self.compute_wte[s:s + self.chunk]           # (c, H) (unit rows if sphere)
            sim = e @ w.t()                                  # (B*T, c) cosine (or dot)
            if self.half_sq is not None:
                # -||w||^2/2 turns the dot product into the euclidean ranking. fp32 because
                # the term reaches ~18 while the margins it decides are far smaller.
                sim = sim.float() - self.half_sq[s:s + self.chunk].unsqueeze(0)
            if self.forbid_mask[s:s + self.chunk].any():
                sim = sim.masked_fill(self.forbid_mask[s:s + self.chunk].unsqueeze(0), float("-inf"))
            cmax, cidx = sim.max(dim=1)
            take = cmax > best_sim
            best_sim = torch.where(take, cmax, best_sim)
            best_idx = torch.where(take, cidx + s, best_idx)
        return best_idx.reshape(B, T)
