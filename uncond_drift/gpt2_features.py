"""Frozen GPT-2 token-embedding table (lookup only -- NO transformer forward).

Per user spec the representation is just 2-gram TOKEN embeddings: for a sequence
its 2-gram at position i is [emb_i || emb_{i+1}] in R^{2D}. Real tokens use the
GPT-2 wte table; generated samples already are continuous embeddings, so their
2-grams are just consecutive concatenations of the generator output. There is no
GPT-2/T5 forward pass anywhere.
"""
import torch
import torch.nn as nn
from transformers import GPT2Model


# common typographic chars that DO occur in clean English text (openwebtext) and so
# are allowed in addition to printable ASCII: curly quotes, en/em dash, ellipsis,
# bullet, non-breaking space, single/double angle quotes, dagger, degree.
_ALLOWED_EXTRA = set("‘’“”–—…• "
                     "‹›«»†‡°")


def _char_ok(c: str) -> bool:
    if c in "\t\n\r":
        return True
    o = ord(c)
    if 0x20 <= o <= 0x7e:                 # printable ASCII
        return True
    return c in _ALLOWED_EXTRA


def _build_illegal_mask(tokenizer, vocab_size: int) -> torch.Tensor:
    """(V,) bool, True = a token that must never be decoded to / used as a candidate.

    ALLOWLIST (denylist was not enough -- it stops control / U+FFFD junk but lets through
    'legal-but-junk' tokens: rare symbols rendered as '◼', Latin-supplement byte proxies
    'ÃÂ', CJK / Arabic fragments, etc). Here a token is legal ONLY if it is not a special
    token, decodes to a non-empty string, and EVERY char is printable ASCII, common
    whitespace (\\t \\n \\r), or a common typographic char (_ALLOWED_EXTRA). Everything
    else -- non-Latin scripts, accented letters, box / symbol glyphs, byte fragments,
    U+FFFD -- is forbidden. Intended for the English sanity-check (dropping non-English
    text is acceptable, per the dataset).
    """
    bad = torch.zeros(vocab_size, dtype=torch.bool)
    special = set(getattr(tokenizer, "all_special_ids", []) or [])
    for i in range(vocab_size):
        if i in special:
            bad[i] = True
            continue
        s = tokenizer.decode([i])
        if s == "" or any(not _char_ok(c) for c in s):
            bad[i] = True
    return bad


class TokenEmbedder(nn.Module):
    def __init__(self, model_name: str = "gpt2", tokenizer=None, mask_illegal: bool = True,
                 sphere: bool = False):
        super().__init__()
        print(f"Loading frozen {model_name} wte (embedding lookup only) ...")
        gpt2 = GPT2Model.from_pretrained(model_name)
        wte = gpt2.wte.weight.detach().clone()        # (V, D)
        del gpt2
        self.sphere = bool(sphere)
        if self.sphere:                               # put token vectors on the unit sphere
            wte = wte / wte.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            print("  sphere mode: wte rows L2-normalised (decode/kernels become angular)")
        self.register_buffer("wte", wte)
        # 0.5*||w||^2 precomputed for L2-nearest decode (argmin_w ||e-w||^2 =
        # argmax_w (e·w - 0.5||w||^2); the ||e||^2 term is constant per row).
        self.register_buffer("wte_half_sqnorm", 0.5 * (wte ** 2).sum(-1))
        self.d_model = wte.shape[1]
        self.vocab_size = wte.shape[0]
        # illegal-token mask (see _build_illegal_mask / cfg.mask_illegal_tokens)
        self.mask_illegal = bool(mask_illegal) and tokenizer is not None
        if self.mask_illegal:
            illegal = _build_illegal_mask(tokenizer, self.vocab_size)
            print(f"  illegal-token mask: {int(illegal.sum())}/{self.vocab_size} tokens "
                  f"forbidden in decode + repair candidates")
        else:
            illegal = torch.zeros(self.vocab_size, dtype=torch.bool)
        self.register_buffer("illegal_mask", illegal)

    @property
    def wte_weight(self) -> torch.Tensor:
        return self.wte

    @staticmethod
    def to_ngrams(emb: torch.Tensor, n: int) -> torch.Tensor:
        """(B, T, D) embeddings -> (B, T-n+1, n*D) consecutive-concat n-grams.

        The n-gram at position p is [emb_p || emb_{p+1} || ... || emb_{p+n-1}].
        Windows overlap, so a gradient on n-gram p touches emb positions p..p+n-1
        (each token participates in up to n n-grams -> smoothing across the span).
        """
        T = emb.shape[1]
        W = T - n + 1                                   # number of windows
        return torch.cat([emb[:, i:i + W, :] for i in range(n)], dim=-1)

    @torch.no_grad()
    def ngrams_from_ids(self, token_ids: torch.Tensor, n: int) -> torch.Tensor:
        """(B, T) ids -> (B, T-n+1, n*D) n-grams of their wte embeddings."""
        return self.to_ngrams(self.wte[token_ids], n)

    # backward-compat 2-gram aliases
    @staticmethod
    def to_twograms(emb: torch.Tensor) -> torch.Tensor:
        return TokenEmbedder.to_ngrams(emb, 2)

    @torch.no_grad()
    def twograms_from_ids(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.ngrams_from_ids(token_ids, 2)

    @torch.no_grad()
    def decode(self, emb: torch.Tensor) -> torch.Tensor:
        """(B, T, D) continuous embeddings -> (B, T) nearest token ids (L2).

        Same L2 metric as drift / positive selection. argmin_w ||e-w||^2.
        """
        B, T, D = emb.shape
        e = emb.reshape(B * T, D).to(self.wte.dtype)             # emb may be bf16
        R = e.shape[0]
        # (R, V) scores would be ~200 GiB at R=1000*1024, V=50k; chunk over rows.
        row_chunk = 16384
        out = torch.empty(R, dtype=torch.long, device=e.device)
        for s in range(0, R, row_chunk):
            sc = e[s:s + row_chunk] @ self.wte.T - self.wte_half_sqnorm   # (c, V)
            if self.mask_illegal:                                # forbid junk tokens
                sc = sc.masked_fill(self.illegal_mask.unsqueeze(0), float("-inf"))
            out[s:s + row_chunk] = sc.argmax(-1)
        return out.view(B, T)
