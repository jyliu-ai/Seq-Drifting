"""Real-sample memory bank: n-gram token embeddings + min-distance positive selection.

The whole sanity-check set is small, so we precompute every real row's n-gram
token embeddings once: real_ng (N, T-n+1, n*D), where n-gram p = [wte[t_p] || ...
|| wte[t_{p+n-1}]]. No GPT-2 forward -- just the wte lookup. The active n is part
of an n-gram CURRICULUM (see config): call set_ngram(n) to rebuild real_ng when n
grows (rare -- only at schedule thresholds).

select_positives implements the user's rule: a generated sample matches a real
row if ONE of its n-grams is close (min n-gram distance) to one of the real row's
n-grams. Keep each generation's top-n_pos real rows; union into the drift cloud.
select_positives_token adds the EXACT same-position n-gram match (n consecutive
tokens equal at some position) as a coarse pre-filter.
"""
import torch

from .drift_loss import min_2gram_dist as min_ngram_dist


class TextMemoryBank:
    def __init__(self, tokens: torch.Tensor, embedder, cfg, device,
                 verbose: bool = True, build_ngram: bool = True):
        self.cfg = cfg
        self.device = device
        self.embedder = embedder
        self.tokens = tokens.to(device).to(torch.int64)     # (N, T)
        self.N, self.T = self.tokens.shape
        self.verbose = verbose
        self.build_ngram = build_ngram
        self.set_ngram(int(getattr(cfg, "ngram_min", 2)))

    def set_ngram(self, n: int):
        """(Re)build the real n-gram bank for the given window size n.

        Cheap (a wte lookup + concat); called only when the curriculum's n grows.
        real_ng: (N, T-n+1, n*D). Skipped entirely when build_ngram=False (gpt2 mode),
        where real_ng is huge and only feeds diagnostics.
        """
        n = max(2, min(int(n), self.T))
        self.n = n
        self.n_ng = self.T - n + 1
        if not self.build_ngram:
            self.real_ng = None
            if self.verbose:
                print(f"Real {n}-gram bank: SKIPPED (build_ngram=False)")
            return
        # PRE-ALLOCATE the result and fill it in row chunks, instead of one-shot
        # ngrams_from_ids(all): the one-shot path materialises the (N,T,D) wte-lookup,
        # its n slices, and the concatenated (N,T-n+1,n*D) tensor all at once -- a peak
        # far larger than the resident bank (e.g. ~130GB at N=50k, n=3) that OOMs the
        # build even though the bank itself fits. The chunked fill peaks at the result
        # plus one small chunk, and is bit-identical to the one-shot build.
        D = int(self.embedder.wte.shape[1])
        self.real_ng = torch.empty((self.N, self.n_ng, n * D),
                                   dtype=self.embedder.wte.dtype, device=self.device)
        step = int(getattr(self.cfg, "bank_build_chunk", 4096))
        for s in range(0, self.N, step):
            self.real_ng[s:s + step] = self.embedder.ngrams_from_ids(self.tokens[s:s + step], n)
        if self.verbose:
            print(f"Real {n}-gram bank: {tuple(self.real_ng.shape)} "
                  f"(no GPT-2 forward, chunked build, chunk={step})")

    @torch.no_grad()
    def select_positives_token(self, gen_tokens: torch.Tensor, gen_ng: torch.Tensor,
                               n_pos: int, per_position: bool = False,
                               return_pos: bool = False):
        """Token-level coarse pre-filter: a real row is a candidate for a generation
        iff they share an EXACT same-position n-gram (gen_tok[i:i+n] == real_tok[i:i+n]
        at some position i, n = self.n). Among a generation's matched reals, keep its
        n_pos closest. If a generation matches NONE, fall back to its nearest. Union ->
        the positive cloud.

        Ranking / fallback distance (Q3):
          per_position=True  -> distance AT THE MATCH POSITION (consistent with the
              per-position force: ||gen[p*]-real[p*]|| where p* = the match position);
          per_position=False -> whole-sequence position-aligned L2 (||gen-real||).

        If return_pos, also return:
          pos_match_pos (G, P): the same-position match position per (gen, union-real)
              -- exact-match position, or position-aligned argmin when no exact match.
          own (G, P) bool: whether THIS gen actually selected union-real j (Q2). The
              per-position attraction must only pull a gen toward ITS OWN picks, not
              toward reals other gens chose (those are unrelated -- there is no class
              condition here to make the shared cloud coherent).
        """
        gen_tokens = gen_tokens.to(self.device).to(torch.int64)
        gen_ng = gen_ng.to(self.device)
        G = gen_tokens.shape[0]
        n, n_ng = self.n, self.n_ng

        # first[g, r] = smallest start i with gen_tok[g, i:i+n] == real_tok[r, i:i+n],
        # else -1. Sliding AND over the n token offsets; reversed so earlier wins.
        first = torch.full((G, self.N), -1, dtype=torch.long, device=self.device)
        for i in reversed(range(n_ng)):
            m = torch.ones(G, self.N, dtype=torch.bool, device=self.device)
            for k in range(n):
                m &= gen_tokens[:, i + k].unsqueeze(1) == self.tokens[:, i + k].unsqueeze(0)
            first = torch.where(m, torch.full_like(first, i), first)
        match = first >= 0                                             # (G, N)

        # per-position-aligned n-gram distances (G, N, n_ng): used for ranking + match pos
        dpos = torch.stack([torch.cdist(gen_ng[:, p, :], self.real_ng[:, p, :])
                            for p in range(n_ng)], dim=-1)
        aligned = dpos.argmin(dim=-1)                                  # (G, N) nearest pos
        match_pos = torch.where(first >= 0, first, aligned)            # (G, N), all >= 0
        if per_position:                                               # Q3: rank at match pos
            dist_rank = dpos.gather(-1, match_pos.unsqueeze(-1)).squeeze(-1)   # (G, N)
        else:                                                          # whole-sequence L2
            dist_rank = torch.sqrt((dpos ** 2).sum(dim=-1).clamp_min(0.0))     # (G, N)

        INF = torch.finfo(dist_rank.dtype).max
        has_match = match.any(dim=1, keepdim=True)                     # (G, 1)
        dist = torch.where(has_match, torch.where(match, dist_rank, torch.full_like(dist_rank, INF)),
                           dist_rank)
        k = min(n_pos, self.N)
        top_d, top_i = (-dist).topk(k, dim=1)
        valid_sel = (-top_d) < INF                                     # (G, k) real picks
        keep = top_i[valid_sel]
        union = torch.unique(keep)
        if not return_pos:
            return union

        pos_match_pos = match_pos[:, union]                           # (G, P)
        # own[g,j]: did gen g itself select union[j]? (Q2)
        selected = torch.zeros(G, self.N, dtype=torch.bool, device=self.device)
        rows = torch.arange(G, device=self.device).unsqueeze(1).expand(G, k)
        selected[rows[valid_sel], top_i[valid_sel]] = True
        own = selected[:, union]                                      # (G, P)
        return union, pos_match_pos, own

    @torch.no_grad()
    def select_positives(self, gen_ng: torch.Tensor, n_pos: int,
                         alpha: float = 0.0, return_pos: bool = False):
        """gen_ng (G, T-n+1, n*D) -> unique real-row indices (the positive cloud).

        Selection distance = blend of two POSITION-ALIGNED metrics (both i=j, NEVER
        cross-position), each mean-normalised so they are comparable:
            d_min = min_p ||gen[p]-real[p]||   (permissive: one SAME-position n-gram
                    matching is enough -- the position-aligned analogue of the old
                    min-n-gram, but the match must be at the same position p)
            d_l2  = ||gen_flat - real_flat||   (strict: all positions match; full
                    position-aligned sequence L2)
            d = (1-alpha) * d_min  +  alpha * d_l2
        alpha=0 -> permissive bootstrap, alpha=1 -> strict full-sequence. Because
        both terms are position-aligned, a degenerate sequence that repeats a common
        token everywhere is FAR from every (diverse) real at every aligned position,
        so it is never selected (the old CROSS-position min-n-gram had the opposite,
        broken property and rewarded repetition).

        If return_pos, also return (Q2-aware):
          pos_match_pos (G, P): POSITION-ALIGNED match position argmin_p ||gen[p]-real[p]||
          own (G, P) bool: whether THIS gen selected union-real j (so the per-position
              attraction only pulls a gen toward its OWN picks, not other gens').
        ``alpha`` already controls the position consistency here (alpha=0 -> rank by the
        match-position distance d_min; alpha=1 -> whole-sequence), so Q3 needs no extra
        flag in this mode.
        """
        gen_ng = gen_ng.to(self.device)
        G = gen_ng.shape[0]
        # per-position-aligned n-gram distance, all positions: (G, N, n_ng)
        dpos = torch.stack([torch.cdist(gen_ng[:, p, :], self.real_ng[:, p, :])
                            for p in range(self.n_ng)], dim=-1)
        if alpha <= 0.0:
            d = dpos.min(dim=-1).values                              # (G, N) position-aligned min
        elif alpha >= 1.0:
            d = torch.sqrt((dpos ** 2).sum(dim=-1).clamp_min(0.0))   # (G, N) full position-aligned L2
        else:
            d_min = dpos.min(dim=-1).values
            d_l2 = torch.sqrt((dpos ** 2).sum(dim=-1).clamp_min(0.0))
            d = ((1 - alpha) * d_min / d_min.mean().clamp_min(1e-6)
                 + alpha * d_l2 / d_l2.mean().clamp_min(1e-6))
        k = min(n_pos, self.N)
        top_idx = (-d).topk(k, dim=-1).indices                       # (G, k) smallest dist
        union = torch.unique(top_idx.reshape(-1))
        if not return_pos:
            return union
        pos_match_pos = dpos[:, union, :].argmin(dim=-1)             # (G, P) position-aligned
        # own[g,j]: did gen g itself select union[j]? (Q2)
        selected = torch.zeros(G, self.N, dtype=torch.bool, device=self.device)
        rows = torch.arange(G, device=self.device).unsqueeze(1).expand(G, k)
        selected[rows, top_idx] = True
        own = selected[:, union]                                     # (G, P)
        return union, pos_match_pos, own

    @torch.no_grad()
    def sample_negatives(self, n_neg: int) -> torch.Tensor:
        k = min(n_neg, self.N)
        return torch.randperm(self.N, device=self.device)[:k]

    def gather_ng(self, indices: torch.Tensor) -> torch.Tensor:
        """Real n-grams for the given rows: (len(indices), T-n+1, n*D)."""
        return self.real_ng[indices]

    # backward-compat alias
    def gather_2g(self, indices: torch.Tensor) -> torch.Tensor:
        return self.gather_ng(indices)
