"""Task configurations and compatibility defaults for released checkpoints."""
from dataclasses import dataclass, field
from typing import Tuple

@dataclass
class TextDriftConfig:
    # ── Data ────────────────────────────────────────────────────────────────
    dataset_name: str = "openwebtext"  # openwebtext | wikitext-2 | wikitext-103 | ptb
    tokenizer_name: str = "gpt2"
    data_path: str = ""               # optional local JSONL file/directory of text records
    seq_len: int = 8                   # CONTENT length (no BOS/EOS in training; wrapped only at eval)
    n_train: int = 1000                # cap training rows (the sanity-check set)
    # skip precomputing the real n-gram bank (real_ng, shape (N, T-n+1, n*D)). In pure
    # gpt2 mode the bank is never used for training, only for the distinct_reals_hit /
    # [batch nn] diagnostics, and at long seq_len it is huge (~100 GB) and OOMs. True =
    # do not build it and skip those two diagnostics; distinct_seqs / gen_ppl still work.
    skip_bank: bool = False
    num_workers: int = 0

    # ── Embedding (frozen GPT-2 wte, lookup only) ───────────────────────────
    embed_model: str = "gpt2"          # provides the wte table; no forward pass
    d_model: int = 768                 # = GPT-2 n_embd; generator output dim
    # decode (continuous emb -> nearest wte) ranges over the FULL 50257 vocab,
    # which includes control-byte / UTF-8-fragment tokens (the '◼ \x10 \x14' junk).
    # When the generator drifts off the clean-token manifold (e.g. gpt2-repair, where
    # plausible positions are unanchored) its nearest wte is easily one of these rare
    # peripheral tokens -> illegal characters in the output AND fed back into GPT-2.
    # True: forbid decode (and gpt2-repair candidates) from picking any token that
    # decodes to a control / non-printable / replacement (U+FFFD) char, or a special
    # token -> the nearest CLEAN token is emitted instead. Also drops non-English
    # byte-fragment tokens (the Arabic/Bengali junk-collapse attractors).
    mask_illegal_tokens: bool = True
    # put EVERYTHING on the unit sphere: L2-normalise the wte rows AND the generator output
    # (per position), so all distances/kernels/decoding become ANGULAR (cosine). Rationale:
    # in L2 space the repulsion leaks into the radial direction (blows the embedding norm up ->
    # the scale/loss explosions); on the sphere there is no radius, so repulsion can ONLY spread
    # directions and can be made strong without inflating. NOTE: since ||a-b||^2 = 2-2cos for
    # unit vectors, the existing L2 code becomes angular automatically once both sides are unit;
    # only the goal is re-projected onto the sphere. Intended for the gpt2-repair path; train
    # from scratch (the n-gram checkpoints were trained in L2 geometry).
    sphere_norm: bool = False
    # sphere step geometry (only used when sphere_norm): how the detached goal moves on the sphere.
    #   "proj":     goal = normalize(old + attract)   -- chord + project; the radial part of the
    #               force distorts the angle moved (and is wasted). (current/default behaviour)
    #   "retract":  tangent-project the force first, then normalize(old + F_tan) -- cleaner, bounded.
    #   "geodesic": tangent-project, then exp-map along the great circle (true geodesic step);
    #               arc length clamped to sphere_geo_max to avoid wrapping past the antipode.
    sphere_step: str = "proj"          # proj | retract | geodesic
    sphere_geo_max: float = 1.5708     # geodesic: max per-step arc length in radians (~pi/2)

    # ── Generator (continuous z -> continuous embedding sequence) ───────────
    noise_dim: int = 128               # dim of the continuous Gaussian z
    nhead: int = 12
    num_layers: int = 6
    ffn_dim: int = 3072
    dropout: float = 0.0               # 0 so eval == train for the sanity check
    vocab_size: int = 50257            # set from tokenizer at runtime (decode)

    # ── Loss选择 ────────────────────────────────────────────────────────────
    # "drift" = drift_loss_2gram (affinity/Sinkhorn);
    # "match" = greedy one-to-one matching (ning MD_loss analogue).
    loss_mode: str = "drift"
    match_n_real: int = 0              # match 模式每步随机抽的候选真实样本数 (0 -> 2*gen_per_step)

    # ── Drift loss (min-2-gram-distance variant) ────────────────────────────
    R_list: Tuple[float, ...] = (0.02, 0.05, 0.2)
    attract_temp: float = 1.0          # affinity temperature; <1 sharpens (->one-hot)
    # optional attraction-temperature warm-up: anneal tau from attr_temp_warmup_start (a
    # softer/larger value) down to attract_temp over attr_temp_warmup_steps, geometric in
    # the step fraction. 0 steps = disabled (tau constant at attract_temp; teacher and all
    # existing runs unchanged). Breaks the init-collapse deadlock when far off-policy
    # positives cannot compete with clustered siblings under a sharp steady tau.
    attr_temp_warmup_start: float = 1.0
    attr_temp_warmup_steps: int = 0
    sinkhorn_iters: int = 0            # >0: balanced (Sinkhorn) affinity, caps per-real intake
    per_position_force: bool = False  # affinity+force per position (each pos pulled only by reals matching there)
    # Q3: rank token-match candidates by the distance AT THE MATCH POSITION (consistent
    # with the per-position force) instead of the whole-sequence L2. WARNING: at small n
    # this rewards trivially-matched common tokens (function words match everywhere with
    # ~0 match-position distance) -> function-word collapse. Default off (whole-sequence
    # ranking acts as a regulariser that forces global similarity).
    rank_at_match_pos: bool = False
    # per-position force routed by the CROSS-position min-2-gram (force at the single argmin
    # 2-gram position) instead of same-position (i=j) alignment; keeps force nonzero at long len.
    drift_cross_2gram: bool = False
    # Q2: restrict each gen's per-position ATTRACTION to the positives IT selected
    # (not the batch-wide union). Off (default) = original drift behaviour: attract
    # over the shared union cloud (the union + Sinkhorn IS the coverage mechanism;
    # far union reals are anyway ~0 affinity). On = only own picks (sharper, but can
    # break cross-gen coverage -> collapse).
    attract_own_only: bool = False
    # 2-gram nearest-neighbour positive selection: per generated sample keep the
    # top-n_pos real rows by smallest min-2-gram distance, union into the cloud.
    # positive selection: "repr_min2gram" (representation-space min-2-gram, with
    # alpha anneal) | "token_2gram_match" (exact same-position token 2-gram match,
    # fallback to repr nearest when a generation matches none).
    select_mode: str = "repr_min2gram"
    # when select_mode="token_2gram_match": use repr min-2gram for the first
    # `token_match_start` steps (bootstrap), then switch to exact token matching
    # (refine, once generations are coherent enough to actually match).
    token_match_start: int = 0
    # ── n-gram curriculum ───────────────────────────────────────────────────
    # The whole pipeline (representation + matching + drift force) uses n-grams
    # = concat of n consecutive token embeddings. n starts at ngram_min and is
    # incremented by 1 at each step threshold in ngram_grow_steps, capped at
    # ngram_max. Longer n -> rarer exact same-position matches -> the generation
    # must reproduce a longer contiguous span -> coherence -> memorisation.
    #   n(step) = min(ngram_max, ngram_min + #{s in ngram_grow_steps : step >= s})
    # e.g. ngram_min=2, ngram_grow_steps=(40000,70000), ngram_max=4:
    #   <40k -> 2-gram, 40k-70k -> 3-gram, >=70k -> 4-gram.
    ngram_min: int = 2
    ngram_max: int = 4
    ngram_grow_steps: Tuple[int, ...] = ()
    # ── GPT-2 distillation positives (alternative to dataset matching) ───────
    # positive_source="gpt2": don't match against the (finite) dataset; use a GPT-2
    # teacher to build coherent positives per generation. Decode the generation,
    # find the FIRST position where the gen token is NOT in GPT-2's top-k (the
    # divergence p*), keep the plausible prefix [0..p*-1], sample gpt2_n_pos top-k
    # tokens at p*, and let GPT-2 continue each to full length -> the positive cloud
    # (union over the batch). Drift uses whole-sequence affinity, with the force
    # MASKED to positions [0..p*] per generation (prefix snap + learn the repair;
    # tail gets no gradient). NOTE: this replaces dataset matching; the dataset bank
    # is still loaded for eval metrics only.
    positive_source: str = "dataset"   # dataset | gpt2
    # diagnostic (b): build the GPT-2 teacher support from REAL data prefixes (off-policy) instead
    # of the model's own generated prefix, to test whether on-policy bootstrapping blocks fluency.
    teacher_real_prefix: bool = False
    # two GPT-2 positive methods (positive_source=gpt2):
    #   "continuation": first-divergence p* + prefix + top-k sample + GPT-2 continuation;
    #                   whole-sequence affinity, force on [0..p*]. (uses gpt2_topk)
    #   "repair":       per-position -- flag every implausible token (prob<=prob_thresh),
    #                   resample plausible tokens there, attract that position by TOKEN
    #                   distance only. (uses gpt2_prob_thresh, gpt2_temp)
    gpt2_method: str = "repair"        # continuation | repair
    # repair-method target construction (cfg.gpt2_method="repair"):
    #   "self":    original -- flag implausible positions (prob<=thresh) and pull each
    #              toward the affinity-weighted nearest PLAUSIBLE token; plausible
    #              positions snap to their OWN token (zero force). Degenerate: any
    #              locally-plausible token (',', '\n') is a zero-force fixed point ->
    #              the batch drifts into punctuation/function-word soup.
    #   "teacher": A2 / reverse-KL. EVERY position is pulled toward the affinity-weighted
    #              centroid of the GPT-2 SUPPORT SET (top-n_pos tokens with prob>thresh,
    #              illegal excluded). The affinity is by the gen's own embedding DISTANCE
    #              (mode-seeking: each gen commits to the nearest teacher-allowed token,
    #              not the teacher mean), so force ~ how far the gen is from what GPT-2
    #              wants -- no implausible flag, no own-token snap, no zero-force soup.
    gpt2_repair_target: str = "self"   # self | teacher
    gpt2_teacher: str = "gpt2"
    gpt2_topk: int = 50                # continuation method: top-k for plausibility + sampling
    gpt2_n_pos: int = 16               # positives / plausible tokens sampled per position
    gpt2_prob_thresh: float = 0.01     # repair method: plausibility probability floor
    gpt2_temp: float = 1.0             # repair method: token-distance affinity temperature
    # (variant 1) how the teacher SUPPORT SET is cut from the top-n_pos tokens:
    #   "thresh":  keep tokens with prob > gpt2_prob_thresh (fixed absolute floor; unfair at
    #              high-entropy positions like the first token -> very few candidates).
    #   "nucleus": sort by prob, keep the smallest prefix whose cumulative prob reaches
    #              gpt2_nucleus_p (top-p). Adaptive: high-entropy positions keep many candidates
    #              (more diversity), low-entropy keep few. Argmax always kept.
    gpt2_support: str = "thresh"       # thresh | nucleus
    gpt2_nucleus_p: float = 0.99       # nucleus cumulative-probability cutoff
    # anti-repetition on the teacher support set. k>=2: before taking the support set, forbid
    # any candidate token that would complete a k-gram already present in the generation's own
    # prefix, so the repeated token is dropped from the candidates and the target becomes a
    # non-repeat. This breaks the repetition feedback loop (a repetitive prefix makes the LM
    # condone more repetition) at its source. Bans the exact k-gram, not the token, so common
    # words are untouched at k>=3. 0 = off.
    gpt2_no_repeat_ngram: int = 0      # 0=off; k>=2 (use 3 or 4)
    gpt2_no_repeat_window: int = 0     # only look back this many tokens (0 = whole prefix)
    # (variant 3) multi-temperature fusion for the attraction affinity (like drifting's multi-R
    # kernel): if non-empty, average the softmax affinity computed at EACH of these temps instead
    # of the single gpt2_temp -> blends sharp (mode-seeking) and soft (broad) scales.
    gpt2_temp_list: Tuple[float, ...] = ()
    # repair method: WHOLE-SEQUENCE repulsion between generations (anti-collapse, like
    # the original drift's gen-gen repulsion). 0 = attraction only. Per-position
    # attraction needs no repulsion, but a sequence-level one keeps gens from collapsing
    # to a single output.
    gpt2_repel: float = 1.0
    # repair-method repulsion scaling. False (original): repulsion RMS is rescaled to the
    # ATTRACTION RMS -- so when attraction vanishes (the self-mode soup fixed point) the
    # repulsion vanishes too and cannot push the collapsed gens apart. True (B): rescale
    # repulsion to the candidate-DISTANCE scale instead, an absolute embedding-space unit
    # that stays finite when attraction is ~0 -> anti-collapse force survives at the
    # degenerate point. Pair with gpt2_repair_target="teacher".
    gpt2_repel_abs: bool = False
    # repair-method repulsion granularity. False (original): WHOLE-SEQUENCE -- gen-gen
    # distance over the flattened (T*D) sequence. Flaw: two gens that share a near-identical
    # PREFIX but differ at the tail already count as "far", so the repulsion is satisfied by
    # tail-only diversity and the prefixes stay nearly identical (-> 'I was..'/'The only..'
    # clones). True: PER-POSITION -- compute gen-gen distance and push apart INDEPENDENTLY at
    # each position, so every position (incl. the prefix) is diversified across the batch.
    gpt2_repel_perpos: bool = False
    # repair-method INTRA-SEQUENCE repulsion (weight; 0 = off). Pushes a generation's OWN T
    # positions apart from each other, so the sequence can't collapse to the same token
    # repeated ('very very very', 'a a a a'). For each gen, position p is pushed away from the
    # other positions q!=p of the SAME sequence that are nearest it (repeated tokens have ~0
    # distance -> strong push). Orthogonal to gpt2_repel (which diversifies ACROSS gens);
    # this one fights WITHIN-sequence repetition.
    gpt2_repel_intra: float = 0.0

    # --- Monte-Carlo repulsion field ---------------------------------------------------
    # gpt2_repel is a per-SEQUENCE interaction: with gen_per_step=G each generation is only
    # pushed away from G-1 peers, so anti-collapse weakens as G shrinks. seq1024 forces G=32
    # for memory where len128 ran G=512 -- a 16x thinner repel field on the very setting that
    # needs it most. These extras are drawn under no_grad and take part in the repulsion ONLY:
    # no attraction, no loss, no gradient. They need neither embedder.decode nor the GPT-2
    # teacher pass -- the repel block reads `old` (the raw embedding) and nothing else -- so
    # they cost one generator forward each and no teacher time at all.
    repel_extra: int = 0               # fresh no-grad samples drawn per step for the repel field
    # MoCo-style FIFO: repel against the last `repel_queue` extras while refreshing only
    # `repel_extra` of them per step, so a large field costs few forwards. 0 = no queue (the
    # fresh extras are the whole field). Costs repel_queue*seq_len*768*4 bytes of VRAM and
    # introduces staleness: the buffer trails the live generator by repel_queue/repel_extra
    # steps, so keep it short while the generator still moves fast.
    repel_queue: int = 0
    # Extras are generated in chunks of this many: a no_grad forward still has a
    # per-layer transient that scales with batch, so drawing all repel_extra at once
    # peaks at repel_extra/gen_per_step times the forward the card was sized for.
    # Only the (K, T, D) result is kept, so chunking costs nothing. 0 = gen_per_step.
    repel_chunk: int = 0

    # Gradient accumulation: run this many micro-steps of gen_per_step before one
    # optimizer step, so the gradient batch is grad_accum*gen_per_step*world_size
    # without raising peak activation memory. Cost is linear -- N micro-steps of the
    # same work -- with no second forward, because every global scalar the goal needs
    # (scale, a_rms, r_rms) is a mean over G*T*D ~ 25M elements and is already stable
    # at G=32. Only the gen-gen repulsion genuinely wants the whole batch; see
    # repel_reuse for how it gets it for free.
    grad_accum: int = 1
    # Feed the repulsion ring from the gradient batch instead of dedicated extras.
    # Those generations are produced for the backward anyway, so the repel field costs
    # zero extra forwards; it lags the live generator by at most one optimizer step.
    # Ring size defaults to grad_accum*gen_per_step. Mutually exclusive with
    # repel_extra (reuse wins).
    repel_reuse: bool = False
    # repair method: leave the first `gpt2_free_prefix` positions FORCE-FREE -- no attraction,
    # no repulsion, no gradient -- keeping whatever the generator already produces there. Reason:
    # after just [BOS] GPT-2's support is pathologically narrow (~7 tokens > thresh), so the
    # teacher force at position 0 destroys the (already diverse, sensible) first token the n-gram
    # bootstrap learned, and that homogenisation cascades. The GPT-2 teacher pass still conditions
    # the LATER positions on [BOS]+the kept first token(s). 0 = off; 1 = free position 0.
    gpt2_free_prefix: int = 0
    # with positive_source="gpt2": use dataset matching for the first gpt2_start_step
    # steps (bootstrap, e.g. through the 2/3-gram curriculum), then switch to the
    # GPT-2 teacher positives. 0 = GPT-2 from the start.
    gpt2_start_step: int = 0
    # initialise generator + EMA weights from a saved checkpoint (skip re-running the
    # 2/3-gram bootstrap). Pair with ngram_min=ngram_max=3, ngram_grow_steps=(),
    # gpt2_start_step=0 to continue straight into the GPT-2 phase at 3-gram.
    init_from: str = ""
    n_pos: int = 32                    # top-k real rows per generation (source pos_per_sample=64)
    n_neg: int = 0                     # random real negatives (0 = off, like cfg=1)
    # Positive-selection metric anneal: alpha goes 0 (min-2-gram, bootstrap) -> 1
    # (position-aligned L2, discriminative) linearly over `select_anneal_steps`.
    select_anneal_steps: int = 10_000

    # ── Optimisation ────────────────────────────────────────────────────────
    steps: int = 20_000
    gen_per_step: int = 64             # generations per step (the drift group)
    lr: float = 2e-4
    weight_decay: float = 0.0
    grad_clip: float = 2.0
    temp: float = 1.0                  # z sampling temperature
    ema_decay: float = 0.999           # EMA of generator params, used at eval
    seed: int = 42
    use_bf16: bool = False

    # ── Logging / eval / checkpoint ─────────────────────────────────────────
    log_every: int = 50
    log_gen_samples: bool = False      # print per-generation repair/positive diagnostics at log_every
    eval_every: int = 1_000
    eval_samples: int = 1000           # samples for gen_ppl / entropy / coverage
    eval_n_show: int = 10              # decoded samples printed at eval
    eval_compute_ppl: bool = True      # load GPT-2 Large for gen_ppl + entropy
    save_every: int = 5_000
    checkpoint_dir: str = "runs/unconditional"
    output_dir: str = "runs/unconditional/output"


@dataclass
class ContinuationConfig:
    # ── Data ────────────────────────────────────────────────────────────────
    dataset_name: str = "owt"          # lm1b | jsonl | owt; empty for unconditional warmup
    owt_dir: str = ""                  # dataset_name=owt: dir of *.jsonl.zst OpenWebText2 shards
    owt_docs: int = 50000              # how many docs to load (prefix->continuation examples)
    jsonl_path: str = ""               # dataset_name=jsonl: path to a {query,response} jsonl
    test_jsonl_path: str = ""          # separate held-out prefix/continuation JSONL
    jsonl_query_key: str = "query"
    jsonl_response_key: str = "response"
    n_train: int = 0                   # cap training pairs (0 = use all)
    query_len: int = 128               # max query tokens (LEFT-padded so the last token is real)
    # GSM8K gold solutions are ~74 tokens on average (median 69, p90 125). At resp_len=192
    # about 62% of positions had NO gold token, so they received only the teacher force
    # conditioned on the model's own garbage prefix -- a self-referential loop that is what
    # produced the multilingual junk tail.
    resp_len: int = 128                # fixed response length (covers ~p90 of gold solutions)
    # The gold response occupies positions 0..L-1, position L holds ONE end-of-text token as
    # the delimiter (supervised -- the model must learn where to stop), and everything after
    # is pad that carries NO target and is excluded from the loss entirely. Over-long
    # solutions are truncated.
    use_chat_template: bool = True     # wrap the query with the Qwen chat template

    # ── Teacher / manifold (Qwen3) ──────────────────────────────────────────
    # The generation manifold IS this model's INPUT embedding matrix: the generator
    # outputs into it and decodes by cosine-NN to it, and the same model is the teacher
    # that scores [query ++ response] and returns the per-position support set. Teacher
    # vocab == generation vocab by construction.
    teacher_model: str = "Qwen/Qwen3-4B"
    embed_dim: int = 0                 # generator output dim == Qwen hidden; 0 -> set at runtime
    vocab_size: int = 0                # set at runtime
    sphere_norm: bool = True           # unit-sphere manifold (angular geometry); on by default here
    sphere_step: str = "geodesic"      # proj | retract | geodesic
    sphere_geo_max: float = 1.5708     # geodesic max per-step arc length (rad)
    decode_chunk: int = 8192           # vocab chunk size for cosine-NN decode (memory)

    # ── Teacher support set (per response position) ─────────────────────────
    n_pos: int = 32                    # support tokens kept per position (memory: rw is G*T*n_pos*H)
    prob_thresh: float = 0.01          # thresh support: keep tokens with prob > this
    support: str = "thresh"            # thresh | nucleus
    nucleus_p: float = 0.99            # nucleus cumulative-prob cutoff
    no_repeat_ngram: int = 4           # 0=off; k>=2 bans completing a k-gram already in the response
    no_repeat_window: int = 0          # only look back this many response tokens (0 = whole response)
    mask_special: bool = True          # never let a candidate be a special/added token
    # ALLOWLIST junk-token mask (printable ASCII + common typographic). Forbids control chars,
    # UTF-8 byte fragments ('ÃÂ'), non-English scripts etc. from both decode and the teacher
    # support set. Without it the drift lands on junk peripheral tokens and cascades into word
    # salad (as in the unconditional version). Building it decodes every vocab token once
    # (~minutes for Qwen's 150k; fast for GPT-2's 50k).
    mask_illegal: bool = True

    # ── Attraction / repulsion (same knobs as the unconditional repair loss) ─
    # GOLD (real response) attraction. Position-aligned pull toward the dataset's real
    # solution tokens. The teacher support set only encodes "plausible here" and carries no
    # notion of the CORRECT answer, so on an accuracy-scored task (GSM8K) pure teacher
    # self-distillation cannot reach the right tokens. This re-introduces the dataset
    # supervision, conditionally. 0 = off (pure teacher, the previous behaviour).
    gold_weight: float = 0.0           # try 2-5; large relative to the teacher force
    teacher_weight: float = 1.0        # scales the Qwen-support attraction; 0 = pure gold supervision
    attract_temp: float = 0.3          # token-distance affinity temperature (mode-seeking when small)
    temp_list: Tuple[float, ...] = ()  # multi-temp fusion; empty = single attract_temp
    repel: float = 5.0                 # gen-gen repulsion weight
    repel_block: bool = True           # True=repel only within same prefix (K samples);
                                       # False=whole-batch repulsion (like unconditional's 2048)
    repel_perpos: bool = True          # per-position gen-gen repulsion
    repel_intra: float = 5.0           # within-response repulsion (anti token-repeat)
    repel_abs: bool = False
    abs_scale: float = 1.0             # set at runtime to the mean pairwise embedding distance if repel_abs
    free_prefix: int = 0               # leave the first N response positions force-free

    # ── Generator (query embeddings + z -> response embeddings, one pass) ────
    d_model: int = 1024                # generator internal width (decoupled from embed_dim)
    nhead: int = 16
    num_layers: int = 12
    ffn_dim: int = 4096
    # query conditioning. In the main stack the query reaches the response only through
    # self-attention, which LayerScale gates by ls_init (~1e-4). That gate is inherited from
    # the unconditional generator, where it is a FIX (z is injected directly at layer 0, so
    # near-identity blocks pass it through un-smoothed). Here the query is NOT direct, so the
    # same gate suppresses conditioning and the model collapses to the query-independent
    # per-position marginal of the gold responses (identical prefix for every query).
    # UNCONDITIONAL warmup: for the first uncond_warmup_steps the context is filled with a
    # CONSTANT special token (warmup_ctx, default pad==eos==<|endoftext|> for GPT-2 = document
    # start) under NORMAL attention -- so the generator/teacher condition on a clean constant
    # prefix (truly unconditional) rather than masking. This lifts the generator into the
    # "coherent English" basin before the real prefix conditioning is turned on, and keeps the
    # conditioning params gradient-fed throughout. 0 = off (condition from step 0).
    uncond_warmup_steps: int = 0
    warmup_ctx: str = "pad"            # warmup fills context with this token (pad|eos|bos), normal attn
    query_cross_attn: bool = True      # ungated cross-attention (response slots read query tokens)
    ls_init: float = 1e-4              # LayerScale init; raise (0.1/1.0) to un-gate the main stack
    noise_dim: int = 128               # continuous z dim (diversity across samples of one query)
    k_samples: int = 4                 # response samples per query (enables gen-gen repulsion)

    # ── Optimisation ────────────────────────────────────────────────────────
    steps: int = 200_000
    queries_per_step: int = 16         # distinct queries per step; batch = queries_per_step * k_samples
    lr: float = 2e-4
    weight_decay: float = 0.0
    grad_clip: float = 2.0
    temp: float = 1.0                  # z sampling temperature
    ema_decay: float = 0.999
    seed: int = 42
    use_bf16: bool = True

    # ── Logging / eval / checkpoint ─────────────────────────────────────────
    log_every: int = 50
    eval_every: int = 1_000
    eval_queries: int = 200            # held-out queries scored at eval
    eval_n_show: int = 6
    eval_accuracy: bool = True         # gsm8k/math: extract final answer and score exact-match
    eval_on_train: bool = False        # overfit diagnostic: eval the TRAIN queries
    eval_gen_ppl: bool = True          # FLM-style generative perplexity of generated text
    eval_ppl_model: str = "gpt2-large" # reference LM for gen-ppl (FLM default; offline-cached)
    save_every: int = 5_000
    checkpoint_dir: str = "runs/cond_drift"


@dataclass
class Seq2SeqDriftConfig:
    # Data follows ELF: no packing, a fixed clean condition, and 64 target slots.
    dataset_name: str = "wmt14_de_en"       # wmt14_de_en | xsum
    data_dir: str = "data/wmt14_de_en"
    train_split: str = "train"
    eval_split: str = "validation"
    query_len: int = 64                     # ELF: 64 for WMT14, 1024 for XSum
    resp_len: int = 64
    n_train: int = 0
    n_eval: int = 0

    # Frozen token manifold and autoregressive support teacher. The student generator
    # remains the original CondDriftGenerator; this model supplies only the tokenizer,
    # embedding manifold, and teacher support distribution.
    teacher_model: str = "gpt2"
    embed_dim: int = 0
    vocab_size: int = 0
    sphere_norm: bool = True
    sphere_step: str = "geodesic"           # proj | retract | geodesic
    sphere_geo_max: float = 1.5708
    decode_chunk: int = 8192
    mask_special: bool = True
    mask_illegal: bool = True

    # Teacher support. For these tasks, point teacher_model at a GPT-2 checkpoint
    # fine-tuned on source -> target pairs with train_teacher.py.
    n_pos: int = 32
    prob_thresh: float = 0.01
    support: str = "thresh"                 # thresh | nucleus
    nucleus_p: float = 0.99
    no_repeat_ngram: int = 4
    no_repeat_window: int = 0

    # Drift objective.
    gold_weight: float = 3.0
    teacher_weight: float = 1.0
    # Per-position ramps: teacher weight decays linearly from 1.0 at position 0
    # to pos_teacher_decay at position T-1; gold weight scales from 1.0 to
    # pos_gold_boost. Both default to 1.0 (flat, no ramp).
    pos_teacher_decay: float = 1.0
    pos_gold_boost: float = 1.0
    # EOS repulsion: push content positions (r < eos_pos) away from the EOS token
    # embedding to prevent the model from drifting toward EOS before finishing the
    # translation. 0.0 = off.
    eos_repel: float = 0.0
    # Restrict EOS repulsion to only the last eos_repel_tail positions before the
    # gold EOS (window [eos_pos-tail, eos_pos)). 0 = all content positions.
    eos_repel_tail: int = 0
    attract_temp: float = 0.3
    temp_list: Tuple[float, ...] = ()
    repel: float = 0.0
    repel_block: bool = True
    repel_perpos: bool = True
    repel_intra: float = 0.0
    repel_abs: bool = False
    abs_scale: float = 1.0
    free_prefix: int = 0
    gold_warmup_steps: int = 10_000
    gold_warmup_ramp: int = 10_000
    # Gold DECAY (separate from the warmup above, which ramps the TEACHER in). The gold
    # pull is position-aligned: it requires position r to be target_ids[r]. diag_prefix
    # shows the student's own tokens are implausible under a GOLD prefix 80-85% of the
    # time late in the sequence, i.e. it writes a coherent paraphrase that no longer
    # lines up with the reference. Holding gold at full strength there penalises a
    # possibly-correct translation for being shifted. Decays gold_weight linearly to
    # gold_weight*gold_min_ratio over gold_decay_steps, counted from the step training
    # RESUMED at (like ce_ramp) so adding it mid-run does not jolt a converged model.
    # Do NOT decay to 0: gold carries the only correctness signal -- the teacher term
    # only says "locally plausible" (its top-1 is the gold token 6% of the time at
    # flagged positions), so with gold off the model drifts to fluent non-translation.
    gold_decay_steps: int = 0               # 0 = off (constant gold_weight)
    gold_min_ratio: float = 1.0             # floor as a fraction of gold_weight

    # Stop training a response at the first token the teacher calls implausible.
    # Motivation: the teacher scores position r conditioned on the STUDENT's own tokens
    # 0..r-1, so once the student has drifted the signal answers "what follows this
    # broken text?" rather than "what belongs here". Measured on a 9.2-BLEU checkpoint:
    # the teacher's top-1 agrees with the student 68% of the time at r=0-3 but only 33%
    # by r=16-31, while the SAME tokens rescored under a gold prefix are implausible 80%
    # of the time -- the student is writing a locally coherent paraphrase that no longer
    # lines up with the reference.
    #   none    -- every supervised position trains (the original behaviour)
    #   prefix  -- positions up to and INCLUDING the first flagged one; nothing after
    #   first   -- only the flagged position itself
    #   teacher -- keep gold everywhere, drop only the TEACHER term past the flag. The
    #              gold term is a clean function of (query, r); the teacher term is
    #              conditioned on a prefix the student cannot observe at inference, since
    #              it emits all positions in one pass.
    div_mask: str = "none"                  # none | prefix | first | teacher

    # Discriminative auxiliary loss. The drift objective is an MSE onto ONE gold
    # embedding per position, so where the true token is multi-modal its optimum is the
    # conditional MEAN of those embeddings -- which, in GPT-2's anisotropic input space,
    # sits next to the vocabulary centroid and decodes to ' the'. Measured: the trained
    # student reaches cos(pred, gold)=0.537 while cos(any token, mean direction)=0.545,
    # and a RANDOM-direction error at that same cosine decodes correctly 100% of the
    # time versus the model's 12.4% -- the error is not large, it is aimed at the
    # centroid. Cross-entropy over cosine logits has a DISTRIBUTION as its optimum
    # rather than a mean, so it does not collapse multiple modes onto their average.
    ce_weight: float = 0.0                  # 0 = off; the drift term is always kept
    ce_tau: float = 0.07                    # cosine logits are in [-1, 1] -> sharpen
    ce_ramp: int = 0                        # linear ramp-in, for resuming mid-run

    # The illegal-token allowlist in qwen_features is a HEURISTIC over decoded strings; the
    # gold references are ground truth. Where they disagree the gold wins: a token that
    # occurs in a real reference translation is legal by construction, and forbidding it
    # makes those positions unlearnable AND undecodable -- a hard ceiling on accuracy.
    unforbid_gold: bool = True

    # Generator.
    d_model: int = 768
    nhead: int = 12
    num_layers: int = 12
    ffn_dim: int = 3072
    query_cross_attn: bool = True
    ls_init: float = 1e-4
    gradient_checkpointing: bool = False
    noise_dim: int = 128
    k_samples: int = 1

    # Optimisation. steps=0 derives the ELF-style 100-epoch budget from effective batch.
    steps: int = 0
    epochs: float = 100.0
    queries_per_step: int = 8               # microbatch per rank
    global_batch_size: int = 512            # gradient accumulation fills this batch
    grad_accum_steps: int = 0               # 0 = infer from global_batch_size
    lr: float = 1e-4
    weight_decay: float = 0.0
    grad_clip: float = 2.0
    # Learning-rate schedule. Separate from gold_warmup_steps, which ramps the gold LOSS
    # WEIGHT; these scale the optimizer lr. "none" reproduces the earlier constant-lr runs.
    lr_schedule: str = "none"               # none | cosine
    lr_warmup: int = 0                      # linear lr ramp, in optimizer steps
    lr_min_ratio: float = 0.1               # lr floor as a fraction of cfg.lr
    lr_decay_steps: int = 0                 # 0 = decay across the whole step budget
    temp: float = 1.0
    ema_decay: float = 0.9999
    seed: int = 42
    use_bf16: bool = True

    # Logging/evaluation.
    log_every: int = 50
    eval_every: int = 2_000
    eval_queries: int = 5_000             # ELF WMT14 default; XSum overrides to 100
    eval_n_show: int = 6
    eval_batch_size: int = 16
    save_every: int = 5_000
    checkpoint_dir: str = "runs/seq2seq"


@dataclass
class ReasoningConfig:
    # ── Data ────────────────────────────────────────────────────────────────
    query_len: int = 128               # max query tokens (LEFT-padded so the last token is real)
    # GSM8K gold solutions are ~74 tokens on average (median 69, p90 125). At resp_len=192
    # about 62% of positions had NO gold token, so they received only the teacher force
    # conditioned on the model's own garbage prefix -- a self-referential loop that is what
    # produced the multilingual junk tail.
    resp_len: int = 128                # fixed response length (covers ~p90 of gold solutions)
    # The gold response occupies positions 0..L-1, position L holds ONE end-of-text token as
    # the delimiter (supervised -- the model must learn where to stop), and everything after
    # is pad that carries NO target and is excluded from the loss entirely. Over-long
    # solutions are truncated.
    # Query prompting. Qwen2.5-0.5B is a BASE (pure continuation) model, so the default is NO chat
    # template: the query is wrapped by base_prompt_template into a plain continuation prompt whose
    # last tokens cue the answer. Set use_chat_template=True (--chat-template) only for the -Instruct
    # variant, which is what the system/user/assistant wrapper is for.
    use_chat_template: bool = False    # wrap the query with the Qwen Instruct chat template
    base_prompt_template: str = "Question: {q}\nAnswer:"   # use_chat_template=False: base continuation cue
    # GSM8K natural-language solutions carry inline '<<a-b=c>>' calculator markup that a base LM
    # would never emit, so the teacher's per-position support set penalises it and fights the gold.
    # strip_calc_annot removes the '<<...>>' spans (keeping the prose and the '#### x' line), leaving
    # clean natural language that the teacher scores in-distribution. Off for equation-only data.
    strip_calc_annot: bool = False

    # ── Teacher / manifold (Qwen3) ──────────────────────────────────────────
    # The generation manifold IS this model's INPUT embedding matrix: the generator
    # outputs into it and decodes by cosine-NN to it, and the same model is the teacher
    # that scores [query ++ response] and returns the per-position support set. Teacher
    # vocab == generation vocab by construction.
    teacher_model: str = "Qwen/Qwen2.5-0.5B"   # teacher + generation manifold (input-embedding space)
    # Generator backbone. qwen_backbone=True builds the one-step generator ON a pretrained Qwen
    # (QwenCondGenerator), initialised from backbone_model (default = teacher_model, so the
    # generator hidden dim matches the manifold). False -> the from-scratch CondDriftGenerator.
    qwen_backbone: bool = True
    backbone_model: str = ""           # "" -> use teacher_model as the generator backbone too
    embed_dim: int = 0                 # generator output dim == Qwen hidden; 0 -> set at runtime
    vocab_size: int = 0                # set at runtime
    sphere_norm: bool = True           # unit-sphere manifold (angular geometry); on by default here
    sphere_step: str = "geodesic"      # proj | retract | geodesic
    sphere_geo_max: float = 1.5708     # geodesic max per-step arc length (rad)
    decode_chunk: int = 8192           # vocab chunk size for cosine-NN decode (memory)

    # ── Teacher support set (per response position) ─────────────────────────
    n_pos: int = 32                    # support tokens kept per position (memory: rw is G*T*n_pos*H)
    prob_thresh: float = 0.01          # thresh support: keep tokens with prob > this
    support: str = "thresh"            # thresh | nucleus
    nucleus_p: float = 0.99            # nucleus cumulative-prob cutoff
    no_repeat_ngram: int = 4           # 0=off; k>=2 bans completing a k-gram already in the response
    no_repeat_window: int = 0          # only look back this many response tokens (0 = whole response)
    mask_special: bool = True          # never let a candidate be a special/added token
    # ALLOWLIST junk-token mask (printable ASCII + common typographic). Forbids control chars,
    # UTF-8 byte fragments ('ÃÂ'), non-English scripts etc. from both decode and the teacher
    # support set. Without it the drift lands on junk peripheral tokens and cascades into word
    # salad (as in the unconditional version). Building it decodes every vocab token once
    # (~minutes for Qwen's 150k; fast for GPT-2's 50k).
    mask_illegal: bool = True

    # ── Attraction / repulsion (same knobs as the unconditional repair loss) ─
    # GOLD (real response) attraction. Position-aligned pull toward the dataset's real
    # solution tokens. The teacher support set only encodes "plausible here" and carries no
    # notion of the CORRECT answer, so on an accuracy-scored task (GSM8K) pure teacher
    # self-distillation cannot reach the right tokens. This re-introduces the dataset
    # supervision, conditionally. 0 = off (pure teacher, the previous behaviour).
    gold_weight: float = 0.0           # try 2-5; large relative to the teacher force
    teacher_weight: float = 1.0        # scales the Qwen-support attraction; 0 = pure gold supervision
    attract_temp: float = 0.3          # token-distance affinity temperature (mode-seeking when small)
    temp_list: Tuple[float, ...] = ()  # multi-temp fusion; empty = single attract_temp
    repel: float = 5.0                 # gen-gen repulsion weight
    repel_block: bool = True           # True=repel only within same prefix (K samples);
                                       # False=whole-batch repulsion (like unconditional's 2048)
    repel_perpos: bool = True          # per-position gen-gen repulsion
    repel_intra: float = 5.0           # within-response repulsion (anti token-repeat)
    repel_abs: bool = False
    abs_scale: float = 1.0             # set at runtime to the mean pairwise embedding distance if repel_abs
    free_prefix: int = 0               # leave the first N response positions force-free

    # ── Generator (query embeddings + z -> response embeddings, one pass) ────
    d_model: int = 1024                # generator internal width (decoupled from embed_dim)
    nhead: int = 16
    num_layers: int = 12
    ffn_dim: int = 4096
    # query conditioning. In the main stack the query reaches the response only through
    # self-attention, which LayerScale gates by ls_init (~1e-4). That gate is inherited from
    # the unconditional generator, where it is a FIX (z is injected directly at layer 0, so
    # near-identity blocks pass it through un-smoothed). Here the query is NOT direct, so the
    # same gate suppresses conditioning and the model collapses to the query-independent
    # per-position marginal of the gold responses (identical prefix for every query).
    # UNCONDITIONAL warmup: for the first uncond_warmup_steps the context is filled with a
    # CONSTANT special token (warmup_ctx, default pad==eos==<|endoftext|> for GPT-2 = document
    # start) under NORMAL attention -- so the generator/teacher condition on a clean constant
    # prefix (truly unconditional) rather than masking. This lifts the generator into the
    # "coherent English" basin before the real prefix conditioning is turned on, and keeps the
    # conditioning params gradient-fed throughout. 0 = off (condition from step 0).
    uncond_warmup_steps: int = 0
    # GOLD warmup: for the first gold_warmup_steps the ONLY force is the gold (ground-truth)
    # attraction -- teacher_weight, repel and repel_intra are held at 0 so the generator first
    # learns to emit the real solution (a reachable, fixed target the loss actually drives to 0).
    # Turning the teacher and the (a_rms-scaled, ~10x) repulsion on from scratch instead drowns the
    # gold signal and the output stays random. After warmup the configured teacher/repel/intra come
    # in -- instantly, or linearly over gold_warmup_ramp steps (smoother). Needs gold_weight > 0.
    gold_warmup_steps: int = 10_000
    gold_warmup_ramp: int = 0          # >0: linearly ramp teacher/repel from 0 to target over N steps
    warmup_ctx: str = "pad"            # warmup fills context with this token (pad|eos|bos), normal attn
    query_cross_attn: bool = True      # ungated cross-attention (response slots read query tokens)
    ls_init: float = 1e-4              # LayerScale init; raise (0.1/1.0) to un-gate the main stack
    noise_dim: int = 128               # continuous z dim (diversity across samples of one query)
    k_samples: int = 1                 # deterministic reasoning target; increase only for diversity experiments
    self_cond: bool = False            # self-conditioning: feed draft output back as slot prior
    self_cond_prob: float = 0.5        # fraction of training steps that use the draft pass
    self_cond_steps: int = 1           # eval-time refine passes (1 = single forward, no self-cond)
    self_cond_thresh: float = 0.9      # cosine-sim threshold: only confident positions feed back

    # ── Optimisation ────────────────────────────────────────────────────────
    steps: int = 200_000
    queries_per_step: int = 16         # distinct queries per step; batch = queries_per_step * k_samples
    lr: float = 2e-4
    weight_decay: float = 0.0
    grad_clip: float = 2.0
    temp: float = 1.0                  # z sampling temperature
    ema_decay: float = 0.999
    seed: int = 42
    use_bf16: bool = True
    # LR schedule: cosine decay from lr to lr*lr_min_ratio over lr_decay_steps, with
    # lr_warmup linear warm-up steps at the start. 0 = flat (constant lr).
    lr_schedule: str = "none"          # none | cosine
    lr_warmup: int = 0                 # linear warm-up steps
    lr_decay_steps: int = 0            # cosine decay horizon; 0 = off (flat lr)
    lr_min_ratio: float = 0.1          # floor = lr * lr_min_ratio
    # Gradient accumulation. global_batch_size / (queries_per_step * k_samples * nproc)
    # microbatches are accumulated before each optimizer step. 0 = no accumulation.
    global_batch_size: int = 0         # 0 = no accumulation (one microbatch per step)
    grad_accum_steps: int = 0          # explicit override; takes precedence over global_batch_size
    gradient_checkpointing: bool = False
    # Gold weight decay: linearly decay gold_weight to gold_weight*gold_min_ratio over
    # gold_decay_steps steps counted from the resume step. 0 = off (constant gold_weight).
    gold_decay_steps: int = 0
    gold_min_ratio: float = 1.0        # floor as a fraction of gold_weight
    # Discriminative auxiliary CE loss. Minimiser is the conditional distribution, not the
    # conditional mean, so it addresses the multi-modal centroid collapse in the drift term.
    ce_weight: float = 0.0             # 0 = off; try 0.3-0.5
    ce_tau: float = 0.07               # cosine logit temperature
    ce_ramp: int = 0                   # ramp in from resume step over this many steps

    # ── Logging / eval / checkpoint ─────────────────────────────────────────
    log_every: int = 50
    eval_every: int = 1_000
    eval_queries: int = 200            # held-out queries scored at eval
    eval_n_show: int = 6
    eval_accuracy: bool = True         # gsm8k/math: extract final answer and score exact-match
    eval_on_train: bool = False        # overfit diagnostic: eval the TRAIN queries
    eval_gen_ppl: bool = False         # FLM-style generative perplexity (slow; needs gpt2-large)
    eval_ppl_model: str = "gpt2-large" # reference LM for gen-ppl (FLM default; offline-cached)
    save_every: int = 5_000
    checkpoint_dir: str = "runs/cond_drift"
