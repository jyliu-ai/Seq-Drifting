"""Config for conditional (query -> response) text drifting with a Qwen3 teacher.

Mirrors the unconditional TextDriftConfig where the mechanism is identical
(sphere geometry, teacher support set, per-position / intra repulsion, no-repeat),
and adds the conditional pieces (dataset, query/response lengths, samples-per-query,
Qwen teacher). d_model here is the GENERATOR's internal width; the generator's
OUTPUT dim (embed_dim) is set from the Qwen input-embedding size at runtime.
"""
from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class CondDriftConfig:
    # ── Data ────────────────────────────────────────────────────────────────
    dataset_name: str = "gsm8k"        # gsm8k | gsm8k_local | math | jsonl | owt
    owt_dir: str = ""                  # dataset_name=owt: dir of *.jsonl.zst OpenWebText2 shards
    owt_docs: int = 50000              # how many docs to load (prefix->continuation examples)
    local_json: str = ""               # dataset_name=gsm8k_local: nested GSM8K.json (offline)
    jsonl_path: str = ""               # dataset_name=jsonl: path to a {query,response} jsonl
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
