"""Configuration for WMT14 De-En and XSum one-step conditional generation."""

from dataclasses import dataclass
from typing import Tuple


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
