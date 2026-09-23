"""Conditional text drifting (query -> response) with a Qwen3 teacher.

A separate package so the reasoning implementation stays isolated.
Reuses the parent package's tested pieces (generator Block, sphere/geodesic goal
geometry, no-repeat-ngram support masking) and only replaces what conditioning +
a Qwen teacher actually change:

  data.py         GSM8K / MATH  (query, response) pairs, tokenised with Qwen.
  qwen_features.py Qwen input-embedding manifold: sphere-normalised rows, decode
                  by cosine-NN, query-token embedding lookup for conditioning.
  qwen_teacher.py build_repairs_cond: Qwen sees [query ++ response prefix] and
                  returns a per-response-position support set (+ no-repeat).
  cond_generator.py CondDriftGenerator: (query embeddings, z) -> response
                  embeddings in ONE forward pass (non-autoregressive).
  cond_drift_loss.py cond_repair_loss: parent's attraction/sphere step, but the
                  gen-gen repulsion is BLOCK-DIAGONAL per query (samples of
                  different queries are not pushed apart).
  train.py / evaluate.py  conditional loop + GSM8K answer-accuracy eval.
"""
