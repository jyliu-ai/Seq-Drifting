"""Load local or Hugging Face checkpoints, including configurations saved by old packages."""
import os
import pickle
import sys
from dataclasses import fields, is_dataclass, MISSING
from pathlib import Path

import torch

from .config import TextDriftConfig, ContinuationConfig, Seq2SeqDriftConfig, ReasoningConfig

DEFAULT_REPO = "jyliuAI/Seq-Drifting"


def resolve_checkpoint(path):
    path = str(path)
    if Path(path).is_file():
        return path
    repo = os.environ.get("HF_REPO", DEFAULT_REPO)
    if path.startswith("hf://"):
        parts = path[5:].split("/", 2)
        if len(parts) != 3:
            raise ValueError("expected hf://owner/repository/filename.pt")
        repo, filename = "/".join(parts[:2]), parts[2]
    elif "/" not in path and "\\" not in path:
        filename = path
    else:
        raise FileNotFoundError(path)
    from huggingface_hub import hf_hub_download
    return hf_hub_download(repo_id=repo, filename=filename,
                           revision=os.environ.get("HF_REVISION", "main"))


class Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        # Released files contain a dataclass cfg, rather than a model object.
        # Redirect only historical config symbols; retain their saved fields.
        if module == "config" or module.endswith(".config"):
            if name == "TextDriftConfig":
                return TextDriftConfig
            if name == "Seq2SeqDriftConfig":
                return Seq2SeqDriftConfig
            if name == "CondDriftConfig":
                if "reason" in module:
                    return ReasoningConfig
                if "seq2seq" in module:
                    return Seq2SeqDriftConfig
                return ContinuationConfig
        return super().find_class(module, name)


# torch.load's legacy serialization path expects these pickle entry points.
load, loads, dump, dumps = pickle.load, pickle.loads, pickle.dump, pickle.dumps


def load_checkpoint(path, map_location="cpu", **kwargs):
    kwargs.pop("weights_only", None)
    state = torch.load(resolve_checkpoint(path), map_location=map_location,
                       pickle_module=sys.modules[__name__], weights_only=False, **kwargs)
    cfg = state.get("cfg") if isinstance(state, dict) else None
    if cfg is not None and is_dataclass(cfg):
        for item in fields(cfg):
            if not hasattr(cfg, item.name):
                if item.default is not MISSING:
                    setattr(cfg, item.name, item.default)
                elif item.default_factory is not MISSING:
                    setattr(cfg, item.name, item.default_factory())
    return state
