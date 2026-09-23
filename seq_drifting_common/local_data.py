"""Read explicitly supplied local corpus shards without changing token packing."""
import io
import json
from pathlib import Path


def iter_texts(path):
    root = Path(path)
    files = sorted([*root.glob("*.jsonl"), *root.glob("*.jsonl.zst")]) if root.is_dir() else [root]
    if not files:
        raise FileNotFoundError(f"No JSONL shards under {root}")
    for file in files:
        with file.open("rb") as handle:
            if file.name.endswith(".zst"):
                import zstandard
                reader = zstandard.ZstdDecompressor().stream_reader(handle)
            else:
                reader = handle
            with io.TextIOWrapper(reader, encoding="utf-8") as lines:
                for index, line in enumerate(lines, 1):
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    text = row.get("text")
                    if not isinstance(text, str):
                        raise ValueError(f"{file}:{index}: expected a text string")
                    yield text
