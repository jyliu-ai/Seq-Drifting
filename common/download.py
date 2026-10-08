"""Download selected release weights into the Hugging Face cache."""
import argparse
import os

from huggingface_hub import snapshot_download

from .checkpoint import DEFAULT_REPO


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=os.environ.get("HF_REPO", DEFAULT_REPO))
    parser.add_argument("--revision", default=os.environ.get("HF_REVISION", "main"))
    parser.add_argument("--include", nargs="+", default=["*.pt"], help="repository filename patterns")
    args = parser.parse_args()
    print(snapshot_download(repo_id=args.repo, revision=args.revision,
                            allow_patterns=args.include))


if __name__ == "__main__":
    main()
