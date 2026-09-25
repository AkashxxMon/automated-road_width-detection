"""Download facebook/sam3 into ./model/sam3 (this project only - not the
shared Hugging Face cache).

facebook/sam3 is gated with *manual* approval from Meta - having a
Hugging Face account and a token isn't enough by itself:

  1. Log into huggingface.co, open https://huggingface.co/facebook/sam3,
     click "Agree and request access", and wait for Meta to approve it
     (this is a manual review queue, not instant).
  2. Once approved, create a token at https://huggingface.co/settings/tokens
     (Read access is enough).
  3. Run this script with that token.

Usage:
    python src/download_sam3.py --token hf_xxxxxxxxxxxx
    # or, once, so you don't have to pass --token every time:
    python -c "from huggingface_hub import login; login()"
    python src/download_sam3.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download
from huggingface_hub.errors import GatedRepoError, HfHubHTTPError

REPO_ID = "facebook/sam3"
LOCAL_DIR = Path("model/sam3")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--token", default=None, help="HF access token (or set HF_TOKEN env var, or run huggingface_hub.login() once)")
    ap.add_argument("--repo-id", default=REPO_ID, help="e.g. facebook/sam3 or facebook/sam3.1")
    args = ap.parse_args()

    LOCAL_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {args.repo_id} -> {LOCAL_DIR}/ ...")
    try:
        path = snapshot_download(
            repo_id=args.repo_id,
            local_dir=str(LOCAL_DIR),
            token=args.token,
        )
        print(f"\nDone. Weights are in {path}")
        print(f"Load with: Sam3Model.from_pretrained({str(LOCAL_DIR)!r})")
    except GatedRepoError:
        print(f"\n[blocked] {args.repo_id} is gated and this token doesn't have approved access yet.")
        print(f"Request access at https://huggingface.co/{args.repo_id} and wait for Meta's approval, then retry.")
        raise SystemExit(1)
    except HfHubHTTPError as e:
        print(f"\n[error] {e}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
