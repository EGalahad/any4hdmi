from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import HfApi


DEFAULT_TOKEN_PATH = Path("~/.cache/huggingface/token").expanduser()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Upload a local any4hdmi dataset folder to a Hugging Face dataset repo."
    )
    parser.add_argument("folder", help="Local folder to upload.")
    parser.add_argument("repo", help="Target Hugging Face repo id, for example elijahgalahad/any4hdmi-lafan.")
    parser.add_argument(
        "--path-in-repo",
        default=None,
        help="Optional subdirectory inside the target repo.",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="Optional target branch or revision. Defaults to the repo default branch.",
    )
    parser.add_argument(
        "--commit-message",
        default=None,
        help="Optional commit message. Defaults to 'Upload <folder-name>'.",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="Optional Hugging Face token. Defaults to HF_TOKEN or ~/.cache/huggingface/token.",
    )
    parser.add_argument(
        "--token-path",
        default=str(DEFAULT_TOKEN_PATH),
        help="Fallback token file path used when --token and HF_TOKEN are unset.",
    )
    return parser.parse_args()


def _resolve_folder(folder: str) -> Path:
    path = Path(folder).expanduser().resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"Upload folder not found: {path}")
    return path


def _resolve_token(*, token: str | None, token_path: str | None) -> str:
    if token:
        return token

    env_token = os.environ.get("HF_TOKEN")
    if env_token:
        return env_token

    if token_path is None:
        raise RuntimeError("No Hugging Face token provided. Set HF_TOKEN or pass --token.")

    path = Path(token_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(
            f"Hugging Face token not found: {path}. Set HF_TOKEN or pass --token."
        )

    resolved = path.read_text(encoding="utf-8").strip()
    if not resolved:
        raise RuntimeError(f"Hugging Face token file is empty: {path}")
    return resolved


def upload_folder_to_hf(
    *,
    folder: Path,
    repo: str,
    token: str,
    path_in_repo: str | None = None,
    revision: str | None = None,
    commit_message: str | None = None,
):
    api = HfApi(token=token)
    api.create_repo(repo_id=repo, repo_type="dataset", exist_ok=True)
    return api.upload_folder(
        repo_id=repo,
        repo_type="dataset",
        folder_path=str(folder),
        path_in_repo=path_in_repo,
        revision=revision,
        commit_message=commit_message or f"Upload {folder.name}",
    )


def main() -> None:
    args = _parse_args()
    folder = _resolve_folder(args.folder)
    token = _resolve_token(token=args.token, token_path=args.token_path)
    commit_info = upload_folder_to_hf(
        folder=folder,
        repo=args.repo,
        token=token,
        path_in_repo=args.path_in_repo,
        revision=args.revision,
        commit_message=args.commit_message,
    )
    oid = getattr(commit_info, "oid", None) or getattr(commit_info, "commit_oid", None)
    print(
        "Uploaded folder:",
        f"folder={folder}",
        f"repo={args.repo}",
        f"path_in_repo={args.path_in_repo or '.'}",
        f"revision={args.revision or 'default'}",
        f"commit={oid or commit_info}",
    )


if __name__ == "__main__":
    main()
