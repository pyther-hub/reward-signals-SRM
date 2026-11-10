"""Minimal experiment logging: console logger + W&B run with config and git hash."""

import logging
import os
import subprocess
from pathlib import Path

import wandb

ROOT = Path(__file__).resolve().parents[1]


def get_logger(name: str = "exp") -> logging.Logger:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")
    return logging.getLogger(name)


def _load_wandb_key() -> str | None:
    """Look for WANDB_API_KEY in: env var -> repo .env -> Kaggle Secrets."""
    if os.environ.get("WANDB_API_KEY"):
        return os.environ["WANDB_API_KEY"]
    env_file = ROOT / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            key, sep, value = line.partition("=")
            if sep and key.strip() == "WANDB_API_KEY":
                return value.strip().strip("'\"")
    try:
        from kaggle_secrets import UserSecretsClient
        return UserSecretsClient().get_secret("WANDB_API_KEY")
    except Exception:
        return None


def git_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def init_wandb(config: dict, project: str = "reward-signals-SRM", name: str | None = None, **kwargs):
    """Start a W&B run. Config + git hash are stored with the run."""
    key = _load_wandb_key()
    if key:
        wandb.login(key=key)
    return wandb.init(project=project, name=name, config={**config, "git_hash": git_hash()}, **kwargs)
