"""Configuration — env-var driven with sensible defaults, no config file needed.

The primary store is the repo's own `models/` folder (flat publisher/model
layout — the thing a future CLI/GUI manages). The HF cache is scanned too so
models downloaded the classic way are also visible.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from .cache import Repo, hub_dir, scan_flat, scan_hf_cache

REPO_ROOT = Path(__file__).resolve().parents[1]


def _env_path(name: str, default: Path) -> Path:
    val = os.environ.get(name)
    return Path(val).expanduser() if val else default


# Canonical store: this project's own `models/` folder (gitignored — never
# committed). Override with MODELCTL_STORE.
DEFAULT_STORE = REPO_ROOT / "models"


@dataclass
class Config:
    store: Path                 # canonical flat store (download target, source of truth)
    extra_stores: list[Path]    # additional flat stores to scan (read-only)
    hub: Path                   # HF hub cache (scanned read-only)
    scan_hub: bool
    lmstudio_dir: Path

    @classmethod
    def load(cls) -> "Config":
        store_env = os.environ.get("MODELCTL_STORE")
        stores = [Path(p).expanduser() for p in store_env.split(":")] if store_env else [DEFAULT_STORE]
        return cls(
            store=stores[0],
            extra_stores=stores[1:],
            hub=hub_dir(),
            scan_hub=os.environ.get("MODELCTL_SCAN_HUB", "1") != "0",
            # LM Studio is pointed at the store, so it reads it natively; only
            # out-of-store models (e.g. in the HF cache) get symlinked in.
            lmstudio_dir=_env_path("MODELCTL_LMSTUDIO_DIR", stores[0]),
        )

    def scan(self) -> list[Repo]:
        """Scan every store, primary first; de-dupe by repo_id (first wins)."""
        repos: list[Repo] = []
        seen: set[str] = set()
        sources = [(self.store, "store")]
        sources += [(p, f"store:{p.name}") for p in self.extra_stores]
        for path, label in sources:
            for r in scan_flat(path, label):
                if r.repo_id not in seen:
                    seen.add(r.repo_id)
                    repos.append(r)
        if self.scan_hub:
            for r in scan_hf_cache(self.hub):
                if r.repo_id not in seen:
                    seen.add(r.repo_id)
                    repos.append(r)
        return repos
