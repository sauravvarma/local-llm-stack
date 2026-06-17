"""Adapter contract + shared symlink helper.

An adapter *projects* the source-of-truth cache into one tool's expected view.
The contract is deliberately tiny:

    accepts(repo)            -> is this repo relevant to my tool?
    sync(repos, dry_run)     -> make my tool's view match the cache (idempotent)
    doctor()                 -> human-readable config/env checks
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from ..cache import Repo


@dataclass
class Action:
    adapter: str
    op: str        # "link" | "copy" | "native" | "skip" | "relink" | "error"
    target: str
    detail: str = ""

    def __str__(self) -> str:
        glyph = {
            "link": "+", "relink": "~", "copy": "C", "native": "=",
            "skip": ".", "error": "!",
        }.get(self.op, "?")
        line = f"  {glyph} [{self.adapter}] {self.target}"
        return f"{line}  ({self.detail})" if self.detail else line


class Adapter:
    name = "base"

    def accepts(self, repo: Repo) -> bool:
        return False

    def sync(self, repos: list[Repo], *, dry_run: bool = False) -> list[Action]:
        return []

    def doctor(self) -> list[str]:
        return []


def ensure_symlink(target: Path, source: Path, adapter: str, *, dry_run: bool) -> Action:
    """Idempotently point `target` (symlink) at `source` (file or directory).
    Never overwrites a real file/dir — only manages symlinks it would itself
    create. A no-op when target already resolves to source (e.g. source already
    lives under the target tree)."""
    rel_target = str(target)
    try:
        if target.exists() and source.exists() and target.resolve() == source.resolve():
            if not target.is_symlink():
                return Action(adapter, "skip", rel_target, "already in place")
    except OSError:
        pass
    if target.is_symlink():
        try:
            current = os.readlink(target)
        except OSError:
            current = None
        if current and Path(current) == source:
            return Action(adapter, "skip", rel_target, "up to date")
        if not dry_run:
            target.unlink()
            target.symlink_to(source)
        return Action(adapter, "relink", rel_target, f"-> {source}")
    if target.exists():
        return Action(adapter, "error", rel_target, "exists and is not a symlink; left untouched")
    if not dry_run:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.symlink_to(source)
    return Action(adapter, "link", rel_target, f"-> {source}")
