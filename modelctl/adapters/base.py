"""Adapter contract + shared symlink helper.

An adapter *projects* the source-of-truth cache into one tool's expected view.
The contract is deliberately tiny:

    accepts(repo)            -> is this repo relevant to my tool?
    sync(repos, dry_run)     -> make my tool's view match the cache (idempotent)
    doctor()                 -> human-readable config/env checks

`sync` takes **options so an adapter can accept a flag (ollama's opt-in
import) without the CLI growing a branch per adapter. Adapters ignore options
they do not recognise.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from ..cache import CACHE_DIRNAME, Repo


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

    def sync(self, repos: list[Repo], *, dry_run: bool = False, **options) -> list[Action]:
        return []

    def doctor(self) -> list[str]:
        return []


def ensure_hardlink_tree(target: Path, source: Path, adapter: str, *, dry_run: bool) -> Action:
    """Mirror `source`'s file tree into `target` with real dirs + hard links,
    reported as ONE action for the whole package.

    Why hard links: the Splash indexer resolves real paths and rejects both a
    symlinked package ("escapes the models directory") and symlinked artifacts
    inside a real package dir ("path escapes its directory"). A hard link has
    no separate real path, so it passes both checks, and on one filesystem it
    shares inodes and so costs no extra bytes.

    Idempotent by inode: a file already hard-linked to its source is left
    alone, and one whose source was REPLACED (a re-download writes a new inode)
    is relinked, so a stale mirror repairs itself on the next sync."""
    if not _same_filesystem(source, target):
        return Action(adapter, "skip", str(target),
                      "cannot hard-link across filesystems; point this app's models dir "
                      "at the store, or copy the package in")
    linked = relinked = uptodate = 0
    for src in sorted(source.rglob("*")):
        if src.is_dir() or CACHE_DIRNAME in src.parts:
            continue
        dst = target / src.relative_to(source)
        try:
            ds, ss = dst.stat(), src.stat()
            if (ds.st_ino, ds.st_dev) == (ss.st_ino, ss.st_dev):
                uptodate += 1
                continue
            stale = True
        except OSError:
            stale = dst.is_symlink()   # a broken symlink still needs replacing
        if not dry_run:
            dst.parent.mkdir(parents=True, exist_ok=True)
            if stale:
                dst.unlink()
            os.link(src, dst)
        if stale:
            relinked += 1
        else:
            linked += 1
    if linked == relinked == 0:
        return Action(adapter, "skip", str(target), f"up to date ({uptodate} hard links)")
    op = "relink" if relinked and not linked else "link"
    bits = [f"{linked} new"] if linked else []
    if relinked:
        bits.append(f"{relinked} stale")
    if uptodate:
        bits.append(f"{uptodate} unchanged")
    return Action(adapter, op, str(target), f"hard links: {', '.join(bits)} (no extra bytes)")


def _same_filesystem(a: Path, b: Path) -> bool:
    """Compare st_dev of `a` and of `b`'s nearest EXISTING ancestor, since the
    target itself usually doesn't exist yet."""
    probe = b
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    try:
        return a.stat().st_dev == probe.stat().st_dev
    except OSError:
        return False


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
