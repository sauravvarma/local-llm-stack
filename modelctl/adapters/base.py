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

import json
import os
from dataclasses import dataclass
from pathlib import Path

from ..cache import CACHE_DIRNAME, Repo


@dataclass
class Action:
    adapter: str
    op: str        # "link" | "copy" | "native" | "skip" | "relink" | "error" | "unlink"
    target: str
    detail: str = ""

    def __str__(self) -> str:
        glyph = {
            "link": "+", "relink": "~", "copy": "C", "native": "=",
            "skip": ".", "error": "!", "unlink": "-",
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


def ensure_hardlink_tree(target: Path, source: Path, adapter: str, *, dry_run: bool,
                         ledger: "MirrorLedger | None" = None) -> Action:
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
    rels: list[str] = []
    for src in sorted(source.rglob("*")):
        if src.is_dir() or CACHE_DIRNAME in src.parts:
            continue
        rels.append(str(src.relative_to(source)))
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
    # Record ownership whenever the mirror is made OR merely confirmed: an
    # up-to-date mirror shares inodes with the store right now, which proves
    # it is ours. That is how mirrors made before the ledger existed get
    # adopted into it.
    if ledger is not None:
        ledger.record(adapter, target, source, rels)
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


# ---------------------------------------------------------------- pruning
#
# `sync` promises to make a tool's view MATCH the store, which includes taking
# away what the store no longer has. The hard part is deleting inside a
# directory an app owns without ever touching something modelctl did not make.
#
# Symlinks prove their own ownership: a link whose target lies inside one of
# modelctl's sources (the stores, the HF cache) was made by modelctl, whether
# its target still exists or not. So those are pruned by rule, with no state.
#
# Hard-link mirrors do not. Once the store copy is deleted, a mirror is just a
# real directory of real files, indistinguishable from a model the app
# downloaded itself. So mirrors are recorded in a small ledger when sync makes
# or confirms them, and only recorded files are ever removed.


def _is_within(path: Path, roots: list[Path]) -> bool:
    for root in roots:
        for form in {root, _safe_resolve(root)}:
            try:
                if path == form or path.is_relative_to(form):
                    return True
            except ValueError:
                pass
    return False


def _safe_resolve(p: Path) -> Path:
    try:
        return p.resolve()
    except OSError:
        return p


def _link_target(link: Path) -> Path:
    raw = Path(os.readlink(link))
    return raw if raw.is_absolute() else link.parent / raw


def _remove_empty_parents(start: Path, stop: Path, *, dry_run: bool) -> None:
    """Remove directories left empty by a prune, walking up to (not
    including) `stop`. Only ever removes a directory that is actually empty."""
    if dry_run:
        return
    d = start
    while d != stop and d.is_relative_to(stop):
        try:
            d.rmdir()                  # fails, harmlessly, unless empty
        except OSError:
            return
        d = d.parent


def prune_links(root: Path, desired: set[Path], *, sources: list[Path],
                protected: list[Path], adapter: str, dry_run: bool) -> list[Action]:
    """Remove symlinks under `root` that point into `sources` but that this
    sync did not ask for: models deleted from the store, and models no longer
    projected (now partial, now shadowed).

    Never runs when `root` is itself a store or inside one: there the directory
    IS the source of truth and its links belong to the user (`adopt --link`
    makes exactly such links). Never descends into a protected directory, and
    never touches anything that is not a symlink."""
    if not root.is_dir() or _is_within(root, protected) or _is_within(_safe_resolve(root), protected):
        return []
    wanted = {str(p) for p in desired}
    actions: list[Action] = []
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        here = Path(dirpath)
        # os.walk lists directory symlinks among dirnames without following
        # them, which is what we want. Do not descend into protected trees.
        dirnames[:] = [d for d in dirnames
                       if (here / d).is_symlink() or not _is_within(here / d, protected)]
        for name in list(dirnames) + filenames:
            link = here / name
            if not link.is_symlink() or str(link) in wanted:
                continue
            try:
                target = _link_target(link)
            except OSError:
                continue
            if not _is_within(target, sources):
                continue               # somebody else's link: not ours to judge
            gone = not link.exists()
            if not dry_run:
                link.unlink()
            actions.append(Action(adapter, "unlink", str(link),
                                  "target no longer in the store" if gone
                                  else "no longer projected"))
            _remove_empty_parents(link.parent, root, dry_run=dry_run)
    return actions


class MirrorLedger:
    """Which hard-link mirrors modelctl made, and exactly which files.

    Lives in the store (`<store>/.modelctl/mirrors.json`), since it describes
    projections OF that store. A missing or unreadable ledger is treated as
    empty: the only cost is that an unrecorded mirror is left alone, which is
    the safe failure."""

    def __init__(self, path: Path | None):
        self.path = path
        self.entries: dict[str, dict] = {}
        if path is not None:
            try:
                data = json.loads(path.read_text())
                if isinstance(data, dict):
                    self.entries = {k: v for k, v in data.items() if isinstance(v, dict)}
            except (OSError, ValueError):
                pass

    def record(self, adapter: str, target: Path, source: Path, files: list[str]) -> None:
        self.entries[str(target)] = {"adapter": adapter, "source": str(source),
                                     "files": sorted(files)}

    def save(self, *, dry_run: bool) -> None:
        if self.path is None or dry_run:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self.entries, indent=2, sort_keys=True))
            tmp.replace(self.path)           # atomic: never a half-written ledger
        except OSError:
            pass

    def prune(self, adapter: str, root: Path, desired: set[Path], *, dry_run: bool) -> list[Action]:
        """Remove recorded mirrors under `root` that this sync did not ask for.
        Deletes only the files it recorded; anything else found inside is left,
        along with the directory holding it."""
        wanted = {str(p) for p in desired}
        actions: list[Action] = []
        for target, entry in list(self.entries.items()):
            t = Path(target)
            if entry.get("adapter") != adapter or target in wanted or not _is_within(t, [root]):
                continue
            recorded = set(entry.get("files", []))
            present = [f for f in t.rglob("*") if f.is_file()] if t.is_dir() else []
            leftovers = [f for f in present if str(f.relative_to(t)) not in recorded]
            removed = 0
            for rel in sorted(recorded):
                f = t / rel
                if f.is_file() and not f.is_symlink():
                    if not dry_run:
                        f.unlink()
                    removed += 1
            if not dry_run:
                for d in sorted((d for d in t.rglob("*") if d.is_dir()), reverse=True):
                    try:
                        d.rmdir()
                    except OSError:
                        pass
                _remove_empty_parents(t, root, dry_run=False)
                del self.entries[target]
            detail = f"removed {removed} hard-linked file(s)"
            if leftovers:
                detail += f"; kept {len(leftovers)} file(s) modelctl did not create"
            actions.append(Action(adapter, "unlink", target, detail))
        return actions
