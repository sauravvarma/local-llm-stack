"""LM Studio-family adapters: LM Studio and Bionic share one engine set.

Both apps ship the same llama.cpp (GGUF) and mlx-llm (MLX) runtimes and index a
`<models_dir>/<publisher>/<model>/` tree, following symlinks. So the projection
is identical and only the models dir differs:

    GGUF:   <models_dir>/<pub>/<model>/<file>.gguf   -> cache blob (per-file symlink)
    MLX :   <models_dir>/<pub>/<model>              -> model dir (directory symlink)
    SPLASH: <models_dir>/<pub>/<model>              -> real dirs + per-file HARD links

Splash is the exception, and not for the reason you'd guess. Its indexer
resolves real paths and enforces containment at two levels, which rules out
symlinks entirely. All three of these were tried against Bionic 1.1.5:

    directory symlink   -> Model package escapes the models directory: <pkg>
    per-file symlinks   -> Model package path escapes its directory: <pkg>/manifest.json
    per-file hard links -> indexed, "format": "splash"

Hard links win because a hard link has no separate real path: it IS the file,
at that path, so both checks pass. On one filesystem it also shares inodes, so
the mirror costs no extra bytes (17.4 GB mirrored with zero change in free
space). GGUF and MLX never reach that code path, which is why they follow
symlinks happily.

Two consequences worth knowing. A re-download replaces a file with a NEW inode,
orphaning the mirror, so the tree is kept fresh by comparing inodes and
relinking what drifted. And because inodes are shared, deleting the package
from the store frees nothing until this mirror goes too, which is why every
mirror is recorded in a ledger: the next sync removes it once its source is
gone (see base.MirrorLedger).

When the package already resolves inside `models_dir` (the app's models dir IS
the store) nothing is needed, and a nested store can still use a plain
directory symlink, since the real path stays contained.

Bytes live once in the store. Full-precision/GPTQ safetensors are skipped,
because neither app's runtimes can load them.
"""

from __future__ import annotations

from pathlib import Path

from ..cache import Repo
from .. import downloads
from .base import (Action, Adapter, MirrorLedger, ensure_hardlink_tree,
                   ensure_symlink, prune_links)


def _resolves_within(child: Path, parent: Path) -> bool:
    """True when `child`'s real path sits inside `parent`'s real path. Both are
    resolved first, because the whole point is what the OS sees after following
    symlinks, which is what the splash indexer checks."""
    try:
        return child.resolve().is_relative_to(parent.resolve())
    except OSError:
        return False


class LMStudioFamilyAdapter(Adapter):
    """Shared projection for any app built on the LM Studio engines."""

    name = "lmstudio-family"

    def __init__(self, models_dir: Path):
        self.models_dir = models_dir

    def accepts(self, repo: Repo) -> bool:
        return repo.fmt in ("gguf", "mlx", "splash")

    def sync(self, repos: list[Repo], *, dry_run: bool = False, **options) -> list[Action]:
        """Project every accepted repo, then (with prune=True) remove what the
        store no longer has. Options: prune, sources, protected, state_dir."""
        state_dir = options.get("state_dir")
        ledger = MirrorLedger(Path(state_dir) / "mirrors.json" if state_dir else None)
        actions: list[Action] = []
        desired: set[Path] = set()      # every path this sync wants to exist
        for repo in repos:
            if not self.accepts(repo):
                continue
            # A bare model (no publisher in its repo id) yields publisher ==
            # model, so it projects to <dir>/<name>/<name>. These apps index a
            # two-level tree and a bare model has no publisher to use, so the
            # name is repeated deliberately; `modelctl adopt` is the way to give
            # it a real publisher.
            target = self.models_dir / repo.publisher / repo.model
            if repo.fmt == "gguf":
                # Per-file links are safe even mid-download: hf writes partial
                # bytes to temp files and renames on completion, so a .gguf at
                # its final path is always whole.
                for f in repo.gguf_files:
                    desired.add(target / f.filename)
                    actions.append(ensure_symlink(
                        target / f.filename, f.path, self.name, dry_run=dry_run))
                continue
            # A directory-level projection exposes the whole model, so a
            # partial one would show up and then fail to load.
            if downloads.inspect(repo.root, active_dirs=set()).partial:
                actions.append(Action(self.name, "skip", str(target),
                                      "incomplete download; not projected until it finishes "
                                      "(see `modelctl status`)"))
                continue
            desired.add(target)
            if repo.fmt == "splash" and not _resolves_within(repo.root, self.models_dir):
                # Symlinks are rejected at both levels, so mirror the package
                # with hard links instead: same inodes, no extra bytes.
                actions.append(ensure_hardlink_tree(
                    target, repo.root, self.name, dry_run=dry_run, ledger=ledger))
            else:  # mlx, or splash already inside the models dir: link the directory
                actions.append(ensure_symlink(target, repo.root, self.name, dry_run=dry_run))
        if options.get("prune"):
            actions += prune_links(self.models_dir, desired,
                                   sources=list(options.get("sources", [])),
                                   protected=list(options.get("protected", [])),
                                   adapter=self.name, dry_run=dry_run)
            actions += ledger.prune(self.name, self.models_dir, desired, dry_run=dry_run)
        ledger.save(dry_run=dry_run)
        return actions

    def doctor(self) -> list[str]:
        state = "exists" if self.models_dir.is_dir() else "missing, created on first sync"
        return [f"models dir: {self.models_dir} ({state})"]


class LMStudioAdapter(LMStudioFamilyAdapter):
    name = "lmstudio"

    def doctor(self) -> list[str]:
        return super().doctor() + [
            "Serves GGUF + MLX by symlink, splash by hard link (same filesystem only). "
            "If LM Studio uses a custom folder, set MODELCTL_LMSTUDIO_DIR.",
        ]
