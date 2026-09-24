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
from the store does not reclaim space until this mirror goes too.

When the package already resolves inside `models_dir` (the app's models dir IS
the store) nothing is needed, and a nested store can still use a plain
directory symlink, since the real path stays contained.

Bytes live once in the store. Full-precision/GPTQ safetensors are skipped,
because neither app's runtimes can load them.
"""

from __future__ import annotations

from pathlib import Path

from ..cache import Repo
from .base import Action, Adapter, ensure_hardlink_tree, ensure_symlink


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
        actions: list[Action] = []
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
                for f in repo.gguf_files:
                    actions.append(ensure_symlink(
                        target / f.filename, f.path, self.name, dry_run=dry_run))
            elif repo.fmt == "splash" and not _resolves_within(repo.root, self.models_dir):
                # Symlinks are rejected at both levels, so mirror the package
                # with hard links instead: same inodes, no extra bytes.
                actions.append(ensure_hardlink_tree(
                    target, repo.root, self.name, dry_run=dry_run))
            else:  # mlx, or splash already inside the models dir: link the directory
                actions.append(ensure_symlink(target, repo.root, self.name, dry_run=dry_run))
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
