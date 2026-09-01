"""LM Studio adapter — serves the two formats LM Studio can run: GGUF and MLX.

    GGUF:  <models_dir>/<pub>/<model>/<file>.gguf   -> cache blob (per-file symlink)
    MLX :  <models_dir>/<pub>/<model>               -> model dir (directory symlink)

LM Studio indexes this tree and follows symlinks, so bytes live once in the store.
Full-precision/GPTQ safetensors are skipped — LM Studio's runtimes can't load them.
"""

from __future__ import annotations

from pathlib import Path

from ..cache import Repo
from .base import Action, Adapter, ensure_symlink, prune_empty, remove_symlink


class LMStudioAdapter(Adapter):
    name = "lmstudio"

    def __init__(self, models_dir: Path):
        self.models_dir = models_dir

    def accepts(self, repo: Repo) -> bool:
        return repo.fmt in ("gguf", "mlx")

    def sync(self, repos: list[Repo], *, dry_run: bool = False) -> list[Action]:
        actions: list[Action] = []
        for repo in repos:
            if not self.accepts(repo):
                continue
            if repo.fmt == "gguf":
                for f in repo.gguf_files:
                    target = self.models_dir / repo.publisher / repo.model / f.filename
                    actions.append(ensure_symlink(target, f.path, self.name, dry_run=dry_run))
            else:  # mlx: link the whole model directory
                target = self.models_dir / repo.publisher / repo.model
                actions.append(ensure_symlink(target, repo.root, self.name, dry_run=dry_run))
        return actions

    def remove(self, repo: Repo, files: list, *, dry_run: bool = False) -> list[Action]:
        """Tear down the symlinks `sync` put here, so deleting the store model
        doesn't leave LM Studio indexing dangling links."""
        if not self.accepts(repo):
            return []
        base = self.models_dir / repo.publisher / repo.model
        whole = len(files) == len(repo.files)
        actions: list[Action] = []
        parents: list[Path] = []
        if repo.fmt == "gguf":
            for f in files:
                if not f.is_gguf:
                    continue
                target = base / f.filename
                a = remove_symlink(target, self.name, dry_run=dry_run)
                if a:
                    actions.append(a)
                    parents.append(target.parent)
        elif whole:  # mlx: one directory symlink
            a = remove_symlink(base, self.name, dry_run=dry_run)
            if a:
                actions.append(a)
                parents.append(base.parent)
        if not dry_run:
            # deepest first, so a quant subfolder goes before its model dir
            for d in sorted(set(parents), key=lambda p: len(p.parts), reverse=True):
                actions += prune_empty(d, self.models_dir, self.name)
        return actions

    def doctor(self) -> list[str]:
        state = "exists" if self.models_dir.is_dir() else "missing — created on first sync"
        return [
            f"models dir: {self.models_dir} ({state})",
            "Serves GGUF + MLX. If LM Studio uses a custom folder, set MODELCTL_LMSTUDIO_DIR.",
        ]
