"""LM Studio adapter — serves the two formats LM Studio can run: GGUF and MLX.

    GGUF:  <models_dir>/<pub>/<model>/<file>.gguf   -> cache blob (per-file symlink)
    MLX :  <models_dir>/<pub>/<model>               -> model dir (directory symlink)

LM Studio indexes this tree and follows symlinks, so bytes live once in the store.
Full-precision/GPTQ safetensors are skipped — LM Studio's runtimes can't load them.
"""

from __future__ import annotations

from pathlib import Path

from ..cache import Repo
from .base import Action, Adapter, ensure_symlink


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

    def doctor(self) -> list[str]:
        state = "exists" if self.models_dir.is_dir() else "missing — created on first sync"
        return [
            f"models dir: {self.models_dir} ({state})",
            "Serves GGUF + MLX. If LM Studio uses a custom folder, set MODELCTL_LMSTUDIO_DIR.",
        ]
