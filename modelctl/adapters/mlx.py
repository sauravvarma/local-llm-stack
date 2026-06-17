"""MLX (mlx_lm) adapter — Apple-Silicon runtime, no projection needed.

mlx_lm loads a model from a local path or HF repo id, so the resolver is just
the model directory. Accepts MLX-quantized models and full-precision safetensors
(mlx_lm runs both). GGUF is out of scope (that's llama.cpp's job).
"""

from __future__ import annotations

import shutil

from ..cache import Repo
from .base import Action, Adapter


class MlxAdapter(Adapter):
    name = "mlx"

    def accepts(self, repo: Repo) -> bool:
        return repo.fmt in ("mlx", "safetensors")

    def sync(self, repos: list[Repo], *, dry_run: bool = False) -> list[Action]:
        return [
            Action(self.name, "native", r.repo_id, f"mlx_lm.generate --model {r.root} --prompt …")
            for r in repos if self.accepts(r)
        ]

    def doctor(self) -> list[str]:
        if shutil.which("mlx_lm.generate") or shutil.which("mlx_lm.server"):
            return ["mlx_lm found. Load by path: mlx_lm.server --model <path-from `modelctl resolve`>"]
        return ["mlx_lm not on PATH (pip install mlx-lm). Loads a model dir by path or repo id."]
