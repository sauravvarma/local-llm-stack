"""MLX (mlx_lm) adapter — Apple-Silicon runtime, no projection needed.

mlx_lm loads a model from a local path or HF repo id, so the resolver is just
the model directory. Accepts MLX-quantized models and full-precision safetensors
(mlx_lm runs both). GGUF is out of scope (that's llama.cpp's job).

Not every safetensors model is loadable though: GPTQ, AWQ and friends are
safetensors too, and mlx_lm cannot read them. Claiming those would print a
launch command that fails, the same mistake the splash format exposed, so the
quantisation method is checked rather than the container format alone.
"""

from __future__ import annotations

import shutil

from ..cache import SERVER_ONLY_QUANTS, Repo
from .base import Action, Adapter


class MlxAdapter(Adapter):
    name = "mlx"

    def accepts(self, repo: Repo) -> bool:
        if repo.fmt == "mlx":
            return True
        return repo.fmt == "safetensors" and repo.quant_method not in SERVER_ONLY_QUANTS

    def sync(self, repos: list[Repo], *, dry_run: bool = False, **options) -> list[Action]:
        return [
            Action(self.name, "native", r.repo_id, f"mlx_lm.generate --model {r.root} --prompt …")
            for r in repos if self.accepts(r)
        ]

    def doctor(self) -> list[str]:
        if shutil.which("mlx_lm.generate") or shutil.which("mlx_lm.server"):
            return ["mlx_lm found. Load by path: mlx_lm.server --model <path-from `modelctl resolve`>"]
        return ["mlx_lm not on PATH (pip install mlx-lm). Loads a model dir by path or repo id."]
