"""vLLM / transformers / TGI / SGLang adapter — no projection needed.

These read a model from a path or an HF repo id directly, so the "resolver" is
just the model's directory. Accepts standard safetensors (full-precision, GPTQ,
AWQ); MLX-quantized weights are excluded — vLLM can't load them.
"""

from __future__ import annotations

import os
import shutil

from ..cache import Repo
from .base import Action, Adapter


class VllmAdapter(Adapter):
    name = "vllm"

    def accepts(self, repo: Repo) -> bool:
        return repo.fmt == "safetensors"

    def sync(self, repos: list[Repo], *, dry_run: bool = False) -> list[Action]:
        return [
            Action(self.name, "native", r.repo_id, f"vllm serve {r.root}")
            for r in repos if self.accepts(r)
        ]

    def doctor(self) -> list[str]:
        msgs = []
        if os.environ.get("HF_HOME") or os.environ.get("HF_HUB_CACHE"):
            msgs.append(f"HF cache: HF_HOME={os.environ.get('HF_HOME','-')} HF_HUB_CACHE={os.environ.get('HF_HUB_CACHE','-')}")
        else:
            msgs.append("HF_HOME/HF_HUB_CACHE not set (default ~/.cache/huggingface). Models in the flat store load by path: `vllm serve <path>`.")
        if shutil.which("vllm") is None:
            msgs.append("vllm not installed. It needs CUDA; on macOS it's CPU-only/experimental — usually run on a Linux+GPU box mounting this store.")
        else:
            msgs.append("vllm found on PATH.")
        return msgs
