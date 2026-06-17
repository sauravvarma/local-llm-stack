from __future__ import annotations

from ..config import Config
from .base import Adapter
from .lmstudio import LMStudioAdapter
from .llamacpp import LlamaCppAdapter
from .mlx import MlxAdapter
from .ollama import OllamaAdapter
from .vllm import VllmAdapter


def build_adapters(cfg: Config) -> dict[str, Adapter]:
    return {
        "vllm": VllmAdapter(),
        "mlx": MlxAdapter(),
        "lmstudio": LMStudioAdapter(cfg.lmstudio_dir),
        "llamacpp": LlamaCppAdapter(),
        "ollama": OllamaAdapter(),
    }
