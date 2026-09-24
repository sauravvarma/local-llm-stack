from __future__ import annotations

from typing import TYPE_CHECKING

from .base import Adapter
from .bionic import BionicAdapter
from .lmstudio import LMStudioAdapter
from .llamacpp import LlamaCppAdapter
from .mlx import MlxAdapter
from .ollama import OllamaAdapter
from .splash import SplashAdapter
from .vllm import VllmAdapter

if TYPE_CHECKING:  # deferred: config resolves adapter dirs, so it imports us
    from ..config import Config


def build_adapters(cfg: "Config") -> dict[str, Adapter]:
    return {
        "vllm": VllmAdapter(),
        "mlx": MlxAdapter(),
        "lmstudio": LMStudioAdapter(cfg.lmstudio_dir),
        "bionic": BionicAdapter(cfg.bionic),
        "splash": SplashAdapter(),
        "llamacpp": LlamaCppAdapter(),
        "ollama": OllamaAdapter(),
    }
