"""Fixtures: build synthetic flat stores and HF caches in temp dirs.

No real models, no network — just small files laid out exactly like the real
thing so scanners/adapters/CLI can run against them deterministically.
"""

from __future__ import annotations

import json
import os
import unittest
from pathlib import Path


def write(path: Path, data: bytes = b"x" * 16) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def write_config(path: Path, cfg: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cfg))
    return path


def make_flat_store(root: Path) -> Path:
    """A store covering every format + edge case the scanner must handle."""
    # gguf repo, two files (main + mmproj)
    write(root / "lmstudio-community" / "gemma-GGUF" / "gemma-Q8_0.gguf")
    write(root / "lmstudio-community" / "gemma-GGUF" / "mmproj-BF16.gguf")
    # gguf repo with a single file (clean `resolve` target)
    write(root / "unsloth" / "tiny-GGUF" / "tiny-Q4_K_M.gguf", b"y" * 32)
    # gguf repo with a quant SUBFOLDER (relative path must be preserved)
    write(root / "bartowski" / "split-GGUF" / "Q4_K_M" / "model-00001-of-00002.gguf")
    # in-progress download (.part) must be ignored
    write(root / "unsloth" / "tiny-GGUF" / "downloading_tiny-Q8_0.gguf.part", b"z" * 64)

    # mlx (quantized): top-level "quantization" key in config
    mlx = root / "mlx-community" / "gemma-8bit"
    write(mlx / "model-00001-of-00002.safetensors")
    write_config(mlx / "config.json", {"model_type": "gemma", "quantization": {"group_size": 64, "bits": 8}})

    # mlx (by name): no quantization key, but "MLX" in the repo name
    mlxn = root / "someone" / "Model-MLX-4bit"
    write(mlxn / "model.safetensors")
    write_config(mlxn / "config.json", {"model_type": "llama"})

    # safetensors (full precision): config without quantization
    st = root / "Qwen" / "Qwen3-7B"
    write(st / "model-00001-of-00002.safetensors")
    write(st / "model-00002-of-00002.safetensors")
    write_config(st / "config.json", {"model_type": "qwen3"})

    # bare model dir (no publisher level): files directly under a top-level dir
    bare = root / "BareModel"
    write(bare / "model.safetensors")
    write_config(bare / "config.json", {"model_type": "mistral"})

    # noise that must be skipped
    write(root / ".DS_Store", b"junk")
    write(root / ".hidden" / "model.safetensors")  # hidden publisher dir ignored
    (root / "empty-publisher").mkdir(parents=True, exist_ok=True)  # no model files
    return root


def make_hf_cache(hub: Path) -> Path:
    """HF hub cache layout: models--org--name/{refs,snapshots,blobs}."""
    repo = hub / "models--org--demo-GGUF"
    (repo / "refs").mkdir(parents=True, exist_ok=True)
    (repo / "refs" / "main").write_text("commitabc")
    snap = repo / "snapshots" / "commitabc"
    snap.mkdir(parents=True, exist_ok=True)
    blobs = repo / "blobs"
    blobs.mkdir(exist_ok=True)
    blob = write(blobs / "sha-deadbeef", b"g" * 128)
    (snap / "demo.gguf").symlink_to(blob)  # cache files are symlinks into blobs

    # empty repo: refs only, no snapshot files -> must be skipped
    empty = hub / "models--Qwen--Empty"
    (empty / "refs").mkdir(parents=True, exist_ok=True)
    (empty / "refs" / "main").write_text("rev")
    return hub


# --- repo lookups used across tests -------------------------------------------

FLAT_FORMATS = {
    "lmstudio-community/gemma-GGUF": "gguf",
    "unsloth/tiny-GGUF": "gguf",
    "bartowski/split-GGUF": "gguf",
    "mlx-community/gemma-8bit": "mlx",
    "someone/Model-MLX-4bit": "mlx",
    "Qwen/Qwen3-7B": "safetensors",
    "BareModel": "safetensors",
}


class EnvTestCase(unittest.TestCase):
    """Snapshot/restore the env vars modelctl reads, so tests don't leak."""

    ENV_KEYS = ("MODELCTL_STORE", "MODELCTL_SCAN_HUB", "MODELCTL_LMSTUDIO_DIR",
                "HF_HOME", "HF_HUB_CACHE")

    def setUp(self):
        self._saved = {k: os.environ.get(k) for k in self.ENV_KEYS}
        for k in self.ENV_KEYS:
            os.environ.pop(k, None)
        # default: isolate from the real HF cache unless a test opts in
        os.environ["MODELCTL_SCAN_HUB"] = "0"

    def tearDown(self):
        for k, v in self._saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
