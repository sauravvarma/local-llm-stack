"""Tests for `bin/llm-probe` — the store→capability registry.

Real GGUFs are 25 GB, so these build byte-exact miniature ones instead: a
valid header, a metadata block, and a tensor table. That's what the prober
actually reads, so the fixtures exercise the real parsing path.
"""

from __future__ import annotations

import json
import os
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PROBE = REPO_ROOT / "bin" / "llm-probe"

_STR, _U32, _U64 = 8, 4, 10


def _kv(key: str, val: str) -> bytes:
    """One string-typed metadata pair."""
    kb = key.encode()
    vb = val.encode()
    return (struct.pack("<Q", len(kb)) + kb
            + struct.pack("<I", _STR)
            + struct.pack("<Q", len(vb)) + vb)


def _tensor(name: str) -> bytes:
    """One tensor descriptor: name, 1 dimension, type, offset."""
    nb = name.encode()
    return (struct.pack("<Q", len(nb)) + nb
            + struct.pack("<I", 1)          # n_dims
            + struct.pack("<Q", 4)          # dim[0]
            + struct.pack("<I", 0)          # ggml type
            + struct.pack("<Q", 0))         # offset


def write_gguf(path: Path, *, arch: str = "llama", tensors=("token_embd.weight",)) -> Path:
    """Write a minimal but structurally valid GGUF."""
    path.parent.mkdir(parents=True, exist_ok=True)
    kvs = _kv("general.architecture", arch)
    body = b"".join(_tensor(t) for t in tensors)
    path.write_bytes(
        b"GGUF" + struct.pack("<I", 3)
        + struct.pack("<Q", len(tensors))
        + struct.pack("<Q", 1)
        + kvs + body
    )
    return path


def write_mlx(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "config.json").write_text(json.dumps(
        {"model_type": "gemma", "quantization": {"bits": 8}}))
    (root / "model.safetensors").write_bytes(b"x" * 32)
    return root


class ProbeTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.store = Path(self._tmp.name) / "models"
        self.state = Path(self._tmp.name) / "state"
        self.store.mkdir(parents=True)
        self.addCleanup(self._tmp.cleanup)

    def probe(self) -> dict:
        env = dict(os.environ,
                   MODELCTL_STORE=str(self.store),
                   LLM_RUN_DIR=str(self.state))
        out = subprocess.run([sys.executable, str(PROBE), "--refresh"],
                             capture_output=True, text=True, env=env, check=True)
        return json.loads(out.stdout)

    # -- capability detection ------------------------------------------------
    def test_plain_gguf_has_no_extras(self):
        write_gguf(self.store / "acme" / "tiny-GGUF" / "tiny-Q4.gguf")
        m = self.probe()["models"]["tiny"]
        self.assertEqual(m["format"], "gguf")
        self.assertFalse(m["mtp"])
        self.assertIsNone(m["mmproj"])
        self.assertIsNone(m["dflash"])
        self.assertEqual(m["backends"], ["llamacpp", "lmstudio"])

    def test_nextn_tensors_mean_mtp(self):
        write_gguf(self.store / "acme" / "big-GGUF" / "big-Q4.gguf",
                   tensors=("token_embd.weight", "blk.64.nextn.eh_proj.weight"))
        self.assertTrue(self.probe()["models"]["big"]["mtp"])

    def test_arch_is_read_from_metadata(self):
        write_gguf(self.store / "acme" / "big-GGUF" / "big-Q4.gguf", arch="qwen35")
        self.assertEqual(self.probe()["models"]["big"]["arch"], "qwen35")

    def test_sibling_mmproj_means_vision(self):
        d = self.store / "acme" / "see-GGUF"
        write_gguf(d / "see-Q4.gguf")
        write_gguf(d / "mmproj-F16.gguf")
        m = self.probe()["models"]["see"]
        self.assertTrue(m["mmproj"].endswith("mmproj-F16.gguf"))
        # the projector must not be mistaken for the weights
        self.assertTrue(m["path"].endswith("see-Q4.gguf"))

    def test_matching_dflash_repo_is_linked(self):
        write_gguf(self.store / "unsloth" / "Qwen3.8-27B-GGUF" / "Qwen3.8-27B-Q6.gguf")
        write_gguf(self.store / "z-lab" / "Qwen3.8-27B-DFlash2-GGUF" / "draft-Q4.gguf")
        models = self.probe()["models"]
        self.assertTrue(models["qwen3.8-27b"]["dflash"].endswith("draft-Q4.gguf"))
        # the drafter is a component, not a servable model
        self.assertNotIn("qwen3.8-27b-dflash2", models)

    def test_dflash_not_linked_to_unrelated_model(self):
        write_gguf(self.store / "acme" / "other-GGUF" / "other-Q4.gguf")
        write_gguf(self.store / "z-lab" / "Qwen3.8-27B-DFlash2-GGUF" / "draft-Q4.gguf")
        self.assertIsNone(self.probe()["models"]["other"]["dflash"])

    # -- formats -------------------------------------------------------------
    def test_mlx_is_lmstudio_only(self):
        write_mlx(self.store / "mlx-community" / "gemma-8bit")
        m = self.probe()["models"]["gemma"]
        self.assertEqual(m["format"], "mlx")
        self.assertEqual(m["backends"], ["lmstudio"])

    # -- alias collisions ----------------------------------------------------
    def test_colliding_aliases_both_survive(self):
        """A GGUF and an MLX build of one model must not shadow each other."""
        write_gguf(self.store / "lmstudio-community" / "gemma-4-12B-it-GGUF" / "g-Q8.gguf")
        write_mlx(self.store / "mlx-community" / "gemma-4-12B-it-8bit")
        models = self.probe()["models"]
        self.assertIn("gemma-4-12b-gguf", models)
        self.assertIn("gemma-4-12b-mlx", models)
        self.assertNotIn("gemma-4-12b", models)

    # -- robustness ----------------------------------------------------------
    def test_unreadable_gguf_still_listed(self):
        """A truncated file must degrade to no-capabilities, not crash."""
        p = self.store / "acme" / "trunc-GGUF" / "trunc-Q4.gguf"
        p.parent.mkdir(parents=True)
        p.write_bytes(b"GGUF" + b"\x00" * 8)
        m = self.probe()["models"]["trunc"]
        self.assertFalse(m["mtp"])
        self.assertIsNone(m["arch"])

    def test_missing_store_is_not_an_error(self):
        self.store.rmdir()
        self.assertEqual(self.probe()["models"], {})

    # -- caching -------------------------------------------------------------
    def test_cache_is_written_and_reused(self):
        write_gguf(self.store / "acme" / "tiny-GGUF" / "tiny-Q4.gguf")
        self.probe()
        cache = self.state / "llm-registry.json"
        self.assertTrue(cache.is_file())

        env = dict(os.environ, MODELCTL_STORE=str(self.store),
                   LLM_RUN_DIR=str(self.state))
        # No --refresh: served from cache, so a store change made without
        # touching directory mtimes is not picked up.
        out = subprocess.run([sys.executable, str(PROBE)],
                             capture_output=True, text=True, env=env, check=True)
        self.assertIn("tiny", json.loads(out.stdout)["models"])


if __name__ == "__main__":
    unittest.main()
