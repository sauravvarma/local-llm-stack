from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from modelctl.adapters.base import ensure_symlink
from modelctl.adapters.llamacpp import LlamaCppAdapter
from modelctl.adapters.lmstudio import LMStudioAdapter
from modelctl.adapters.mlx import MlxAdapter
from modelctl.adapters.ollama import OllamaAdapter, _tag
from modelctl.adapters.vllm import VllmAdapter
from modelctl.cache import scan_flat
from tests.helpers import make_flat_store, write


class EnsureSymlinkTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.d = Path(self.tmp.name)
        self.src = write(self.d / "src.bin", b"data")

    def test_creates_link(self):
        tgt = self.d / "sub" / "link.bin"
        a = ensure_symlink(tgt, self.src, "t", dry_run=False)
        self.assertEqual(a.op, "link")
        self.assertTrue(tgt.is_symlink())
        self.assertEqual(tgt.resolve(), self.src.resolve())

    def test_dry_run_creates_nothing(self):
        tgt = self.d / "link.bin"
        a = ensure_symlink(tgt, self.src, "t", dry_run=True)
        self.assertEqual(a.op, "link")
        self.assertFalse(tgt.exists())

    def test_idempotent(self):
        tgt = self.d / "link.bin"
        ensure_symlink(tgt, self.src, "t", dry_run=False)
        a = ensure_symlink(tgt, self.src, "t", dry_run=False)
        self.assertEqual(a.op, "skip")

    def test_relink_when_pointing_elsewhere(self):
        other = write(self.d / "other.bin", b"x")
        tgt = self.d / "link.bin"
        tgt.symlink_to(other)
        a = ensure_symlink(tgt, self.src, "t", dry_run=False)
        self.assertEqual(a.op, "relink")
        self.assertEqual(tgt.resolve(), self.src.resolve())

    def test_real_file_is_error_and_untouched(self):
        tgt = write(self.d / "real.bin", b"keep")
        a = ensure_symlink(tgt, self.src, "t", dry_run=False)
        self.assertEqual(a.op, "error")
        self.assertFalse(tgt.is_symlink())
        self.assertEqual(tgt.read_bytes(), b"keep")

    def test_already_in_place_when_target_is_source(self):
        a = ensure_symlink(self.src, self.src, "t", dry_run=False)
        self.assertEqual(a.op, "skip")
        self.assertIn("already in place", a.detail)


class AdapterBase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        self.store = make_flat_store(self.base / "store")
        self.repos = scan_flat(self.store)

    def repo(self, rid):
        return next(r for r in self.repos if r.repo_id == rid)


class LMStudioAdapterTest(AdapterBase):
    def test_accepts_only_gguf_and_mlx(self):
        ad = LMStudioAdapter(self.base / "lm")
        self.assertTrue(ad.accepts(self.repo("unsloth/tiny-GGUF")))
        self.assertTrue(ad.accepts(self.repo("mlx-community/gemma-8bit")))
        self.assertFalse(ad.accepts(self.repo("Qwen/Qwen3-7B")))

    def test_gguf_per_file_symlink_preserves_path(self):
        lm = self.base / "lm"
        LMStudioAdapter(lm).sync(self.repos, dry_run=False)
        link = lm / "bartowski" / "split-GGUF" / "Q4_K_M" / "model-00001-of-00002.gguf"
        self.assertTrue(link.is_symlink())
        self.assertTrue(link.resolve().is_file())

    def test_mlx_directory_symlink(self):
        lm = self.base / "lm"
        LMStudioAdapter(lm).sync(self.repos, dry_run=False)
        link = lm / "mlx-community" / "gemma-8bit"
        self.assertTrue(link.is_symlink())
        self.assertTrue((link / "config.json").exists())

    def test_safetensors_not_projected(self):
        lm = self.base / "lm"
        actions = LMStudioAdapter(lm).sync(self.repos, dry_run=False)
        self.assertFalse(any("Qwen3-7B" in a.target for a in actions))

    def test_already_in_place_when_dir_equals_store(self):
        actions = LMStudioAdapter(self.store).sync(self.repos, dry_run=False)
        gguf = [a for a in actions if a.target.endswith("tiny-Q4_K_M.gguf")][0]
        self.assertEqual(gguf.op, "skip")


class VllmAdapterTest(AdapterBase):
    def test_accepts_safetensors_only(self):
        ad = VllmAdapter()
        self.assertTrue(ad.accepts(self.repo("Qwen/Qwen3-7B")))
        self.assertFalse(ad.accepts(self.repo("mlx-community/gemma-8bit")))
        self.assertFalse(ad.accepts(self.repo("unsloth/tiny-GGUF")))

    def test_native_action_points_at_dir(self):
        actions = VllmAdapter().sync(self.repos)
        st = [a for a in actions if a.target == "Qwen/Qwen3-7B"][0]
        self.assertEqual(st.op, "native")
        self.assertIn("vllm serve", st.detail)


class MlxAdapterTest(AdapterBase):
    def test_accepts_mlx_and_safetensors(self):
        ad = MlxAdapter()
        self.assertTrue(ad.accepts(self.repo("mlx-community/gemma-8bit")))
        self.assertTrue(ad.accepts(self.repo("Qwen/Qwen3-7B")))
        self.assertFalse(ad.accepts(self.repo("unsloth/tiny-GGUF")))

    def test_native_action(self):
        actions = MlxAdapter().sync(self.repos)
        targets = {a.target for a in actions}
        self.assertIn("mlx-community/gemma-8bit", targets)
        self.assertTrue(all(a.op == "native" for a in actions))


class LlamaCppAdapterTest(AdapterBase):
    def test_accepts_gguf_only(self):
        ad = LlamaCppAdapter()
        self.assertTrue(ad.accepts(self.repo("unsloth/tiny-GGUF")))
        self.assertFalse(ad.accepts(self.repo("Qwen/Qwen3-7B")))

    def test_native_action_has_load_command(self):
        actions = LlamaCppAdapter().sync(self.repos)
        self.assertTrue(actions)
        self.assertTrue(all(a.op == "native" and "llama-server -m" in a.detail for a in actions))


class OllamaAdapterTest(AdapterBase):
    def test_accepts_gguf_only(self):
        self.assertTrue(OllamaAdapter().accepts(self.repo("unsloth/tiny-GGUF")))
        self.assertFalse(OllamaAdapter().accepts(self.repo("mlx-community/gemma-8bit")))

    def test_sync_default_skips_with_import_hint(self):
        actions = OllamaAdapter().sync(self.repos)
        self.assertTrue(actions)
        self.assertTrue(all(a.op == "skip" for a in actions))
        self.assertIn("copies bytes", actions[0].detail)

    def test_tag_derivation(self):
        self.assertEqual(_tag("gemma-4-E2B-it-GGUF", "gemma-UD-Q4_K_XL.gguf"), "gemma-4-e2b-it-gguf:q4_k_xl")
        self.assertEqual(_tag("Some Model", "weights-f16.gguf"), "some-model:f16")
        self.assertTrue(_tag("m", "no-quant.gguf").endswith(":latest"))

    def test_import_errors_without_ollama_binary(self):
        with mock.patch("modelctl.adapters.ollama.shutil.which", return_value=None):
            f = self.repo("unsloth/tiny-GGUF").gguf_files[0]
            a = OllamaAdapter().import_file(f, "tiny")
            self.assertEqual(a.op, "error")

    def test_import_dry_run_does_not_invoke_ollama(self):
        with mock.patch("modelctl.adapters.ollama.shutil.which", return_value="/usr/bin/ollama"), \
             mock.patch("modelctl.adapters.ollama.subprocess.run") as run:
            f = self.repo("unsloth/tiny-GGUF").gguf_files[0]
            a = OllamaAdapter().import_file(f, "tiny", dry_run=True)
            self.assertEqual(a.op, "copy")
            run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
