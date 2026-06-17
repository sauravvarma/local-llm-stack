from __future__ import annotations

import contextlib
import io
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from modelctl.cli import main
from tests.helpers import EnvTestCase, make_flat_store


def run(*argv):
    """Invoke the CLI, capturing (exit_code, stdout, stderr)."""
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = main(list(argv))
    return code, out.getvalue(), err.getvalue()


class CliTest(EnvTestCase):
    def setUp(self):
        super().setUp()
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        self.store = make_flat_store(self.base / "store")
        self.lm = self.base / "lm"
        os.environ["MODELCTL_STORE"] = str(self.store)
        os.environ["MODELCTL_LMSTUDIO_DIR"] = str(self.lm)
        os.environ["MODELCTL_SCAN_HUB"] = "0"

    def test_list(self):
        code, out, _ = run("list")
        self.assertEqual(code, 0)
        self.assertIn("Qwen/Qwen3-7B", out)
        self.assertIn("[mlx]", out)
        self.assertIn("[gguf]", out)

    def test_list_files(self):
        code, out, _ = run("list", "-f")
        self.assertEqual(code, 0)
        self.assertIn("model-00001-of-00002.safetensors", out)

    def test_resolve_gguf_single_file(self):
        code, out, _ = run("resolve", "unsloth/tiny-GGUF")
        self.assertEqual(code, 0)
        self.assertTrue(out.strip().endswith("tiny-Q4_K_M.gguf"))

    def test_resolve_mlx_returns_dir(self):
        code, out, _ = run("resolve", "mlx-community/gemma-8bit")
        self.assertEqual(code, 0)
        self.assertTrue(out.strip().endswith("mlx-community/gemma-8bit"))

    def test_resolve_safetensors_returns_dir(self):
        code, out, _ = run("resolve", "Qwen/Qwen3-7B")
        self.assertEqual(code, 0)
        self.assertTrue(out.strip().endswith("Qwen/Qwen3-7B"))

    def test_resolve_with_file_arg(self):
        code, out, _ = run("resolve", "lmstudio-community/gemma-GGUF", "mmproj-BF16.gguf")
        self.assertEqual(code, 0)
        self.assertTrue(out.strip().endswith("mmproj-BF16.gguf"))

    def test_resolve_unknown_exits_1(self):
        code, _, err = run("resolve", "no/such")
        self.assertEqual(code, 1)
        self.assertIn("not found", err)

    def test_sync_dry_run_changes_nothing(self):
        code, out, _ = run("sync", "-n")
        self.assertEqual(code, 0)
        self.assertIn("DRY RUN", out)
        self.assertFalse((self.lm / "unsloth").exists())

    def test_sync_creates_symlinks(self):
        code, _, _ = run("sync", "-a", "lmstudio")
        self.assertEqual(code, 0)
        self.assertTrue((self.lm / "unsloth" / "tiny-GGUF" / "tiny-Q4_K_M.gguf").is_symlink())

    def test_sync_unknown_adapter_exits_2(self):
        with self.assertRaises(SystemExit) as ctx:
            run("sync", "-a", "bogus")
        self.assertEqual(ctx.exception.code, 2)

    def test_doctor(self):
        code, out, _ = run("doctor")
        self.assertEqual(code, 0)
        self.assertIn("Stores", out)
        self.assertIn("[vllm]", out)
        self.assertIn("[mlx]", out)

    def test_env(self):
        code, out, _ = run("env")
        self.assertEqual(code, 0)
        self.assertIn("export HF_HOME=", out)
        self.assertIn("MODELCTL_STORE=", out)

    def test_download_invokes_hf_with_local_dir(self):
        with mock.patch("modelctl.cli.shutil.which", return_value="/usr/bin/hf"), \
             mock.patch("modelctl.cli.subprocess.run") as run_mock:
            run_mock.return_value = mock.Mock(returncode=0)
            code, out, _ = run("download", "org/new-model", "--no-sync")
        self.assertEqual(code, 0)
        args = run_mock.call_args[0][0]
        self.assertEqual(args[:3], ["hf", "download", "org/new-model"])
        self.assertIn("--local-dir", args)
        target = args[args.index("--local-dir") + 1]
        self.assertTrue(target.endswith("org/new-model"))

    def test_download_missing_hf_exits_1(self):
        with mock.patch("modelctl.cli.shutil.which", return_value=None):
            code, _, err = run("download", "org/x")
        self.assertEqual(code, 1)
        self.assertIn("hf", err)

    def test_ollama_import_dry_run(self):
        with mock.patch("modelctl.adapters.ollama.shutil.which", return_value="/usr/bin/ollama"), \
             mock.patch("modelctl.adapters.ollama.subprocess.run") as run_mock:
            code, out, _ = run("ollama-import", "unsloth/tiny-GGUF", "--dry-run")
            run_mock.assert_not_called()
        self.assertEqual(code, 0)
        self.assertIn("would import", out)

    def test_ollama_import_unknown_repo_exits_1(self):
        code, _, err = run("ollama-import", "no/such")
        self.assertEqual(code, 1)

    # --- adopt -----------------------------------------------------------

    def _make_hub_with(self, repo_id):
        from tests.helpers import make_hf_cache
        hub = make_hf_cache(self.base / "hub")
        org, name = repo_id.split("/")
        (hub / f"models--{org}--{name}" / "refs").mkdir(parents=True)
        (hub / f"models--{org}--{name}" / "refs" / "main").write_text("rev")
        os.environ["HF_HUB_CACHE"] = str(hub)

    def test_adopt_moves_bare_dir_to_publisher_model(self):
        # add a bare model dir to the store
        bare = self.store / "Qwen3-Demo"
        (bare).mkdir()
        (bare / "model.safetensors").write_bytes(b"w")
        (bare / "config.json").write_text('{"model_type":"qwen3"}')
        self._make_hub_with("Qwen/Qwen3-Demo")
        code, out, _ = run("adopt", "Qwen3-Demo")
        self.assertEqual(code, 0)
        self.assertFalse(bare.exists())
        self.assertTrue((self.store / "Qwen" / "Qwen3-Demo" / "config.json").exists())

    def test_adopt_link_is_nondestructive(self):
        bare = self.store / "Qwen3-Demo"
        bare.mkdir()
        (bare / "config.json").write_text('{"model_type":"qwen3"}')
        (bare / "model.safetensors").write_bytes(b"w")
        self._make_hub_with("Qwen/Qwen3-Demo")
        code, _, _ = run("adopt", "Qwen3-Demo", "--link")
        self.assertEqual(code, 0)
        self.assertTrue(bare.exists())  # original kept
        self.assertTrue((self.store / "Qwen" / "Qwen3-Demo").is_symlink())

    def test_adopt_explicit_publisher(self):
        bare = self.store / "Mystery"
        bare.mkdir()
        (bare / "config.json").write_text('{"model_type":"llama"}')
        (bare / "model.safetensors").write_bytes(b"w")
        code, _, _ = run("adopt", "Mystery", "--publisher", "acme")
        self.assertEqual(code, 0)
        self.assertTrue((self.store / "acme" / "Mystery" / "config.json").exists())

    def test_adopt_unknown_publisher_exits_1(self):
        bare = self.store / "Orphan"
        bare.mkdir()
        (bare / "config.json").write_text('{"model_type":"llama"}')
        (bare / "model.safetensors").write_bytes(b"w")
        code, _, err = run("adopt", "Orphan")
        self.assertEqual(code, 1)
        self.assertIn("publisher", err)

    def test_adopt_already_nested_is_noop(self):
        code, out, _ = run("adopt", "Qwen/Qwen3-7B")
        self.assertEqual(code, 0)
        self.assertIn("already", out)


if __name__ == "__main__":
    unittest.main()
