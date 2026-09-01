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

    # --- rm ---------------------------------------------------------------

    def test_rm_deletes_model_dir_and_reports_size(self):
        run("sync", "-a", "lmstudio")
        target = self.store / "unsloth" / "tiny-GGUF"
        self.assertTrue(target.is_dir())
        code, out, _ = run("rm", "unsloth/tiny-GGUF", "-y")
        self.assertEqual(code, 0)
        self.assertFalse(target.exists())
        self.assertIn("reclaimed", out)
        self.assertIn("32B", out)  # tiny-Q4_K_M.gguf is 32 bytes
        # empty publisher dir pruned, the rest of the store untouched
        self.assertFalse((self.store / "unsloth").exists())
        self.assertTrue((self.store / "Qwen" / "Qwen3-7B").is_dir())

    def test_rm_takes_down_lmstudio_projection(self):
        run("sync", "-a", "lmstudio")
        link = self.lm / "lmstudio-community" / "gemma-GGUF" / "gemma-Q8_0.gguf"
        self.assertTrue(link.is_symlink())
        code, out, _ = run("rm", "lmstudio-community/gemma-GGUF", "-y")
        self.assertEqual(code, 0)
        self.assertIn("[lmstudio]", out)
        self.assertFalse(link.is_symlink())
        self.assertFalse((self.lm / "lmstudio-community").exists())  # pruned

    def test_rm_mlx_dir_symlink_projection(self):
        run("sync", "-a", "lmstudio")
        link = self.lm / "mlx-community" / "gemma-8bit"
        self.assertTrue(link.is_symlink())
        code, _, _ = run("rm", "mlx-community/gemma-8bit", "-y")
        self.assertEqual(code, 0)
        self.assertFalse(link.is_symlink())
        self.assertFalse((self.store / "mlx-community" / "gemma-8bit").exists())

    def test_rm_specific_file_keeps_the_rest(self):
        run("sync", "-a", "lmstudio")
        code, out, _ = run("rm", "lmstudio-community/gemma-GGUF", "mmproj-BF16.gguf", "-y")
        self.assertEqual(code, 0)
        self.assertIn("1 file", out)
        gone = self.store / "lmstudio-community" / "gemma-GGUF" / "mmproj-BF16.gguf"
        kept = self.store / "lmstudio-community" / "gemma-GGUF" / "gemma-Q8_0.gguf"
        self.assertFalse(gone.exists())
        self.assertTrue(kept.is_file())
        self.assertFalse((self.lm / "lmstudio-community" / "gemma-GGUF" / "mmproj-BF16.gguf").is_symlink())
        self.assertTrue((self.lm / "lmstudio-community" / "gemma-GGUF" / "gemma-Q8_0.gguf").is_symlink())

    def test_rm_unknown_file_exits_1(self):
        code, _, err = run("rm", "unsloth/tiny-GGUF", "nope.gguf", "-y")
        self.assertEqual(code, 1)
        self.assertIn("not in", err)
        self.assertTrue((self.store / "unsloth" / "tiny-GGUF").is_dir())

    def test_rm_dry_run_touches_nothing(self):
        run("sync", "-a", "lmstudio")
        code, out, _ = run("rm", "unsloth/tiny-GGUF", "-n")
        self.assertEqual(code, 0)
        self.assertIn("DRY RUN", out)
        self.assertIn(str(self.store / "unsloth" / "tiny-GGUF"), out)
        self.assertTrue((self.store / "unsloth" / "tiny-GGUF" / "tiny-Q4_K_M.gguf").is_file())
        self.assertTrue((self.lm / "unsloth" / "tiny-GGUF" / "tiny-Q4_K_M.gguf").is_symlink())

    def test_rm_dry_run_never_prompts(self):
        with mock.patch("builtins.input", side_effect=AssertionError("prompted!")):
            code, _, _ = run("rm", "unsloth/tiny-GGUF", "-n")
        self.assertEqual(code, 0)

    def test_rm_prompts_and_abort_keeps_everything(self):
        with mock.patch("builtins.input", return_value="n"):
            code, out, _ = run("rm", "unsloth/tiny-GGUF")
        self.assertEqual(code, 1)
        self.assertIn("aborted", out)
        self.assertTrue((self.store / "unsloth" / "tiny-GGUF").is_dir())

    def test_rm_prompt_yes_deletes(self):
        with mock.patch("builtins.input", return_value="y"):
            code, _, _ = run("rm", "unsloth/tiny-GGUF")
        self.assertEqual(code, 0)
        self.assertFalse((self.store / "unsloth" / "tiny-GGUF").exists())

    def test_rm_remove_alias(self):
        code, _, _ = run("remove", "unsloth/tiny-GGUF", "-y")
        self.assertEqual(code, 0)
        self.assertFalse((self.store / "unsloth" / "tiny-GGUF").exists())

    def test_rm_refuses_hub_cache_only_repo(self):
        from tests.helpers import make_hf_cache
        hub = make_hf_cache(self.base / "hub")
        os.environ["HF_HUB_CACHE"] = str(hub)
        os.environ["MODELCTL_SCAN_HUB"] = "1"
        code, _, err = run("rm", "org/demo-GGUF", "-y")
        self.assertEqual(code, 1)
        self.assertIn("read-only", err)
        self.assertIn("hf cache delete", err)
        self.assertTrue((hub / "models--org--demo-GGUF" / "blobs" / "sha-deadbeef").is_file())

    def test_rm_refuses_secondary_store(self):
        extra = make_flat_store(self.base / "extra2")
        # a model only the secondary store has
        (extra / "acme" / "Only-There").mkdir(parents=True)
        (extra / "acme" / "Only-There" / "model.safetensors").write_bytes(b"w")
        (extra / "acme" / "Only-There" / "config.json").write_text('{"model_type":"llama"}')
        os.environ["MODELCTL_STORE"] = f"{self.store}:{extra}"
        code, _, err = run("rm", "acme/Only-There", "-y")
        self.assertEqual(code, 1)
        self.assertIn("read-only", err)
        self.assertTrue((extra / "acme" / "Only-There").is_dir())

    def test_rm_unknown_repo_exits_1(self):
        code, _, err = run("rm", "no/such", "-y")
        self.assertEqual(code, 1)
        self.assertIn("not found", err)

    def test_rm_warns_about_ollama_copy(self):
        with mock.patch("modelctl.adapters.ollama.OllamaAdapter.imported_names",
                        return_value={"tiny-gguf:q4_k_m"}):
            code, out, _ = run("rm", "unsloth/tiny-GGUF", "-y")
        self.assertEqual(code, 0)
        self.assertIn("ollama rm tiny-gguf:q4_k_m", out)
        self.assertIn("NOT reclaimed", out)

    def test_rm_no_ollama_warning_when_not_imported(self):
        with mock.patch("modelctl.adapters.ollama.OllamaAdapter.imported_names", return_value=set()):
            code, out, _ = run("rm", "unsloth/tiny-GGUF", "-y")
        self.assertEqual(code, 0)
        self.assertNotIn("ollama rm", out)

    def test_rm_symlinked_model_dir_only_unlinks(self):
        # `modelctl adopt --link` leaves a symlinked model dir; the bytes it
        # points at live outside the store and must survive.
        outside = self.base / "outside" / "Linked"
        outside.mkdir(parents=True)
        (outside / "config.json").write_text('{"model_type":"llama"}')
        (outside / "model.safetensors").write_bytes(b"w" * 8)
        link = self.store / "acme" / "Linked"
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to(outside)
        code, out, _ = run("rm", "acme/Linked", "-y")
        self.assertEqual(code, 0)
        self.assertIn("symlink only", out)
        self.assertFalse(link.exists())
        self.assertTrue((outside / "model.safetensors").is_file())

    def test_rm_hidden_cache_files_are_not_listed(self):
        junk = self.store / "unsloth" / "tiny-GGUF" / ".cache" / "huggingface" / "download"
        junk.mkdir(parents=True)
        (junk / "tiny-Q4_K_M.gguf.metadata").write_text("bookkeeping")
        code, out, _ = run("list", "-f")
        self.assertEqual(code, 0)
        self.assertNotIn(".metadata", out)
        self.assertIn("tiny-Q4_K_M.gguf", out)


class RmGuardTest(unittest.TestCase):
    """The path guard is the last thing standing between a typo and 150GB."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        self.store = self.base / "store"
        (self.store / "acme" / "m").mkdir(parents=True)

    def check(self, path, **kw):
        from modelctl.cli import _check_removable
        return _check_removable(self.store, Path(path), **kw)

    def test_ok_inside_store(self):
        self.assertIsNone(self.check(self.store / "acme" / "m"))

    def test_rejects_dotdot_escape(self):
        self.assertIsNotNone(self.check(self.store / "acme" / ".." / ".." / "elsewhere"))

    def test_rejects_absolute_outside(self):
        self.assertIsNotNone(self.check("/etc"))

    def test_rejects_store_root_and_fs_root(self):
        self.assertIsNotNone(self.check(self.store))
        self.assertIsNotNone(self.check("/"))

    def test_rejects_relative_path(self):
        self.assertIsNotNone(self.check("acme/m"))

    def test_rejects_symlink_escape_via_parent_dir(self):
        # a real dir inside the store whose *parent* chain leaves the store
        outside = self.base / "outside"
        outside.mkdir()
        (self.store / "acme" / "escape").symlink_to(outside)
        self.assertIsNotNone(self.check(self.store / "acme" / "escape" / "victim"))

    def test_symlink_itself_is_allowed(self):
        outside = self.base / "outside2"
        outside.mkdir()
        link = self.store / "acme" / "link"
        link.symlink_to(outside)
        self.assertIsNone(self.check(link))  # unlinked, never followed

    def test_confine_keeps_files_inside_their_model(self):
        model = self.store / "acme" / "m"
        other = self.store / "acme" / "other"
        other.mkdir()
        self.assertIsNone(self.check(model / "a.gguf", confine=model))
        self.assertIsNotNone(self.check(other / "a.gguf", confine=model))



if __name__ == "__main__":
    unittest.main()
