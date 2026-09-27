from __future__ import annotations

import io
import os
import tempfile
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

from modelctl import cli, downloads
from modelctl.cli import main
from tests.helpers import EnvTestCase, make_flat_store, make_hf_cache, write
from tests.test_downloads import make_scratch


class PruneCliTest(EnvTestCase):
    """End to end in a sandbox: a store, plus Bionic and oMLX dirs to project
    into. Nothing here touches a real store."""

    def setUp(self):
        super().setUp()
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        base = Path(self.tmp.name)
        self.store = make_flat_store(base / "store")
        self.bionic = base / "bionic"
        self.omlx = base / "omlx"
        os.environ.update(MODELCTL_STORE=str(self.store), MODELCTL_BIONIC_DIR=str(self.bionic),
                          MODELCTL_OMLX_DIR=str(self.omlx), MODELCTL_LMSTUDIO_DIR=str(self.store))
        self.run_cli("sync")                        # project everything first

    def run_cli(self, *argv, usable=False, answer=None):
        out, err = io.StringIO(), io.StringIO()
        patches = [mock.patch.object(cli.picker, "usable", return_value=usable)]
        if answer is not None:
            patches.append(mock.patch("builtins.input", return_value=answer))
        for p in patches:
            p.start()
        try:
            with redirect_stdout(out), redirect_stderr(err):
                code = main(list(argv))
        finally:
            for p in patches:
                p.stop()
        return code, out.getvalue(), err.getvalue()

    # ---- the happy path
    def test_yes_deletes_the_model_and_every_projection(self):
        model = self.store / "mlx-community" / "gemma-8bit"
        self.assertTrue((self.bionic / "mlx-community" / "gemma-8bit").is_symlink())
        self.assertTrue((self.omlx / "mlx-community" / "gemma-8bit").is_symlink())
        code, out, _ = self.run_cli("prune", "mlx-community/gemma-8bit", "--yes")
        self.assertEqual(code, 0, out)
        self.assertFalse(model.exists())
        self.assertFalse((self.bionic / "mlx-community" / "gemma-8bit").is_symlink())
        self.assertFalse((self.omlx / "mlx-community" / "gemma-8bit").is_symlink())
        self.assertIn("deleted mlx-community/gemma-8bit", out)

    def test_empty_publisher_folder_goes_but_a_shared_one_stays(self):
        self.run_cli("prune", "mlx-community/gemma-8bit", "--yes")
        self.assertFalse((self.store / "mlx-community").exists())
        self.run_cli("prune", "someone/Model-MLX-4bit", "--yes")
        self.assertTrue((self.store / "someone" / "Not-Splash").is_dir(),
                        "the publisher still holds another model")

    def test_splash_mirror_is_removed_so_the_bytes_are_actually_freed(self):
        """Deleting the store copy alone frees nothing: the hard-link mirror
        still pins the inodes. prune's sync must take the mirror too."""
        mirror = self.bionic / "incoai" / "Demo-Splash"
        self.assertTrue((mirror / "manifest.json").is_file())
        size = sum(f.stat().st_size for f in (self.store / "incoai" / "Demo-Splash").rglob("*")
                   if f.is_file())
        code, out, _ = self.run_cli("prune", "incoai/Demo-Splash", "--yes")
        self.assertEqual(code, 0)
        self.assertFalse((self.store / "incoai" / "Demo-Splash").exists())
        self.assertFalse(mirror.exists())
        from modelctl.cache import human_size
        self.assertIn(f"Freed {human_size(size)}", out,
                      "both links removed, so the full size comes back")

    def test_several_at_once_and_duplicates_collapse(self):
        code, out, _ = self.run_cli("prune", "unsloth/tiny-GGUF", "tiny-GGUF",
                                    "Qwen/Qwen3-7B", "--yes")
        self.assertEqual(code, 0)
        self.assertEqual(out.count("deleted unsloth/tiny-GGUF"), 1)
        self.assertFalse((self.store / "Qwen" / "Qwen3-7B").exists())

    # ---- previews and confirmation
    def test_dry_run_lists_projections_and_deletes_nothing(self):
        code, out, _ = self.run_cli("prune", "mlx-community/gemma-8bit", "-n")
        self.assertEqual(code, 0)
        self.assertIn("DRY RUN", out)
        self.assertIn(str(self.bionic / "mlx-community" / "gemma-8bit"), out)
        self.assertTrue((self.store / "mlx-community" / "gemma-8bit").is_dir())
        self.assertTrue((self.bionic / "mlx-community" / "gemma-8bit").is_symlink())

    def test_without_a_terminal_it_needs_yes(self):
        code, _, err = self.run_cli("prune", "mlx-community/gemma-8bit", usable=False)
        self.assertEqual(code, 2)
        self.assertIn("--yes", err)
        self.assertTrue((self.store / "mlx-community" / "gemma-8bit").is_dir())

    def test_answering_no_deletes_nothing(self):
        code, _, err = self.run_cli("prune", "mlx-community/gemma-8bit", usable=True, answer="n")
        self.assertEqual(code, 1)
        self.assertIn("nothing deleted", err)
        self.assertTrue((self.store / "mlx-community" / "gemma-8bit").is_dir())

    def test_answering_yes_deletes(self):
        code, _, _ = self.run_cli("prune", "mlx-community/gemma-8bit", usable=True, answer="y")
        self.assertEqual(code, 0)
        self.assertFalse((self.store / "mlx-community" / "gemma-8bit").exists())

    # ---- refusals
    def test_unknown_model_refused(self):
        code, _, err = self.run_cli("prune", "nobody/nothing", "--yes")
        self.assertEqual(code, 1)
        self.assertIn("not found", err)

    def test_one_bad_name_blocks_the_whole_batch(self):
        """Deleting some of what you asked for and failing on the rest is
        worse than deleting nothing."""
        code, _, _ = self.run_cli("prune", "mlx-community/gemma-8bit", "nobody/nothing", "--yes")
        self.assertEqual(code, 1)
        self.assertTrue((self.store / "mlx-community" / "gemma-8bit").is_dir())

    def test_active_download_refused(self):
        target = str(self.store / "mlx-community" / "gemma-8bit")
        with mock.patch.object(downloads, "active_download_dirs", return_value={target}):
            code, _, err = self.run_cli("prune", "mlx-community/gemma-8bit", "--yes")
        self.assertEqual(code, 1)
        self.assertIn("being downloaded", err)
        self.assertTrue(Path(target).is_dir())

    def test_extra_store_is_read_only(self):
        extra = make_flat_store(Path(self.tmp.name) / "extra")
        (extra / "mlx-community" / "gemma-8bit").rename(extra / "mlx-community" / "only-in-extra")
        os.environ["MODELCTL_STORE"] = f"{self.store}:{extra}"
        code, _, err = self.run_cli("prune", "mlx-community/only-in-extra", "--yes")
        self.assertEqual(code, 1)
        self.assertIn("read-only", err)
        self.assertTrue((extra / "mlx-community" / "only-in-extra").is_dir())

    def test_hf_cache_model_pointed_at_hf_cache_rm(self):
        hub = make_hf_cache(Path(self.tmp.name) / "hub")
        os.environ.update(MODELCTL_SCAN_HUB="1", HF_HUB_CACHE=str(hub))
        code, _, err = self.run_cli("prune", "org/demo-GGUF", "--yes")
        self.assertEqual(code, 1)
        self.assertIn("hf cache rm", err)

    def test_symlinked_store_entry_loses_the_link_not_the_target(self):
        """`adopt --link` puts symlinks in the store; their targets live
        elsewhere and are not the store's to delete."""
        elsewhere = Path(self.tmp.name) / "elsewhere" / "Linked-MLX"
        write(elsewhere / "model.safetensors")
        (elsewhere / "config.json").write_text('{"quantization": {"bits": 4}}')
        (self.store / "linker").mkdir()
        (self.store / "linker" / "Linked-MLX").symlink_to(elsewhere)
        code, _, _ = self.run_cli("prune", "linker/Linked-MLX", "--yes")
        self.assertEqual(code, 0)
        self.assertFalse((self.store / "linker" / "Linked-MLX").is_symlink())
        self.assertTrue((elsewhere / "model.safetensors").is_file())

    def test_nothing_to_do_without_arguments(self):
        code, _, err = self.run_cli("prune")
        self.assertEqual(code, 2)

    # ---- stale temp files
    def test_stale_removes_only_proven_leftovers(self):
        model = self.store / "mlx-community" / "gemma-8bit"
        scr = downloads.scratch_dir(model)
        scr.mkdir(parents=True)
        (scr / "a.safetensors.metadata").write_text("c\naaa111\n1\n")
        stale = scr / "h=.aaa111.x1.incomplete"
        live = scr / "h=.bbb222.x2.incomplete"
        stale.write_bytes(b"s" * 10)
        live.write_bytes(b"l" * 10)
        code, out, _ = self.run_cli("prune", "--stale", "--yes")
        self.assertEqual(code, 0, out)
        self.assertFalse(stale.exists())
        self.assertTrue(live.exists(), "a live partial is what a resume keeps")
        self.assertTrue((scr / "a.safetensors.metadata").exists())
        self.assertTrue(model.is_dir(), "--stale alone never deletes a model")

    def test_stale_with_nothing_to_remove(self):
        code, out, _ = self.run_cli("prune", "--stale", "--yes")
        self.assertEqual(code, 0)
        self.assertIn("no stale temp files", out)


class ReclaimableTest(EnvTestCase):
    def setUp(self):
        super().setUp()
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.d = Path(self.tmp.name)

    def test_single_link_file_is_freed(self):
        write(self.d / "a" / "f", b"x" * 100)
        self.assertEqual(cli._reclaimable([self.d / "a"]), 100)

    def test_hard_link_freed_only_when_every_link_goes(self):
        """The Splash case: store copy + Bionic mirror share inodes."""
        write(self.d / "store" / "f", b"x" * 100)
        (self.d / "mirror").mkdir()
        os.link(self.d / "store" / "f", self.d / "mirror" / "f")
        self.assertEqual(cli._reclaimable([self.d / "store"]), 0,
                         "the mirror still holds the bytes")
        self.assertEqual(cli._reclaimable([self.d / "store", self.d / "mirror"]), 100,
                         "counted once, not twice")

    def test_symlinked_root_frees_nothing(self):
        write(self.d / "real" / "f", b"x" * 100)
        (self.d / "link").symlink_to(self.d / "real")
        self.assertEqual(cli._reclaimable([self.d / "link"]), 0)
