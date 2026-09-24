from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from modelctl.adapters import bionic
from modelctl.adapters.base import ensure_symlink
from modelctl.adapters.bionic import BionicAdapter
from modelctl.adapters.llamacpp import LlamaCppAdapter
from modelctl.adapters.lmstudio import LMStudioAdapter
from modelctl.adapters.mlx import MlxAdapter
from modelctl.adapters.ollama import OllamaAdapter, _tag
from modelctl.adapters.splash import SplashAdapter
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


class BionicResolutionTest(unittest.TestCase):
    """Bionic reads its OWN settings under <home>/apps/bionic; getting this
    wrong means syncing into a folder the app never looks at."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.fake_home = Path(self.tmp.name) / "home"
        (self.fake_home).mkdir(parents=True)
        self._saved = os.environ.pop("MODELCTL_BIONIC_DIR", None)
        self.addCleanup(self._restore)
        patcher = mock.patch.object(Path, "home", staticmethod(lambda: self.fake_home))
        patcher.start()
        self.addCleanup(patcher.stop)

    def _restore(self):
        if self._saved is None:
            os.environ.pop("MODELCTL_BIONIC_DIR", None)
        else:
            os.environ["MODELCTL_BIONIC_DIR"] = self._saved

    def write_settings(self, home, folder):
        s = bionic.settings_path(home)
        s.parent.mkdir(parents=True, exist_ok=True)
        s.write_text(json.dumps({"downloadsFolder": str(folder)}))
        return s

    def test_home_defaults_to_dot_lmstudio(self):
        self.assertEqual(bionic.lmstudio_home(), self.fake_home / ".lmstudio")

    def test_home_pointer_wins(self):
        elsewhere = Path(self.tmp.name) / "external-home"
        (self.fake_home / ".lmstudio-home-pointer").write_text(f"{elsewhere}\n")
        self.assertEqual(bionic.lmstudio_home(), elsewhere)

    def test_reads_downloads_folder_from_bionic_settings(self):
        home = self.fake_home / ".lmstudio"
        self.write_settings(home, "/tmp/bionic-models")
        where = bionic.models_dir()
        self.assertEqual(where.path, Path("/tmp/bionic-models"))
        self.assertIn("downloadsFolder", where.source)
        self.assertTrue(where.configured)

    def test_ignores_lmstudios_own_settings(self):
        """LM Studio's settings.json sits one level up and must not be read."""
        home = self.fake_home / ".lmstudio"
        home.mkdir(parents=True, exist_ok=True)
        (home / "settings.json").write_text(json.dumps({"downloadsFolder": "/tmp/lmstudio-store"}))
        self.assertEqual(bionic.models_dir().path, home / "models")

    def test_falls_back_to_home_models_when_settings_missing(self):
        where = bionic.models_dir()
        self.assertEqual(where.path, self.fake_home / ".lmstudio" / "models")
        self.assertIn("default", where.source)
        self.assertFalse(where.configured)

    def test_falls_back_when_settings_is_malformed(self):
        home = self.fake_home / ".lmstudio"
        s = bionic.settings_path(home)
        s.parent.mkdir(parents=True, exist_ok=True)
        s.write_text("{not json")
        where = bionic.models_dir()
        self.assertEqual(where.path, home / "models")
        self.assertFalse(where.configured)

    def test_non_dict_settings_does_not_crash(self):
        """`Config.load()` calls this, so every modelctl command depends on it.
        json.loads("null") is VALID json and returns None, which is not an
        error and so slipped past an except (OSError, ValueError)."""
        home = self.fake_home / ".lmstudio"
        s = bionic.settings_path(home)
        s.parent.mkdir(parents=True, exist_ok=True)
        for payload in ("null", "[1, 2, 3]", '"a string"', "42"):
            with self.subTest(payload=payload):
                s.write_text(payload)
                where = bionic.models_dir()
                self.assertEqual(where.path, home / "models")
                self.assertFalse(where.configured)

    def test_settings_without_downloads_folder_is_still_configured(self):
        """Settings we read fine but that simply omit the key must not be
        reported as unreadable: the app IS installed."""
        home = self.fake_home / ".lmstudio"
        s = bionic.settings_path(home)
        s.parent.mkdir(parents=True, exist_ok=True)
        s.write_text(json.dumps({"language": "en"}))
        where = bionic.models_dir()
        self.assertEqual(where.path, home / "models")
        self.assertTrue(where.configured)
        self.assertIn("no downloadsFolder", where.source)

    def test_non_string_downloads_folder_falls_back(self):
        home = self.fake_home / ".lmstudio"
        s = bionic.settings_path(home)
        s.parent.mkdir(parents=True, exist_ok=True)
        s.write_text(json.dumps({"downloadsFolder": {"nested": "object"}}))
        self.assertEqual(bionic.models_dir().path, home / "models")

    def test_doctor_uses_injected_provenance_not_the_real_filesystem(self):
        """doctor() used to stat the REAL ~/.lmstudio path regardless of the
        directory it was constructed with, so it could contradict itself."""
        ad = BionicAdapter(bionic.BionicDir(self.fake_home / "injected", "test", False))
        self.assertTrue(any("Bionic settings not found" in l for l in ad.doctor()))
        ad = BionicAdapter(bionic.BionicDir(self.fake_home / "injected", "test", True))
        self.assertFalse(any("Bionic settings not found" in l for l in ad.doctor()))

    def test_env_var_overrides_settings(self):
        self.write_settings(self.fake_home / ".lmstudio", "/tmp/from-settings")
        os.environ["MODELCTL_BIONIC_DIR"] = "/tmp/from-env"
        where = bionic.models_dir()
        self.assertEqual(where.path, Path("/tmp/from-env"))
        self.assertEqual(where.source, "MODELCTL_BIONIC_DIR")


class BionicAdapterTest(AdapterBase):
    def adapter(self, d):
        return BionicAdapter(bionic.BionicDir(d, "test", True))

    def test_accepts_only_gguf_and_mlx(self):
        ad = self.adapter(self.base / "bio")
        self.assertTrue(ad.accepts(self.repo("unsloth/tiny-GGUF")))
        self.assertTrue(ad.accepts(self.repo("mlx-community/gemma-8bit")))
        self.assertFalse(ad.accepts(self.repo("Qwen/Qwen3-7B")))

    def test_projects_gguf_and_mlx_like_lmstudio(self):
        bio = self.base / "bio"
        self.adapter(bio).sync(self.repos, dry_run=False)
        gguf = bio / "bartowski" / "split-GGUF" / "Q4_K_M" / "model-00001-of-00002.gguf"
        self.assertTrue(gguf.is_symlink())
        self.assertTrue(gguf.resolve().is_file())
        mlx = bio / "mlx-community" / "gemma-8bit"
        self.assertTrue(mlx.is_symlink())
        self.assertTrue((mlx / "config.json").exists())

    def test_actions_are_labelled_bionic(self):
        actions = self.adapter(self.base / "bio").sync(self.repos, dry_run=True)
        self.assertTrue(actions)
        self.assertEqual({a.adapter for a in actions}, {"bionic"})

    def test_matches_lmstudio_projection_exactly(self):
        """Same engines, same tree: the two adapters must not drift apart.
        Compares targets AND ops, so a divergence in the splash rule shows up."""
        lm = LMStudioAdapter(self.base / "lm").sync(self.repos, dry_run=True)
        bio = self.adapter(self.base / "bio").sync(self.repos, dry_run=True)
        strip = lambda acts, d: sorted((a.target.replace(str(d), ""), a.op) for a in acts)
        self.assertEqual(strip(lm, self.base / "lm"), strip(bio, self.base / "bio"))

    def test_safetensors_not_projected(self):
        actions = self.adapter(self.base / "bio").sync(self.repos, dry_run=False)
        self.assertFalse(any("Qwen3-7B" in a.target for a in actions))

    def test_idempotent(self):
        bio = self.base / "bio"
        self.adapter(bio).sync(self.repos, dry_run=False)
        again = self.adapter(bio).sync(self.repos, dry_run=False)
        self.assertEqual({a.op for a in again}, {"skip"})

    def test_dry_run_creates_nothing(self):
        bio = self.base / "bio"
        self.adapter(bio).sync(self.repos, dry_run=True)
        self.assertFalse(bio.exists())


class SplashAdapterTest(AdapterBase):
    def test_accepts_only_splash(self):
        ad = SplashAdapter()
        self.assertTrue(ad.accepts(self.repo("incoai/Demo-Splash")))
        self.assertFalse(ad.accepts(self.repo("unsloth/tiny-GGUF")))
        self.assertFalse(ad.accepts(self.repo("Qwen/Qwen3-7B")))
        self.assertFalse(ad.accepts(self.repo("mlx-community/gemma-8bit")))

    def test_native_launch_command(self):
        actions = SplashAdapter().sync(self.repos)
        self.assertEqual(len(actions), 1)
        a = actions[0]
        self.assertEqual(a.op, "native")
        self.assertIn("splash serve --model", a.detail)
        self.assertIn("Demo-Splash", a.detail)

    def test_creates_nothing_on_disk(self):
        before = sorted(self.base.rglob("*"))
        SplashAdapter().sync(self.repos, dry_run=False)
        self.assertEqual(before, sorted(self.base.rglob("*")))


class SplashRoutingTest(AdapterBase):
    """A Splash package must reach the engines that can load it, and only those.
    Before splash existed it classified as `safetensors`, so vLLM and mlx_lm both
    offered launch commands that cannot work."""

    def test_engines_that_cannot_load_splash_reject_it(self):
        repo = self.repo("incoai/Demo-Splash")
        self.assertFalse(VllmAdapter().accepts(repo))
        self.assertFalse(MlxAdapter().accepts(repo))
        self.assertFalse(LlamaCppAdapter().accepts(repo))
        self.assertFalse(OllamaAdapter().accepts(repo))

    def test_engines_that_can_load_splash_accept_it(self):
        repo = self.repo("incoai/Demo-Splash")
        self.assertTrue(SplashAdapter().accepts(repo))
        self.assertTrue(LMStudioAdapter(self.base / "lm").accepts(repo))
        self.assertTrue(BionicAdapter(bionic.BionicDir(self.base / "bio", "test", True)).accepts(repo))

    def test_splash_outside_the_models_dir_is_hardlinked_not_symlinked(self):
        """The indexer rejects a symlinked package ("escapes the models
        directory") AND symlinked artifacts inside a real package dir ("path
        escapes its directory"). Hard links have no separate real path, so they
        pass both. All three verified against Bionic 1.1.5."""
        bio = self.base / "bio"
        actions = BionicAdapter(bionic.BionicDir(bio, "test", True)).sync(self.repos, dry_run=False)
        splash = [a for a in actions if "Demo-Splash" in a.target]
        self.assertEqual(len(splash), 1, "one action for the whole package, not per file")
        self.assertEqual(splash[0].op, "link")
        self.assertIn("hard links", splash[0].detail)

        pkg = bio / "incoai" / "Demo-Splash"
        self.assertTrue(pkg.is_dir())
        self.assertFalse(pkg.is_symlink(), "the package dir itself must be real")
        for rel in ("manifest.json", "target/layer-0.bin", "draft/layer-0.bin"):
            f = pkg / rel
            self.assertTrue(f.is_file(), rel)
            self.assertFalse(f.is_symlink(), f"{rel} must be a hard link, not a symlink")
            self.assertEqual(f.stat().st_ino,
                             (self.repo("incoai/Demo-Splash").root / rel).stat().st_ino,
                             f"{rel} must share the source inode (no extra bytes)")

    def test_splash_hardlink_tree_is_idempotent(self):
        bio = self.base / "bio-idem"
        ad = BionicAdapter(bionic.BionicDir(bio, "test", True))
        ad.sync(self.repos, dry_run=False)
        again = [a for a in ad.sync(self.repos, dry_run=False) if "Demo-Splash" in a.target]
        self.assertEqual(again[0].op, "skip")
        self.assertIn("up to date", again[0].detail)

    def test_splash_hardlink_repairs_a_replaced_source(self):
        """A re-download writes a NEW inode, orphaning the mirror. The next
        sync must notice by inode and relink, or Bionic serves stale bytes."""
        bio = self.base / "bio-stale"
        ad = BionicAdapter(bionic.BionicDir(bio, "test", True))
        ad.sync(self.repos, dry_run=False)
        src = self.repo("incoai/Demo-Splash").root / "target" / "layer-0.bin"
        src.unlink()
        src.write_bytes(b"replaced-by-a-redownload")       # new inode
        after = [a for a in ad.sync(self.repos, dry_run=False) if "Demo-Splash" in a.target]
        self.assertEqual(after[0].op, "relink")
        self.assertIn("1 stale", after[0].detail)
        mirrored = bio / "incoai" / "Demo-Splash" / "target" / "layer-0.bin"
        self.assertEqual(mirrored.read_bytes(), b"replaced-by-a-redownload")
        self.assertEqual(mirrored.stat().st_ino, src.stat().st_ino)

    def test_splash_hardlink_dry_run_creates_nothing(self):
        bio = self.base / "bio-dry"
        BionicAdapter(bionic.BionicDir(bio, "test", True)).sync(self.repos, dry_run=True)
        self.assertFalse((bio / "incoai" / "Demo-Splash").exists())

    def test_splash_already_inside_the_models_dir_is_served(self):
        """The working configuration: the app's models dir IS the store, so the
        package is already contained and no link is needed."""
        actions = BionicAdapter(bionic.BionicDir(self.store, "test", True)).sync(self.repos)
        splash = [a for a in actions if "Demo-Splash" in a.target]
        self.assertEqual(len(splash), 1)
        self.assertEqual(splash[0].op, "skip")
        self.assertIn("already in place", splash[0].detail)

    def test_splash_served_when_store_is_nested_under_the_models_dir(self):
        """Containment, not equality: a store beneath the models dir still
        resolves inside it, so the symlink is safe."""
        outer = self.base / "outer"
        repos = scan_flat(make_flat_store(outer / "nested-store"))
        actions = BionicAdapter(bionic.BionicDir(outer, "test", True)).sync(repos, dry_run=False)
        splash = [a for a in actions if "Demo-Splash" in a.target]
        self.assertEqual(len(splash), 1)
        self.assertIn(splash[0].op, ("link", "skip"))
        self.assertNotIn("can't be symlinked", splash[0].detail)

    def test_gguf_and_mlx_are_unaffected_by_the_splash_rule(self):
        bio = self.base / "bio2"
        BionicAdapter(bionic.BionicDir(bio, "test", True)).sync(self.repos, dry_run=False)
        self.assertTrue((bio / "mlx-community" / "gemma-8bit").is_symlink())
        self.assertTrue((bio / "unsloth" / "tiny-GGUF" / "tiny-Q4_K_M.gguf").is_symlink())


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
