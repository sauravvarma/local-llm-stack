from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from modelctl.adapters import omlx
from modelctl.adapters.omlx import OmlxAdapter, OmlxDirs
from modelctl.cache import scan_flat
from tests.helpers import EnvTestCase, make_flat_store
from tests.test_downloads import make_scratch


class HomeAndDirsTest(EnvTestCase):
    """Resolution must match oMLX 0.7's settings.py exactly, or modelctl and
    the app disagree about where models live."""

    def setUp(self):
        super().setUp()
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.home = Path(self.tmp.name) / "home"
        self.home.mkdir()
        patcher = mock.patch.object(Path, "home", staticmethod(lambda: self.home))
        patcher.start()
        self.addCleanup(patcher.stop)

    def settings(self, obj, home=None):
        home = home or (self.home / ".omlx")
        home.mkdir(parents=True, exist_ok=True)
        (home / "settings.json").write_text(obj if isinstance(obj, str) else json.dumps(obj))

    # ---- home
    def test_home_defaults_to_dot_omlx(self):
        self.assertEqual(omlx.omlx_home(), self.home / ".omlx")

    def test_bootstrap_file_relocates_home(self):
        boot = self.home / "Library" / "Application Support" / "oMLX" / "base-path"
        boot.parent.mkdir(parents=True)
        boot.write_text("/elsewhere/omlx\n")
        self.assertEqual(omlx.omlx_home(), Path("/elsewhere/omlx"))

    def test_env_beats_bootstrap(self):
        boot = self.home / "Library" / "Application Support" / "oMLX" / "base-path"
        boot.parent.mkdir(parents=True)
        boot.write_text("/from/bootstrap")
        os.environ["OMLX_BASE_PATH"] = "/from/env"
        self.assertEqual(omlx.omlx_home(), Path("/from/env"))

    # ---- model dirs
    def test_model_dirs_list_in_scan_order(self):
        self.settings({"model": {"model_dirs": ["/a", "/b"]}})
        where = omlx.model_dirs()
        self.assertEqual(where.dirs, [Path("/a"), Path("/b")])
        self.assertTrue(where.configured)

    def test_deprecated_model_dir_still_honoured(self):
        self.settings({"model": {"model_dir": "/legacy"}})
        self.assertEqual(omlx.model_dirs().dirs, [Path("/legacy")])

    def test_list_beats_deprecated_single(self):
        self.settings({"model": {"model_dirs": ["/new"], "model_dir": "/old"}})
        self.assertEqual(omlx.model_dirs().dirs, [Path("/new")])

    def test_omlx_model_dir_env_is_comma_separated(self):
        self.settings({"model": {"model_dirs": ["/from/settings"]}})
        os.environ["OMLX_MODEL_DIR"] = "/x, /y"
        self.assertEqual(omlx.model_dirs().dirs, [Path("/x"), Path("/y")])

    def test_modelctl_override_beats_everything(self):
        os.environ["OMLX_MODEL_DIR"] = "/x"
        os.environ["MODELCTL_OMLX_DIR"] = "/mine"
        self.assertEqual(omlx.model_dirs().dirs, [Path("/mine")])

    def test_missing_settings_falls_back_unconfigured(self):
        where = omlx.model_dirs()
        self.assertEqual(where.dirs, [self.home / ".omlx" / "models"])
        self.assertFalse(where.configured)

    def test_malformed_settings_never_raise(self):
        """Config.load() calls this for every command: a bad file must not
        take the CLI down (the Bionic lesson)."""
        for payload in ("null", "[1, 2]", "{not json", '"str"', '{"model": null}',
                        '{"model": {"model_dirs": [1, null, ""]}}'):
            with self.subTest(payload=payload):
                self.settings(payload)
                where = omlx.model_dirs()
                self.assertEqual(where.dirs, [self.home / ".omlx" / "models"])


class OmlxSyncTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        self.store = make_flat_store(self.base / "store")
        self.repos = scan_flat(self.store)
        self.dir = self.base / "omlx-models"

    def repo(self, rid):
        return next(r for r in self.repos if r.repo_id == rid)

    def adapter(self, *dirs):
        return OmlxAdapter(OmlxDirs(list(dirs) or [self.dir], "test", True))

    def test_accepts_what_mlx_lm_loads(self):
        ad = self.adapter()
        self.assertTrue(ad.accepts(self.repo("mlx-community/gemma-8bit")))
        self.assertTrue(ad.accepts(self.repo("Qwen/Qwen3-7B")))

    def test_rejects_what_omlx_cannot_load(self):
        ad = self.adapter()
        for rid in ("unsloth/tiny-GGUF",        # no GGUF loader
                    "incoai/Demo-Splash",       # no root config.json
                    "TheBloke/Demo-GPTQ"):      # server-only quantisation
            with self.subTest(rid=rid):
                self.assertFalse(ad.accepts(self.repo(rid)))

    def test_projects_publisher_model_directory_symlinks(self):
        self.adapter().sync(self.repos, dry_run=False)
        link = self.dir / "mlx-community" / "gemma-8bit"
        self.assertTrue(link.is_symlink())
        self.assertTrue((link / "config.json").is_file())

    def test_idempotent_and_restart_note_only_on_change(self):
        first = self.adapter().sync(self.repos, dry_run=False)
        self.assertTrue(any(a.target == "oMLX server" for a in first))
        again = self.adapter().sync(self.repos, dry_run=False)
        self.assertFalse(any(a.target == "oMLX server" for a in again),
                         "nothing changed, so no restart is needed")
        self.assertTrue(all(a.op == "skip" for a in again))

    def test_store_listed_in_model_dirs_needs_no_projection(self):
        actions = self.adapter(self.store).sync(self.repos, dry_run=False)
        mlx = [a for a in actions if "gemma-8bit" in a.target]
        self.assertEqual(mlx[0].op, "skip")
        self.assertIn("already visible", mlx[0].detail)
        self.assertFalse(any(a.op == "link" for a in actions))

    def test_store_nested_too_deep_is_not_visible(self):
        """oMLX scans exactly two levels; containment alone is not enough."""
        actions = self.adapter(self.base).sync(self.repos, dry_run=True)
        mlx = [a for a in actions if "gemma-8bit" in a.target]
        self.assertEqual(mlx[0].op, "link")

    def test_folder_name_collision_is_reported_not_linked(self):
        """oMLX keys models by folder name alone; the second one would be
        silently shadowed."""
        other = self.store / "someone-else" / "gemma-8bit"
        other.mkdir(parents=True)
        (other / "config.json").write_text(json.dumps(
            {"model_type": "gemma", "quantization": {"bits": 4}}))
        (other / "model.safetensors").write_bytes(b"x")
        actions = self.adapter().sync(scan_flat(self.store), dry_run=False)
        clash = [a for a in actions if a.target.endswith("gemma-8bit")]
        self.assertEqual(sorted(a.op for a in clash), ["link", "skip"])
        self.assertIn("shadowed", [a for a in clash if a.op == "skip"][0].detail)

    def test_partial_download_is_held_back(self):
        make_scratch(self.repo("mlx-community/gemma-8bit").root, incomplete=1)
        actions = self.adapter().sync(self.repos, dry_run=False)
        gemma = [a for a in actions if "gemma-8bit" in a.target][0]
        self.assertEqual(gemma.op, "skip")
        self.assertIn("incomplete", gemma.detail)
        self.assertFalse((self.dir / "mlx-community" / "gemma-8bit").exists())
