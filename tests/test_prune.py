from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from modelctl.adapters import bionic
from modelctl.adapters.base import MirrorLedger, prune_links
from modelctl.adapters.bionic import BionicAdapter
from modelctl.adapters.omlx import OmlxAdapter, OmlxDirs
from modelctl.cache import scan_flat
from tests.helpers import EnvTestCase, make_flat_store
from tests.test_downloads import make_scratch


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        self.store = make_flat_store(self.base / "store")
        self.app = self.base / "app-models"
        self.app.mkdir()

    def link(self, rel, target):
        p = self.app / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.symlink_to(target)
        return p

    def prune(self, desired=(), root=None, dry_run=False, protected=None):
        sources = [self.store]
        return prune_links(root or self.app, set(desired), sources=sources,
                           protected=sources if protected is None else protected,
                           adapter="t", dry_run=dry_run)


class PruneLinksTest(Base):
    def test_removes_a_link_whose_model_left_the_store(self):
        gone = self.link("pub/deleted", self.store / "pub" / "deleted")
        actions = self.prune()
        self.assertFalse(gone.is_symlink())
        self.assertEqual(actions[0].op, "unlink")
        self.assertIn("no longer in the store", actions[0].detail)

    def test_removes_a_live_link_no_longer_wanted(self):
        """A model still in the store but no longer projected (became partial,
        got shadowed) must lose its link too, or the view does not match."""
        live = self.link("mlx-community/gemma-8bit", self.store / "mlx-community" / "gemma-8bit")
        actions = self.prune()
        self.assertFalse(live.is_symlink())
        self.assertIn("no longer projected", actions[0].detail)

    def test_keeps_what_this_sync_wanted(self):
        keep = self.link("mlx-community/gemma-8bit", self.store / "mlx-community" / "gemma-8bit")
        self.assertEqual(self.prune(desired={keep}), [])
        self.assertTrue(keep.is_symlink())

    def test_never_touches_links_that_point_elsewhere(self):
        """Pointing outside modelctl's sources means someone else made it."""
        elsewhere = self.base / "user-stuff"
        elsewhere.mkdir()
        theirs = self.link("theirs/model", elsewhere)
        broken_theirs = self.link("theirs/broken", self.base / "nowhere")
        self.assertEqual(self.prune(), [])
        self.assertTrue(theirs.is_symlink())
        self.assertTrue(broken_theirs.is_symlink())

    def test_never_touches_real_files(self):
        real = self.app / "pub" / "downloaded-by-the-app" / "model.gguf"
        real.parent.mkdir(parents=True)
        real.write_bytes(b"x")
        self.prune()
        self.assertTrue(real.is_file())

    def test_never_prunes_inside_a_store(self):
        """When the app's dir IS the store (LM Studio pointed at it), its
        links belong to the user: `adopt --link` makes exactly such links."""
        adopted = self.store / "pub" / "adopted"
        adopted.parent.mkdir(parents=True, exist_ok=True)
        adopted.symlink_to(self.store / "gone")
        self.assertEqual(self.prune(root=self.store), [])
        self.assertTrue(adopted.is_symlink())

    def test_does_not_descend_into_a_store_nested_under_the_root(self):
        nested = self.app / "nested-store"
        shutil.copytree(self.store, nested, symlinks=True)
        inner = nested / "pub" / "inner-link"
        inner.parent.mkdir(parents=True, exist_ok=True)
        inner.symlink_to(self.store / "gone")
        self.prune(protected=[self.store, nested])
        self.assertTrue(inner.is_symlink())

    def test_removes_directories_it_emptied_but_not_the_root(self):
        self.link("pub/model/a.gguf", self.store / "gone" / "a.gguf")
        self.link("pub/model/b.gguf", self.store / "gone" / "b.gguf")
        self.link("pub/other", self.store / "gone2")
        self.prune()
        self.assertFalse((self.app / "pub").exists())
        self.assertTrue(self.app.is_dir())

    def test_keeps_directories_that_still_hold_something(self):
        self.link("pub/gone", self.store / "gone")
        (self.app / "pub" / "keep.txt").write_text("mine")
        self.prune()
        self.assertTrue((self.app / "pub" / "keep.txt").is_file())

    def test_dry_run_reports_but_removes_nothing(self):
        gone = self.link("pub/deleted", self.store / "gone")
        actions = self.prune(dry_run=True)
        self.assertEqual(len(actions), 1)
        self.assertTrue(gone.is_symlink())


class MirrorLedgerTest(Base):
    """Hard-link mirrors cannot prove their own ownership once the store copy
    is gone, so only ledger-recorded files are ever removed."""

    def adapter(self):
        return BionicAdapter(bionic.BionicDir(self.app, "test", True))

    def sync(self, repos=None, **kw):
        opts = dict(prune=True, sources=[self.store], protected=[self.store],
                    state_dir=self.store / ".modelctl")
        opts.update(kw)
        return self.adapter().sync(repos if repos is not None else scan_flat(self.store), **opts)

    def mirror(self):
        return self.app / "incoai" / "Demo-Splash"

    def test_sync_records_the_mirror(self):
        self.sync()
        led = json.loads((self.store / ".modelctl" / "mirrors.json").read_text())
        self.assertIn(str(self.mirror()), led)
        self.assertIn("manifest.json", led[str(self.mirror())]["files"])

    def test_mirror_of_a_deleted_model_is_removed(self):
        self.sync()
        self.assertTrue((self.mirror() / "manifest.json").is_file())
        shutil.rmtree(self.store / "incoai" / "Demo-Splash")
        actions = self.sync()
        unlinked = [a for a in actions if a.op == "unlink" and "Demo-Splash" in a.target]
        self.assertEqual(len(unlinked), 1)
        self.assertFalse(self.mirror().exists(), "hard links still pin the bytes")
        led = json.loads((self.store / ".modelctl" / "mirrors.json").read_text())
        self.assertNotIn(str(self.mirror()), led)

    def test_files_modelctl_did_not_create_are_kept(self):
        self.sync()
        (self.mirror() / "user-notes.txt").write_text("mine")
        shutil.rmtree(self.store / "incoai" / "Demo-Splash")
        actions = self.sync()
        self.assertTrue((self.mirror() / "user-notes.txt").is_file())
        self.assertFalse((self.mirror() / "manifest.json").exists())
        self.assertIn("did not create", [a for a in actions if a.op == "unlink"][0].detail)

    def test_an_unrecorded_real_package_is_never_touched(self):
        """E.g. a Splash model Bionic downloaded itself: indistinguishable from
        an orphaned mirror, so it must be left alone."""
        own = self.app / "someone" / "Own-Splash"
        own.mkdir(parents=True)
        (own / "manifest.json").write_text("{}")
        self.sync()
        self.assertTrue((own / "manifest.json").is_file())

    def test_corrupt_ledger_is_treated_as_empty(self):
        (self.store / ".modelctl").mkdir()
        (self.store / ".modelctl" / "mirrors.json").write_text("{not json")
        self.sync()                                    # must not raise
        self.assertTrue((self.mirror() / "manifest.json").is_file())

    def test_dry_run_keeps_ledger_and_mirror(self):
        self.sync()
        shutil.rmtree(self.store / "incoai" / "Demo-Splash")
        self.sync(dry_run=True)
        self.assertTrue((self.mirror() / "manifest.json").is_file())
        led = MirrorLedger(self.store / ".modelctl" / "mirrors.json")
        self.assertIn(str(self.mirror()), led.entries)


class PartialProjectionTest(Base):
    def test_partial_mlx_is_not_projected_and_its_old_link_goes(self):
        ad = BionicAdapter(bionic.BionicDir(self.app, "test", True))
        opts = dict(prune=True, sources=[self.store], protected=[self.store])
        ad.sync(scan_flat(self.store), **opts)
        link = self.app / "mlx-community" / "gemma-8bit"
        self.assertTrue(link.is_symlink())
        make_scratch(self.store / "mlx-community" / "gemma-8bit", incomplete=1)
        actions = ad.sync(scan_flat(self.store), **opts)
        self.assertFalse(link.is_symlink(), "a half model must not stay visible")
        self.assertTrue(any("incomplete" in a.detail for a in actions))

    def test_gguf_files_are_still_linked_during_a_download(self):
        """A .gguf at its final path is always whole (hf renames on
        completion), so another quant arriving must not hide this one."""
        make_scratch(self.store / "unsloth" / "tiny-GGUF", incomplete=1)
        ad = BionicAdapter(bionic.BionicDir(self.app, "test", True))
        ad.sync(scan_flat(self.store), prune=True, sources=[self.store], protected=[self.store])
        self.assertTrue((self.app / "unsloth" / "tiny-GGUF" / "tiny-Q4_K_M.gguf").is_symlink())


class OmlxPruneTest(Base):
    def test_dangling_link_removed_and_restart_suggested(self):
        self.link("pub/deleted", self.store / "pub" / "deleted")
        ad = OmlxAdapter(OmlxDirs([self.app], "test", True))
        ad.sync(scan_flat(self.store), prune=True, sources=[self.store], protected=[self.store])
        again = OmlxAdapter(OmlxDirs([self.app], "test", True))
        (self.app / "pub2").mkdir()
        (self.app / "pub2" / "gone").symlink_to(self.store / "gone")
        actions = again.sync(scan_flat(self.store), prune=True,
                             sources=[self.store], protected=[self.store])
        self.assertTrue(any(a.op == "unlink" for a in actions))
        self.assertTrue(any(a.target == "oMLX server" for a in actions),
                        "removing a model also needs a restart to take effect")


class CliPruneTest(EnvTestCase):
    def test_no_prune_leaves_projections_in_place(self):
        import io
        from contextlib import redirect_stdout
        import os
        from modelctl.cli import main
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        base = Path(tmp.name)
        store = make_flat_store(base / "store")
        app = base / "bionic"
        (app / "pub").mkdir(parents=True)
        dead = app / "pub" / "gone"
        dead.symlink_to(store / "pub" / "gone")
        os.environ.update(MODELCTL_STORE=str(store), MODELCTL_BIONIC_DIR=str(app),
                          MODELCTL_OMLX_DIR=str(base / "omlx"), MODELCTL_LMSTUDIO_DIR=str(store))
        with redirect_stdout(io.StringIO()):
            main(["sync", "-a", "bionic", "--no-prune"])
        self.assertTrue(dead.is_symlink())
        with redirect_stdout(io.StringIO()) as out:
            main(["sync", "-a", "bionic"])
        self.assertFalse(dead.is_symlink())
        self.assertIn("- [bionic]", out.getvalue())
