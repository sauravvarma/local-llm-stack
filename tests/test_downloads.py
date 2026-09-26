from __future__ import annotations

import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from modelctl import downloads, picker


def make_scratch(model: Path, *, incomplete: int = 0, completed: int = 0,
                 incomplete_bytes: int = 1024, age: float = 0.0) -> Path:
    """Lay out `hf download`'s bookkeeping the way it really appears."""
    d = downloads.scratch_dir(model) / "target"
    d.mkdir(parents=True, exist_ok=True)
    for i in range(completed):
        (d / f"layer-{i}.bin.metadata").write_text("commit\netag\n1789893836.0\n")
    for i in range(incomplete):
        p = d / f"opaque{i}.sha{i}.incomplete"
        p.write_bytes(b"x" * incomplete_bytes)
        if age:
            past = time.time() - age
            os.utime(p, (past, past))
    return model


class StatusTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.model = Path(self.tmp.name) / "pub" / "model"
        self.model.mkdir(parents=True)

    def test_no_scratch_is_complete(self):
        st = downloads.inspect(self.model, active_dirs=set())
        self.assertEqual(st.status, "complete")
        self.assertFalse(st.partial)

    def test_metadata_only_is_complete(self):
        make_scratch(self.model, completed=3)
        st = downloads.inspect(self.model, active_dirs=set())
        self.assertEqual(st.status, "complete")
        self.assertEqual(st.completed, 3)

    def test_partial_with_no_process_is_interrupted(self):
        """The common case here: closed lid or a network change."""
        make_scratch(self.model, incomplete=2, completed=5)
        st = downloads.inspect(self.model, active_dirs=set())
        self.assertEqual(st.status, "interrupted")
        self.assertIn("resumes", st.hint)

    def test_partial_with_fresh_process_is_downloading(self):
        make_scratch(self.model, incomplete=1)
        st = downloads.inspect(self.model, active_dirs={str(self.model)})
        self.assertEqual(st.status, "downloading")

    def test_partial_with_idle_process_is_stalled(self):
        make_scratch(self.model, incomplete=1, age=downloads.STALL_SECONDS + 60)
        st = downloads.inspect(self.model, active_dirs={str(self.model)})
        self.assertEqual(st.status, "stalled")
        self.assertIn("kill", st.hint)

    def test_pending_bytes_counts_what_a_resume_keeps(self):
        make_scratch(self.model, incomplete=3, incomplete_bytes=2048)
        st = downloads.inspect(self.model, active_dirs=set())
        self.assertEqual(st.pending_bytes, 3 * 2048)

    def test_trailing_slash_still_matches_a_running_process(self):
        make_scratch(self.model, incomplete=1)
        st = downloads.inspect(self.model, active_dirs={str(self.model) + "/"})
        self.assertEqual(st.status, "downloading")

    def test_unreadable_scratch_does_not_raise(self):
        make_scratch(self.model, incomplete=1)
        with mock.patch.object(Path, "rglob", side_effect=OSError("boom")):
            st = downloads.inspect(self.model, active_dirs=set())
        self.assertEqual(st.status, "complete")


class StaleTempFileTest(unittest.TestCase):
    """hf 2.0 starts a fresh temp file per retry and abandons the losers when
    one attempt finishes, so temp files can outlive a COMPLETE model. Found in
    the wild: a verified-complete 68GB model reported as interrupted."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.model = Path(self.tmp.name) / "pub" / "model"
        self.scr = downloads.scratch_dir(self.model)
        self.scr.mkdir(parents=True)

    def done(self, name, etag):
        (self.scr / f"{name}.metadata").write_text(f"commit\n{etag}\n1790000000.0\n")

    def temp(self, name, size=100):
        p = self.scr / name
        p.write_bytes(b"x" * size)
        return p

    def test_leftover_from_a_finished_file_is_stale_not_partial(self):
        self.done("model-00001.safetensors", "aaa111")
        self.temp("hashA=.aaa111.1f2e3d4c.incomplete")        # hf 2.x naming
        self.temp("hashA=.aaa111.9a8b7c6d.incomplete", size=0)
        st = downloads.inspect(self.model, active_dirs=set())
        self.assertEqual(st.status, "complete")
        self.assertEqual(len(st.stale), 2)
        self.assertIn("stale", st.summary())
        self.assertIn("reclaimable", st.summary())

    def test_temp_for_an_unfinished_file_is_still_partial(self):
        self.done("model-00001.safetensors", "aaa111")
        self.temp("hashB=.bbb222.1f2e3d4c.incomplete")
        st = downloads.inspect(self.model, active_dirs=set())
        self.assertEqual(st.status, "interrupted")
        self.assertEqual(len(st.incomplete), 1)

    def test_hf1_naming_is_understood_too(self):
        self.done("model.safetensors", "ccc333")
        self.temp("hashC=.ccc333.incomplete")                 # hf 1.x naming
        self.assertEqual(downloads.inspect(self.model, active_dirs=set()).status, "complete")

    def test_unparseable_temp_name_counts_as_pending(self):
        """When in doubt, say incomplete: a false alarm costs a re-run, a false
        all-clear costs a model that will not load."""
        self.temp("weird.incomplete")
        self.assertEqual(downloads.inspect(self.model, active_dirs=set()).status, "interrupted")

    def test_mixed_stale_and_live(self):
        self.done("a.safetensors", "aaa111")
        self.temp("h1=.aaa111.x1.incomplete")
        self.temp("h2=.bbb222.x2.incomplete")
        st = downloads.inspect(self.model, active_dirs=set())
        self.assertEqual((len(st.incomplete), len(st.stale)), (1, 1))
        self.assertTrue(st.partial)


class ScanStoreTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.store = Path(self.tmp.name)

    def test_finds_models_at_publisher_depth(self):
        make_scratch(self.store / "pub" / "a", completed=1)
        make_scratch(self.store / "pub" / "b", incomplete=1)
        states = {s.root.name: s for s in downloads.scan_store(self.store)}
        self.assertEqual(set(states), {"a", "b"})
        self.assertEqual(states["b"].status, "interrupted")

    def test_finds_a_download_that_died_before_any_file_landed(self):
        """Only scratch exists, so the model scanner would not see it at all.
        This is exactly the case worth reporting."""
        make_scratch(self.store / "pub" / "ghost", incomplete=1)
        states = downloads.scan_store(self.store)
        self.assertEqual(len(states), 1)
        self.assertTrue(states[0].partial)

    def test_ignores_directories_without_bookkeeping(self):
        (self.store / "pub" / "plain").mkdir(parents=True)
        (self.store / "pub" / "plain" / "model.gguf").write_bytes(b"x")
        self.assertEqual(downloads.scan_store(self.store), [])

    def test_missing_store_is_empty_not_an_error(self):
        self.assertEqual(downloads.scan_store(self.store / "nope"), [])


class ActiveDirsTest(unittest.TestCase):
    def test_parses_local_dir_from_a_process_line(self):
        line = "/path/python /path/hf download org/model --local-dir /store/org/model\n"
        with mock.patch.object(downloads.subprocess, "run",
                               return_value=mock.Mock(stdout=line)):
            self.assertEqual(downloads.active_download_dirs(), {"/store/org/model"})

    def test_non_string_stdout_degrades_to_empty(self):
        """A patched-out subprocess must not take the caller down."""
        with mock.patch.object(downloads.subprocess, "run",
                               return_value=mock.Mock(stdout=mock.Mock())):
            self.assertEqual(downloads.active_download_dirs(), set())

    def test_ps_failure_degrades_to_empty(self):
        with mock.patch.object(downloads.subprocess, "run", side_effect=OSError):
            self.assertEqual(downloads.active_download_dirs(), set())


class DurationTest(unittest.TestCase):
    def test_scales(self):
        self.assertEqual(downloads.human_duration(45), "45s")
        self.assertEqual(downloads.human_duration(120), "2m")
        self.assertEqual(downloads.human_duration(3700), "1h01m")
        self.assertEqual(downloads.human_duration(90000), "1d01h")


class PickerPureTest(unittest.TestCase):
    """The key handling is exercised through a pty by hand; these cover the
    pure helpers, where the off-by-one bugs actually lived."""

    def test_viewport_shows_everything_when_it_fits(self):
        self.assertEqual(picker._viewport(0, 5, 10), (0, 5))

    def test_viewport_scrolls_and_stays_in_range(self):
        for cursor in range(30):
            top, bottom = picker._viewport(cursor, 30, 10)
            self.assertGreaterEqual(top, 0)
            self.assertLessEqual(bottom, 30)
            self.assertEqual(bottom - top, 10)
            self.assertTrue(top <= cursor < bottom, f"cursor {cursor} outside viewport")

    def test_terminal_size_has_a_floor(self):
        """A pty with no winsize reports 0 columns; truncating to columns-1
        then eats a character off every row."""
        with mock.patch.object(picker.shutil, "get_terminal_size",
                               return_value=os.terminal_size((0, 0))):
            size = picker._size()
        self.assertGreaterEqual(size.columns, 40)
        self.assertGreaterEqual(size.lines, 10)

    def test_not_usable_without_a_tty(self):
        with mock.patch.object(picker.sys, "stdin", mock.Mock(isatty=lambda: False)):
            self.assertFalse(picker.usable())
