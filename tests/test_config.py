from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from modelctl.config import DEFAULT_STORE, Config
from tests.helpers import EnvTestCase, make_flat_store, make_hf_cache


class ConfigTest(EnvTestCase):
    def setUp(self):
        super().setUp()
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)

    def test_default_store_is_repo_models(self):
        os.environ.pop("MODELCTL_STORE", None)
        cfg = Config.load()
        self.assertEqual(cfg.store, DEFAULT_STORE)
        self.assertEqual(cfg.store.name, "models")

    def test_store_env_override(self):
        os.environ["MODELCTL_STORE"] = str(self.base / "s1")
        self.assertEqual(Config.load().store, self.base / "s1")

    def test_multiple_stores_colon_separated(self):
        os.environ["MODELCTL_STORE"] = f"{self.base/'s1'}:{self.base/'s2'}"
        cfg = Config.load()
        self.assertEqual(cfg.store, self.base / "s1")
        self.assertEqual(cfg.extra_stores, [self.base / "s2"])

    def test_lmstudio_defaults_to_store(self):
        os.environ["MODELCTL_STORE"] = str(self.base / "s1")
        self.assertEqual(Config.load().lmstudio_dir, self.base / "s1")

    def test_lmstudio_override(self):
        os.environ["MODELCTL_STORE"] = str(self.base / "s1")
        os.environ["MODELCTL_LMSTUDIO_DIR"] = str(self.base / "lm")
        self.assertEqual(Config.load().lmstudio_dir, self.base / "lm")

    def test_scan_hub_disabled(self):
        store = make_flat_store(self.base / "store")
        hub = make_hf_cache(self.base / "hub")
        os.environ["MODELCTL_STORE"] = str(store)
        os.environ["HF_HUB_CACHE"] = str(hub)
        os.environ["MODELCTL_SCAN_HUB"] = "0"
        ids = {r.repo_id for r in Config.load().scan()}
        self.assertNotIn("org/demo-GGUF", ids)

    def test_scan_includes_hub_when_enabled(self):
        store = make_flat_store(self.base / "store")
        hub = make_hf_cache(self.base / "hub")
        os.environ["MODELCTL_STORE"] = str(store)
        os.environ["HF_HUB_CACHE"] = str(hub)
        os.environ["MODELCTL_SCAN_HUB"] = "1"
        repos = Config.load().scan()
        ids = {r.repo_id for r in repos}
        self.assertIn("org/demo-GGUF", ids)
        self.assertIn("Qwen/Qwen3-7B", ids)

    def test_dedup_first_store_wins(self):
        s1 = make_flat_store(self.base / "s1")
        # second store also has a repo with the same id but different content
        s2 = self.base / "s2"
        (s2 / "Qwen" / "Qwen3-7B").mkdir(parents=True)
        (s2 / "Qwen" / "Qwen3-7B" / "model.safetensors").write_bytes(b"q")
        (s2 / "Qwen" / "Qwen3-7B" / "config.json").write_text('{"model_type":"qwen3"}')
        os.environ["MODELCTL_STORE"] = f"{s1}:{s2}"
        repos = [r for r in Config.load().scan() if r.repo_id == "Qwen/Qwen3-7B"]
        self.assertEqual(len(repos), 1)
        self.assertEqual(repos[0].store, "store")  # primary label, not "store:s2"


if __name__ == "__main__":
    unittest.main()
