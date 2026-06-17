from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from modelctl.cache import (
    classify, find_repo, human_size, scan_flat, scan_hf_cache,
)
from tests.helpers import FLAT_FORMATS, make_flat_store, make_hf_cache, write, write_config


class ClassifyTest(unittest.TestCase):
    def _repo(self, files):
        # build a throwaway dir with the given filenames to classify
        d = Path(self.tmp.name)
        from modelctl.cache import ModelFile
        mfs = [ModelFile("r/x", f, d / f, 0) for f in files]
        return d, mfs

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def test_gguf_wins(self):
        from modelctl.cache import ModelFile
        files = [ModelFile("r/x", "a.gguf", self.root / "a.gguf", 0)]
        self.assertEqual(classify("r/x", self.root, files), "gguf")

    def test_mlx_by_quantization_key(self):
        write_config(self.root / "config.json", {"model_type": "gemma", "quantization": {"bits": 8, "group_size": 64}})
        from modelctl.cache import ModelFile
        files = [ModelFile("r/x", "model.safetensors", self.root / "model.safetensors", 0)]
        self.assertEqual(classify("r/x", self.root, files), "mlx")

    def test_quantization_config_is_not_mlx(self):
        # HF GPTQ/AWQ use quantization_config -> still safetensors, not mlx
        write_config(self.root / "config.json", {"model_type": "qwen3", "quantization_config": {"bits": 4}})
        from modelctl.cache import ModelFile
        files = [ModelFile("r/x", "model.safetensors", self.root / "model.safetensors", 0)]
        self.assertEqual(classify("r/x", self.root, files), "safetensors")

    def test_mlx_by_name(self):
        write_config(self.root / "config.json", {"model_type": "llama"})
        from modelctl.cache import ModelFile
        files = [ModelFile("a/Model-MLX-4bit", "model.safetensors", self.root / "model.safetensors", 0)]
        self.assertEqual(classify("a/Model-MLX-4bit", self.root, files), "mlx")

    def test_safetensors(self):
        write_config(self.root / "config.json", {"model_type": "qwen3"})
        from modelctl.cache import ModelFile
        files = [ModelFile("r/x", "model.safetensors", self.root / "model.safetensors", 0)]
        self.assertEqual(classify("r/x", self.root, files), "safetensors")

    def test_other(self):
        from modelctl.cache import ModelFile
        files = [ModelFile("r/x", "README.md", self.root / "README.md", 0)]
        self.assertEqual(classify("r/x", self.root, files), "other")


class ScanFlatTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = make_flat_store(Path(self.tmp.name))
        self.repos = {r.repo_id: r for r in scan_flat(self.root)}

    def test_finds_all_expected_repos(self):
        self.assertEqual(set(self.repos), set(FLAT_FORMATS))

    def test_formats(self):
        for rid, fmt in FLAT_FORMATS.items():
            self.assertEqual(self.repos[rid].fmt, fmt, rid)

    def test_ignores_part_files(self):
        names = [f.basename for f in self.repos["unsloth/tiny-GGUF"].files]
        self.assertIn("tiny-Q4_K_M.gguf", names)
        self.assertNotIn("downloading_tiny-Q8_0.gguf.part", names)

    def test_preserves_subfolder_path(self):
        f = self.repos["bartowski/split-GGUF"].files[0]
        self.assertEqual(f.filename, "Q4_K_M/model-00001-of-00002.gguf")

    def test_bare_dir_repo_id(self):
        self.assertIn("BareModel", self.repos)
        self.assertEqual(self.repos["BareModel"].publisher, "BareModel")

    def test_skips_hidden_and_empty(self):
        self.assertNotIn(".hidden/model", self.repos)
        for rid in self.repos:
            self.assertFalse(rid.startswith("."))

    def test_size_is_sum_of_files(self):
        r = self.repos["lmstudio-community/gemma-GGUF"]
        self.assertEqual(r.size, sum(f.size for f in r.files))
        self.assertGreater(r.size, 0)

    def test_missing_store_is_empty(self):
        self.assertEqual(scan_flat(Path(self.tmp.name) / "nope"), [])


class ScanHfCacheTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.hub = make_hf_cache(Path(self.tmp.name))
        self.repos = {r.repo_id: r for r in scan_hf_cache(self.hub)}

    def test_decodes_repo_id(self):
        self.assertIn("org/demo-GGUF", self.repos)

    def test_skips_empty_refs_only_repo(self):
        self.assertNotIn("Qwen/Empty", self.repos)

    def test_classified_and_sized(self):
        r = self.repos["org/demo-GGUF"]
        self.assertEqual(r.fmt, "gguf")
        self.assertEqual(r.files[0].size, 128)  # blob size, via resolved symlink

    def test_resolves_through_symlink(self):
        r = self.repos["org/demo-GGUF"]
        self.assertTrue(r.files[0].path.is_file())


class FindRepoAndUtilTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.repos = scan_flat(make_flat_store(Path(self.tmp.name)))

    def test_exact_match(self):
        self.assertIsNotNone(find_repo(self.repos, "Qwen/Qwen3-7B"))

    def test_unique_model_name_fallback(self):
        self.assertEqual(find_repo(self.repos, "Qwen3-7B").repo_id, "Qwen/Qwen3-7B")

    def test_unknown_is_none(self):
        self.assertIsNone(find_repo(self.repos, "does/not-exist"))

    def test_human_size(self):
        self.assertEqual(human_size(0), "0B")
        self.assertEqual(human_size(1024), "1.0K")
        self.assertEqual(human_size(3 * 1024 ** 3), "3.0G")


if __name__ == "__main__":
    unittest.main()
