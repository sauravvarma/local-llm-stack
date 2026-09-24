from __future__ import annotations

import unittest

from modelctl.quants import (
    Variant, default_selection, group, needs_selection, parse_size, quants,
)

# A real multi-quant GGUF repo, trimmed: unsloth/Qwen3.8-27B-GGUF.
REAL = [
    (".gitattributes", 4300),
    ("BF16/Qwen3.8-27B-BF16-00001-of-00002.gguf", 50_000_000_000),
    ("BF16/Qwen3.8-27B-BF16-00002-of-00002.gguf", 4_700_000_000),
    ("MTP/mtp-Qwen3.8-27B-Q4_0.gguf", 1_400_000_000),
    ("Qwen3.8-27B-Q4_0.gguf", 16_100_000_000),
    ("Qwen3.8-27B-Q8_0.gguf", 29_000_000_000),
    ("Qwen3.8-27B-UD-Q4_K_M.gguf", 16_500_000_000),
    ("Qwen3.8-27B-UD-IQ2_XXS.gguf", 7_300_000_000),
    ("README.md", 7500),
    ("config.json", 3800),
    ("imatrix_unsloth.gguf", 13_600_000),
    ("mmproj-BF16.gguf", 931_000_000),
    ("mmproj-F16.gguf", 927_000_000),
]


def by_key(variants):
    return {v.key: v for v in variants}


class ParseSizeTest(unittest.TestCase):
    def test_units(self):
        self.assertEqual(parse_size("1K"), 1024)
        self.assertEqual(parse_size("4.2K"), int(4.2 * 1024))
        self.assertEqual(parse_size("50.0G"), int(50 * 1024 ** 3))
        self.assertEqual(parse_size("512B"), 512)

    def test_garbage_is_zero_not_an_error(self):
        for bad in ("", "  ", "unknown", None, "??"):
            self.assertEqual(parse_size(bad), 0)


class GroupingTest(unittest.TestCase):
    def setUp(self):
        self.v = group(REAL)
        self.keyed = by_key(self.v)

    def test_mmproj_is_a_projector_not_an_f16_quant(self):
        """The trap: mmproj-F16 looks like an F16 quant. Picking it instead of
        the model would silently fetch a 900MB projector."""
        self.assertEqual(self.keyed["mmproj-F16"].kind, "projector")
        self.assertEqual(self.keyed["mmproj-BF16"].kind, "projector")
        self.assertNotIn("F16", [v.key for v in quants(self.v)])

    def test_imatrix_is_not_runnable(self):
        self.assertEqual(self.keyed["imatrix_unsloth"].kind, "imatrix")

    def test_mtp_folder_is_a_draft_model(self):
        draft = [v for v in self.v if v.kind == "draft"]
        self.assertEqual(len(draft), 1)
        self.assertIn("mtp", draft[0].key.lower())

    def test_shards_collapse_into_one_variant(self):
        bf16 = self.keyed["BF16"]
        self.assertEqual(bf16.kind, "quant")
        self.assertEqual(len(bf16.files), 2)
        self.assertEqual(bf16.size, 54_700_000_000)

    def test_quant_tags_parsed_from_filenames(self):
        for key in ("Q4_0", "Q8_0", "UD-Q4_K_M", "UD-IQ2_XXS"):
            self.assertIn(key, self.keyed, key)
            self.assertEqual(self.keyed[key].kind, "quant")

    def test_non_gguf_collapses_into_one_aux_row(self):
        aux = [v for v in self.v if v.kind == "aux"]
        self.assertEqual(len(aux), 1)
        self.assertEqual(len(aux[0].files), 3)

    def test_quants_sorted_small_to_large(self):
        sizes = [v.size for v in quants(self.v)]
        self.assertEqual(sizes, sorted(sizes))

    def test_every_file_lands_in_exactly_one_variant(self):
        seen = [f for v in self.v for f in v.files]
        self.assertEqual(sorted(seen), sorted(p for p, _ in REAL))
        self.assertEqual(len(seen), len(set(seen)))


class SubfolderQuantTest(unittest.TestCase):
    def test_bartowski_style_quant_subfolders(self):
        v = by_key(group([
            ("Q4_K_M/model-00001-of-00002.gguf", 10),
            ("Q4_K_M/model-00002-of-00002.gguf", 10),
            ("Q8_0/model.gguf", 30),
        ]))
        self.assertEqual(set(v), {"Q4_K_M", "Q8_0"})
        self.assertEqual(len(v["Q4_K_M"].files), 2)


class SelectionTest(unittest.TestCase):
    def test_needs_selection_only_when_more_than_one_quant(self):
        self.assertTrue(needs_selection(group(REAL)))
        single = group([("model-Q4_K_M.gguf", 10), ("config.json", 1)])
        self.assertFalse(needs_selection(single))

    def test_splash_style_repo_needs_no_selection(self):
        """A non-GGUF package is one thing; never prompt for it."""
        v = group([("manifest.json", 10), ("target/layer-0.bin", 20)])
        self.assertFalse(needs_selection(v))

    def test_default_selection_preselects_addons_but_no_quant(self):
        v = group(REAL)
        picked = {v[i].kind for i in default_selection(v)}
        self.assertNotIn("quant", picked, "the user must choose a quant deliberately")
        self.assertIn("projector", picked)
        self.assertNotIn("imatrix", picked, "calibration data is not runnable")


class VariantLabelTest(unittest.TestCase):
    def test_shard_count_shown(self):
        self.assertIn("[2 files]", Variant("BF16", "quant", ["a", "b"]).label)

    def test_kind_explained_for_non_quants(self):
        self.assertIn("projector", Variant("mmproj-F16", "projector", ["a"]).label)
        self.assertEqual(Variant("Q4_K_M", "quant", ["a"]).label, "Q4_K_M")
