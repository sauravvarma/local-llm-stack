"""Group a repo's files into the variants a user actually chooses between.

A GGUF repo is usually one model published at a dozen-plus quantisations, and
nothing about `hf download <repo>` hints at that: unsloth/Qwen3.8-27B-GGUF is
33 files and 472 GB if you take the default. You want one quant, typically
16-30 GB of it.

Splitting the file list into variants is mostly filename archaeology, and the
traps are the files that LOOK like quants but are not:

    mmproj-F16.gguf        vision projector, pairs with ANY quant
    imatrix_unsloth.gguf   calibration data, not runnable
    MTP/mtp-...-Q4_0.gguf  a draft model for speculative decoding
    BF16/...-00001-of-00002.gguf   one variant split across files

Treating `mmproj-F16` as "the F16 quant" would be the obvious bug: pick F16 and
you would get a 900 MB projector instead of the model. So kinds are classified
before quant tags are parsed, and only `quant` variants are mutually exclusive
choices. Everything else is offered alongside them.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import PurePosixPath

GGUF_SUFFIX = ".gguf"

# Quant tags as they appear in filenames: an optional UD- (unsloth dynamic)
# prefix, then Q4_K_M / IQ2_XXS / Q8_0 style tags, or a plain float format.
_QUANT_RE = re.compile(
    r"(?P<tag>(?:UD-)?(?:I?Q\d+(?:_[A-Za-z0-9]+)*|BF16|F16|F32|FP16))(?=[.\-_]|$)",
    re.IGNORECASE,
)
_SHARD_RE = re.compile(r"-\d{5}-of-\d{5}$")


@dataclass
class Variant:
    """One thing a user can choose to download."""

    key: str
    kind: str                       # quant | projector | draft | imatrix | aux
    files: list[str] = field(default_factory=list)
    size: int = 0

    @property
    def is_choice(self) -> bool:
        """Quants are the mutually exclusive picks; the rest are add-ons."""
        return self.kind == "quant"

    @property
    def label(self) -> str:
        note = {
            "projector": "  (vision projector, pairs with any quant)",
            "draft": "  (draft model for speculative decoding)",
            "imatrix": "  (calibration data, not runnable)",
            "aux": "  (config / readme)",
        }.get(self.kind, "")
        shards = f"  [{len(self.files)} files]" if len(self.files) > 1 else ""
        return f"{self.key}{shards}{note}"


def parse_size(text: str) -> int:
    """Parse `hf`'s human sizes ("50.0G", "4.2K", "931.1M") into bytes."""
    text = (text or "").strip()
    m = re.fullmatch(r"([\d.]+)\s*([BKMGT]?)i?B?", text, re.IGNORECASE)
    if not m:
        return 0
    value = float(m.group(1))
    return int(value * (1024 ** "BKMGT".index((m.group(2) or "B").upper())))


def _stem(name: str) -> str:
    stem = name[: -len(GGUF_SUFFIX)] if name.lower().endswith(GGUF_SUFFIX) else name
    return _SHARD_RE.sub("", stem)     # collapse -00001-of-00002 shards


def _classify(path: str) -> str:
    p = PurePosixPath(path)
    low = p.name.lower()
    if not low.endswith(GGUF_SUFFIX):
        return "aux"
    if "mmproj" in low or "mproj" in low:
        return "projector"
    if "imatrix" in low:
        return "imatrix"
    # A draft/MTP model is a small companion, flagged by name or by living in
    # its own MTP/draft folder.
    if low.startswith("mtp") or "draft" in low or p.parent.name.lower() in ("mtp", "draft"):
        return "draft"
    return "quant"


def _quant_key(path: str) -> str:
    """The variant a quant file belongs to.

    A quant in its own subfolder is keyed by that folder (bartowski publishes
    `Q4_K_M/model-00001-of-00002.gguf`), otherwise by the tag in the filename.
    Falling back to the whole stem keeps unrecognised naming as its own row
    rather than silently merging distinct files."""
    p = PurePosixPath(path)
    if p.parent.name and p.parent.name != ".":
        return p.parent.name
    m = _QUANT_RE.search(_stem(p.name))
    return m.group("tag") if m else _stem(p.name)


def group(files: list[tuple[str, int]]) -> list[Variant]:
    """Group `(path, size)` pairs into variants, largest quant last.

    Ordering puts quants first (that is what the user is choosing), then the
    add-ons, and sorts quants by size so the list reads small to large."""
    variants: dict[tuple[str, str], Variant] = {}
    for path, size in files:
        kind = _classify(path)
        if kind == "quant":
            key = _quant_key(path)
        elif kind == "aux":
            key = "other files"
        else:
            key = _stem(PurePosixPath(path).name)
        slot = variants.setdefault((kind, key), Variant(key=key, kind=kind))
        slot.files.append(path)
        slot.size += size
    order = {"quant": 0, "projector": 1, "draft": 2, "imatrix": 3, "aux": 4}
    return sorted(variants.values(), key=lambda v: (order[v.kind], v.size, v.key))


def quants(variants: list[Variant]) -> list[Variant]:
    return [v for v in variants if v.is_choice]


def needs_selection(variants: list[Variant]) -> bool:
    """Only worth asking when there is a real choice to make."""
    return len(quants(variants)) > 1


def default_selection(variants: list[Variant]) -> set[int]:
    """What to preselect: nothing among the quants (the user must choose), but
    every non-quant add-on, since those are small and usually wanted."""
    return {i for i, v in enumerate(variants) if not v.is_choice and v.kind != "imatrix"}
