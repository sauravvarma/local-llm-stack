"""Read-only registry over one or more model *stores* (the source of truth).

Two store layouts are supported, no dependencies (stock python3):

  flat      <root>/<publisher>/<model>/<files…>     (what `hf download --local-dir`
            <root>/<model>/<files…>                   and LM Studio produce — GUI-friendly)
  hf-cache  <root>/models--<org>--<name>/snapshots/<commit>/…  (HF hub cache layout)

Every store yields the same `Repo`/`ModelFile` view, classified into one of four
formats so adapters know which tools can consume each model:

  gguf         -> llama.cpp / LM Studio / ollama
  mlx          -> MLX-quantized safetensors; mlx_lm / LM Studio only
  safetensors  -> full-precision / GPTQ / AWQ; vLLM / transformers / mlx_lm
  other        -> unclassified
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

GGUF_EXTS = {".gguf"}
WEIGHT_EXTS = {".safetensors", ".bin", ".pt", ".pth", ".npz"}
SKIP_NAMES = {".DS_Store"}
# in-progress / scratch files an in-flight download leaves behind
SKIP_SUFFIXES = {".part", ".incomplete", ".tmp", ".lock"}


@dataclass
class ModelFile:
    repo_id: str
    filename: str  # path relative to the model root
    path: Path     # absolute, symlinks resolved
    size: int

    @property
    def basename(self) -> str:
        return Path(self.filename).name

    @property
    def is_gguf(self) -> bool:
        return Path(self.filename).suffix.lower() in GGUF_EXTS


@dataclass
class Repo:
    repo_id: str          # "publisher/model" (or bare "model")
    fmt: str              # gguf | mlx | safetensors | other
    store: str            # which store it came from (label)
    root: Path            # the model's directory (what tools load)
    files: list[ModelFile] = field(default_factory=list)
    revision: str = ""

    @property
    def publisher(self) -> str:
        return self.repo_id.split("/")[0]

    @property
    def model(self) -> str:
        return self.repo_id.split("/", 1)[1] if "/" in self.repo_id else self.repo_id

    @property
    def size(self) -> int:
        return sum(f.size for f in self.files)

    @property
    def gguf_files(self) -> list[ModelFile]:
        return [f for f in self.files if f.is_gguf]


# ---------------------------------------------------------------- classification


def _read_config(root: Path) -> dict | None:
    cfg = root / "config.json"
    if cfg.is_file():
        try:
            return json.loads(cfg.read_text())
        except (ValueError, OSError):
            return {}
    return None


def classify(repo_id: str, root: Path, files: list[ModelFile]) -> str:
    if any(f.is_gguf for f in files):
        return "gguf"
    has_weights = any(Path(f.filename).suffix.lower() in WEIGHT_EXTS for f in files)
    cfg = _read_config(root)
    if cfg is not None:
        # MLX writes a top-level "quantization" block (HF/transformers uses
        # "quantization_config" instead, so a bare "quantization" key is the tell).
        if isinstance(cfg.get("quantization"), dict):
            return "mlx"
    if "mlx" in repo_id.lower():
        return "mlx"
    if has_weights:
        return "safetensors"
    return "other"


# ---------------------------------------------------------------- store scanners


def _collect_files(repo_id: str, root: Path) -> list[ModelFile]:
    out: list[ModelFile] = []
    for p in sorted(root.rglob("*")):
        if p.is_dir() or p.name in SKIP_NAMES or p.name.startswith("."):
            continue
        if p.suffix.lower() in SKIP_SUFFIXES:
            continue
        real = p.resolve()
        try:
            size = real.stat().st_size
        except OSError:
            size = 0
        out.append(ModelFile(repo_id, str(p.relative_to(root)), real, size))
    return out


def _has_model_files(d: Path, *, recursive: bool = False) -> bool:
    for p in (d.rglob("*") if recursive else d.iterdir()):
        if (
            p.is_file()
            and not p.name.startswith(".")
            and p.suffix.lower() not in SKIP_SUFFIXES
            and (p.suffix.lower() in GGUF_EXTS or p.suffix.lower() in WEIGHT_EXTS or p.name == "config.json")
        ):
            return True
    return False


def scan_flat(root: Path, label: str = "flat") -> list[Repo]:
    """Scan a `<publisher>/<model>/` (or bare `<model>/`) store. A top-level dir
    with model files directly inside is a bare model; otherwise each subdir that
    contains model files anywhere beneath it (e.g. quant subfolders) is a model.
    repo_id is the model's path under root."""
    repos: list[Repo] = []
    if not root.is_dir():
        return repos
    for pub in sorted(root.iterdir()):
        if not pub.is_dir() or pub.name.startswith("."):
            continue
        if _has_model_files(pub):  # bare <model>/ at the top level
            _add_flat(repos, pub.name, pub, label)
            continue
        for model in sorted(pub.iterdir()):  # <publisher>/<model>/
            if model.is_dir() and not model.name.startswith(".") and _has_model_files(model, recursive=True):
                _add_flat(repos, f"{pub.name}/{model.name}", model, label)
    return repos


def _add_flat(repos: list[Repo], repo_id: str, root: Path, label: str) -> None:
    files = _collect_files(repo_id, root)
    if files:
        repos.append(Repo(repo_id, classify(repo_id, root, files), label, root, files))


def _current_revision(repo_dir: Path) -> str | None:
    main = repo_dir / "refs" / "main"
    if main.is_file() and main.read_text().strip():
        return main.read_text().strip()
    snaps = repo_dir / "snapshots"
    if snaps.is_dir():
        subs = [p for p in snaps.iterdir() if p.is_dir()]
        if subs:
            subs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            return subs[0].name
    return None


def scan_hf_cache(hub: Path, label: str = "hf-cache") -> list[Repo]:
    repos: list[Repo] = []
    if not hub.is_dir():
        return repos
    for repo_dir in sorted(hub.glob("models--*")):
        rev = _current_revision(repo_dir)
        if not rev:
            continue
        snap = repo_dir / "snapshots" / rev
        if not snap.is_dir():
            continue
        repo_id = repo_dir.name[len("models--"):].replace("--", "/")
        files = _collect_files(repo_id, snap)
        if files:
            repos.append(Repo(repo_id, classify(repo_id, snap, files), label, snap, files, rev))
    return repos


def hf_cache_repo_ids(hub: Path) -> list[str]:
    """All repo_ids present in the cache by directory name — including
    incomplete/refs-only repos that `scan_hf_cache` skips. Used to recover a
    publisher for a bare-named model."""
    if not hub.is_dir():
        return []
    return [d.name[len("models--"):].replace("--", "/") for d in hub.glob("models--*")]


def hub_dir() -> Path:
    if os.environ.get("HF_HUB_CACHE"):
        return Path(os.environ["HF_HUB_CACHE"]).expanduser()
    if os.environ.get("HF_HOME"):
        return Path(os.environ["HF_HOME"]).expanduser() / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def find_repo(repos: list[Repo], repo_id: str) -> Repo | None:
    for r in repos:
        if r.repo_id == repo_id:
            return r
    # fall back to a unique model-name match (publisher omitted)
    matches = [r for r in repos if r.model == repo_id]
    return matches[0] if len(matches) == 1 else None


def human_size(n: int) -> str:
    f = float(n)
    for unit in ("B", "K", "M", "G", "T"):
        if f < 1024 or unit == "T":
            return f"{f:.0f}{unit}" if unit == "B" else f"{f:.1f}{unit}"
        f /= 1024
    return f"{f:.1f}T"
