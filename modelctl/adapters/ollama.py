"""Ollama adapter — the outlier.

Ollama stores models in a content-addressed blob store (blobs/sha256-… +
manifests) that nothing else reads, and it will NOT run a model from an external
symlink. The only way to reuse a GGUF is to *import* it via a Modelfile, which
copies the bytes into Ollama's store. So `sync` reports these as skipped by
default; import is explicit/opt-in.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from ..cache import ModelFile, Repo
from .base import Action, Adapter


def _tag(model: str, basename: str) -> str:
    name = re.sub(r"[^a-z0-9._-]+", "-", model.lower()).strip("-")
    m = re.search(r"(Q\d[\w]*|f16|bf16|f32)", basename, re.IGNORECASE)
    return f"{name}:{m.group(1).lower() if m else 'latest'}"


class OllamaAdapter(Adapter):
    name = "ollama"

    def accepts(self, repo: Repo) -> bool:
        return repo.fmt == "gguf"

    def sync(self, repos: list[Repo], *, dry_run: bool = False, do_import: bool = False) -> list[Action]:
        actions: list[Action] = []
        for repo in repos:
            if not self.accepts(repo):
                continue
            for f in repo.gguf_files:
                name = _tag(repo.model, f.basename)
                if do_import:
                    actions.append(self.import_file(f, repo.model, name=name, dry_run=dry_run))
                else:
                    actions.append(Action(
                        self.name, "skip", name,
                        "import copies bytes; `modelctl sync --import-ollama` or `modelctl ollama-import <repo>`",
                    ))
        return actions

    def import_file(self, f: ModelFile, model: str, *, name: str | None = None, dry_run: bool = False) -> Action:
        name = name or _tag(model, f.basename)
        if shutil.which("ollama") is None:
            return Action(self.name, "error", name, "ollama not on PATH")
        if dry_run:
            return Action(self.name, "copy", name, f"would import {f.path}")
        with tempfile.NamedTemporaryFile("w", suffix=".Modelfile", delete=False) as mf:
            mf.write(f"FROM {f.path}\n")
            modelfile = mf.name
        try:
            subprocess.run(["ollama", "create", name, "-f", modelfile],
                           check=True, capture_output=True, text=True)
            return Action(self.name, "copy", name, "imported (copied into ollama store)")
        except subprocess.CalledProcessError as e:
            last = (e.stderr or e.stdout or "ollama create failed").strip().splitlines()[-1]
            return Action(self.name, "error", name, last)
        finally:
            Path(modelfile).unlink(missing_ok=True)

    def doctor(self) -> list[str]:
        found = "found" if shutil.which("ollama") else "not on PATH"
        return [
            f"ollama {found}.",
            "Cannot share via symlink — imports COPY bytes into ~/.ollama/models (set OLLAMA_MODELS to relocate).",
        ]
