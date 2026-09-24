"""llama.cpp adapter — native, loads a GGUF by path.

`llama-server -m <path>` takes any path, and the canonical store already keeps
GGUFs in a readable `<store>/<pub>/<model>/file.gguf` tree, so there's nothing to
project. `modelctl resolve <repo>` prints the path to feed `-m`.
"""

from __future__ import annotations

import shutil

from ..cache import Repo
from .base import Action, Adapter


class LlamaCppAdapter(Adapter):
    name = "llamacpp"

    def accepts(self, repo: Repo) -> bool:
        return repo.fmt == "gguf"

    def sync(self, repos: list[Repo], *, dry_run: bool = False, **options) -> list[Action]:
        return [
            # f.path is resolved, so in the HF cache it is blobs/<sha>: show the
            # repo-relative name instead or the target column is a bare hash.
            Action(self.name, "native", f.basename, f"llama-server -m {f.path}")
            for r in repos if self.accepts(r) for f in r.gguf_files
        ]

    def doctor(self) -> list[str]:
        found = "found" if shutil.which("llama-server") else "not on PATH"
        return [f"llama-server {found}. Load by path: `llama-server -m $(modelctl resolve <repo>)`"]
