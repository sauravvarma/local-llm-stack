"""Splash adapter: Inco AI's Apple-silicon engine, loads a splash package by path.

`splash serve --model <path>` takes a path, and the canonical store already
holds the package as `<store>/<pub>/<model>/` (manifest.json + target/ + draft/
+ vision/ + tokenizer/), so there's nothing to project. `modelctl resolve
<repo>` prints the path to feed `--model`.

Splash packages are also loadable inside LM Studio / Bionic via the `splash`
backend extension, which is why the LM Studio family accepts splash too; that
path is a symlink projection and lives in `lmstudio.py`.
"""

from __future__ import annotations

import shutil

from ..cache import Repo
from .base import Action, Adapter


class SplashAdapter(Adapter):
    name = "splash"

    def accepts(self, repo: Repo) -> bool:
        return repo.fmt == "splash"

    def sync(self, repos: list[Repo], *, dry_run: bool = False, **options) -> list[Action]:
        return [
            Action(self.name, "native", r.repo_id, f"splash serve --model {r.root}")
            for r in repos if self.accepts(r)
        ]

    def doctor(self) -> list[str]:
        found = "found" if shutil.which("splash") else "not on PATH (brew install incoai/tap/splash)"
        return [
            f"splash {found}. Load by path: `splash serve --model $(modelctl resolve <repo>)`",
            "Packed splash weights only. Needs Apple silicon; the engine maps the shards directly from disk.",
        ]
