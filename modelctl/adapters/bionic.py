"""Bionic adapter: LM Studio's sibling app, same engines, its own models dir.

Bionic (`ai.elementlabs.bionic`) is built from the LM Studio codebase and loads
the same runtimes (llama.cpp/GGUF + mlx-llm/MLX), so the projection is exactly
LM Studio's. What differs is *where* it looks: both apps share one LM Studio
home, but Bionic keeps its own settings under `apps/bionic/`, and its
`downloadsFolder` defaults to `<home>/models` rather than wherever LM Studio is
pointed. Reading that setting (instead of assuming the default) is what keeps
`sync` from silently populating a folder Bionic isn't reading.

Bionic also ships an ExecuTorch ASR runtime (`.pte` speech models). Those aren't
a format the store classifies or that Bionic downloads into `downloadsFolder`,
so they're out of scope here; `doctor` just says so.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import NamedTuple

from .lmstudio import LMStudioFamilyAdapter

HOME_POINTER = ".lmstudio-home-pointer"
DEFAULT_HOME = ".lmstudio"
APP_SUBDIR = ("apps", "bionic")


def lmstudio_home() -> Path:
    """Resolve the shared LM Studio home the way the apps themselves do:
    `~/.lmstudio-home-pointer` if present, else `~/.lmstudio`."""
    pointer = Path.home() / HOME_POINTER
    try:
        target = pointer.read_text().strip()
        if target:
            return Path(target).expanduser()
    except OSError:
        pass
    return Path.home() / DEFAULT_HOME


def settings_path(home: Path | None = None) -> Path:
    return (home or lmstudio_home()).joinpath(*APP_SUBDIR, "settings.json")


class BionicDir(NamedTuple):
    """Where Bionic's models live, and how we worked that out.

    `configured` records whether the path came from a real source (the env var
    or a settings file we actually read) rather than the built-in default, so
    `doctor` can say something useful without re-deriving it or sniffing
    `source` for substrings."""

    path: Path
    source: str
    configured: bool


def models_dir(home: Path | None = None) -> BionicDir:
    """Bionic's models dir plus its provenance.

    Precedence: MODELCTL_BIONIC_DIR > its settings.json `downloadsFolder` >
    `<home>/models` (the app's own default).

    Every failure degrades to the default. This runs from `Config.load()`, so
    it is on the path of EVERY modelctl command: a settings file Bionic wrote
    badly, or half-wrote while quitting, must never take the CLI down with it."""
    env = os.environ.get("MODELCTL_BIONIC_DIR")
    if env:
        return BionicDir(Path(env).expanduser(), "MODELCTL_BIONIC_DIR", True)
    home = home or lmstudio_home()
    settings = settings_path(home)
    default = home / "models"
    try:
        raw = settings.read_text()
    except OSError:
        return BionicDir(default, f"default (no Bionic settings at {settings})", False)
    try:
        data = json.loads(raw)
    except ValueError:
        return BionicDir(default, f"default (unreadable settings at {settings})", False)
    # Anything but an object means the file isn't what we think it is. Note
    # json.loads("null") returns None, which is valid JSON and NOT an error.
    if not isinstance(data, dict):
        return BionicDir(default, f"default (unexpected settings shape in {settings})", False)
    folder = data.get("downloadsFolder")
    if not isinstance(folder, str) or not folder:
        return BionicDir(default, f"default (no downloadsFolder in {settings})", True)
    return BionicDir(Path(folder).expanduser(), f"downloadsFolder in {settings}", True)


class BionicAdapter(LMStudioFamilyAdapter):
    name = "bionic"

    def __init__(self, where: BionicDir | None = None):
        self.where = where or models_dir()
        super().__init__(self.where.path)

    def doctor(self) -> list[str]:
        lines = super().doctor() + [f"resolved from: {self.where.source}"]
        if not self.where.configured:
            lines.append("Bionic settings not found. Is Bionic installed and launched once? "
                         "Set MODELCTL_BIONIC_DIR to project anyway.")
        lines.append("Serves GGUF + MLX by symlink. Splash needs the experimental splash backend, "
                     "and is mirrored with HARD links because the indexer rejects symlinked "
                     "packages; same filesystem only, and it costs no extra bytes.")
        lines.append("Its ASR runtime (ExecuTorch .pte) is managed by the app, not the store.")
        return lines
