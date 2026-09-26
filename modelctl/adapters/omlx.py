"""oMLX adapter: an MLX inference server for Apple silicon (app.omlx).

oMLX loads MLX checkpoints, so it accepts exactly what mlx_lm does: MLX-quantised
models and full-precision safetensors, but not GPTQ/AWQ, and not GGUF (it has no
GGUF loader at all) or Splash packages (no root config.json, so it never sees
them). Everything below comes from reading oMLX 0.7's own discovery code.

Where it looks. The home resolves like the app does:

    OMLX_BASE_PATH  >  ~/Library/Application Support/oMLX/base-path  >  ~/.omlx

and the model directories, in scan order:

    OMLX_MODEL_DIR (comma-separated)  >  settings.json model.model_dirs
        >  model.model_dir (deprecated)  >  <home>/models

What it finds. Each model dir is scanned exactly two levels deep: a child with a
config.json is a model, otherwise the child is an organisation folder whose
children are models. That is the store's `<publisher>/<model>` layout, and the
scan follows symlinks (plain is_dir/iterdir, no containment check), so a
directory symlink is enough, unlike Splash in LM Studio.

Two consequences shape `sync`:

  - `model_dirs` is a list, so the zero-projection setup is adding the store to
    it. If the store is already one of them, there is nothing to do. modelctl
    never edits settings.json itself: it holds the API and secret keys, and the
    app rewrites it.
  - oMLX keys a model by its FOLDER NAME alone, not publisher/model, and the
    first one found wins. Two publishers' `Foo-4bit` would shadow each other,
    so a collision is reported instead of silently linked.

oMLX discovers models at startup (or on an admin reload), not by watching the
directory, so new links need a restart to appear.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import NamedTuple

from .. import downloads
from ..cache import SERVER_ONLY_QUANTS, Repo
from .base import Action, Adapter, ensure_symlink, prune_links


def _bootstrap_file() -> Path:
    # Computed per call, not at import, so tests that move Path.home() work.
    return Path.home() / "Library" / "Application Support" / "oMLX" / "base-path"


def omlx_home() -> Path:
    """Resolve oMLX's home exactly as oMLX's settings.resolve_default_base_path
    does, so modelctl and the app agree on where settings.json lives."""
    env = os.environ.get("OMLX_BASE_PATH")
    if env:
        return Path(env).expanduser()
    try:
        raw = _bootstrap_file().read_text(encoding="utf-8").strip()
    except OSError:
        raw = ""
    if raw:
        return Path(raw).expanduser()
    return Path.home() / ".omlx"


class OmlxDirs(NamedTuple):
    """oMLX's model directories in its own scan order, plus provenance.

    The first entry is where modelctl projects. `configured` is False only when
    we fell back to the built-in default because nothing said otherwise."""

    dirs: list[Path]
    source: str
    configured: bool


def model_dirs(home: Path | None = None) -> OmlxDirs:
    """Where oMLX looks for models.

    Called from Config.load(), so it is on the path of every modelctl command:
    every failure degrades to the default rather than raising."""
    override = os.environ.get("MODELCTL_OMLX_DIR")
    if override:
        return OmlxDirs([Path(override).expanduser()], "MODELCTL_OMLX_DIR", True)
    env = os.environ.get("OMLX_MODEL_DIR")
    if env:
        dirs = [Path(d.strip()).expanduser() for d in env.split(",") if d.strip()]
        if dirs:
            return OmlxDirs(dirs, "OMLX_MODEL_DIR", True)
    home = home or omlx_home()
    settings = home / "settings.json"
    default = OmlxDirs([home / "models"], f"default (no settings at {settings})", False)
    try:
        data = json.loads(settings.read_text())
    except OSError:
        return default
    except ValueError:
        return default._replace(source=f"default (unreadable settings at {settings})")
    model = data.get("model") if isinstance(data, dict) else None
    if not isinstance(model, dict):
        return default._replace(source=f"default (no model section in {settings})",
                                configured=isinstance(data, dict))
    listed = model.get("model_dirs")
    if isinstance(listed, list):
        dirs = [Path(d).expanduser() for d in listed if isinstance(d, str) and d.strip()]
        if dirs:
            return OmlxDirs(dirs, f"model.model_dirs in {settings}", True)
    single = model.get("model_dir")
    if isinstance(single, str) and single.strip():
        return OmlxDirs([Path(single).expanduser()], f"model.model_dir in {settings}", True)
    return OmlxDirs([home / "models"], f"default (no model dirs set in {settings})", True)


def _discoverable_from(root: Path, model_dir: Path) -> bool:
    """Would oMLX's two-level scan of `model_dir` reach the model at `root`?

    Containment is not enough: a store nested deeper than <dir>/<org>/<model>
    is simply never visited."""
    try:
        r, d = root.resolve(), model_dir.resolve()
    except OSError:
        return False
    return r.parent == d or r.parent.parent == d


class OmlxAdapter(Adapter):
    name = "omlx"

    def __init__(self, where: OmlxDirs | None = None):
        self.where = where or model_dirs()

    @property
    def primary(self) -> Path:
        return self.where.dirs[0]

    def accepts(self, repo: Repo) -> bool:
        if repo.fmt == "mlx":
            return True
        return repo.fmt == "safetensors" and repo.quant_method not in SERVER_ONLY_QUANTS

    def sync(self, repos: list[Repo], *, dry_run: bool = False, **options) -> list[Action]:
        actions: list[Action] = []
        claimed: dict[str, str] = {}     # folder name -> repo_id that owns it in oMLX
        desired: set[Path] = set()       # every link this sync wants to exist
        changed = False
        for repo in repos:
            if not self.accepts(repo):
                continue
            target = self.primary / repo.publisher / repo.model
            owner = claimed.get(repo.model)
            if owner is not None:
                actions.append(Action(
                    self.name, "skip", str(target),
                    f"oMLX names models by folder alone and '{repo.model}' is already "
                    f"{owner}; it would be shadowed. Rename one to serve both."))
                continue
            claimed[repo.model] = repo.repo_id
            if any(_discoverable_from(repo.root, d) for d in self.where.dirs):
                actions.append(Action(self.name, "skip", repo.repo_id,
                                      "already visible: the store is one of oMLX's model dirs"))
                continue
            # oMLX's own completeness check only covers shards named by an index
            # that has already landed, so it can list a half-downloaded model
            # and fail at load time. Hold partials back here instead. Only
            # `partial` matters, so skip the process scan.
            if downloads.inspect(repo.root, active_dirs=set()).partial:
                actions.append(Action(
                    self.name, "skip", str(target),
                    "incomplete download; not projected until it finishes "
                    "(see `modelctl status`)"))
                continue
            desired.add(target)
            action = ensure_symlink(target, repo.root, self.name, dry_run=dry_run)
            changed = changed or action.op in ("link", "relink")
            actions.append(action)
        if options.get("prune"):
            for d in self.where.dirs:
                pruned = prune_links(d, desired, sources=list(options.get("sources", [])),
                                     protected=list(options.get("protected", [])),
                                     adapter=self.name, dry_run=dry_run)
                changed = changed or bool(pruned)
                actions += pruned
        if changed:
            actions.append(Action(
                self.name, "native", "oMLX server",
                "discovers models at startup: `omlx restart`, or restart it from the menu bar"))
        return actions

    def doctor(self) -> list[str]:
        dirs = ", ".join(str(d) for d in self.where.dirs)
        lines = [f"model dirs (scan order): {dirs}",
                 f"resolved from: {self.where.source}",
                 f"projects into: {self.primary}"]
        if not self.where.configured:
            lines.append("oMLX settings not found. Is oMLX installed and launched once? "
                         "Set MODELCTL_OMLX_DIR to project anyway.")
        found = "found" if shutil.which("omlx") else "not on PATH (brew install omlx, or the app)"
        lines.append(f"omlx {found}. Serves MLX + full-precision safetensors; no GGUF, no Splash.")
        lines.append("Zero-projection option: add the store to model.model_dirs in oMLX's "
                     "settings, and sync becomes a no-op for it.")
        return lines
