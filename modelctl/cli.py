from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

from . import __version__
from .adapters import build_adapters
from .adapters.ollama import OllamaAdapter
from .cache import ModelFile, Repo, find_repo, hf_cache_repo_ids, human_size
from .config import Config


def cmd_list(cfg: Config, args) -> int:
    repos = cfg.scan()
    if not repos:
        print(f"No models found in {cfg.store} (or HF cache).")
        return 0
    print(f"Primary store: {cfg.store}\n")
    for r in repos:
        print(f"{r.repo_id}  [{r.fmt}]  {human_size(r.size)}  ({r.store})")
        if args.files:
            for f in r.files:
                print(f"     {f.filename}  ({human_size(f.size)})")
    return 0


def cmd_resolve(cfg: Config, args) -> int:
    repo = find_repo(cfg.scan(), args.repo)
    if not repo:
        print(f"not found: {args.repo}  (try: modelctl download {args.repo})", file=sys.stderr)
        return 1
    if args.file:
        for f in repo.files:
            if f.basename == args.file or f.filename == args.file:
                print(f.path)
                return 0
        print(f"file not found in {args.repo}: {args.file}", file=sys.stderr)
        return 1
    if repo.fmt == "gguf":
        ggufs = repo.gguf_files
        if len(ggufs) == 1:
            print(ggufs[0].path)
        else:
            for f in ggufs:
                print(f"{f.basename}\t{f.path}")
    else:  # mlx / safetensors: tools load the directory
        print(repo.root)
    return 0


def _selected(cfg: Config, names):
    adapters = build_adapters(cfg)
    if not names:
        return adapters
    bad = [n for n in names if n not in adapters]
    if bad:
        print(f"unknown adapter(s): {', '.join(bad)}; have: {', '.join(adapters)}", file=sys.stderr)
        sys.exit(2)
    return {n: adapters[n] for n in names}


def cmd_sync(cfg: Config, args) -> int:
    repos = cfg.scan()
    adapters = _selected(cfg, args.adapter)
    print(f"{'DRY RUN — ' if args.dry_run else ''}syncing {len(repos)} model(s)\n")
    total = 0
    for name, adapter in adapters.items():
        if name == "ollama":
            actions = adapter.sync(repos, dry_run=args.dry_run, do_import=args.import_ollama)
        else:
            actions = adapter.sync(repos, dry_run=args.dry_run)
        for a in actions:
            print(a)
            total += 1
    if total == 0:
        print("  (nothing to project)")
    return 0


def cmd_download(cfg: Config, args) -> int:
    if shutil.which("hf") is None:
        print("`hf` CLI not found (install: uv tool install huggingface-hub)", file=sys.stderr)
        return 1
    pub, _, model = args.repo.partition("/")
    target = cfg.store / pub / model if model else cfg.store / pub
    cmd = ["hf", "download", args.repo] + args.files + ["--local-dir", str(target)]
    print("+", " ".join(cmd))
    rc = subprocess.run(cmd).returncode
    if rc != 0:
        return rc
    if not args.no_sync:
        print()
        return cmd_sync(cfg, argparse.Namespace(adapter=None, dry_run=False, import_ollama=False))
    return 0


def cmd_ollama_import(cfg: Config, args) -> int:
    repo = find_repo(cfg.scan(), args.repo)
    if not repo or not repo.gguf_files:
        print(f"no GGUF found for {args.repo}", file=sys.stderr)
        return 1
    files = repo.gguf_files
    if args.file:
        files = [f for f in files if f.basename == args.file]
        if not files:
            print(f"file not found: {args.file}", file=sys.stderr)
            return 1
    ad = OllamaAdapter()
    for f in files:
        print(ad.import_file(f, repo.model, name=args.name, dry_run=args.dry_run))
    return 0


def _recover_publisher(cfg: Config, model: str) -> str | None:
    """Find a unique publisher for a bare model name, via the HF cache refs."""
    cands = {rid.split("/", 1)[0] for rid in hf_cache_repo_ids(cfg.hub)
             if "/" in rid and rid.split("/", 1)[1] == model}
    return cands.pop() if len(cands) == 1 else None


def cmd_adopt(cfg: Config, args) -> int:
    """Normalize a store model into canonical <store>/<publisher>/<model> layout
    so path-native tools (LM Studio) index it."""
    repo = find_repo(cfg.scan(), args.repo)
    if not repo:
        print(f"not found in store: {args.repo}", file=sys.stderr)
        return 1
    if "/" in repo.repo_id and not args.publisher:
        print(f"{repo.repo_id} is already in publisher/model layout; nothing to do.")
        return 0
    publisher = args.publisher or _recover_publisher(cfg, repo.model)
    if not publisher:
        print(f"could not determine publisher for '{repo.model}' "
              f"(not found in HF cache). Pass --publisher.", file=sys.stderr)
        return 1
    target = cfg.store / publisher / repo.model
    if target.exists():
        print(f"target already exists: {target}", file=sys.stderr)
        return 1
    verb = "link" if args.link else "move"
    print(f"{'DRY RUN — ' if args.dry_run else ''}{verb}: {repo.root}  ->  {target}")
    if args.dry_run:
        return 0
    target.parent.mkdir(parents=True, exist_ok=True)
    if args.link:
        target.symlink_to(repo.root)
    else:
        repo.root.rename(target)  # same filesystem: atomic, no copy
    print(f"adopted as {publisher}/{repo.model}")
    return 0


# ------------------------------------------------------------------------ rm

def _check_removable(store: Path, path: Path, *, confine: Path | None = None) -> str | None:
    """Paranoid guard run before anything is deleted. `path` must be absolute,
    must not be the filesystem root or the store root, must sit lexically inside
    the primary store (so `..` or an absolute arg can't escape), and, unless it
    is itself a symlink (which is only unlinked, never followed), must still be
    inside the store after resolving. `confine` additionally pins individual
    files inside their own model dir. Returns an error message, or None if OK."""
    p = Path(os.path.normpath(str(path)))
    store = Path(os.path.normpath(str(store.expanduser())))
    if not p.is_absolute() or str(p) in ("", p.anchor):
        return f"refusing to remove unsafe path: {path}"
    if p == store or store not in p.parents:
        return f"refusing: {p} is not inside the primary store {store}"
    if confine is not None:
        confine = Path(os.path.normpath(str(confine)))
        if p == confine or confine not in p.parents:
            return f"refusing: {p} is not inside {confine}"
    if not p.is_symlink():
        try:
            real, store_real = p.resolve(), store.resolve()
        except OSError:
            return f"refusing: cannot resolve {p}"
        if real == store_real or store_real not in real.parents:
            return f"refusing: {p} resolves outside the store ({real})"
    return None


def _prune_empty_dirs(start: Path, stop: Path) -> None:
    """Drop directories left empty under `stop` (exclusive) after a removal."""
    p = start
    while p != stop and stop in p.parents and p.is_dir() and not p.is_symlink():
        try:
            p.rmdir()
        except OSError:
            return
        p = p.parent


def _pick_files(repo: Repo, names: list[str]) -> tuple[list[ModelFile], list[str]]:
    """Resolve file arguments (basename or store-relative path) to ModelFiles."""
    chosen: dict[str, ModelFile] = {}
    missing: list[str] = []
    for n in names:
        hits = [f for f in repo.files if f.basename == n or f.filename == n]
        if not hits:
            missing.append(n)
        for f in hits:
            chosen[f.filename] = f
    return [f for f in repo.files if f.filename in chosen], missing


def _confirm(question: str) -> bool:
    try:
        return input(f"{question} [y/N] ").strip().lower() in ("y", "yes")
    except EOFError:
        return False


def cmd_rm(cfg: Config, args) -> int:
    """Delete a model (or specific files) from the primary store, taking the
    projections down with it. Read-only sources (HF cache, extra stores) are
    refused rather than touched."""
    repo = find_repo(cfg.scan(), args.repo)
    if not repo:
        print(f"not found: {args.repo}  (try: modelctl list)", file=sys.stderr)
        return 1
    if repo.store == "hf-cache":
        print(f"{repo.repo_id} lives only in the HF hub cache ({cfg.hub}), which modelctl "
              f"scans read-only and never deletes from.\n"
              f"Remove it with `hf cache delete` (or delete the models--* dir under "
              f"{cfg.hub} yourself).", file=sys.stderr)
        return 1
    if repo.store != "store":
        print(f"{repo.repo_id} lives in a secondary store ({repo.root}), which is scanned "
              f"read-only; `modelctl rm` only removes from the primary store {cfg.store}.",
              file=sys.stderr)
        return 1

    files = repo.files
    if args.files:
        files, missing = _pick_files(repo, args.files)
        if missing:
            print(f"file(s) not in {repo.repo_id}: {', '.join(missing)}", file=sys.stderr)
            return 1
    whole = len(files) == len(repo.files)

    targets = [repo.root] if whole else [repo.root / f.filename for f in files]
    for t in targets:
        err = _check_removable(cfg.store, t, confine=None if whole else repo.root)
        if err:
            print(err, file=sys.stderr)
            return 1

    # A symlinked model dir (`modelctl adopt --link`) only costs the link itself.
    linked = whole and repo.root.is_symlink()
    reclaim = 0 if linked else sum(f.size for f in files)

    prefix = "DRY RUN — " if args.dry_run else ""
    scope = "all files" if whole else f"{len(files)} file{'' if len(files) == 1 else 's'}"
    print(f"{prefix}remove {repo.repo_id}  [{repo.fmt}]  {scope}  {human_size(reclaim)}")
    for t in targets:
        print(f"  x {t}{'  (symlink only; bytes stay at the target)' if linked else ''}")

    # Undo every tool's projection first, so nothing is left pointing at bytes
    # that are about to disappear. Planned (dry) here; executed after confirming.
    adapters = build_adapters(cfg)
    plan = [a for ad in adapters.values() for a in ad.remove(repo, files, dry_run=True)]
    for a in plan:
        print(a)

    if args.dry_run:
        return 0
    if not args.yes and not _confirm(f"Delete {repo.repo_id} from {cfg.store}?"):
        print("aborted.")
        return 1

    for ad in adapters.values():
        for a in ad.remove(repo, files, dry_run=False):
            if a.op == "error":
                print(a)

    for t in targets:
        if t.is_symlink() or t.is_file():
            t.unlink()
        else:
            shutil.rmtree(t)
        _prune_empty_dirs(t.parent, cfg.store)

    print(f"removed {repo.repo_id}; {human_size(reclaim)} reclaimed")
    return 0


def cmd_doctor(cfg: Config, args) -> int:
    print("Stores (scanned in order, first match wins):")
    print(f"  - {cfg.store}  (primary / download target){'' if cfg.store.is_dir() else '  [missing]'}")
    for p in cfg.extra_stores:
        print(f"  - {p}  (extra)")
    if cfg.scan_hub:
        print(f"  - {cfg.hub}  (HF cache){'' if cfg.hub.is_dir() else '  [missing]'}")
    print(f"\nModels discovered: {len(cfg.scan())}\n")
    for name, adapter in build_adapters(cfg).items():
        print(f"[{name}]")
        for line in adapter.doctor():
            print(f"  - {line}")
        print()
    return 0


def cmd_env(cfg: Config, args) -> int:
    print("# Point HF-aware tools (vLLM, transformers, mlx_lm) at the shared cache:")
    print(f"export HF_HOME={cfg.hub.parent}")
    print(f"# Primary model store (download target): {cfg.store}")
    print(f"export MODELCTL_STORE={cfg.store}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="modelctl", description="One model store, projected into every tool.")
    p.add_argument("--version", action="version", version=f"modelctl {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("list", help="list models across all stores")
    sp.add_argument("-f", "--files", action="store_true", help="also list files")
    sp.set_defaults(func=cmd_list)

    sp = sub.add_parser("resolve", help="print the path a tool should load")
    sp.add_argument("repo")
    sp.add_argument("file", nargs="?")
    sp.set_defaults(func=cmd_resolve)

    sp = sub.add_parser("sync", help="project the store into each tool's view")
    sp.add_argument("-a", "--adapter", action="append", help="limit to adapter(s); repeatable")
    sp.add_argument("-n", "--dry-run", action="store_true")
    sp.add_argument("--import-ollama", action="store_true", help="also import GGUFs into ollama (COPIES bytes)")
    sp.set_defaults(func=cmd_sync)

    sp = sub.add_parser("download", help="hf download into the primary store, then sync")
    sp.add_argument("repo")
    sp.add_argument("files", nargs="*", help="specific files (e.g. one .gguf)")
    sp.add_argument("--no-sync", action="store_true")
    sp.set_defaults(func=cmd_download)

    sp = sub.add_parser("ollama-import", help="import a GGUF into ollama (copies)")
    sp.add_argument("repo")
    sp.add_argument("file", nargs="?")
    sp.add_argument("--name", help="ollama model name:tag")
    sp.add_argument("-n", "--dry-run", action="store_true")
    sp.set_defaults(func=cmd_ollama_import)

    sp = sub.add_parser("adopt", help="normalize a store model into <publisher>/<model> layout")
    sp.add_argument("repo", help="current repo id / bare model name in the store")
    sp.add_argument("--publisher", help="publisher to file it under (else recovered from HF cache)")
    sp.add_argument("--link", action="store_true", help="symlink instead of moving (non-destructive)")
    sp.add_argument("-n", "--dry-run", action="store_true")
    sp.set_defaults(func=cmd_adopt)

    sp = sub.add_parser("rm", aliases=["remove"], help="delete a model from the primary store")
    sp.add_argument("repo")
    sp.add_argument("files", nargs="*", help="specific files (else the whole model dir)")
    sp.add_argument("-n", "--dry-run", action="store_true", help="print what would go; delete nothing")
    sp.add_argument("-y", "--yes", action="store_true", help="skip the confirmation prompt")
    sp.set_defaults(func=cmd_rm)

    sp = sub.add_parser("doctor", help="show stores + per-tool checks")
    sp.set_defaults(func=cmd_doctor)

    sp = sub.add_parser("env", help="print shell exports for a shared store")
    sp.set_defaults(func=cmd_env)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(Config.load(), args)
