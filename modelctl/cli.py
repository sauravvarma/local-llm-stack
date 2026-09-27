from __future__ import annotations

import argparse
import json
import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path

from . import __version__, downloads, picker, quants
from .adapters import build_adapters
from .adapters.ollama import OllamaAdapter
from .cache import find_repo, hf_cache_repo_ids, human_size
from .config import Config

DESCRIPTION = """\
One model store, projected into every local LLM tool.

Models are downloaded once into a single store. Tools that read a path
(llama.cpp, mlx_lm, vLLM, Splash) are pointed at it; tools that want their own
directory (LM Studio, Bionic, oMLX) get a symlink or hard-link projection, so the
bytes are never copied. `modelctl sync` makes every tool's view match the
store, and is safe to re-run.
"""

EPILOG = """\
commands by what they do:

  look at things       list, resolve, status, verify, doctor
  change things        download, sync, prune, adopt, ollama-import
  set things up        env

typical session:

  modelctl download unsloth/Qwen3.8-27B-GGUF   pick a quant, fetch it, sync
  modelctl list                                what is in the store
  modelctl status                              anything half-downloaded?
  modelctl sync -n                             preview what sync would project
  modelctl resolve <repo>                      the path to hand a tool

run `modelctl <command> --help` for a command's own options and examples.
"""


def _fmt(prog):
    return argparse.RawDescriptionHelpFormatter(prog, max_help_position=32)


# ----------------------------------------------------------------- inspect


def cmd_list(cfg: Config, args) -> int:
    repos = cfg.scan()
    if not repos:
        print(f"No models found in {cfg.store} (or the HF cache).")
        print("Fetch one with:  modelctl download <publisher>/<model>")
        return 0
    active = downloads.active_download_dirs()
    print(f"Primary store: {cfg.store}\n")
    for r in repos:
        state = downloads.inspect(r.root, active_dirs=active)
        flag = "" if not state.partial else f"  << {state.status.upper()}"
        print(f"{r.repo_id}  [{r.fmt}]  {human_size(r.size)}  ({r.store}){flag}")
        if args.files:
            for f in r.files:
                print(f"     {f.filename}  ({human_size(f.size)})")
    if any(downloads.inspect(r.root, active_dirs=active).partial for r in repos):
        print("\nSome models are incomplete. `modelctl status` explains and says what to do.")
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
    else:  # mlx / safetensors / splash: tools load the directory
        print(repo.root)
    return 0


def cmd_status(cfg: Config, args) -> int:
    """Which models are whole, and which died partway through a download."""
    stores = [cfg.store, *cfg.extra_stores]
    states: list[downloads.DownloadState] = []
    for store in stores:
        states.extend(downloads.scan_store(store))
    if args.repo:
        states = [s for s in states if args.repo in str(s.root)]
    if not states:
        print("No download bookkeeping found. Nothing has been fetched into this store by `hf`.")
        return 0
    partial = [s for s in states if s.partial]
    for s in states:
        if args.all or s.partial:
            name = s.root.name if s.root.parent.name in (".",) else f"{s.root.parent.name}/{s.root.name}"
            print(f"{name}\n    {s.summary()}")
            if s.hint:
                print(f"    -> {s.hint}")
    if not partial:
        print(f"All {len(states)} download(s) complete."
              + ("" if args.all else "  (pass --all to list them)"))
        return 0
    print(f"\n{len(partial)} of {len(states)} model(s) incomplete.")
    print("A resume re-uses the partial bytes; nothing already fetched is downloaded twice.")
    return 1 if args.exit_code else 0


def cmd_verify(cfg: Config, args) -> int:
    """Check a model's files against the Hub, via `hf cache verify`."""
    if shutil.which("hf") is None:
        print("`hf` CLI not found (install: uv tool install huggingface-hub)", file=sys.stderr)
        return 1
    repo = find_repo(cfg.scan(), args.repo)
    if not repo:
        print(f"not found in store: {args.repo}", file=sys.stderr)
        return 1
    cmd = ["hf", "cache", "verify", repo.repo_id, "--local-dir", str(repo.root)]
    if args.strict:
        cmd += ["--fail-on-missing-files", "--fail-on-extra-files"]
    else:
        cmd.append("--fail-on-missing-files")
    print("+", " ".join(cmd), flush=True)
    return subprocess.run(cmd).returncode


def cmd_doctor(cfg: Config, args) -> int:
    print("Stores (scanned in order, first match wins):")
    print(f"  - {cfg.store}  (primary / download target){'' if cfg.store.is_dir() else '  [missing]'}")
    for p in cfg.extra_stores:
        print(f"  - {p}  (extra)")
    if cfg.scan_hub:
        print(f"  - {cfg.hub}  (HF cache){'' if cfg.hub.is_dir() else '  [missing]'}")
    print(f"\nModels discovered: {len(cfg.scan())}")

    partial = [s for s in downloads.scan_store(cfg.store) if s.partial]
    if partial:
        print(f"\nIncomplete downloads: {len(partial)}  (run `modelctl status` for detail)")
        for s in partial[:5]:
            print(f"  - {s.root.name}: {s.summary()}")
    else:
        print("Incomplete downloads: none")

    print(f"\nhf CLI: {'found' if shutil.which('hf') else 'NOT FOUND (uv tool install huggingface-hub)'}")
    print()
    for name, adapter in build_adapters(cfg).items():
        print(f"[{name}]")
        for line in adapter.doctor():
            print(f"  - {line}")
        print()
    return 0


def cmd_env(cfg: Config, args) -> int:
    print("# Point HF-aware tools (vLLM, transformers, mlx_lm) at the shared cache:")
    print(f"export HF_HOME={cfg.hub.parent}")
    print("\n# The store modelctl downloads into and projects from:")
    print(f"export MODELCTL_STORE={cfg.store}")
    print("\n# Where each app-managed tool keeps its models (override if you move them):")
    print(f"export MODELCTL_LMSTUDIO_DIR={cfg.lmstudio_dir}")
    print(f"export MODELCTL_BIONIC_DIR={cfg.bionic.path}")
    print(f"export MODELCTL_OMLX_DIR={cfg.omlx.dirs[0]}")
    return 0


# ----------------------------------------------------------------- download


def _repo_files(repo_id: str, revision: str | None) -> list[tuple[str, int]] | None:
    """List a repo's files without downloading, via `hf download --dry-run`.

    Returns None when the listing fails (no network, no such repo, needs auth),
    so the caller can fall back rather than crash."""
    cmd = ["hf", "download", repo_id, "--dry-run", "--json"]
    if revision:
        cmd += ["--revision", revision]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    except (OSError, subprocess.SubprocessError):
        return None
    try:
        if out.returncode != 0:
            return None
        data = json.loads(out.stdout)
    except (ValueError, TypeError):     # not JSON, or a patched-out subprocess
        return None
    if not isinstance(data, list):
        return None
    files = []
    for entry in data:
        if isinstance(entry, dict) and entry.get("file"):
            files.append((entry["file"], quants.parse_size(str(entry.get("size", "")))))
    return files or None


def _print_variants(variants, *, prefix: str = "  ", file=None) -> None:
    for v in variants:
        print(f"{prefix}{human_size(v.size):>8}  {v.label}", file=file or sys.stdout)


def _choose_files(repo_id: str, variants) -> list[str] | None:
    """Ask which variants to fetch. Returns filenames, or None to abort."""
    choices = [
        picker.Choice(label=v.label, detail=human_size(v.size), weight=v.size,
                      selected=i in quants.default_selection(variants))
        for i, v in enumerate(variants)
    ]
    title = (f"{repo_id} publishes {len(quants.quants(variants))} quantisations.\n"
             f"Select what to download:\n")
    footer = ("\n  ↑/↓ move   space toggle   a all   n none   enter confirm   q cancel")
    picked = picker.select(choices, title=title, footer=footer, total=human_size)
    if picked is None:
        return None
    return [f for i in picked for f in variants[i].files]


def cmd_download(cfg: Config, args) -> int:
    if shutil.which("hf") is None:
        print("`hf` CLI not found (install: uv tool install huggingface-hub)", file=sys.stderr)
        return 1
    pub, _, model = args.repo.partition("/")
    target = cfg.store / pub / model if model else cfg.store / pub

    state = downloads.inspect(target)
    if state.partial:
        print(f"Resuming an {state.status} download: {state.summary()}")
        print("Already-fetched bytes are re-used.\n")

    files = list(args.files)
    selecting = not files and not args.include and not args.all
    if selecting or args.dry_run:
        listing = _repo_files(args.repo, args.revision)
        if listing is None:
            if args.dry_run:
                print(f"could not list {args.repo} (network? auth? typo?)", file=sys.stderr)
                return 1
            print(f"note: could not list {args.repo} to offer a choice; downloading everything.",
                  file=sys.stderr)
            selecting = False
        else:
            variants = quants.group(listing)
            total = sum(v.size for v in variants)
            if args.dry_run:
                print(f"{args.repo}: {len(listing)} files, {human_size(total)} total\n")
                _print_variants(variants)
                if quants.needs_selection(variants):
                    print(f"\n{len(quants.quants(variants))} quantisations. Downloading without "
                          f"a selection would fetch all {human_size(total)}.")
                return 0
            if selecting and quants.needs_selection(variants):
                if picker.usable():
                    chosen = _choose_files(args.repo, variants)
                    if chosen is None:
                        print("cancelled.", file=sys.stderr)
                        return 130
                    if not chosen:
                        print("nothing selected.", file=sys.stderr)
                        return 1
                    files = chosen
                else:
                    # Refusing beats silently pulling every quant: this repo is
                    # 472 GB if you take them all.
                    print(f"{args.repo} publishes {len(quants.quants(variants))} quantisations "
                          f"({human_size(total)} in total):\n", file=sys.stderr)
                    _print_variants(quants.quants(variants), file=sys.stderr)
                    print("\nNo terminal to ask on. Choose with one of:", file=sys.stderr)
                    print(f"  modelctl download {args.repo} <file> [<file> ...]", file=sys.stderr)
                    print(f"  modelctl download {args.repo} --include '*Q4_K_M*'", file=sys.stderr)
                    print(f"  modelctl download {args.repo} --all      # everything", file=sys.stderr)
                    return 2

    cmd = ["hf", "download", args.repo, *files, "--local-dir", str(target)]
    for pattern in args.include:
        cmd += ["--include", pattern]
    for pattern in args.exclude:
        cmd += ["--exclude", pattern]
    if args.revision:
        cmd += ["--revision", args.revision]
    if args.max_workers:
        cmd += ["--max-workers", str(args.max_workers)]
    print("+", " ".join(cmd), flush=True)       # flush: the child writes to the same stdout
    rc = subprocess.run(cmd).returncode
    if rc != 0:
        after = downloads.inspect(target)
        if after.partial:
            print(f"\nDownload did not finish: {after.summary()}", file=sys.stderr)
            print(f"-> {after.hint}", file=sys.stderr)
        return rc
    if not args.no_sync:
        print()
        return cmd_sync(cfg, argparse.Namespace(adapter=None, dry_run=False,
                                                import_ollama=False, no_prune=False))
    return 0


# ----------------------------------------------------------------- change


def _selected(cfg: Config, names):
    adapters = build_adapters(cfg)
    if not names:
        return adapters
    bad = [n for n in names if n not in adapters]
    if bad:
        print(f"unknown adapter(s): {', '.join(bad)}; have: {', '.join(adapters)}", file=sys.stderr)
        sys.exit(2)
    return {n: adapters[n] for n in names}


def _sync_actions(cfg: Config, repos, *, dry_run: bool, adapter_names=None,
                  import_ollama: bool = False, prune: bool = True):
    """Run every selected adapter over `repos` and return its actions.

    Taking the repo list as an argument (rather than rescanning) is what lets
    `prune` preview removals honestly: it syncs the store MINUS the models it
    is about to delete, and the unlinks that produces are exactly the ones a
    real run will make."""
    # Everything modelctl projects FROM. A link pointing into one of these was
    # made by modelctl; the stores themselves are never pruned inside.
    stores = [cfg.store, *cfg.extra_stores]
    sources = stores + ([cfg.hub] if cfg.scan_hub else [])
    options = dict(import_ollama=import_ollama, prune=prune,
                   sources=sources, protected=sources,
                   state_dir=cfg.store / ".modelctl")
    actions = []
    for _, adapter in _selected(cfg, adapter_names).items():
        actions += adapter.sync(repos, dry_run=dry_run, **options)
    return actions


def cmd_sync(cfg: Config, args) -> int:
    repos = cfg.scan()
    print(f"{'DRY RUN - ' if args.dry_run else ''}syncing {len(repos)} model(s)\n")
    actions = _sync_actions(cfg, repos, dry_run=args.dry_run, adapter_names=args.adapter,
                            import_ollama=getattr(args, "import_ollama", False),
                            prune=not getattr(args, "no_prune", False))
    for a in actions:
        print(a)
    if not actions:
        print("  (nothing to project)")
    return 0


# ----------------------------------------------------------------- prune


def _confirm(question: str) -> bool:
    print(f"{question} [y/N] ", end="", file=sys.stderr, flush=True)
    try:
        return input().strip().lower() in ("y", "yes")
    except EOFError:
        return False


def _prune_refusal(cfg: Config, repo, active: set[str]) -> str | None:
    """Why this repo must not be deleted by prune, or None if it may be."""
    if repo.store == "hf-cache":
        return (f"lives in the HF cache, whose blobs can be shared between revisions; "
                f"use `hf cache rm model/{repo.repo_id}`")
    if repo.store != "store":        # Config.scan labels the primary store "store"
        return "is in an extra store, which modelctl treats as read-only"
    if not repo.root.is_symlink() and repo.root.resolve() == cfg.store.resolve():
        return "resolves to the store itself"
    if downloads.inspect(repo.root, active_dirs=active).active:
        return "is being downloaded right now; stop that download first"
    return None


def _delete_model(root: Path, store: Path) -> None:
    """Delete one model from the store.

    A store entry that is itself a symlink (`adopt --link` makes these) is
    removed as a link: its target lives elsewhere and is not ours to delete.
    rmtree never follows symlinks inside the tree either. Then drop the
    publisher folder if that left it empty."""
    if root.is_symlink():
        root.unlink()
    else:
        shutil.rmtree(root)
    parent = root.parent
    if parent != store and parent.is_dir() and not any(parent.iterdir()):
        parent.rmdir()


def _prune_stale(cfg: Config, *, dry_run: bool) -> tuple[int, int]:
    """Delete temp files abandoned by retried downloads. Only files whose etag
    matches a COMPLETED file are touched, so a live partial is never removed."""
    count = size = 0
    for state in downloads.scan_store(cfg.store):
        for p in state.stale:
            try:
                size += p.stat().st_size
                if not dry_run:
                    p.unlink()
                count += 1
            except OSError:
                pass
    return count, size


def _reclaimable(roots: list[Path]) -> int:
    """Bytes that deleting every file under `roots` actually frees.

    A file's data is freed only when its LAST hard link goes. A Splash model is
    the case that matters: its store copy and Bionic mirror share inodes, so
    the model's bytes come back only because prune removes both. Counting links
    removed per inode against st_nlink gets that right, where "free space
    before vs after" would also count whatever else wrote to the disk."""
    inodes: dict[tuple[int, int], list[int]] = {}   # (dev, ino) -> [size, nlink, removed]
    for root in roots:
        if root.is_symlink() or not root.is_dir():
            continue
        for dirpath, _, files in os.walk(root, followlinks=False):
            for name in files:
                try:
                    st = os.lstat(os.path.join(dirpath, name))
                except OSError:
                    continue
                if not stat.S_ISREG(st.st_mode):
                    continue
                entry = inodes.setdefault((st.st_dev, st.st_ino), [st.st_size, st.st_nlink, 0])
                entry[2] += 1
    return sum(size for size, nlink, removed in inodes.values() if removed >= nlink)


def cmd_prune(cfg: Config, args) -> int:
    if not args.repos and not args.stale:
        print("nothing to prune: name one or more models, or pass --stale", file=sys.stderr)
        return 2
    all_repos = cfg.scan()
    active = downloads.active_download_dirs()
    targets, problems = [], []
    for name in args.repos:
        repo = find_repo(all_repos, name)
        if repo is None:
            problems.append(f"  {name}: not found in any store")
            continue
        why = _prune_refusal(cfg, repo, active)
        if why:
            problems.append(f"  {repo.repo_id}: {why}")
        elif repo not in targets:
            targets.append(repo)
    if problems:
        print("refusing to prune:", file=sys.stderr)
        print("\n".join(problems), file=sys.stderr)
        return 1

    # The plan: the models, then exactly what sync will take away once they are
    # gone, computed by syncing the store without them.
    remaining = [r for r in all_repos if r not in targets]
    unlinks = []
    if targets:
        unlinks = [a for a in _sync_actions(cfg, remaining, dry_run=True) if a.op == "unlink"]
    stale_count, stale_size = _prune_stale(cfg, dry_run=True) if args.stale else (0, 0)
    # Mirror directories sync will delete count toward what gets freed; plain
    # link removals do not free anything.
    mirrors = [Path(a.target) for a in unlinks
               if Path(a.target).is_dir() and not Path(a.target).is_symlink()]
    freed = _reclaimable([r.root for r in targets] + mirrors) + stale_size
    if not targets and not stale_count:
        print("nothing to prune: no stale temp files found.")
        return 0

    if targets:
        print(f"Delete from the store ({human_size(sum(r.size for r in targets))}):")
        for r in targets:
            kind = "symlink, target kept" if r.root.is_symlink() else human_size(r.size)
            print(f"  {r.repo_id}  [{r.fmt}]  {kind}")
            print(f"      {r.root}")
        if unlinks:
            print("\nThen sync removes these projections:")
            for a in unlinks:
                # The preview runs while the model still exists, so the action's
                # own reason ("no longer projected") would read oddly here.
                print(f"  - [{a.adapter}] {a.target}")
    if stale_count:
        print(f"{chr(10) if targets else ''}Stale temp files from retried downloads: "
              f"{stale_count} ({human_size(stale_size)})")
    print(f"\nFrees {human_size(freed)}.")
    if args.dry_run:
        print("DRY RUN - nothing deleted.")
        return 0

    if not args.yes:
        if not picker.usable():
            print("\nNo terminal to confirm on. Re-run with --yes to delete, "
                  "or -n to preview.", file=sys.stderr)
            return 2
        if not _confirm("\nDelete permanently?"):
            print("cancelled; nothing deleted.", file=sys.stderr)
            return 1

    for r in targets:
        _delete_model(r.root, cfg.store)
        print(f"deleted {r.repo_id}")
    if stale_count:
        count, size = _prune_stale(cfg, dry_run=False)
        print(f"deleted {count} stale temp file(s), {human_size(size)}")
    if targets:
        # Show only what sync changed; the unchanged rest is noise here.
        changed = [a for a in _sync_actions(cfg, cfg.scan(), dry_run=False)
                   if a.op in ("link", "relink", "unlink", "copy", "error")]
        print(f"\nsync: {len(changed)} change(s)")
        for a in changed:
            print(a)
        if any(a.adapter == "omlx" for a in changed):
            print("  (oMLX discovers models at startup: `omlx restart`)")
    print(f"\nFreed {human_size(freed)}.")
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
    print(f"{'DRY RUN - ' if args.dry_run else ''}{verb}: {repo.root}  ->  {target}")
    if args.dry_run:
        return 0
    target.parent.mkdir(parents=True, exist_ok=True)
    if args.link:
        target.symlink_to(repo.root)
    else:
        repo.root.rename(target)  # same filesystem: atomic, no copy
    print(f"adopted as {publisher}/{repo.model}")
    return 0


# ----------------------------------------------------------------- parser


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="modelctl", description=DESCRIPTION, epilog=EPILOG, formatter_class=_fmt)
    p.add_argument("--version", action="version", version=f"modelctl {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True, metavar="<command>")

    # ---- look at things
    sp = sub.add_parser(
        "list", help="list every model across all stores", formatter_class=_fmt,
        description="List models in the primary store, any extra stores, and the HF cache.\n"
                    "Incomplete downloads are flagged; `modelctl status` explains them.",
        epilog="examples:\n  modelctl list\n  modelctl list -f      # also list each file\n")
    sp.add_argument("-f", "--files", action="store_true",
                    help="also list every file in each model, with sizes")
    sp.set_defaults(func=cmd_list)

    sp = sub.add_parser(
        "resolve", help="print the path a tool should load", formatter_class=_fmt,
        description="Print the filesystem path to hand to a tool's --model/-m flag.\n"
                    "GGUF resolves to a .gguf file; mlx, safetensors and splash to the\n"
                    "model directory, which is what those runtimes load.",
        epilog="examples:\n"
               "  llama-server -m $(modelctl resolve unsloth/Qwen3.8-27B-GGUF)\n"
               "  splash serve --model $(modelctl resolve incoai/Qwen3.8-27B-Splash)\n"
               "  modelctl resolve <repo> mmproj-F16.gguf   # a specific file\n")
    sp.add_argument("repo", help="repo id (publisher/model), or a unique bare model name")
    sp.add_argument("file", nargs="?",
                    help="optional: a specific file within the model, by name")
    sp.set_defaults(func=cmd_resolve)

    sp = sub.add_parser(
        "status", help="show which downloads are partial, stalled or interrupted",
        formatter_class=_fmt,
        description="Report download state per model.\n\n"
                    "  downloading   partial, a live `hf` owns it, bytes still moving\n"
                    "  stalled       partial, a live `hf` owns it, nothing moving\n"
                    "  interrupted   partial, nothing running (closed lid, network change)\n"
                    "  complete      nothing pending\n\n"
                    "Only complete models are safe to load. Re-running the download\n"
                    "resumes from the partial bytes.",
        epilog="examples:\n  modelctl status\n  modelctl status --all\n"
               "  modelctl status Qwen3.8\n")
    sp.add_argument("repo", nargs="?", help="only models whose path contains this text")
    sp.add_argument("-a", "--all", action="store_true",
                    help="also list downloads that are complete")
    sp.add_argument("--exit-code", action="store_true",
                    help="exit 1 if anything is incomplete (for scripts)")
    sp.set_defaults(func=cmd_status)

    sp = sub.add_parser(
        "verify", help="check a model's files against the Hub", formatter_class=_fmt,
        description="Verify local files against the Hub's checksums, via `hf cache verify`.\n"
                    "Catches truncation and corruption that a file's mere presence hides.",
        epilog="examples:\n  modelctl verify incoai/Qwen3.8-27B-Splash\n"
               "  modelctl verify <repo> --strict    # also fail on unexpected extra files\n")
    sp.add_argument("repo", help="repo id of a model in the store")
    sp.add_argument("--strict", action="store_true",
                    help="also fail when local files are not present on the remote")
    sp.set_defaults(func=cmd_verify)

    sp = sub.add_parser(
        "doctor", help="show stores, incomplete downloads, and per-tool checks",
        formatter_class=_fmt,
        description="Print the resolved configuration: which stores are scanned, how many\n"
                    "models were found, what is half-downloaded, and for each tool whether\n"
                    "it is installed and where modelctl projects into.")
    sp.set_defaults(func=cmd_doctor)

    # ---- change things
    sp = sub.add_parser(
        "download", help="fetch a model into the store (asks which quant)",
        formatter_class=_fmt,
        description="Download a model into <store>/<publisher>/<model>, then sync.\n\n"
                    "A GGUF repo usually publishes one model at a dozen-plus quantisations,\n"
                    "so taking the whole repo can mean hundreds of GB. When there is more\n"
                    "than one, modelctl lists them and asks which to fetch. Give filenames,\n"
                    "--include or --all to skip the prompt; with no terminal it refuses\n"
                    "rather than guess.\n\n"
                    "Re-running resumes: already-fetched bytes are never downloaded twice.",
        epilog="examples:\n"
               "  modelctl download unsloth/Qwen3.8-27B-GGUF          # pick a quant\n"
               "  modelctl download <repo> -n                         # list variants, fetch nothing\n"
               "  modelctl download <repo> --include '*Q4_K_M*'       # by pattern\n"
               "  modelctl download <repo> model-Q4_K_M.gguf          # by exact name\n"
               "  modelctl download <repo> --all --no-sync            # everything, don't project\n")
    sp.add_argument("repo", help="repo id on the Hub, e.g. unsloth/Qwen3.8-27B-GGUF")
    sp.add_argument("files", nargs="*", metavar="FILE",
                    help="specific files to fetch; skips the quant prompt")
    sp.add_argument("--include", action="append", default=[], metavar="GLOB",
                    help="only fetch files matching this glob; repeatable")
    sp.add_argument("--exclude", action="append", default=[], metavar="GLOB",
                    help="skip files matching this glob; repeatable")
    sp.add_argument("--all", action="store_true",
                    help="fetch every file, no prompt (can be very large)")
    sp.add_argument("-n", "--dry-run", action="store_true",
                    help="list the repo's variants and sizes, download nothing")
    sp.add_argument("--revision", metavar="REF",
                    help="branch, tag or commit to fetch instead of main")
    sp.add_argument("--max-workers", type=int, metavar="N",
                    help="parallel download workers (hf default: 8)")
    sp.add_argument("--no-sync", action="store_true",
                    help="do not run sync after downloading")
    sp.set_defaults(func=cmd_download)

    sp = sub.add_parser(
        "sync", help="project the store into every tool's view", formatter_class=_fmt,
        description="Make each tool's view MATCH the store: add what is new, repair what\n"
                    "drifted, and remove projections of models the store no longer has.\n"
                    "Idempotent. It only ever removes what it can prove it made (links\n"
                    "pointing into the store, recorded hard-link mirrors), never a real\n"
                    "file, and never anything inside a store.\n\n"
                    "  +  created a link      ~  repointed a stale link\n"
                    "  -  removed a link      .  already correct\n"
                    "  =  nothing to do, here is the launch command\n"
                    "  C  copied bytes        !  refused, something real is in the way\n",
        epilog="examples:\n  modelctl sync\n  modelctl sync -n            # preview\n"
               "  modelctl sync -a bionic     # one adapter only\n")
    sp.add_argument("-a", "--adapter", action="append", metavar="NAME",
                    help="limit to these adapters; repeatable "
                         "(vllm, mlx, omlx, lmstudio, bionic, splash, llamacpp, ollama)")
    sp.add_argument("-n", "--dry-run", action="store_true",
                    help="show what would change without touching the filesystem")
    sp.add_argument("--import-ollama", action="store_true",
                    help="also import GGUFs into ollama, which COPIES the bytes")
    sp.add_argument("--no-prune", action="store_true",
                    help="only add and repair; leave projections of deleted models in place")
    sp.set_defaults(func=cmd_sync)

    sp = sub.add_parser(
        "prune", help="delete models from the store, then sync", formatter_class=_fmt,
        description="Permanently delete models from the primary store, then run sync so\n"
                    "every tool stops showing them. It lists exactly what will go (each\n"
                    "model, its size, and every projection sync will remove) and asks\n"
                    "before deleting anything. With no terminal it needs --yes.\n\n"
                    "Refused: models in the HF cache (use `hf cache rm`), in an extra\n"
                    "store (read-only), or being downloaded right now. A store entry\n"
                    "that is a symlink is removed as a link; its target is kept.\n\n"
                    "--stale deletes temp files that retried downloads abandoned beside\n"
                    "a complete model; only files matched to a finished download go.",
        epilog="examples:\n"
               "  modelctl prune unsloth/Qwen3.8-27B-GGUF -n    # preview, delete nothing\n"
               "  modelctl prune unsloth/Qwen3.8-27B-GGUF       # asks, then deletes + syncs\n"
               "  modelctl prune <repo> <repo> --yes            # several, no prompt\n"
               "  modelctl prune --stale                        # clear retry leftovers\n")
    sp.add_argument("repos", nargs="*", metavar="REPO",
                    help="models to delete (publisher/model, or a unique model name)")
    sp.add_argument("--stale", action="store_true",
                    help="also delete temp files abandoned by retried downloads")
    sp.add_argument("-n", "--dry-run", action="store_true",
                    help="show what would be deleted and unlinked, change nothing")
    sp.add_argument("-y", "--yes", action="store_true",
                    help="do not ask for confirmation")
    sp.set_defaults(func=cmd_prune)

    sp = sub.add_parser(
        "adopt", help="normalize a model into <publisher>/<model> layout",
        formatter_class=_fmt,
        description="Move a bare model directory into the canonical\n"
                    "<store>/<publisher>/<model> layout that path-native tools index.\n"
                    "The publisher is recovered from the HF cache when possible.\n\n"
                    "This MOVES the directory by default. --link leaves the original\n"
                    "in place and creates a symlink instead.",
        epilog="examples:\n  modelctl adopt MyModel --publisher someone\n"
               "  modelctl adopt MyModel -n        # show what would move\n")
    sp.add_argument("repo", help="current repo id or bare model name in the store")
    sp.add_argument("--publisher", metavar="NAME",
                    help="publisher to file it under (else recovered from the HF cache)")
    sp.add_argument("--link", action="store_true",
                    help="symlink instead of moving (non-destructive)")
    sp.add_argument("-n", "--dry-run", action="store_true",
                    help="show what would happen, change nothing")
    sp.set_defaults(func=cmd_adopt)

    sp = sub.add_parser(
        "ollama-import", help="import a GGUF into ollama (copies bytes)",
        formatter_class=_fmt,
        description="Import a GGUF into ollama's content-addressed store.\n\n"
                    "This is the one projection that COPIES: ollama will not run a model\n"
                    "from an external symlink, so reuse means duplicating the bytes into\n"
                    "~/.ollama/models. That is why it is opt-in rather than part of sync.",
        epilog="examples:\n  modelctl ollama-import unsloth/Qwen3.8-27B-GGUF\n"
               "  modelctl ollama-import <repo> <file> --name qwen:q4\n")
    sp.add_argument("repo", help="repo id of a GGUF model in the store")
    sp.add_argument("file", nargs="?", help="which .gguf, if the repo has several")
    sp.add_argument("--name", metavar="NAME:TAG",
                    help="ollama model name (default: derived from the filename)")
    sp.add_argument("-n", "--dry-run", action="store_true",
                    help="show what would be imported, copy nothing")
    sp.set_defaults(func=cmd_ollama_import)

    # ---- set things up
    sp = sub.add_parser(
        "env", help="print shell exports for a shared store", formatter_class=_fmt,
        description="Print export lines that point HF-aware tools at the same cache and\n"
                    "record where modelctl projects. Add to your shell profile:\n\n"
                    "  modelctl env >> ~/.zshrc")
    sp.set_defaults(func=cmd_env)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(Config.load(), args)
    except KeyboardInterrupt:
        print("\ninterrupted.", file=sys.stderr)
        return 130
