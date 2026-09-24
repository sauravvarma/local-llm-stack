from __future__ import annotations

import argparse
import json
import shutil
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
directory (LM Studio, Bionic) get a symlink or hard-link projection, so the
bytes are never copied. `modelctl sync` makes every tool's view match the
store, and is safe to re-run.
"""

EPILOG = """\
commands by what they do:

  look at things       list, resolve, status, verify, doctor
  change things        download, sync, adopt, ollama-import
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
        return cmd_sync(cfg, argparse.Namespace(adapter=None, dry_run=False, import_ollama=False))
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


def cmd_sync(cfg: Config, args) -> int:
    repos = cfg.scan()
    adapters = _selected(cfg, args.adapter)
    print(f"{'DRY RUN - ' if args.dry_run else ''}syncing {len(repos)} model(s)\n")
    total = 0
    for name, adapter in adapters.items():
        for a in adapter.sync(repos, dry_run=args.dry_run,
                              import_ollama=getattr(args, "import_ollama", False)):
            print(a)
            total += 1
    if total == 0:
        print("  (nothing to project)")
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
        description="Make each tool's view match the store. Idempotent, and it never\n"
                    "overwrites a real file: a target that exists and is not a link is\n"
                    "reported as an error and left alone.\n\n"
                    "  +  created a link      ~  repointed a stale link\n"
                    "  .  already correct     =  nothing to do, here is the launch command\n"
                    "  C  copied bytes        !  refused, something real is in the way\n",
        epilog="examples:\n  modelctl sync\n  modelctl sync -n            # preview\n"
               "  modelctl sync -a bionic     # one adapter only\n")
    sp.add_argument("-a", "--adapter", action="append", metavar="NAME",
                    help="limit to these adapters; repeatable "
                         "(vllm, mlx, lmstudio, bionic, splash, llamacpp, ollama)")
    sp.add_argument("-n", "--dry-run", action="store_true",
                    help="show what would change without touching the filesystem")
    sp.add_argument("--import-ollama", action="store_true",
                    help="also import GGUFs into ollama, which COPIES the bytes")
    sp.set_defaults(func=cmd_sync)

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
