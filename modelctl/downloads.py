"""Download state: is this model whole, still arriving, or abandoned midway?

A `--local-dir` download leaves its bookkeeping beside the files, under
`<model>/.cache/huggingface/download/`:

    <relpath>.metadata    one per COMPLETED file (commit, etag, timestamp)
    <relpath>.lock        held while a file is being fetched
    <opaque>.incomplete   the partial bytes of a file still arriving

So a model with any `.incomplete` file is not whole, and the set of them says
exactly what is missing. That is worth surfacing, because the usual way a
download dies here is not a crash: the laptop lid closes, or the network moves
from wifi to tethering, and `hf` sits on a dead socket. Nothing fails loudly,
the directory keeps its partial bytes, and `modelctl list` would happily report
a model that cannot load.

The distinction that matters to a user is whether anything is still trying:

    downloading   partial, a live `hf` process owns it, bytes moving
    stalled       partial, a live `hf` process owns it, nothing moving
    interrupted   partial, nothing running: resume it
    complete      no partial files left

`stalled` and `interrupted` are both fixed by re-running the download, which
resumes from the partial bytes, but they need different words: one means kill
it first, the other means just start it.
"""

from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

SCRATCH = (".cache", "huggingface", "download")
INCOMPLETE_SUFFIX = ".incomplete"
METADATA_SUFFIX = ".metadata"

# How long a live download may go without writing a byte before we call it
# stalled rather than merely slow. A 200MB shard on a poor connection can be
# quiet for a while, so this is deliberately not tight.
STALL_SECONDS = 300


@dataclass
class DownloadState:
    """What the scratch directory says about one model."""

    root: Path
    incomplete: list[Path] = field(default_factory=list)
    completed: int = 0
    active: bool = False          # a live `hf download` owns this directory
    now: float = field(default_factory=time.time)

    @property
    def partial(self) -> bool:
        return bool(self.incomplete)

    @property
    def pending_bytes(self) -> int:
        """Bytes already fetched for files that are still incomplete. Useful
        because it is what a resume gets to keep."""
        total = 0
        for p in self.incomplete:
            try:
                total += p.stat().st_size
            except OSError:
                pass
        return total

    @property
    def last_activity(self) -> float | None:
        mtimes = []
        for p in self.incomplete:
            try:
                mtimes.append(p.stat().st_mtime)
            except OSError:
                pass
        return max(mtimes) if mtimes else None

    @property
    def idle_seconds(self) -> float | None:
        last = self.last_activity
        return None if last is None else max(0.0, self.now - last)

    @property
    def status(self) -> str:
        if not self.partial:
            return "complete"
        if not self.active:
            return "interrupted"
        idle = self.idle_seconds
        return "stalled" if idle is not None and idle >= STALL_SECONDS else "downloading"

    @property
    def hint(self) -> str:
        """What the user should actually do about it."""
        return {
            "complete": "",
            "downloading": "in progress; leave it running",
            "stalled": "no bytes for a while; kill the `hf download` process, "
                       "then `modelctl download <repo>` to resume",
            "interrupted": "nothing is running; `modelctl download <repo>` resumes "
                           "from the partial bytes",
        }[self.status]

    def summary(self) -> str:
        if not self.partial:
            return "complete"
        bits = [f"{len(self.incomplete)} file(s) partial"]
        if self.pending_bytes:
            bits.append(f"{human_duration_bytes(self.pending_bytes)} already fetched")
        idle = self.idle_seconds
        if idle is not None:
            bits.append(f"idle {human_duration(idle)}")
        return f"{self.status}: " + ", ".join(bits)


def human_duration(seconds: float) -> str:
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m"
    if seconds < 86400:
        return f"{seconds // 3600}h{(seconds % 3600) // 60:02d}m"
    return f"{seconds // 86400}d{(seconds % 86400) // 3600:02d}h"


def human_duration_bytes(n: int) -> str:
    f = float(n)
    for unit in ("B", "K", "M", "G", "T"):
        if f < 1024 or unit == "T":
            return f"{f:.0f}{unit}" if unit == "B" else f"{f:.1f}{unit}"
        f /= 1024
    return f"{f:.1f}T"


def scratch_dir(root: Path) -> Path:
    return root.joinpath(*SCRATCH)


def active_download_dirs() -> set[str]:
    """Directories named on the command line of a running `hf download`.

    Shelling out to `ps` keeps this dependency-free. A failure here only costs
    us the active/abandoned distinction, so it degrades to "nothing running"
    rather than raising."""
    try:
        out = subprocess.run(["ps", "-axo", "command"], capture_output=True,
                             text=True, timeout=10).stdout
    except (OSError, subprocess.SubprocessError):
        return set()
    if not isinstance(out, str):        # patched-out subprocess, or no stdout
        return set()
    dirs: set[str] = set()
    for line in out.splitlines():
        if "hf" not in line or "download" not in line:
            continue
        parts = line.split()
        for i, tok in enumerate(parts):
            if tok == "--local-dir" and i + 1 < len(parts):
                dirs.add(parts[i + 1].rstrip("/"))
    return dirs


def inspect(root: Path, *, active_dirs: set[str] | None = None) -> DownloadState:
    """Read one model directory's download state. Never raises on a missing or
    unreadable scratch dir: no scratch simply means nothing is pending."""
    scratch = scratch_dir(root)
    incomplete: list[Path] = []
    completed = 0
    if scratch.is_dir():
        try:
            for p in scratch.rglob("*"):
                if not p.is_file():
                    continue
                if p.name.endswith(INCOMPLETE_SUFFIX):
                    incomplete.append(p)
                elif p.name.endswith(METADATA_SUFFIX):
                    completed += 1
        except OSError:
            pass
    if active_dirs is None:
        active_dirs = active_download_dirs()
    # Normalise BOTH sides: a path may reach us with a trailing slash from a
    # command line or from a caller, and only one side was being stripped.
    active = str(root).rstrip("/") in {d.rstrip("/") for d in active_dirs}
    return DownloadState(root=root, incomplete=sorted(incomplete),
                         completed=completed, active=active)


def scan_store(root: Path, *, max_depth: int = 3) -> list[DownloadState]:
    """Every model directory under `root` that has download bookkeeping.

    This walks for scratch directories rather than reusing the model scanner on
    purpose: a download that died before its first file landed leaves ONLY
    scratch, so the scanner would not see a model there at all, and that is
    precisely the case worth reporting."""
    states: list[DownloadState] = []
    if not root.is_dir():
        return states
    active = active_download_dirs()

    def walk(d: Path, depth: int) -> None:
        if depth > max_depth:
            return
        try:
            entries = sorted(d.iterdir())
        except OSError:
            return
        if scratch_dir(d).is_dir():
            states.append(inspect(d, active_dirs=active))
            return                      # a model dir; don't descend further
        for p in entries:
            if p.is_dir() and not p.name.startswith("."):
                walk(p, depth + 1)

    walk(root, 0)
    return states
