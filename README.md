# local-llm-stack — one model store, many tools

**`modelctl`** keeps your models in one human-readable store and *projects* each
one into the layout every tool expects — zero-copy via symlinks wherever the
tool allows it. `hf` does the downloading under the hood.

## Source of truth

This project's **`models/` folder** (gitignored; flat `publisher/model/…` layout
— what `hf download --local-dir` and LM Studio produce, and what a future CLI/GUI
will browse). Default `<repo>/models`, override with `MODELCTL_STORE`. The HF hub
cache is also scanned read-only, so models pulled the classic way still show up.

```
models/
├── Qwen3.6-27B/                              safetensors (full precision)
├── mlx-community/gemma-4-12B-it-8bit/        mlx (quantized)
└── lmstudio-community/gemma-4-12B-it-GGUF/   gguf
```

## Formats → which tools can run each

Each model is classified from its files/`config.json`; adapters accept by capability.

| format | how detected | vLLM | mlx_lm | splash | LM Studio / Bionic | llama.cpp | ollama |
|--------|--------------|:----:|:------:|:------:|:------------------:|:---------:|:------:|
| **gguf** | `*.gguf` | – | – | – | ✓ symlink | ✓ symlink | ✓ *copy* |
| **mlx** | top-level `quantization` in config / name `*MLX*` | – | ✓ path | – | ✓ dir symlink | – | – |
| **splash** | root `manifest.json` with `format.name` + `artifacts` | – | – | ✓ path | ✓ hard link¹ | – | – |
| **safetensors** | `*.safetensors` + `config.json` (full / GPTQ / AWQ) | ✓ path | ✓ path | – | – | – | – |

¹ Splash is the one format the LM Studio family cannot reach by symlink. Its
indexer resolves real paths and enforces containment twice, so `sync` mirrors
the package with real directories plus per-file **hard links** instead. Tried
against Bionic 1.1.5:

| projection | result |
|---|---|
| directory symlink | `Model package escapes the models directory: <pkg>` |
| per-file symlinks | `Model package path escapes its directory: <pkg>/manifest.json` |
| per-file hard links | indexed, `"format": "splash"` |

A hard link has no separate real path, so both checks pass, and sharing inodes
means the mirror costs nothing (17.4 GB mirrored with no change in free space).
It needs one filesystem: across filesystems `sync` skips with an explanation.
Because inodes are shared, deleting the package from the store does not reclaim
space until the mirror goes too, and a re-download (new inode) is repaired by
inode comparison on the next sync.

How each tool is served:

- **vLLM / transformers / mlx_lm** — read the model directory directly; no
  projection. `modelctl resolve <repo>` prints the path; the adapter prints the
  launch command (`vllm serve <path>`, `mlx_lm.generate --model <path>`).
- **LM Studio** — symlinked into `~/.lmstudio/models/<pub>/<model>` (a directory
  for MLX, per-file for GGUF). LM Studio follows the symlinks.
- **Bionic**: LM Studio's sibling app (same llama.cpp + mlx-llm runtimes), so
  the same projection. It shares the LM Studio home but keeps its own settings
  at `<home>/apps/bionic/settings.json`, and its `downloadsFolder` defaults to
  `<home>/models` rather than wherever LM Studio points, so it usually needs a
  real sync even when LM Studio reads the store natively. The adapter reads that
  setting (and `~/.lmstudio-home-pointer`) instead of assuming the default, so
  changing the folder in Bionic's UI doesn't leave `sync` writing somewhere dead.
  Its ExecuTorch ASR runtime (`.pte` speech models) is app-managed and out of scope.
  Both apps can also load `splash` via the `splash` backend extension, which
  installs into the shared `<home>/extensions/backends/` behind the
  `splashEngine` experiment flag. Splash is not symlinkable though (see note ¹),
  so in practice an app serves splash only when its models dir is the store.
  That's true of LM Studio here and not of Bionic, whose `downloadsFolder`
  defaults to `<home>/models`.
- **Splash** (`splash`): Inco AI's Apple-silicon engine. Reads the package
  directory directly, so no projection: `splash serve --model <path>`. A splash
  package is `manifest.json` + `target/` + `draft/` + `vision/` + `tokenizer/`,
  with plain `.bin` shards and no root `config.json`, so it's detected by its
  manifest rather than by a file extension. It loads nowhere else: vLLM and
  mlx_lm are explicitly excluded, even though the `.bin` shards would otherwise
  look like generic weights.
- **llama.cpp**: reads a GGUF by path, so nothing is projected.
  `modelctl resolve <repo>` prints the path for `-m`.
- **Ollama** — the outlier: its content-addressed blob store can't symlink, so
  reuse means *importing* (copying) via a Modelfile. Opt-in only.

## Requirements

Python 3.14 (what `.python-version` pins and what `bin/` runs), plus the
`hf` CLI for `download` and `verify`:

```sh
uv python install 3.14 && uv venv --python 3.14
uv tool install huggingface-hub          # provides `hf`
```

modelctl itself imports nothing outside the standard library. The suite is
verified on 3.9.6, 3.11.15, 3.12.13, 3.13.13 and 3.14.5, so the declared floor
of 3.9 is measured rather than assumed; 3.14 is simply the newest stable that
every piece accepts (`hf` needs >=3.10, and 3.15 is still a beta).

`bin/modelctl` and `bin/test` prefer `.venv/bin/python` and fall back to
`python3`, so they do not silently run on whatever the system ships (on macOS
that is still 3.9).

## Usage

```sh
# look at things
bin/modelctl list                       # every model, with incomplete ones flagged
bin/modelctl list -f                    # also list each file
bin/modelctl status                     # partial / stalled / interrupted downloads
bin/modelctl verify <repo>              # check files against the Hub's checksums
bin/modelctl resolve <repo> [file]      # the path to hand a tool
bin/modelctl doctor                     # stores, downloads, per-tool checks

# change things
bin/modelctl download <repo>            # pick a quant interactively, fetch, sync
bin/modelctl download <repo> -n         # list the repo's variants, fetch nothing
bin/modelctl download <repo> --include '*Q4_K_M*'
bin/modelctl sync                       # project the store into every tool
bin/modelctl sync -n                    # dry run
bin/modelctl adopt <model> --publisher <name>   # MOVES into <pub>/<model> layout
bin/modelctl ollama-import <repo>       # opt-in, COPIES bytes

# set things up
bin/modelctl env                        # shell exports
```

Every command has `--help` with its own options and examples.

### Picking a quant

A GGUF repo is usually one model at a dozen-plus quantisations:
`unsloth/Qwen3.8-27B-GGUF` is 33 files and **472 GB** if you take the default.
So `download` lists what the repo actually publishes and asks:

```
unsloth/Qwen3.8-27B-GGUF publishes 25 quantisations.
Select what to download:

   [ ] UD-IQ2_XXS                           7.3G
 › [x] UD-Q4_K_M                            16.5G
   [ ] UD-Q6_K_XL                           25.3G
   [x] mmproj-F16  (vision projector)      927.6M
    … 22 more below

  2 selected, 17.4G
  ↑/↓ move   space toggle   a all   n none   enter confirm   q cancel
```

Variants that are not quants are classified separately, because picking
`mmproj-F16` thinking it is "the F16 quant" would fetch a 900 MB projector
instead of a model. Pass filenames, `--include` or `--all` to skip the prompt.
With no terminal, `download` refuses rather than silently fetching everything.

### Interrupted downloads

Downloads here usually die because a laptop lid closed or the network moved,
not because anything crashed: `hf` sits on a dead socket, the partial bytes
stay on disk, and the model looks present but cannot load. `status` names that:

```
$ modelctl status
unsloth/Qwen3.8-27B-GGUF
    interrupted: 3 file(s) partial, 1.2G already fetched, idle 4h12m
    -> nothing is running; `modelctl download <repo>` resumes from the partial bytes
```

`downloading` and `stalled` mean a process still owns it (kill it first);
`interrupted` means nothing does. Re-running the download resumes either way.


## Configuration (env vars, all optional)

| Var | Default | Meaning |
|-----|---------|---------|
| `MODELCTL_STORE` | `<repo>/models` | primary store + download target (colon-separate for several) |
| `MODELCTL_SCAN_HUB` | `1` | also scan the HF cache (`0` to disable) |
| `HF_HOME` / `HF_HUB_CACHE` | `~/.cache/huggingface` | HF cache location |
| `MODELCTL_LMSTUDIO_DIR` | `~/.lmstudio/models` | LM Studio's models root |
| `MODELCTL_BIONIC_DIR` | Bionic's `downloadsFolder` | Bionic's models root (overrides its settings) |

> The `models/` folder is gitignored (large binaries, never committed) and so
> lives only in your primary checkout — Conductor worktrees won't have it. Point
> `MODELCTL_STORE` at the store when running from a worktree or elsewhere.

## Tests

```sh
bin/test            # full suite
bin/test -v         # verbose
bin/test -k resolve # filter by name
```

Stdlib `unittest`, zero dependencies. Synthetic stores/caches are built in temp
dirs (`tests/helpers.py`) — the suite never touches the real HF cache, LM Studio
dir, or any model bytes, and external tools (`hf`, `ollama`) are mocked. Covers
the registry/classification, config + store de-dup, `ensure_symlink` semantics,
every adapter's accept/sync behavior, and every CLI command.

## Adding a tool

Implement `accepts` / `sync` / `doctor` in `modelctl/adapters/`, register it in
`adapters/__init__.py`. Every adapter receives the same classified `Repo` view
from `modelctl/cache.py`, so a new adapter is ~30 lines.

## Roadmap

`modelctl` is the engine for a future CLI/GUI model manager: browse/search the
store, download with quant/format pickers, dedupe, disk accounting, one-click
"serve in <tool>". The library/CLI split keeps `hf` as the download backend and
the adapters as the projection layer a GUI would drive.

## Notes

- No dependencies — pure `python3` (3.9+) reading documented on-disk layouts, so
  it never fights a `huggingface_hub` version and runs anywhere.
- `modelctl` only ever creates/replaces symlinks it would make itself; a real
  file/dir at a target path is reported as an error and left untouched.
- Interrupted downloads (`*.part`) and hidden files are ignored by the scanner.
- vLLM needs CUDA; on macOS it's CPU-only/experimental — typically run on a
  Linux+GPU box mounting the same store.
