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
- **llama.cpp** — `modelctl resolve <repo>` → path for `-m`, plus a flat symlink
  library at `~/models/gguf/`.
- **Ollama** — the outlier: its content-addressed blob store can't symlink, so
  reuse means *importing* (copying) via a Modelfile. Opt-in only.

## Usage

```sh
bin/modelctl list -f                      # everything across all stores
bin/modelctl sync                         # project the store into every tool
bin/modelctl sync -n                      # dry run
bin/modelctl resolve <repo> [file]        # path to load (dir for mlx/safetensors, file for gguf)
bin/modelctl download <repo> [file …]     # hf download into models/<pub>/<model>, then sync
bin/modelctl doctor                       # stores + per-tool checks
bin/modelctl env                          # shell exports

# Ollama copies, so it's explicit:
bin/modelctl sync --import-ollama
bin/modelctl ollama-import <repo> [file] --name name:tag
```

## Configuration (env vars, all optional)

| Var | Default | Meaning |
|-----|---------|---------|
| `MODELCTL_STORE` | `<repo>/models` | primary store + download target (colon-separate for several) |
| `MODELCTL_SCAN_HUB` | `1` | also scan the HF cache (`0` to disable) |
| `HF_HOME` / `HF_HUB_CACHE` | `~/.cache/huggingface` | HF cache location |
| `MODELCTL_LMSTUDIO_DIR` | `~/.lmstudio/models` | LM Studio's models root |
| `MODELCTL_BIONIC_DIR` | Bionic's `downloadsFolder` | Bionic's models root (overrides its settings) |
| `MODELCTL_GGUF_DIR` | `~/models/gguf` | flat GGUF library for llama.cpp |

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
