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

| format | how detected | vLLM | mlx_lm | LM Studio | llama.cpp | ollama |
|--------|--------------|:----:|:------:|:---------:|:---------:|:------:|
| **gguf** | `*.gguf` | – | – | ✓ symlink | ✓ symlink | ✓ *copy* |
| **mlx** | top-level `quantization` in config / name `*MLX*` | – | ✓ path | ✓ dir symlink | – | – |
| **safetensors** | `*.safetensors` + `config.json` (full / GPTQ / AWQ) | ✓ path | ✓ path | – | – | – |

How each tool is served:

- **vLLM / transformers / mlx_lm** — read the model directory directly; no
  projection. `modelctl resolve <repo>` prints the path; the adapter prints the
  launch command (`vllm serve <path>`, `mlx_lm.generate --model <path>`).
- **LM Studio** — symlinked into `~/.lmstudio/models/<pub>/<model>` (a directory
  for MLX, per-file for GGUF). LM Studio follows the symlinks.
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

## Serving — `llm-serve`

`modelctl` puts models where tools expect them; `llm-serve` actually runs one
on an OpenAI-compatible endpoint, on either llama.cpp or LM Studio.

```sh
bin/llm-serve list                    # store contents + what each model can do
bin/llm-serve                         # default model, best available mode
bin/llm-serve <model> [mode]          # llama.cpp   (port 8080)
bin/llm-serve lms <model> [mode]      # LM Studio   (port 1234)
bin/llm-serve hermes                  # point Hermes Agent at whatever is running
bin/llm-serve status | stop
```

Model names are the short aliases from `llm-serve list`; a unique prefix works.
Modes are `auto` (default), `fast`, `mtp`, `vision`, `plain`, and `auto` picks
the fastest the model and machine actually support — a DFlash2 drafter if one
is in the store and a build that supports it is installed, else a built-in MTP
head, else plain. MLX models are LM Studio only; asking llama.cpp for one
fails with the `lms` command to use instead.

Every mode serves with `--jinja` (tool-calling from the model's own chat
template) and `--reasoning-format deepseek` (thinking kept out of
`message.content`), which is what agent frameworks expect.

### `llm-probe`

`llm-serve` has no hardcoded model list. `bin/llm-probe` walks the store and
reads each GGUF's real header — metadata for architecture and context length,
then the tensor table for `nextn.*` names that mean a built-in MTP head.
Vision comes from a sibling `mmproj-*.gguf`, a DFlash drafter from a matching
`*-DFlash2-GGUF` repo. Output is JSON, cached and invalidated on store mtime.

```sh
bin/llm-probe             # the registry
bin/llm-probe --refresh   # re-parse, ignoring the cache
```

Download a model with `modelctl download` and it shows up in `llm-serve list`
with correct capabilities, no edits needed.

## Configuration (env vars, all optional)

| Var | Default | Meaning |
|-----|---------|---------|
| `MODELCTL_STORE` | `<repo>/models` | primary store + download target (colon-separate for several) |
| `MODELCTL_SCAN_HUB` | `1` | also scan the HF cache (`0` to disable) |
| `HF_HOME` / `HF_HUB_CACHE` | `~/.cache/huggingface` | HF cache location |
| `MODELCTL_LMSTUDIO_DIR` | `~/.lmstudio/models` | LM Studio's models root |
| `MODELCTL_GGUF_DIR` | `~/models/gguf` | flat GGUF library for llama.cpp |
| `LLM_MODEL` | `qwen3.8-27b` | `llm-serve` default model |
| `LLM_PORT` / `LMS_PORT` | `8080` / `1234` | llama.cpp and LM Studio ports |
| `LLM_CTX` | `65536` | context length (Hermes Agent requires ≥64K) |
| `LLM_RUN_DIR` | `$XDG_STATE_HOME/llm-serve` | pid, log, state, registry cache |

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
