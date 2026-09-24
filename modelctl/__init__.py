"""modelctl: one model store, projected into every tool's view.

The canonical store is a flat `<publisher>/<model>/` tree (by default this
repo's own `models/`); the HuggingFace cache is scanned read-only alongside it.
Tools that load a path are pointed straight at the store. Tools that insist on
their own directory get a thin *adapter* that projects the store into the
layout they expect, zero-copy: symlinks where the tool follows them, hard links
where it refuses to (Splash), and a byte copy only for ollama, which allows
nothing else.
"""

__version__ = "0.1.0"
