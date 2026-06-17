"""modelctl — one model store (the HF cache), projected into every tool's view.

The HuggingFace cache is the single source of truth. Tools that don't read it
natively get a thin *resolver adapter* that projects the cache into the layout
they expect — zero-copy via symlinks wherever the tool allows it.
"""

__version__ = "0.1.0"
