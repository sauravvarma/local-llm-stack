"""A checkbox list for the terminal, stdlib only.

modelctl has no dependencies, so this is a small raw-mode reader rather than a
curses or prompt-toolkit screen. It deliberately does not take over the
terminal: it draws the list in place, redraws on each keypress, and leaves the
final state on screen so the transcript shows what was chosen.

The list scrolls, because the case that motivated this (25 quants plus add-ons)
is taller than a standard terminal and a naive redraw would scroll the cursor
away from its own output.

Callers must check `usable()` first: with no TTY there is nobody to ask, and
prompting would hang a script or a CI run.
"""

from __future__ import annotations

import os
import shutil
import sys
from dataclasses import dataclass

ESC = "\x1b"
HIDE_CURSOR = f"{ESC}[?25l"
SHOW_CURSOR = f"{ESC}[?25h"
CLEAR_LINE = f"{ESC}[2K"


@dataclass
class Choice:
    label: str
    detail: str = ""
    selected: bool = False
    enabled: bool = True     # False renders as a fixed, unselectable row
    weight: int = 0          # summed into the running total (e.g. bytes)


def usable() -> bool:
    """True when there is a human on both ends to draw to and read from."""
    try:
        return sys.stdin.isatty() and sys.stderr.isatty()
    except (AttributeError, ValueError):
        return False


def _read_key(fd: int) -> str:
    ch = os.read(fd, 1).decode("utf-8", "replace")
    if ch != ESC:
        return ch
    # An escape sequence: arrow keys arrive as ESC [ A..D. A bare ESC (nothing
    # buffered behind it) means cancel.
    seq = os.read(fd, 2).decode("utf-8", "replace")
    return {"[A": "up", "[B": "down", "[C": "right", "[D": "left"}.get(seq, ESC)


def _size() -> os.terminal_size:
    """Terminal size with a floor. A pty with no winsize set reports 0 columns,
    and truncating to `columns - 1` then silently eats a character off every
    row, which is how the size column lost its unit suffix."""
    size = shutil.get_terminal_size((80, 24))
    return os.terminal_size((max(size.columns, 40), max(size.lines, 10)))


def _viewport(cursor: int, total: int, height: int) -> tuple[int, int]:
    if total <= height:
        return 0, total
    top = max(0, min(cursor - height // 2, total - height))
    return top, top + height


def select(choices: list[Choice], *, title: str = "", footer: str = "",
           total: "callable | None" = None) -> list[int] | None:
    """Run the picker. Returns the selected indices, or None if cancelled.

    Drawn on stderr so that stdout stays clean for whatever the command is
    actually producing."""
    import termios
    import tty

    if not choices:
        return []
    out = sys.stderr
    fd = sys.stdin.fileno()
    saved = termios.tcgetattr(fd)
    cursor = 0
    drawn = 0
    # Pad to the longest label, but cap it: one long add-on label should not
    # push every size column far to the right.
    width = min(max((len(c.label) for c in choices), default=0), 34)

    def rows() -> int:
        return max(3, _size().lines - 8)

    def render() -> None:
        nonlocal drawn
        if drawn:                                  # rewind over the previous frame
            out.write(f"{ESC}[{drawn}A")
        lines: list[str] = []
        if title:
            lines.append(title)
        top, bottom = _viewport(cursor, len(choices), rows())
        if top > 0:
            lines.append(f"    … {top} more above")
        for i in range(top, bottom):
            c = choices[i]
            pointer = "›" if i == cursor else " "
            if not c.enabled:
                box = "──"
            else:
                box = "[x]" if c.selected else "[ ]"
            line = f" {pointer} {box} {c.label:<{width}}"
            if c.detail:
                line += f"   {c.detail:>8}"
            lines.append(line[: _size().columns - 1])
        if bottom < len(choices):
            lines.append(f"    … {len(choices) - bottom} more below")
        if total is not None:
            picked = [c for c in choices if c.selected and c.enabled]
            lines.append(f"\n  {len(picked)} selected, {total(sum(c.weight for c in picked))}")
        if footer:
            lines.append(footer)
        for line in lines:
            out.write(f"{CLEAR_LINE}{line}\n")
        out.flush()
        # Count terminal ROWS, not list entries: a title or footer may embed
        # newlines, and rewinding by the wrong count makes the frame crawl down
        # the screen on every keypress.
        drawn = sum(1 + line.count("\n") for line in lines)

    try:
        out.write(HIDE_CURSOR)
        tty.setcbreak(fd)
        while True:
            render()
            key = _read_key(fd)
            if key in ("\r", "\n"):
                return [i for i, c in enumerate(choices) if c.selected and c.enabled]
            if key in ("q", ESC, "\x03"):          # q, Esc, Ctrl-C
                return None
            if key in ("up", "k"):
                cursor = (cursor - 1) % len(choices)
            elif key in ("down", "j"):
                cursor = (cursor + 1) % len(choices)
            elif key == " ":
                c = choices[cursor]
                if c.enabled:
                    c.selected = not c.selected
            elif key == "a":
                enabled = [c for c in choices if c.enabled]
                turn_on = not all(c.selected for c in enabled)
                for c in enabled:
                    c.selected = turn_on
            elif key == "n":
                for c in choices:
                    c.selected = False
            elif key == "g":
                cursor = 0
            elif key == "G":
                cursor = len(choices) - 1
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, saved)
        sys.stderr.write(SHOW_CURSOR)
        sys.stderr.flush()
