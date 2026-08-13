"""Render real command output as a terminal-style PNG screenshot.

Not a permanent tool -- one-off helper for assembling submission evidence
from genuine command output (this repo has no interactive terminal to
literally screen-capture; this renders actual captured stdout instead of
fabricating any).
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

FONT_CANDIDATES = [
    r"C:\Windows\Fonts\consola.ttf",
    r"C:\Windows\Fonts\cascadiamono.ttf",
    r"C:\Windows\Fonts\lucon.ttf",
]
FONT_SIZE = 16
PADDING = 24
LINE_SPACING = 6
BG = (13, 17, 23)
FG = (201, 209, 217)
TITLE_FG = (139, 148, 158)
PROMPT_FG = (88, 166, 255)


def _load_font() -> ImageFont.FreeTypeFont:
    for candidate in FONT_CANDIDATES:
        if Path(candidate).is_file():
            return ImageFont.truetype(candidate, FONT_SIZE)
    return ImageFont.load_default()


def render(title: str, command: str, output: str, out_path: Path, max_width_chars: int = 120) -> None:
    font = _load_font()
    lines: list[str] = []
    for raw_line in output.splitlines() or [""]:
        while len(raw_line) > max_width_chars:
            lines.append(raw_line[:max_width_chars])
            raw_line = raw_line[max_width_chars:]
        lines.append(raw_line)

    dummy = Image.new("RGB", (10, 10))
    draw = ImageDraw.Draw(dummy)
    char_w = draw.textlength("M", font=font)
    line_h = FONT_SIZE + LINE_SPACING

    content_lines = [f"$ {command}", ""] + lines
    width = int(max(char_w * max(len(l) for l in content_lines + [title]), 400)) + PADDING * 2
    height = PADDING * 2 + line_h * (len(content_lines) + 2)

    img = Image.new("RGB", (width, height), BG)
    draw = ImageDraw.Draw(img)
    draw.text((PADDING, PADDING - line_h), title, font=font, fill=TITLE_FG)

    y = PADDING + line_h
    draw.text((PADDING, y), f"$ {command}", font=font, fill=PROMPT_FG)
    y += line_h + LINE_SPACING
    for line in lines:
        draw.text((PADDING, y), line, font=font, fill=FG)
        y += line_h

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)
    print(f"wrote {out_path} ({width}x{height})")


def run_and_render(title: str, command: str, out_path: Path, cwd: str | None = None, shell_cmd: list[str] | None = None) -> None:
    proc = subprocess.run(
        shell_cmd or command,
        shell=shell_cmd is None,
        cwd=cwd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    output = (proc.stdout or "") + (proc.stderr or "")
    render(title, command, output.strip(), out_path)


if __name__ == "__main__":
    # title | command | output_file | actual_shell_command
    title, command, out_file, *shell_parts = sys.argv[1:]
    run_and_render(title, command, Path(out_file), shell_cmd=shell_parts if shell_parts else None)
