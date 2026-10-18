"""Text, Markdown and JSON reports."""

from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, List, Optional

from . import __version__
from .checks import FAIL, INFO, PASS, SKIP, WARN, Result, summarize

COLORS = {PASS: "32", WARN: "33", FAIL: "31", SKIP: "90", INFO: "36"}


def use_color(stream) -> bool:
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("FORCE_COLOR"):
        return True
    return hasattr(stream, "isatty") and stream.isatty()


def paint(text: str, status: str, color: bool) -> str:
    return "\033[%sm%s\033[0m" % (COLORS[status], text) if color else text


def header(base_url: str, model: str, note: str = "") -> str:
    line = "oaiprobe %s  %s  model=%s" % (__version__, base_url, model)
    return line + (("\n" + note) if note else "")


def text_row(r: Result, color: bool, width: int = 22) -> str:
    t = "%5.2fs" % r.elapsed if r.elapsed is not None else "     -"
    return "%s  %s  %s  %s" % (r.id.ljust(width), paint(r.status.ljust(4), r.status, color), t, r.detail)


def verdict(results: List[Result]) -> str:
    s = summarize(results)
    parts = ["%d pass" % s["pass"], "%d warn" % s["warn"], "%d fail" % s["fail"]]
    if s["skip"]:
        parts.append("%d skipped" % s["skip"])
    if s["fail"]:
        word = "not compatible enough for drop-in use"
    elif s["warn"]:
        word = "compatible, with caveats"
    else:
        word = "fully compatible with every check run"
    return "%s: %s" % (", ".join(parts), word)


def to_markdown(results: List[Result], base_url: str, model: str) -> str:
    icons = {PASS: "✅", WARN: "⚠️", FAIL: "❌", SKIP: "⏭️", INFO: "ℹ️"}
    lines = [
        "### oaiprobe report",
        "",
        "Endpoint `%s`, model `%s`, oaiprobe %s" % (base_url, model, __version__),
        "",
        "| Check | Result | Detail |",
        "|---|---|---|",
    ]
    for r in results:
        detail = r.detail.replace("|", "\\|").replace("\n", " ")
        lines.append("| `%s` | %s %s | %s |" % (r.id, icons[r.status], r.status, detail))
    lines += ["", "**%s**" % verdict(results), ""]
    return "\n".join(lines)


def to_json(results: List[Result], base_url: str, model: str, extra: Optional[Dict[str, Any]] = None) -> str:
    obj: Dict[str, Any] = {
        "tool": "oaiprobe",
        "version": __version__,
        "base_url": base_url,
        "model": model,
        "summary": summarize(results),
        "results": [r.to_dict() for r in results],
    }
    if extra:
        obj.update(extra)
    return json.dumps(obj, indent=2, ensure_ascii=False)
