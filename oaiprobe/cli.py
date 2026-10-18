"""Command line interface."""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Optional

from . import __version__
from .checks import CHECKS, FAIL, WARN, Context, run, select
from .http import Client, ConnectError, ProbeError
from .mock import DEFAULT_KEY, DEFAULT_MODEL, QUIRKS, MockServer
from .report import header, text_row, to_json, to_markdown, use_color, verdict

EPILOG = """examples:
  oaiprobe -u https://api.deepseek.com -m deepseek-chat        # key from $OPENAI_API_KEY
  oaiprobe -u http://localhost:11434/v1 -m qwen2.5:7b          # Ollama
  oaiprobe -u http://localhost:8000/v1 --quick -f markdown     # vLLM, first listed model
  oaiprobe -u https://openrouter.ai/api/v1 -m qwen/qwen3-32b -p temperature=0
  oaiprobe --mock                                              # try it with no endpoint at all
  oaiprobe --mock substitute_model,cumulative_args             # see what broken gateways look like
  oaiprobe --serve-mock 127.0.0.1:8080 --quirks no_done        # run the mock for your own tests
"""


def split_list(values: Optional[List[str]]) -> List[str]:
    out: List[str] = []
    for v in values or []:
        out += [x.strip() for x in v.split(",") if x.strip()]
    return out


def parse_headers(values: Optional[List[str]]) -> Dict[str, str]:
    headers = {}
    for h in values or []:
        if ":" not in h:
            raise ValueError("header must look like 'Name: value', got %r" % h)
        k, v = h.split(":", 1)
        headers[k.strip()] = v.strip()
    return headers


def parse_params(values: Optional[List[str]]) -> Dict[str, object]:
    params: Dict[str, object] = {}
    for item in values or []:
        if "=" not in item:
            raise ValueError("--param must look like key=value, got %r" % item)
        k, v = item.split("=", 1)
        try:
            params[k.strip()] = json.loads(v)
        except ValueError:
            params[k.strip()] = v
    return params


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="oaiprobe",
        description="Check how OpenAI-compatible an LLM endpoint really is: streaming, usage, tool calls, "
        "JSON mode, structured outputs, limits, errors and auth.",
        epilog=EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("-u", "--base-url", default=os.environ.get("OPENAI_BASE_URL"), help="API base URL, e.g. https://api.deepseek.com/v1 (default: $OPENAI_BASE_URL)")
    p.add_argument("-k", "--api-key", help="API key (default: $OAIPROBE_API_KEY, then $OPENAI_API_KEY). Never printed.")
    p.add_argument("-m", "--model", help="model to test (default: first model from GET /models)")
    p.add_argument("-H", "--header", action="append", metavar="'NAME: VALUE'", help="extra request header, repeatable")
    p.add_argument("-p", "--param", action="append", metavar="KEY=VALUE", help="extra field sent with every chat request, value parsed as JSON if possible (e.g. temperature=0, reasoning_effort=low)")
    p.add_argument("--quick", action="store_true", help="run only the core checks (about 9 requests)")
    p.add_argument("--only", action="append", metavar="IDS", help="comma separated check ids to run")
    p.add_argument("--skip", action="append", metavar="IDS", help="comma separated check ids to skip")
    p.add_argument("--list-checks", action="store_true", help="list checks and exit")
    p.add_argument("--timeout", type=float, default=60.0, help="per-request timeout in seconds (default 60)")
    p.add_argument("-f", "--format", choices=["text", "json", "markdown"], default="text")
    p.add_argument("--strict", action="store_true", help="exit 1 on warnings too")
    p.add_argument("--mock", nargs="?", const="", metavar="QUIRKS", help="probe the built-in mock server (optionally with comma separated quirks)")
    p.add_argument("--serve-mock", metavar="HOST:PORT", help="run the mock server in the foreground")
    p.add_argument("--quirks", action="append", metavar="QUIRKS", help="quirks for --serve-mock")
    p.add_argument("--list-quirks", action="store_true", help="list mock quirks and exit")
    p.add_argument("-V", "--version", action="version", version="oaiprobe " + __version__)
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    for stream in (sys.stdout, sys.stderr):
        try:  # never crash on a legacy Windows console code page
            stream.reconfigure(errors="replace")
        except (AttributeError, ValueError):
            pass
    out = sys.stdout
    err = sys.stderr

    if args.list_checks:
        for c in CHECKS:
            out.write("%-22s %s%s\n" % (c.id, c.title, "  [quick]" if c.quick else ""))
        return 0
    if args.list_quirks:
        for name, desc in QUIRKS.items():
            out.write("%-30s %s\n" % (name, desc))
        return 0

    if args.serve_mock:
        host, _, port = args.serve_mock.rpartition(":")
        try:
            srv = MockServer(split_list(args.quirks), host=host or "127.0.0.1", port=int(port))
        except ValueError as exc:
            err.write("oaiprobe: %s\n" % exc)
            return 2
        err.write("oaiprobe mock listening on %s (model %s, key %s)\n" % (srv.url, DEFAULT_MODEL, DEFAULT_KEY))
        try:
            srv.serve_forever()
        except KeyboardInterrupt:
            pass
        return 0

    try:
        only, skip = split_list(args.only), split_list(args.skip)
        checks = select(only or None, skip or None, quick=args.quick)
        headers = parse_headers(args.header)
        params = parse_params(args.param)
    except ValueError as exc:
        err.write("oaiprobe: %s\n" % exc)
        return 2

    mock = None
    if args.mock is not None:
        try:
            mock = MockServer(split_list([args.mock])).start()
        except ValueError as exc:
            err.write("oaiprobe: %s (see --list-quirks)\n" % exc)
            return 2
        base_url, api_key, model = mock.url, DEFAULT_KEY, args.model or DEFAULT_MODEL
    else:
        base_url = args.base_url
        api_key = args.api_key or os.environ.get("OAIPROBE_API_KEY") or os.environ.get("OPENAI_API_KEY")
        model = args.model
        if not base_url:
            err.write("oaiprobe: give --base-url (or set OPENAI_BASE_URL), or try --mock\n")
            return 2

    try:
        client = Client(base_url, api_key, headers, timeout=args.timeout)
        note = ""
        if mock is not None:
            note = "built-in mock server" + (" with quirks: %s" % args.mock if args.mock else "")
        if not model:
            model, why = discover_model(client)
            if not model:
                err.write("oaiprobe: %s; pass --model\n" % why)
                return 2
            note = (note + "; " if note else "") + "model picked from GET /models (use -m to choose)"
        return probe(client, model, checks, args, note, params)
    except ConnectError as exc:
        err.write("oaiprobe: %s\n" % exc)
        return 2
    finally:
        if mock is not None:
            mock.stop()


def discover_model(client: Client):
    try:
        resp = client.get("/models")
    except ProbeError as exc:
        if isinstance(exc, ConnectError):
            raise
        return None, "could not list models (%s)" % exc
    if not resp.ok:
        return None, "GET /models returned HTTP %d" % resp.status
    try:
        data = resp.json().get("data") or []
        ids = [m["id"] for m in data if isinstance(m, dict) and m.get("id")]
    except (ProbeError, AttributeError):
        ids = []
    if not ids:
        return None, "GET /models returned no models"
    return ids[0], ""


def probe(client: Client, model: str, checks, args, note: str, params=None) -> int:
    out = sys.stdout
    ctx = Context(client, model, params=dict(params or {}))
    color = use_color(out)
    if args.format == "text":
        out.write(header(client.base_url, model, note) + "\n\n")
        out.flush()
        width = max([len(c.id) for c in checks] + [5])
        results = run(ctx, checks, on_result=lambda r: (out.write(text_row(r, color, width) + "\n"), out.flush()))
        out.write("\n" + verdict(results) + "\n")
    else:
        results = run(ctx, checks)
        if args.format == "json":
            out.write(to_json(results, client.base_url, model, {"note": note} if note else None) + "\n")
        else:
            out.write(to_markdown(results, client.base_url, model) + "\n")
    statuses = {r.status for r in results}
    if FAIL in statuses or (args.strict and WARN in statuses):
        return 1
    return 0
