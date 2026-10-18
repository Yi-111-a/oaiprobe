"""The compatibility checks. Each check sends at most a couple of tiny requests."""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from .http import Client, ConnectError, ProbeError, Response, Stream, excerpt

PASS, WARN, FAIL, SKIP, INFO = "PASS", "WARN", "FAIL", "SKIP", "INFO"
STATUSES = (PASS, WARN, FAIL, SKIP, INFO)
FINISH_REASONS = {"stop", "length", "tool_calls", "content_filter", "function_call"}
BAD_MODEL = "oaiprobe-no-such-model-7f3a"

WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name, e.g. Paris"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
            },
            "required": ["city"],
        },
    },
}
FORCE_WEATHER = {"type": "function", "function": {"name": "get_weather"}}
CITY_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "country": {"type": "string"},
        "population": {"type": "integer"},
    },
    "required": ["name", "country", "population"],
    "additionalProperties": False,
}


class Skip(Exception):
    pass


@dataclass
class Result:
    id: str
    title: str
    status: str
    detail: str
    elapsed: Optional[float] = None
    data: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = {"id": self.id, "title": self.title, "status": self.status, "detail": self.detail}
        if self.elapsed is not None:
            d["elapsed_s"] = round(self.elapsed, 3)
        if self.data:
            d["data"] = self.data
        return d


@dataclass
class Context:
    client: Client
    model: str
    cache: Dict[str, Any] = field(default_factory=dict)
    params: Dict[str, Any] = field(default_factory=dict)  # extra fields sent with every chat request

    def chat(self, messages: List[Dict[str, Any]], model: Optional[str] = None, auth: bool = True, **params) -> Response:
        payload = dict(self.params)
        payload.update({"model": model or self.model, "messages": messages})
        payload.update(params)
        return self.client.post("/chat/completions", payload, auth=auth)

    def chat_stream(self, messages: List[Dict[str, Any]], **params) -> Stream:
        payload = dict(self.params)
        payload.update({"model": self.model, "messages": messages, "stream": True})
        payload.update(params)
        return self.client.stream("/chat/completions", payload)


@dataclass
class Check:
    id: str
    title: str
    func: Callable[[Context], Tuple[str, str]]
    quick: bool
    requests: int


CHECKS: List[Check] = []


def check(id: str, title: str, quick: bool = False, requests: int = 1):
    def deco(func):
        CHECKS.append(Check(id, title, func, quick, requests))
        return func

    return deco


def get_check(id: str) -> Check:
    for c in CHECKS:
        if c.id == id:
            return c
    raise KeyError(id)


# ---------------------------------------------------------------- helpers

def plural(n: int, word: str) -> str:
    return "%d %s%s" % (n, word, "" if n == 1 else "s")


def user(text: str) -> List[Dict[str, Any]]:
    return [{"role": "user", "content": text}]


def http_fail(resp: Response, what: str = "request") -> str:
    return "%s rejected: HTTP %d: %s" % (what, resp.status, resp.error_message())


def first_choice(obj: Any) -> Dict[str, Any]:
    if not isinstance(obj, dict):
        raise ProbeError("response is not a JSON object")
    choices = obj.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise ProbeError("response has no choices: %s" % excerpt(json.dumps(obj)))
    return choices[0]


def message_of(obj: Any) -> Dict[str, Any]:
    msg = first_choice(obj).get("message")
    if not isinstance(msg, dict):
        raise ProbeError("choices[0].message missing")
    return msg


def content_of(obj: Any) -> str:
    content = message_of(obj).get("content")
    if isinstance(content, list):  # some servers return content parts
        return "".join(p.get("text", "") for p in content if isinstance(p, dict))
    return content or ""


def usage_tokens(obj: Any, key: str) -> Optional[int]:
    usage = obj.get("usage") if isinstance(obj, dict) else None
    if isinstance(usage, dict) and isinstance(usage.get(key), int):
        return usage[key]
    return None


def strip_think(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()


def parse_json_content(text: str) -> Tuple[Any, bool]:
    """Returns (value, was_fenced). Raises ValueError when not JSON."""
    t = strip_think(text).strip()
    fenced = False
    m = re.match(r"^```(?:json)?\s*(.*?)\s*```$", t, flags=re.S)
    if m:
        t, fenced = m.group(1), True
    return json.loads(t), fenced


def stream_text(st: Stream) -> Tuple[str, Optional[str], List[Dict[str, Any]]]:
    """Concatenate streamed content. Returns (text, finish_reason, chunks_with_choices)."""
    parts: List[str] = []
    finish = None
    with_choices = []
    for ev in st.events:
        if not isinstance(ev, dict):
            continue
        for ch in ev.get("choices") or []:
            if not isinstance(ch, dict):
                continue
            with_choices.append(ev)
            delta = ch.get("delta") or {}
            if isinstance(delta.get("content"), str):
                parts.append(delta["content"])
            if ch.get("finish_reason"):
                finish = ch["finish_reason"]
    return "".join(parts), finish, with_choices


def assemble_stream_tools(st: Stream) -> Dict[str, Any]:
    calls: Dict[int, Dict[str, Any]] = {}
    fragments: Dict[int, List[str]] = {}
    problems: List[str] = []
    missing_index = False
    object_args = False
    finish = None
    for ev in st.events:
        if not isinstance(ev, dict):
            continue
        for ch in ev.get("choices") or []:
            if not isinstance(ch, dict):
                continue
            if ch.get("finish_reason"):
                finish = ch["finish_reason"]
            for tc in (ch.get("delta") or {}).get("tool_calls") or []:
                if not isinstance(tc, dict):
                    continue
                idx = tc.get("index")
                if not isinstance(idx, int):
                    missing_index = True
                    idx = 0
                call = calls.setdefault(idx, {"id": None, "name": "", "arguments": ""})
                if tc.get("id") and not call["id"]:
                    call["id"] = tc["id"]
                fn = tc.get("function") or {}
                if fn.get("name"):
                    call["name"] = fn["name"] if call["name"] in ("", fn["name"]) else call["name"] + fn["name"]
                args = fn.get("arguments")
                if isinstance(args, (dict, list)):
                    object_args = True
                    args = json.dumps(args)
                if isinstance(args, str) and args:
                    fragments.setdefault(idx, []).append(args)
                    call["arguments"] += args
    if missing_index:
        problems.append("tool_call deltas have no 'index'")
    return {"calls": calls, "fragments": fragments, "object_args": object_args, "finish": finish, "problems": problems}


def model_note(ctx: Context, obj: Any) -> Tuple[Optional[str], str]:
    """(warning or None, note) about the 'model' field the server reports."""
    served = obj.get("model") if isinstance(obj, dict) else None
    if not served:
        return "response has no 'model' field", ""
    if served == ctx.model:
        return None, ""
    a, b = served.lower(), ctx.model.lower()
    if a.startswith(b) or b.startswith(a) or a.split("/")[-1] == b.split("/")[-1]:
        return None, "served as '%s'" % served
    return "served by '%s' although '%s' was requested" % (served, ctx.model), ""


# ---------------------------------------------------------------- checks

@check("models", "GET /models lists models", quick=True)
def c_models(ctx: Context):
    resp = ctx.client.get("/models")
    if resp.status == 404:
        hint = "" if ctx.client.base_url.endswith("/v1") else "; does the base URL need a /v1 suffix?"
        return WARN, "GET /models not available (404)" + hint
    if not resp.ok:
        return FAIL, http_fail(resp, "GET /models")
    data = resp.json()
    items = data.get("data") if isinstance(data, dict) else None
    if not isinstance(items, list):
        return FAIL, "body has no 'data' list (not OpenAI-shaped): %s" % excerpt(resp.text)
    ids = [m.get("id") for m in items if isinstance(m, dict)]
    ctx.cache["model_ids"] = ids
    if ctx.model in ids:
        return PASS, "%s, '%s' is listed" % (plural(len(ids), "model"), ctx.model)
    return WARN, "%s, but '%s' is not listed (alias or hidden model?)" % (plural(len(ids), "model"), ctx.model)


@check("chat", "Basic chat completion", quick=True)
def c_chat(ctx: Context):
    resp = ctx.chat(
        [{"role": "system", "content": "You are a terse assistant."}, {"role": "user", "content": "Reply with the single word: pong"}]
    )
    if not resp.ok:
        return FAIL, http_fail(resp)
    obj = resp.json()
    ctx.cache["chat"] = obj
    msg = message_of(obj)
    warns: List[str] = []
    notes: List[str] = []
    text = content_of(obj)
    if not text.strip():
        if msg.get("reasoning_content") or msg.get("reasoning"):
            warns.append("content empty, only reasoning returned")
        else:
            return FAIL, "message.content is empty"
    if msg.get("role") != "assistant":
        warns.append("message.role is %r, expected 'assistant'" % msg.get("role"))
    fr = first_choice(obj).get("finish_reason")
    if fr not in FINISH_REASONS:
        warns.append("finish_reason %r is not a standard value" % fr)
    if obj.get("object") != "chat.completion":
        warns.append("object is %r, expected 'chat.completion'" % obj.get("object"))
    if not obj.get("id"):
        warns.append("no response id")
    p, c, t = (usage_tokens(obj, k) for k in ("prompt_tokens", "completion_tokens", "total_tokens"))
    if None in (p, c, t):
        warns.append("usage missing or incomplete")
    elif p + c != t:
        warns.append("usage total_tokens %d != prompt %d + completion %d" % (t, p, c))
    else:
        notes.append("usage %d+%d tokens" % (p, c))
    mw, mn = model_note(ctx, obj)
    if mw:
        warns.append(mw)
    if mn:
        notes.append(mn)
    if "pong" in strip_think(text).lower():
        notes.insert(0, "replied 'pong'")
    if warns:
        return WARN, "; ".join(warns + notes)
    return PASS, ", ".join(notes) or "ok"


@check("stream", "Streaming (SSE) chat completion", quick=True)
def c_stream(ctx: Context):
    st = ctx.chat_stream(user("Reply with the single word: pong"))
    if not st.ok:
        return FAIL, http_fail(st.as_response(), "stream request")
    if not st.events:
        return FAIL, "no SSE data events received (got %s)" % excerpt(st.body.decode("utf-8", "replace") or "empty body")
    fails, warns = [], []
    if st.bad_lines:
        fails.append("%d data lines are not JSON (e.g. %r)" % (len(st.bad_lines), st.bad_lines[0]))
    text, finish, with_choices = stream_text(st)
    if not text.strip() and not any(
        (ch.get("delta") or {}).get("reasoning_content") for ev in with_choices for ch in ev.get("choices") or []
    ):
        fails.append("streamed content is empty")
    ctype = st.headers.get("content-type", "")
    if "text/event-stream" not in ctype:
        warns.append("Content-Type is %r, not text/event-stream" % ctype)
    objs = {ev.get("object") for ev in st.events if isinstance(ev, dict)}
    if objs != {"chat.completion.chunk"}:
        warns.append("chunk object types %s" % sorted(map(str, objs)))
    if not st.done:
        warns.append("stream did not end with 'data: [DONE]'")
    if not finish:
        warns.append("no finish_reason in any chunk")
    elif finish not in FINISH_REASONS:
        warns.append("finish_reason %r is not standard" % finish)
    ttft = st.first_content if st.first_content is not None else st.first_event
    note = "%s, first token after %.2fs" % (plural(len(st.events), "chunk"), ttft or 0.0)
    if fails:
        return FAIL, "; ".join(fails + warns)
    if warns:
        return WARN, "; ".join(warns) + " (" + note + ")"
    return PASS, note


@check("stream_usage", "stream_options.include_usage returns token usage", quick=True)
def c_stream_usage(ctx: Context):
    st = ctx.chat_stream(user("Reply with the single word: pong"), stream_options={"include_usage": True})
    if not st.ok:
        return FAIL, http_fail(st.as_response(), "stream_options") + " (LangChain, Vercel AI SDK and others send it)"
    usage = [ev for ev in st.events if isinstance(ev, dict) and isinstance(ev.get("usage"), dict) and ev["usage"]]
    if not usage:
        return WARN, "include_usage ignored: no usage chunk, so streamed calls cannot be metered"
    last = usage[-1]
    p, c = usage_tokens(last, "prompt_tokens"), usage_tokens(last, "completion_tokens")
    if p is None or c is None:
        return WARN, "usage chunk lacks prompt_tokens/completion_tokens"
    if last.get("choices"):
        return PASS, "usage %d+%d tokens (sent in a chunk that also has choices; OpenAI sends choices: [])" % (p, c)
    return PASS, "usage %d+%d tokens in final chunk" % (p, c)


LIMIT = 8


def _limit_check(ctx: Context, param: str):
    resp = ctx.chat(user("Write a 300-word story about a lighthouse keeper."), **{param: LIMIT})
    if not resp.ok:
        return None, resp
    obj = resp.json()
    fr = first_choice(obj).get("finish_reason")
    c = usage_tokens(obj, "completion_tokens")
    words = len(content_of(obj).split())
    over = (c is not None and c > LIMIT + 2) or (c is None and words > LIMIT * 3)
    size = ("%d completion tokens" % c) if c is not None else ("%d words" % words)
    return (fr, over, size), resp


@check("max_tokens", "max_tokens limits the output")
def c_max_tokens(ctx: Context):
    res, resp = _limit_check(ctx, "max_tokens")
    if res is None:
        return FAIL, http_fail(resp, "max_tokens")
    fr, over, size = res
    if over:
        return FAIL, "max_tokens=%d ignored: got %s" % (LIMIT, size)
    if fr != "length":
        return WARN, "output was cut but finish_reason is %r, expected 'length' (%s)" % (fr, size)
    return PASS, "stopped at %s with finish_reason 'length'" % size


@check("max_completion_tokens", "max_completion_tokens (newer OpenAI name) is accepted")
def c_max_completion_tokens(ctx: Context):
    res, resp = _limit_check(ctx, "max_completion_tokens")
    if res is None:
        return WARN, http_fail(resp, "max_completion_tokens") + " (newer OpenAI SDKs and o-series code send it)"
    fr, over, size = res
    if over:
        return WARN, "accepted but ignored: got %s" % size
    if fr != "length":
        return WARN, "output was cut but finish_reason is %r (%s)" % (fr, size)
    return PASS, "honoured, stopped at %s" % size


@check("stop", "stop sequences are honoured")
def c_stop(ctx: Context):
    resp = ctx.chat(
        user("Count from 1 to 10, separated by single spaces. Output only the numbers."), stop=["6"]
    )
    if not resp.ok:
        return FAIL, http_fail(resp, "stop")
    obj = resp.json()
    text = strip_think(content_of(obj))
    fr = first_choice(obj).get("finish_reason")
    if "6" in text:
        return FAIL, "stop=['6'] ignored: %r" % excerpt(text, 60)
    if "5" not in text:
        return INFO, "model did not count as asked (%r); stop not verifiable" % excerpt(text, 60)
    if fr != "stop":
        return WARN, "cut at the stop sequence but finish_reason is %r" % fr
    return PASS, "output %r stopped before '6'" % excerpt(text, 40)


def _validate_tool_call(tc: Any, want_name: str = "get_weather") -> Tuple[List[str], List[str], Optional[dict]]:
    fails, warns = [], []
    args_obj = None
    if not isinstance(tc, dict):
        return ["tool_call is not an object"], [], None
    if not tc.get("id"):
        warns.append("tool_call has no id (needed to send the tool result back)")
    if tc.get("type") != "function":
        warns.append("tool_call.type is %r, expected 'function'" % tc.get("type"))
    fn = tc.get("function") or {}
    if fn.get("name") != want_name:
        fails.append("function.name is %r, expected %r" % (fn.get("name"), want_name))
    args = fn.get("arguments")
    if isinstance(args, (dict, list)):
        fails.append("function.arguments is a JSON object, the spec says a JSON-encoded string")
        args_obj = args if isinstance(args, dict) else None
    elif not isinstance(args, str):
        fails.append("function.arguments missing")
    else:
        try:
            args_obj = json.loads(args)
        except ValueError:
            fails.append("function.arguments is not valid JSON: %r" % excerpt(args, 60))
        else:
            if not isinstance(args_obj, dict):
                fails.append("function.arguments is not a JSON object")
                args_obj = None
    if isinstance(args_obj, dict) and "city" not in args_obj:
        warns.append("arguments lack the required 'city' field: %s" % excerpt(json.dumps(args_obj), 60))
    return fails, warns, args_obj


@check("tools", "Tool calling with a forced function", quick=True)
def c_tools(ctx: Context):
    resp = ctx.chat(user("What is the weather in Paris right now?"), tools=[WEATHER_TOOL], tool_choice=FORCE_WEATHER)
    if not resp.ok:
        return FAIL, http_fail(resp, "tools")
    obj = resp.json()
    msg = message_of(obj)
    calls = msg.get("tool_calls")
    if not calls:
        text = content_of(obj)
        if "get_weather" in text or "<tool_call>" in text:
            return FAIL, "tool call returned as plain text instead of tool_calls: %r" % excerpt(text, 80)
        return FAIL, "no tool_calls returned (tools or tool_choice ignored)"
    fails, warns, args = _validate_tool_call(calls[0])
    fr = first_choice(obj).get("finish_reason")
    if fr not in ("tool_calls", "stop"):
        warns.append("finish_reason is %r" % fr)
    if not fails:
        ctx.cache["tool_call"] = calls[0]
    if fails:
        return FAIL, "; ".join(fails + warns)
    detail = "get_weather(%s)" % excerpt(json.dumps(args), 60)
    if warns:
        return WARN, "; ".join(warns) + " [" + detail + "]"
    return PASS, detail


@check("tool_choice_required", "tool_choice='required' is supported")
def c_tool_required(ctx: Context):
    resp = ctx.chat(user("What is the weather in Paris right now?"), tools=[WEATHER_TOOL], tool_choice="required")
    if not resp.ok:
        return WARN, http_fail(resp, "tool_choice='required'")
    calls = message_of(resp.json()).get("tool_calls")
    if not calls:
        return WARN, "accepted but no tool call was made"
    fails, warns, _ = _validate_tool_call(calls[0])
    if fails:
        return FAIL, "; ".join(fails)
    return (WARN, "; ".join(warns)) if warns else (PASS, "forced a tool call")


@check("tool_roundtrip", "Sending a tool result back works", quick=True)
def c_tool_roundtrip(ctx: Context):
    tc = ctx.cache.get("tool_call") or {
        "id": "call_oaiprobe_1",
        "type": "function",
        "function": {"name": "get_weather", "arguments": json.dumps({"city": "Paris"})},
    }
    tc = dict(tc)
    tc.setdefault("id", "call_oaiprobe_1")
    tc["type"] = "function"
    fn = dict(tc.get("function") or {})
    if not isinstance(fn.get("arguments"), str):
        fn["arguments"] = json.dumps(fn.get("arguments") or {"city": "Paris"})
    tc["function"] = fn
    messages = [
        {"role": "user", "content": "What is the weather in Paris right now?"},
        {"role": "assistant", "content": None, "tool_calls": [tc]},
        {"role": "tool", "tool_call_id": tc["id"], "content": json.dumps({"city": "Paris", "temp_c": 21, "condition": "sunny"})},
    ]
    resp = ctx.chat(messages, tools=[WEATHER_TOOL])
    if not resp.ok:
        return FAIL, http_fail(resp, "tool result message")
    obj = resp.json()
    text = strip_think(content_of(obj))
    if not text:
        if message_of(obj).get("tool_calls"):
            return WARN, "model called the tool again instead of answering"
        return FAIL, "empty answer after tool result"
    if "21" in text:
        return PASS, "answer uses the tool result: %r" % excerpt(text, 60)
    return WARN, "answered but did not use the tool result: %r" % excerpt(text, 60)


@check("stream_tools", "Streamed tool-call arguments assemble into JSON", quick=True)
def c_stream_tools(ctx: Context):
    st = ctx.chat_stream(user("What is the weather in Paris right now?"), tools=[WEATHER_TOOL], tool_choice=FORCE_WEATHER)
    if not st.ok:
        return FAIL, http_fail(st.as_response(), "streamed tools")
    info = assemble_stream_tools(st)
    calls = info["calls"]
    if not calls:
        return FAIL, "no tool_calls deltas in the stream"
    call = calls[min(calls)]
    frags = info["fragments"].get(min(calls), [])
    fails, warns = [], list(info["problems"])
    if info["object_args"]:
        fails.append("arguments streamed as JSON objects instead of string fragments")
    if call["name"] != "get_weather":
        fails.append("function name assembled to %r" % call["name"])
    if not call["id"]:
        warns.append("no tool_call id in the stream")
    try:
        args = json.loads(call["arguments"])
    except ValueError:
        if len(frags) > 1:
            try:
                json.loads(frags[-1])
                fails.append("each chunk repeats the full arguments so far instead of sending deltas")
            except ValueError:
                fails.append("concatenated arguments are not valid JSON: %r" % excerpt(call["arguments"], 60))
        else:
            fails.append("arguments are not valid JSON: %r" % excerpt(call["arguments"], 60))
        args = None
    if info["finish"] not in ("tool_calls", "stop"):
        warns.append("finish_reason is %r" % info["finish"])
    if fails:
        return FAIL, "; ".join(fails + warns)
    detail = "%s -> get_weather(%s)" % (plural(len(frags), "argument fragment"), excerpt(json.dumps(args), 50))
    if warns:
        return WARN, "; ".join(warns) + " [" + detail + "]"
    return PASS, detail


@check("parallel_tools", "Parallel tool calls in one turn")
def c_parallel(ctx: Context):
    resp = ctx.chat(
        user("What is the weather in Paris and in Tokyo? Call get_weather once for each city, in parallel."),
        tools=[WEATHER_TOOL],
    )
    if not resp.ok:
        return FAIL, http_fail(resp, "parallel tools")
    calls = message_of(resp.json()).get("tool_calls") or []
    if not calls:
        return WARN, "no tool calls made"
    ids = [tc.get("id") for tc in calls if isinstance(tc, dict)]
    if len(calls) > 1 and len(set(ids)) != len(ids):
        return FAIL, "%d calls share duplicate ids %s" % (len(calls), ids)
    if len(calls) == 1:
        return WARN, "only one tool call (parallel calls unsupported or model chose not to)"
    return PASS, "%d tool calls with distinct ids" % len(calls)


@check("json_mode", "response_format json_object", quick=True)
def c_json_mode(ctx: Context):
    resp = ctx.chat(
        user('Return a JSON object with a single key "answer" whose value is "pong".'),
        response_format={"type": "json_object"},
    )
    if not resp.ok:
        return WARN, http_fail(resp, "json_object")
    text = content_of(resp.json())
    try:
        val, fenced = parse_json_content(text)
    except ValueError:
        return FAIL, "content is not valid JSON: %r" % excerpt(text, 60)
    if not isinstance(val, dict):
        return FAIL, "JSON is not an object: %r" % excerpt(text, 60)
    if fenced:
        return WARN, "JSON is wrapped in a markdown code fence; json.loads() on content will fail"
    if strip_think(text) != text:
        return WARN, "content contains <think> reasoning before the JSON"
    return PASS, "content parses: %s" % excerpt(json.dumps(val), 50)


def _schema_problems(val: Any) -> List[str]:
    if not isinstance(val, dict):
        return ["not an object"]
    out = []
    types = {"string": str, "integer": int}
    for key, spec in CITY_SCHEMA["properties"].items():
        if key not in val:
            out.append("missing %r" % key)
        elif not isinstance(val[key], types[spec["type"]]) or isinstance(val[key], bool):
            out.append("%r is %s, expected %s" % (key, type(val[key]).__name__, spec["type"]))
    extra = set(val) - set(CITY_SCHEMA["properties"])
    if extra:
        out.append("extra keys %s" % sorted(extra))
    return out


@check("json_schema", "Structured outputs (json_schema, strict)")
def c_json_schema(ctx: Context):
    resp = ctx.chat(
        user("Give basic facts about Paris."),
        response_format={"type": "json_schema", "json_schema": {"name": "city_info", "strict": True, "schema": CITY_SCHEMA}},
    )
    if not resp.ok:
        return WARN, http_fail(resp, "json_schema")
    text = content_of(resp.json())
    try:
        val, fenced = parse_json_content(text)
    except ValueError:
        return FAIL, "content is not valid JSON: %r" % excerpt(text, 60)
    probs = _schema_problems(val)
    if probs:
        return WARN, "accepted but schema not enforced: %s" % "; ".join(probs)
    if fenced:
        return WARN, "matches the schema but is wrapped in a markdown fence"
    return PASS, "matches the schema: %s" % excerpt(json.dumps(val), 60)


@check("logprobs", "logprobs / top_logprobs")
def c_logprobs(ctx: Context):
    resp = ctx.chat(user("Reply with the single word: pong"), logprobs=True, top_logprobs=2)
    if not resp.ok:
        return WARN, http_fail(resp, "logprobs")
    lp = first_choice(resp.json()).get("logprobs")
    content = lp.get("content") if isinstance(lp, dict) else None
    if not isinstance(content, list) or not content:
        return WARN, "accepted but no choices[0].logprobs.content returned"
    tok = content[0] if isinstance(content[0], dict) else {}
    if not isinstance(tok.get("logprob"), (int, float)):
        return WARN, "logprobs entries lack a numeric 'logprob'"
    tops = tok.get("top_logprobs")
    if not isinstance(tops, list) or not tops:
        return WARN, "%d token logprobs but top_logprobs missing" % len(content)
    return PASS, "%s with logprobs, %s each" % (plural(len(content), "token"), plural(len(tops), "alternative"))


@check("n", "n > 1 returns several choices")
def c_n(ctx: Context):
    resp = ctx.chat(user("Name a random color. One word."), n=2)
    if not resp.ok:
        return WARN, http_fail(resp, "n=2")
    choices = resp.json().get("choices") or []
    if len(choices) == 2:
        idx = sorted(ch.get("index") for ch in choices if isinstance(ch, dict))
        if idx != [0, 1]:
            return WARN, "2 choices but indexes are %s" % idx
        return PASS, "2 choices returned"
    return WARN, "n=2 accepted but %d choice(s) returned" % len(choices)


@check("developer_role", "'developer' role messages are accepted")
def c_developer(ctx: Context):
    resp = ctx.chat(
        [{"role": "developer", "content": "Answer in uppercase letters only."}, {"role": "user", "content": "Say hello."}]
    )
    if not resp.ok:
        return WARN, http_fail(resp, "'developer' role") + " (OpenAI SDKs send it for reasoning models)"
    text = strip_think(content_of(resp.json()))
    if text and text == text.upper():
        return PASS, "accepted and followed: %r" % excerpt(text, 40)
    return PASS, "accepted: %r" % excerpt(text, 40)


@check("bad_model", "Unknown model names are rejected", quick=True)
def c_bad_model(ctx: Context):
    resp = ctx.chat(user("Reply with the single word: pong"), model=BAD_MODEL, max_tokens=5)
    if resp.ok:
        try:
            served = resp.json().get("model")
        except ProbeError:
            served = None
        return WARN, (
            "accepted model %r and answered%s: wrong model names are silently served by some other model "
            "(normal for single-model servers like llama.cpp, a red flag for gateways and resellers)"
        ) % (BAD_MODEL, " as %r" % served if served else "")
    if resp.status >= 500:
        return WARN, "HTTP %d instead of a 4xx: %s" % (resp.status, resp.error_message())
    try:
        body = json.loads(resp.text)
    except ValueError:
        body = None
    err = body.get("error") if isinstance(body, dict) else None
    if not (isinstance(err, dict) and err.get("message")):
        return WARN, "HTTP %d but the error body is not OpenAI-shaped {'error': {'message': ...}}: %s" % (
            resp.status,
            excerpt(resp.text, 80),
        )
    return PASS, "HTTP %d: %s" % (resp.status, excerpt(err["message"], 70))


@check("auth", "Requests without the API key are rejected")
def c_auth(ctx: Context):
    if not ctx.client.api_key and not any(
        k.lower() in ("authorization", "api-key", "x-api-key") for k in ctx.client.extra_headers
    ):
        raise Skip("no API key given")
    resp = ctx.chat(user("Reply with the single word: pong"), auth=False, max_tokens=5)
    if resp.status in (401, 403):
        return PASS, "HTTP %d without a key" % resp.status
    if resp.ok:
        return WARN, "answered without any API key: anyone who finds this URL can use it"
    return WARN, "unexpected HTTP %d without a key: %s" % (resp.status, resp.error_message())


@check("extras", "Non-standard fields worth knowing about", requests=0)
def c_extras(ctx: Context):
    obj = ctx.cache.get("chat")
    if not isinstance(obj, dict):
        raise Skip("needs a successful 'chat' check")
    found = []
    msg = message_of(obj)
    for key in ("reasoning_content", "reasoning", "reasoning_details", "refusal", "annotations"):
        if msg.get(key):
            found.append("message.%s" % key)
    if "<think>" in (content_of(obj) or ""):
        found.append("<think> tags inside content")
    usage = obj.get("usage") or {}
    for key in ("prompt_cache_hit_tokens", "prompt_cache_miss_tokens", "cache_read_input_tokens"):
        if key in usage:
            found.append("usage.%s" % key)
    for parent in ("prompt_tokens_details", "completion_tokens_details"):
        det = usage.get(parent)
        if isinstance(det, dict):
            for k, v in det.items():
                if v:
                    found.append("usage.%s.%s" % (parent, k))
    if obj.get("system_fingerprint"):
        found.append("system_fingerprint")
    for key in sorted(set(obj) - {"id", "object", "created", "model", "choices", "usage", "system_fingerprint", "service_tier"}):
        found.append(key)
    return INFO, (", ".join(found) if found else "none: plain OpenAI response shape")


# ---------------------------------------------------------------- runner

def select(only: Optional[List[str]] = None, skip: Optional[List[str]] = None, quick: bool = False) -> List[Check]:
    ids = {c.id for c in CHECKS}
    for name in (only or []) + (skip or []):
        if name not in ids:
            raise ValueError("unknown check %r (see --list-checks)" % name)
    out = [c for c in CHECKS if (not quick or c.quick)]
    if only:
        out = [c for c in CHECKS if c.id in only]
    if skip:
        out = [c for c in out if c.id not in skip]
    return out


def run_check(ctx: Context, chk: Check) -> Result:
    start = time.perf_counter()
    try:
        status, detail = chk.func(ctx)
    except Skip as exc:
        status, detail = SKIP, str(exc)
    except ConnectError:
        raise
    except ProbeError as exc:
        status, detail = FAIL, str(exc)
    except Exception as exc:  # a malformed response must never crash the run
        status, detail = FAIL, "unexpected response shape (%s: %s)" % (type(exc).__name__, exc)
    elapsed = time.perf_counter() - start if chk.requests else None
    return Result(chk.id, chk.title, status, detail, elapsed)


def run(ctx: Context, checks: List[Check], on_result: Optional[Callable[[Result], None]] = None) -> List[Result]:
    results: List[Result] = []
    for chk in checks:
        try:
            res = run_check(ctx, chk)
        except ConnectError as exc:
            if not results:
                raise
            res = Result(chk.id, chk.title, FAIL, str(exc))
        results.append(res)
        if on_result:
            on_result(res)
    return results


def summarize(results: List[Result]) -> Dict[str, int]:
    out = {s.lower(): 0 for s in STATUSES}
    for r in results:
        out[r.status.lower()] += 1
    return out
