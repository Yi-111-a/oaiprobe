import pytest

from oaiprobe import checks
from oaiprobe.checks import FAIL, INFO, PASS, SKIP, WARN
from oaiprobe.mock import QUIRKS


def test_reference_mock_passes_everything(probe):
    results = probe()
    assert len(results) == len(checks.CHECKS)
    bad = {k: (r.status, r.detail) for k, r in results.items() if r.status not in (PASS, INFO)}
    assert bad == {}
    assert results["extras"].status == INFO


# quirk -> {check id: expected status}
CASES = {
    "no_done": {"stream": WARN},
    "no_stream_usage": {"stream_usage": WARN},
    "reject_stream_options": {"stream_usage": FAIL},
    "ignore_max_tokens": {"max_tokens": FAIL, "max_completion_tokens": WARN},
    "reject_max_completion_tokens": {"max_completion_tokens": WARN, "max_tokens": PASS},
    "ignore_stop": {"stop": FAIL},
    "no_tools": {"tools": FAIL, "stream_tools": FAIL, "parallel_tools": WARN, "tool_choice_required": WARN},
    "args_object": {"tools": FAIL, "stream_tools": FAIL, "tool_choice_required": FAIL},
    "cumulative_args": {"stream_tools": FAIL, "tools": PASS},
    "reject_tool_required": {"tool_choice_required": WARN, "tools": PASS},
    "single_tool_call": {"parallel_tools": WARN},
    "substitute_model": {"chat": WARN, "bad_model": WARN},
    "accept_any_model": {"bad_model": WARN, "chat": PASS},
    "no_auth": {"auth": WARN},
    "no_json_mode": {"json_mode": WARN, "json_schema": WARN},
    "fenced_json": {"json_mode": WARN, "json_schema": WARN},
    "ignore_schema": {"json_schema": WARN, "json_mode": PASS},
    "reasoning": {"extras": INFO, "chat": PASS},
    "no_logprobs": {"logprobs": WARN},
    "ignore_n": {"n": WARN},
    "reject_developer": {"developer_role": WARN},
    "html_errors": {"bad_model": WARN},
    "bad_usage": {"chat": WARN},
}


def test_every_quirk_has_a_case():
    assert set(CASES) == set(QUIRKS)


@pytest.mark.parametrize("quirk", sorted(CASES))
def test_quirk_is_detected(probe, quirk):
    expected = CASES[quirk]
    results = probe([quirk], ids=expected)
    got = {k: results[k].status for k in expected}
    assert got == expected, {k: results[k].detail for k in expected}


def test_messages_are_specific(probe):
    r = probe(["cumulative_args"], ids=["stream_tools"])["stream_tools"]
    assert "repeats the full arguments" in r.detail
    r = probe(["substitute_model"], ids=["bad_model"])["bad_model"]
    assert "silently served" in r.detail and "cheap-mini" in r.detail
    r = probe(["reasoning"], ids=["chat", "extras"])["extras"]
    assert "message.reasoning_content" in r.detail
    assert "usage.prompt_cache_hit_tokens" in r.detail
    assert "usage.completion_tokens_details.reasoning_tokens" in r.detail


def test_auth_skipped_without_key(probe):
    r = probe(["no_auth"], ids=["auth", "chat"], key=None)
    assert r["auth"].status == SKIP
    assert r["chat"].status == PASS


def test_wrong_key_fails_cleanly(probe):
    r = probe(ids=["chat", "stream"], key="sk-wrong")
    assert r["chat"].status == FAIL and "401" in r["chat"].detail
    assert r["stream"].status == FAIL and "401" in r["stream"].detail


def test_unlisted_model_warns(probe):
    r = probe(["accept_any_model"], ids=["models"], model="other-model")
    assert r["models"].status == WARN and "not listed" in r["models"].detail


def test_extras_skips_without_chat(probe):
    assert probe(ids=["extras"])["extras"].status == SKIP


def test_roundtrip_works_without_prior_tools_check(probe):
    assert probe(ids=["tool_roundtrip"])["tool_roundtrip"].status == PASS


def test_select():
    assert [c.id for c in checks.select(quick=True)] == [c.id for c in checks.CHECKS if c.quick]
    assert [c.id for c in checks.select(only=["stop", "chat"])] == ["chat", "stop"]
    assert "auth" not in [c.id for c in checks.select(skip=["auth"])]
    with pytest.raises(ValueError):
        checks.select(only=["nope"])


def test_parse_json_content():
    assert checks.parse_json_content('```json\n{"a": 1}\n```') == ({"a": 1}, True)
    assert checks.parse_json_content('<think>hmm</think>{"a": 1}') == ({"a": 1}, False)
    with pytest.raises(ValueError):
        checks.parse_json_content("not json")


def test_malformed_response_never_crashes():
    class FakeClient:
        base_url = "http://x/v1"
        api_key = "k"
        extra_headers = {}

        def post(self, path, payload, auth=True):
            from oaiprobe.http import Response

            return Response(200, {}, b'{"choices": "weird"}', 0.0)

    ctx = checks.Context(FakeClient(), "m")
    for cid in ("chat", "tools", "json_mode", "n", "logprobs"):
        res = checks.run_check(ctx, checks.get_check(cid))
        assert res.status in (FAIL, WARN), (cid, res)
