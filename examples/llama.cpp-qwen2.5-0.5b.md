### oaiprobe report

Endpoint `http://127.0.0.1:8080/v1`, model `qwen2.5-0.5b`, oaiprobe 0.1.0

| Check | Result | Detail |
|---|---|---|
| `models` | ✅ PASS | 1 model, 'qwen2.5-0.5b' is listed |
| `chat` | ✅ PASS | replied 'pong', usage 26+4 tokens |
| `stream` | ✅ PASS | 3 chunks, first token after 0.03s |
| `stream_usage` | ✅ PASS | usage 36+2 tokens in final chunk |
| `max_tokens` | ✅ PASS | stopped at 8 completion tokens with finish_reason 'length' |
| `max_completion_tokens` | ✅ PASS | honoured, stopped at 8 completion tokens |
| `stop` | ✅ PASS | output '1 2 3 4 5' stopped before '6' |
| `tools` | ✅ PASS | get_weather({"city": "Paris"}) |
| `tool_choice_required` | ✅ PASS | forced a tool call |
| `tool_roundtrip` | ✅ PASS | answer uses the tool result: 'The current weather in Paris is sunny with a temperature ...' |
| `stream_tools` | ✅ PASS | 7 argument fragments -> get_weather({"city": "Paris"}) |
| `parallel_tools` | ✅ PASS | 2 tool calls with distinct ids |
| `json_mode` | ✅ PASS | content parses: {"answer": "pong"} |
| `json_schema` | ✅ PASS | matches the schema: {"name": "Paris", "country": "France", "population": 2480... |
| `logprobs` | ✅ PASS | 2 tokens with logprobs, 2 alternatives each |
| `n` | ✅ PASS | 2 choices returned |
| `developer_role` | ✅ PASS | accepted: 'Hello.' |
| `bad_model` | ⚠️ WARN | accepted model 'oaiprobe-no-such-model-7f3a' and answered as 'qwen2.5-0.5b': wrong model names are silently served by some other model (normal for single-model servers like llama.cpp, a red flag for gateways and resellers) |
| `auth` | ✅ PASS | HTTP 401 without a key |
| `extras` | ℹ️ INFO | usage.prompt_tokens_details.cached_tokens, system_fingerprint, timings |

**18 pass, 1 warn, 0 fail: compatible, with caveats**

