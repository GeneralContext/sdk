# generalcontext

Record your agent's model calls for the General Context platform. No dependencies, and never
in your agent's way: calls are sent from a background thread, and our failures are logged, not raised.

```bash
pip install generalcontext
export GENERALCONTEXT_API_KEY=gck_...   # from your workspace's Connect page
```

```python
import generalcontext
generalcontext.initialize(app_name="support-agent")   # once per process, at startup

# Optional: group every call in one conversation under your own id
with generalcontext.session(conversation_id, user_id=customer_id, channel="chat"):
    with generalcontext.task("triage"):
        reply = client.chat.completions.create(...)   # any OpenAI or Anthropic client
```

`initialize()` records every call in the process from these providers, sync and async, streamed
or not, from clients created before or after it:

| Provider | Client | Recorded |
| --- | --- | --- |
| OpenAI, including Azure OpenAI | `openai.OpenAI`, `AzureOpenAI` and async versions | `chat.completions` create, parse and stream; `responses` create, parse and stream |
| Anthropic, including Bedrock and Vertex | `anthropic.Anthropic`, `AnthropicBedrock`, `AnthropicVertex` and async versions | `messages` and `beta.messages` create, parse and stream |
| Google Gemini | `google-genai` (`google.genai.Client`) | `models.generate_content` and `generate_content_stream`, also under `client.aio` |
| AWS Bedrock Converse | boto3 `bedrock-runtime` | `converse` and `converse_stream` (images and documents are sent as `{"omitted": "bytes"}`) |
| Anything else | OpenTelemetry | Send GenAI spans to General Context's OTel endpoint |

Streams reach your code unchanged, and each is recorded once as one assembled reply.

Supported versions: Python 3.9 to 3.13 with `openai` >= 1.0.0, `anthropic` >= 0.20.0,
`google-genai` >= 1.0.0 and `boto3` >= 1.34.116 (the first with Bedrock `converse`), up to the
latest releases (tested: openai 3.48.0, anthropic 2.7.0, google-genai 3.0.0, boto3 1.43.127).
Call styles a release does not have yet, such as `responses` in openai 1.0, are simply absent.
See [stress/compat/RESULTS.md](https://github.com/GeneralContext/sdk/blob/main/stress/compat/RESULTS.md).

Providers that are not installed are skipped. Raw `httpx` and `requests` POSTs to `/chat/completions`
are recorded only with `instrument_http=True`. Call it in every process: queue workers and forked
children do not inherit it.

## Flush before exit

On serverless, in jobs, and before `os._exit()`, send what is queued:

```python
generalcontext.flush()   # synchronous, waits up to 10s, never raises
```

## Outcomes and events

```python
generalcontext.report_outcome(conversation_id, "fail", note="customer reopened the ticket")   # or "success"
generalcontext.log_event(conversation_id, "order.refunded", {"amount": 42.5}, event_id=f"refund:{refund.id}")
generalcontext.log_event(conversation_id, "handoff.to_human", {"reason": "billing"})
```

Both need the conversation to exist on our side first, so they flush pending calls before
sending. Both return `True` when accepted and never raise. Pass a stable `event_id` so a
retried event is counted once. Properties are flat strings, numbers and booleans.

## Redaction

```python
generalcontext.initialize(redact=True)   # emails, phones, card- and SSN-like numbers
generalcontext.initialize(redact=fn)     # or your own: str -> str, applied to every string
```

Values become stable placeholders (`<phone:3f2a1c>`) before anything leaves your process. Set
`GENERALCONTEXT_REDACT_SALT` to one secret in every process so placeholders match.

## Configuration

| Option | Default | Meaning |
| --- | --- | --- |
| `api_key` | `GENERALCONTEXT_API_KEY` | Workspace key. Without it nothing is sent. |
| `base_url` | `GENERALCONTEXT_BASE_URL`, else `https://generalcontextlabs.com` | Must be `https://` (plain `http://` only for localhost). |
| `app_name` | `None` | Label for this app. |
| `redact` | `False` | `True` for built-in PII redaction, or a `str -> str` function. Also applied to outcomes and events. |
| `before_send` | `None` | `fn(call) -> dict or None`, runs on the sender thread before redaction. `None` or an exception drops the call. |
| `instrument_http` | `False` | Also record raw `httpx`/`requests` POSTs to `/chat/completions`. |
| `disable_batch` | `False` | Send each call on its own. |

| Env var | Meaning |
| --- | --- |
| `GENERALCONTEXT_API_KEY` | Default `api_key`. |
| `GENERALCONTEXT_BASE_URL` | Default `base_url`. |
| `GENERALCONTEXT_REDACT_SALT` | Shared secret so redaction placeholders match across processes. |
| `GENERALCONTEXT_DISABLED` | `1`, `true` or `yes` makes `initialize()` do nothing. |

`HTTPS_PROXY` and `SSL_CERT_FILE` are honored via `urllib`. Redirects are never followed.
Calls over 4 MB are dropped, and the in-memory queue is capped at 64 MB. A second `initialize()`
is ignored. `wrap(client)` and `conversation(...)` from 0.1 still work.

Tests: `cd python && python -m unittest test_sdk test_evals -v`.

## Run your benchmark

Tasks built from your traffic run anywhere, with no access to your systems: each carries what your
tools returned in production, and the environment replays it. A model plays the person, and a
judge grades the task's rubric for a reward from 0 to 1.

```bash
pip install generalcontext openai anthropic
python -m generalcontext.evals benchmark.json --agent my_agent:handle --trials 3
```

`handle(messages, toolbox)` runs your agent loop, calls tools through `toolbox.call(name,
arguments)`, and returns its final text. `--agent openai:<model>`, `claude-code:<model>` (the
local Claude Code CLI, on its signed-in plan) or `anthropic:<model>` (the API) runs a
model on your agent's recorded instructions instead. Results land in `runs/runs.json`
(`generalcontext.runs/v1`), ready for **Import results** on the Tasks page. A lookup production
never made is answered as unrecorded and counted in each run's note.

Add `--upload` to also send the results to General Context. It uses `GENERALCONTEXT_API_KEY`
and `GENERALCONTEXT_BASE_URL` (https only), like the SDK. `generalcontext-evals` is the same
command as `python -m generalcontext.evals`.

## Command line

`pip install generalcontext` adds a `generalcontext` command. It reads `GENERALCONTEXT_API_KEY`
and `GENERALCONTEXT_BASE_URL`, and exits 0 on success, 1 on a failed check, 2 on a usage or
configuration error.

```bash
generalcontext verify                       # the key works, and when the last call arrived
generalcontext verify --session chat:42     # that one conversation arrived (add --json for scripts)
generalcontext benchmark pull               # writes benchmark.json (needs a key that runs the benchmark)
generalcontext evals run benchmark.json --agent openai:gpt-6.1-terra --upload
```

## License

Apache License 2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
