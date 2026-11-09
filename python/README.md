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

`initialize()` records every OpenAI (`chat.completions`, `responses`) and Anthropic (`messages`)
call in the process, sync and async, streamed or not, from clients created before or after it.
It also records raw `httpx` and `requests` POSTs to `/chat/completions`
(`instrument_http=False` turns that off). Call it in every process: queue workers and forked
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

`initialize(api_key, base_url, app_name, redact, instrument_http, disable_batch)`. Omitted keys
fall back to `GENERALCONTEXT_API_KEY` and `GENERALCONTEXT_BASE_URL`. A second call is ignored.
`wrap(client)` and `conversation(...)` from 0.1 still work.

Tests: `cd python && python -m unittest test_sdk -v`.

## License

Apache License 2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
