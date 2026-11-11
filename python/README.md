# generalcontext

Record your AI agent's model calls for [General Context](https://generalcontextlabs.com/docs). We
read your agent's production conversations against your policies, and each failure you confirm
becomes an RL and eval task built from the conversation. Your tasks together are your benchmark.

Recording never gets in your agent's way: calls are sent in the background, a failure on our side
is logged and dropped, never raised, and there are no runtime dependencies.

## Install

```bash
pip install generalcontext
export GENERALCONTEXT_API_KEY=gck_...   # from your workspace's Connect page
```

## Record

```python
import generalcontext

generalcontext.initialize(app_name="support-agent")   # once per process, at startup

# Group every call in one conversation under the id you already use for it
with generalcontext.session(conversation_id, user_id=customer_id, channel="chat"):
    reply = handle(message)

generalcontext.flush()   # before a serverless function, job or script exits
```

`initialize()` records every call these clients make in the process, sync and async, streamed or
not, from clients created before or after it. Call it in every process, including workers.

| Provider | Client |
| --- | --- |
| OpenAI, including Azure OpenAI | `openai` (chat completions and responses, with parse and stream) |
| Anthropic, including Bedrock and Vertex | `anthropic` (messages and beta messages) |
| Google Gemini | `google-genai` |
| AWS Bedrock Converse | `boto3` `bedrock-runtime` (`converse`, `converse_stream`) |
| Anything else | Send [OpenTelemetry](https://generalcontextlabs.com/docs/opentelemetry) |

**Voice (OpenAI Realtime).** Connections from `client.realtime.connect()` (sync and async, needs
`openai[realtime]`) are recorded too. Open the connection inside `session(...)` and the whole call lands
in that conversation. Each `response.done` is recorded as one `openai-realtime` call shaped like a
Responses call: the instructions and tools, the conversation so far as text (caller turns from the input
transcription, agent turns from the output transcript, tool calls and their results), and a `voice` field
with `speech_stopped_at`, `first_audio_at`, `interrupted` and `truncated_at_ms`. Turn on input audio
transcription in the session to record what the caller said. Audio is never recorded. Pass a connection to
`wrap()` to record only that one.

Python 3.9 to 3.13. Supported client versions, redaction, outcomes and events, and every option:
[SDK reference](https://generalcontextlabs.com/docs/sdk).

## Run your benchmark

Each task runs anywhere, with no access to your systems: it carries what your agent's tools
returned in production, a model plays the person, and a judge grades the rubric for a reward from
0 to 1.

```bash
pip install generalcontext openai anthropic
generalcontext benchmark pull --out benchmark.json   # needs a key that can run the benchmark
generalcontext evals run benchmark.json --agent my_agent:handle --trials 3 --upload
```

`my_agent:handle` is your agent, called as `handle(messages, toolbox)`. See
[Tasks and environments](https://generalcontextlabs.com/docs/tasks) and the
[command line](https://generalcontextlabs.com/docs/cli).

## License

Apache License 2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
