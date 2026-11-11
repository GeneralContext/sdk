# General Context SDK

Record your AI agent's model calls so [General Context](https://generalcontextlabs.com/docs)
can turn its production workflows into RL and eval tasks, and those tasks into its benchmark.

One call at startup records every OpenAI, Anthropic, Google Gemini and AWS Bedrock call in the
process, streamed or not.
Recording happens in the background: your agent never waits on us, and a failure on our side is
logged and dropped, never raised into your code. No runtime dependencies.

| Language | Package | Source |
| --- | --- | --- |
| Python 3.9+ | [`generalcontext`](https://pypi.org/project/generalcontext/) | [`python/`](python) |
| Node.js 18+ | [`generalcontext`](https://www.npmjs.com/package/generalcontext) | [`node/`](node) |

## Quickstart

Create an API key in your workspace (**Connect** or **API keys**), then:

```bash
pip install generalcontext
export GENERALCONTEXT_API_KEY=gck_...
```

```python
import generalcontext

generalcontext.initialize()   # once per process, before any model client is created

# Optional: group every call in one conversation under your own id
with generalcontext.session(conversation_id):
    handle(message)
```

```bash
npm install generalcontext
```

```ts
import OpenAI from 'openai';
import Anthropic from '@anthropic-ai/sdk';
import { initialize } from 'generalcontext';

initialize({ instrumentModules: { openAI: OpenAI, anthropic: Anthropic } });
```

Already tracing with OpenTelemetry? You don't need this SDK: point an OTLP exporter at
`https://generalcontextlabs.com/api/platform-collect/v1/traces`. See the
[docs](https://generalcontextlabs.com/docs/opentelemetry). To group conversations by your own
customer, set the span or resource attribute `generalcontext.account` to that customer's id or name.

## What is sent

For each model call: the request (messages, tools, model and settings), the response, timing,
token usage, any error, and the session tags you set. Nothing else from your process.
`redact=True` replaces emails, phone numbers and card or SSN-like numbers before anything
leaves the process. Inline images, audio and files (base64 data URLs and base64 `data` fields)
are never sent: the recorded copy holds `[omitted N bytes sha256:...]` in their place, keeping
the media type. Any single string over 200,000 characters is recorded as its first 100,000 and
last 20,000 characters. A call still over 4,000,000 bytes after that is dropped with a warning.
Your requests to the model provider are never changed. Full details: [docs](https://generalcontextlabs.com/docs/sdk).

## Repository

- [`python/`](python) and [`node/`](node) are released separately; see [RELEASING.md](RELEASING.md).
- Report a vulnerability: [SECURITY.md](SECURITY.md). Contributing: [CONTRIBUTING.md](CONTRIBUTING.md).

## License

Apache License 2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).

## For coding agents

[`skills/generalcontext/SKILL.md`](skills/generalcontext/SKILL.md) is one page with everything a coding agent needs to connect an app: finding every model call, setup, sessions, workers and serverless, outcomes, redaction, verifying, and common issues. It is also served at https://generalcontextlabs.com/docs/agents.md. Install it as a skill:

```bash
mkdir -p .claude/skills/generalcontext && curl -fsSL https://generalcontextlabs.com/docs/agents.md -o .claude/skills/generalcontext/SKILL.md
```
