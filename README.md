# General Context SDK

Record your AI agent's model calls so [General Context](https://generalcontextlabs.com/platform)
can review its real conversations and turn the failures into tests it has to pass.

One call at startup records every OpenAI and Anthropic call in the process, streamed or not.
Recording happens in the background: your agent never waits on us, and a failure on our side is
logged and dropped, never raised into your code. No runtime dependencies.

| Language | Package | Source |
| --- | --- | --- |
| Python 3.9+ | [`generalcontext`](https://pypi.org/project/generalcontext/) | [`python/`](python) |
| Node.js 18+ | [`@generalcontext/sdk`](https://www.npmjs.com/package/@generalcontext/sdk) | [`node/`](node) |

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
npm install @generalcontext/sdk
```

```ts
import OpenAI from 'openai';
import Anthropic from '@anthropic-ai/sdk';
import { initialize } from '@generalcontext/sdk';

initialize({ instrumentModules: { openAI: OpenAI, anthropic: Anthropic } });
```

Already tracing with OpenTelemetry? You don't need this SDK: point an OTLP exporter at
`https://generalcontextlabs.com/api/platform-collect/v1/traces`. See the
[docs](https://generalcontextlabs.com/platform/docs/opentelemetry).

## What is sent

For each model call: the request (messages, tools, model and settings), the response, timing,
token usage, any error, and the session tags you set. Nothing else from your process.
`redact=True` replaces emails, phone numbers and card or SSN-like numbers before anything
leaves the process. Full details: [docs](https://generalcontextlabs.com/platform/docs/sdk).

## Repository

- [`python/`](python) and [`node/`](node) are released separately; see [RELEASING.md](RELEASING.md).
- Report a vulnerability: [SECURITY.md](SECURITY.md). Contributing: [CONTRIBUTING.md](CONTRIBUTING.md).

## License

Apache License 2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
