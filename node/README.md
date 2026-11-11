# generalcontext

Record your AI agent's model calls for [General Context](https://generalcontextlabs.com/docs). We read your agent's production conversations and find where it goes wrong, and each failure you confirm becomes an RL and eval task built from the conversation. Your tasks together are your benchmark.

Recording never gets in your agent's way: calls are sent in the background, a failure on our side is logged and dropped, never thrown, and there are no runtime dependencies.

## Install

```sh
npm install generalcontext
```

Set `GENERALCONTEXT_API_KEY` to the key from your workspace's Connect page.

## Record

Call `initialize()` once per process, before your agent makes model calls. Pass the client modules you use: it is required whenever a bundler is involved.

```ts
import OpenAI from 'openai';
import Anthropic from '@anthropic-ai/sdk';
import { initialize, withSession, flush } from 'generalcontext';

initialize({ appName: 'support-agent', instrumentModules: { openAI: OpenAI, anthropic: Anthropic } });

// Group every call in one conversation under the id you already use for it
await withSession({ sessionId: conversationId, userId: customerId, channel: 'chat' }, () => handle(message));

await flush(); // before a serverless function, job or script exits
```

If your app serves several companies of its own, pass `account` to `withSession` (the id or name you already use for that company) so conversations are grouped by account.

| Provider | Package | `instrumentModules` key |
| - | - | - |
| OpenAI, including Azure OpenAI | `openai` | `openAI` |
| Anthropic, including Bedrock, Vertex and beta | `@anthropic-ai/sdk` | `anthropic` |
| Google Gemini | `@google/genai` | `googleGenAI` |
| AWS Bedrock Converse | `@aws-sdk/client-bedrock-runtime` | `bedrockRuntime` |
| Anything else | Send [OpenTelemetry](https://generalcontextlabs.com/docs/opentelemetry) | |

**Voice (OpenAI Realtime).** `OpenAIRealtimeWS` (`openai/realtime/ws`) and `OpenAIRealtimeWebSocket`
(`openai/realtime/websocket`) connections are recorded too. Create the connection inside `withSession(...)` and
the whole call lands in that conversation. Each `response.done` is recorded as one `openai-realtime` call shaped
like a Responses call: the instructions and tools, the conversation so far as text (caller turns from the input
transcription, agent turns from the output transcript, tool calls and their results), and a `voice` field with
`speech_stopped_at`, `first_audio_at`, `interrupted` and `truncated_at_ms`. Turn on input audio transcription in
the session to record what the caller said. Audio is never recorded. Pass a connection to `wrap()` to record only
that one.

Node 18 to 24, Node.js runtime only (not Edge). Supported client versions, Next.js, proxies, redaction, outcomes and events, and every option: [SDK reference](https://generalcontextlabs.com/docs/sdk).

## Command line

```sh
npx generalcontext verify                      # the key works, and when the last call arrived
npx generalcontext verify --session conv-123   # that one conversation arrived
npx generalcontext benchmark pull --out benchmark.json
```

Running the benchmark is in the Python package. See the [command line docs](https://generalcontextlabs.com/docs/cli).

## License

Apache License 2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
