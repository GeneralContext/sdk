# generalcontext

Record your agent's model calls for General Context. Your agent never waits on us, and a failure on our side is logged and dropped, never thrown. No runtime dependencies. Supports OpenAI, Anthropic, Google Gemini and AWS Bedrock Converse, streamed or not. See [Supported providers](#supported-providers).

## Install

```sh
npm install generalcontext
```

Set `GENERALCONTEXT_API_KEY` to the key from your workspace's Connect page.

## Supported providers

| Provider | Package | Recorded |
| - | - | - |
| OpenAI, including Azure OpenAI | `openai` | `chat.completions.create`, `.parse`, `.stream`, `runTools`, and `responses.create`, `.parse`, `.stream` |
| Anthropic, including Bedrock, Vertex and beta | `@anthropic-ai/sdk` (also used by `@anthropic-ai/bedrock-sdk` and `@anthropic-ai/vertex-sdk`) | `messages.create`, `.stream`, and `beta.messages.create`, `.parse`, `.stream`, `toolRunner` |
| Google Gemini | `@google/genai` | `models.generateContent`, `models.generateContentStream` |
| AWS Bedrock Converse | `@aws-sdk/client-bedrock-runtime` | `ConverseCommand`, `ConverseStreamCommand`. Images and documents are sent as `{"omitted": "bytes"}`. Other commands, such as `InvokeModel`, are not recorded. |
| Anything else | | Export spans with OpenTelemetry to General Context. |

Each call is recorded once, with the request as you passed it and the provider's reply. Streams pass through unchanged and are recorded as one assembled reply when they finish.

Supported versions: Node 18 to 24 with `openai` >= 4.0.0, `@anthropic-ai/sdk` >= 0.20.0, `@google/genai` >= 1.0.0 and `@aws-sdk/client-bedrock-runtime` >= 3.587.0 (the first with `ConverseCommand`), up to the latest releases (tested: openai 7.54.0, @anthropic-ai/sdk 1.7.0, @google/genai 3.0.0, client-bedrock-runtime 3.1165.0; each package's own Node minimum still applies, for example openai 7 and @google/genai 3 need Node 22). See [stress/compat/RESULTS.md](https://github.com/GeneralContext/sdk/blob/main/stress/compat/RESULTS.md).

## Initialize

Call once per process, before your agent makes model calls. Every client of a supported SDK is recorded, including ones created earlier.

```ts
import OpenAI from 'openai';
import Anthropic from '@anthropic-ai/sdk';
import { initialize } from 'generalcontext';

initialize({ appName: 'support-agent', instrumentModules: { openAI: OpenAI, anthropic: Anthropic } });
```

For Gemini and Bedrock, pass the modules the same way:

```ts
import * as googleGenAI from '@google/genai';
import * as bedrockRuntime from '@aws-sdk/client-bedrock-runtime';

initialize({ instrumentModules: { googleGenAI, bedrockRuntime } });
```

## Configuration

| Option | Default | |
| - | - | - |
| `apiKey` | `GENERALCONTEXT_API_KEY` | |
| `baseUrl` | `GENERALCONTEXT_BASE_URL`, else `https://generalcontextlabs.com` | The option wins over the env var. Must be `https://`, except `http://localhost`, `127.0.0.1` or `[::1]`. Other URLs are refused and nothing is sent. |
| `appName` | none | Sent with every batch. |
| `redact` | `false` | See [Redaction](#redaction). |
| `beforeSend` | none | `(call) => call \| null`. Runs at send time on each call, before redaction. Return the call (edited or not) to send it, or `null` to drop it. A throw drops the call. |
| `instrumentModules` | installed `openai`, `@anthropic-ai/sdk`, `@google/genai` and `@aws-sdk/client-bedrock-runtime` | Keys: `openAI`, `anthropic`, `googleGenAI`, `bedrockRuntime`. Pass these whenever a bundler is involved. |
| `instrumentFetch` | `false` | SDK clients are always recorded. Set `true` to also record raw `fetch` POSTs to any `/chat/completions` URL. |
| `disableBatch` | `false` | Send each call right away. |

| Env var | |
| - | - |
| `GENERALCONTEXT_API_KEY` | API key, if `apiKey` is not passed. |
| `GENERALCONTEXT_BASE_URL` | Collector URL, if `baseUrl` is not passed. |
| `GENERALCONTEXT_REDACT_SALT` | Secret that keeps redaction placeholders stable across processes. |
| `GENERALCONTEXT_DISABLED` | `1`, `true` or `yes` makes `initialize()` do nothing. |

Requests never follow redirects. A call over 4 MB is dropped, and the queue holds at most 5000 calls or 64 MB. Pending calls are flushed when the process is about to exit, for at most 2 seconds.

## Proxies

Node's built-in `fetch` ignores `HTTPS_PROXY` by default. On Node 24+, set `NODE_USE_ENV_PROXY=1`. On older versions, install a global dispatcher (for example undici's `setGlobalDispatcher(new EnvHttpProxyAgent())`). If your proxy inspects TLS, point `NODE_EXTRA_CA_CERTS` at its CA certificate.

## Next.js

Initialize in `instrumentation.ts` at the project root, not in a route file:

```ts
export async function register() {
  if (process.env.NEXT_RUNTIME === 'nodejs') {
    const { initialize } = await import('generalcontext');
    const { OpenAI } = await import('openai');
    const anthropic = await import('@anthropic-ai/sdk');
    initialize({ appName: 'my-agent', instrumentModules: { openAI: OpenAI, anthropic } });
  }
}
```

Keep the SDKs out of the bundle so routes use the instance we patched:

```ts
// next.config.ts
export default { serverExternalPackages: ['openai', '@anthropic-ai/sdk'] };
```

Use the Node.js runtime. The Edge runtime is not supported.

## Sessions and tasks

Tag calls with your own ids. Tags carry across `await`.

```ts
import { withSession, task } from 'generalcontext';

await withSession({ sessionId: conversationId, userId: 'u_42', channel: 'chat' }, async () => {
  await task('triage', () => client.chat.completions.create({ ... }));
});
```

`withSession(conversationId, fn)` also works. `conversation({ id, ... }, fn)` is the older name.

## Serverless and scripts

Calls send in the background. Before a function or script exits:

```ts
await flush(); // waits up to 10 s (flush(ms) to change), resolves true when sent, never throws
```

## Outcomes and events

Both flush pending calls first, so the session exists before the outcome or event arrives. Both return `true` when accepted and never throw.

```ts
await reportOutcome(conversationId, 'fail', 'customer reopened the ticket');

await logEvent({ sessionId: conversationId, eventType: 'order.refunded', eventId: `refund-${refundId}`, properties: { amount: 42.5 } });
await logEvent({ sessionId: conversationId, eventType: 'handoff.to_human', eventId: `handoff-${ticketId}` });
```

Pass a stable `eventId` so retries dedupe. `properties` are flat strings, numbers and booleans. `timestamp` defaults to now.

## Redaction

`initialize({ redact: true })` replaces emails, phone numbers and card or SSN-like numbers with placeholders such as `<email:3f9a1c>` before anything leaves your process. Set `GENERALCONTEXT_REDACT_SALT` to one secret in every process so the same value gets the same placeholder. Or pass your own `(text) => text`. Redaction also applies to `reportOutcome` and `logEvent`, so a redacted session id still matches its calls.

## One client only

`wrap(client, { redact })` records a single OpenAI, Anthropic, `GoogleGenAI` or `BedrockRuntimeClient` instance without patching the SDK. Its `redact` applies to that client only. If `initialize()` already patched the SDK, the client is recorded with the `initialize()` settings.

## Command line

The package installs a `generalcontext` command. It reads `GENERALCONTEXT_API_KEY` and `GENERALCONTEXT_BASE_URL`, with the same https rule as the SDK.

```sh
npx generalcontext verify                      # workspace, key scopes, when the last call arrived
npx generalcontext verify --session conv-123   # did this conversation arrive? exits 1 if not yet
npx generalcontext benchmark pull --out benchmark.json
```

`--json` prints the raw status. `benchmark pull` needs a key that can run the benchmark. Running the benchmark is in the Python package: `pip install generalcontext && generalcontext evals run ...`. Exit codes: 0 success, 1 failed check, 2 usage or configuration error.

## License

Apache License 2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
