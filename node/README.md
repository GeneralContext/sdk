# generalcontext

Record your agent's model calls for General Context. Your agent never waits on us, and a failure on our side is logged and dropped, never thrown. No runtime dependencies. Supports OpenAI (`chat.completions`, `responses`) and Anthropic (`messages`), streamed or not.

## Install

```sh
npm install generalcontext
```

Set `GENERALCONTEXT_API_KEY` to the key from your workspace's Connect page.

## Initialize

Call once per process, before your agent makes model calls. Every OpenAI and Anthropic client is recorded, including ones created earlier.

```ts
import OpenAI from 'openai';
import Anthropic from '@anthropic-ai/sdk';
import { initialize } from 'generalcontext';

initialize({ appName: 'support-agent', instrumentModules: { openAI: OpenAI, anthropic: Anthropic } });
```

## Configuration

| Option | Default | |
| - | - | - |
| `apiKey` | `GENERALCONTEXT_API_KEY` | |
| `baseUrl` | `GENERALCONTEXT_BASE_URL`, else `https://generalcontextlabs.com` | The option wins over the env var. Must be `https://`, except `http://localhost`, `127.0.0.1` or `[::1]`. Other URLs are refused and nothing is sent. |
| `appName` | none | Sent with every batch. |
| `redact` | `false` | See [Redaction](#redaction). |
| `beforeSend` | none | `(call) => call \| null`. Runs at send time on each call, before redaction. Return the call (edited or not) to send it, or `null` to drop it. A throw drops the call. |
| `instrumentModules` | installed `openai` and `@anthropic-ai/sdk` | Pass these whenever a bundler is involved. |
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
await flush(); // never throws
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

`wrap(client, { redact })` records a single client instance without patching the SDK. Its `redact` applies to that client only. If `initialize()` already patched the SDK, the client is recorded with the `initialize()` settings.

## License

Apache License 2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
