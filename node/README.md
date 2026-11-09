# @generalcontext/sdk

Record your agent's model calls for General Context. Your agent never waits on us, and a failure on our side is logged and dropped, never thrown. No runtime dependencies. Supports OpenAI (`chat.completions`, `responses`) and Anthropic (`messages`), streamed or not.

## Install

```sh
npm install @generalcontext/sdk
```

Set `GENERALCONTEXT_API_KEY` to the key from your workspace's Connect page.

## Initialize

Call once per process, before your agent makes model calls. Every OpenAI and Anthropic client is recorded, including ones created earlier.

```ts
import OpenAI from 'openai';
import Anthropic from '@anthropic-ai/sdk';
import { initialize } from '@generalcontext/sdk';

initialize({ appName: 'support-agent', instrumentModules: { openAI: OpenAI, anthropic: Anthropic } });
```

| Option | Default | |
| - | - | - |
| `apiKey` | `GENERALCONTEXT_API_KEY` | |
| `baseUrl` | `GENERALCONTEXT_BASE_URL`, else `https://generalcontextlabs.com` | |
| `appName` | none | Sent with every batch. |
| `instrumentModules` | installed `openai` and `@anthropic-ai/sdk` | Pass these whenever a bundler is involved. |
| `instrumentFetch` | `true` | Also records raw `fetch` POSTs to any `/chat/completions` URL. |
| `redact` | `false` | See [Redaction](#redaction). |
| `disableBatch` | `false` | Send each call right away. |

## Next.js

Initialize in `instrumentation.ts` at the project root, not in a route file:

```ts
export async function register() {
  if (process.env.NEXT_RUNTIME === 'nodejs') {
    const { initialize } = await import('@generalcontext/sdk');
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
import { withSession, task } from '@generalcontext/sdk';

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

`initialize({ redact: true })` replaces emails, phone numbers and card or SSN-like numbers with placeholders such as `<email:3f9a1c>` before anything leaves your process. Set `GENERALCONTEXT_REDACT_SALT` to one secret in every process so the same value gets the same placeholder. Or pass your own `(text) => text`.

## One client only

`wrap(client)` records a single client instance without patching the SDK.

## License

Apache License 2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
