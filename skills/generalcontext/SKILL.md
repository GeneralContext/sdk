---
name: generalcontext
description: Connect an AI agent to General Context: install the generalcontext SDK (Node or Python), record every OpenAI, Anthropic, Gemini and Bedrock call, group calls into conversations with sessions and named steps, flush before exit, and report outcomes and events. Use when connecting an app to General Context or debugging missing conversations, calls, outcomes or events.
---

# General Context integration

> This document lives at https://generalcontextlabs.com/docs/agents.md and in the https://github.com/GeneralContext/sdk repo (`skills/generalcontext/SKILL.md`). It is written for coding agents: everything needed to connect an app to General Context, in one page. It also works as a skill file: install it into the codebase as `.claude/skills/generalcontext/SKILL.md` (or your agent's equivalent) to keep it available offline:
>
> ```bash
> mkdir -p .claude/skills/generalcontext && curl -fsSL https://generalcontextlabs.com/docs/agents.md -o .claude/skills/generalcontext/SKILL.md
> ```

General Context turns an AI agent's production workflows into RL and eval tasks. The SDK records every OpenAI (including Azure), Anthropic (including Bedrock, Vertex and beta clients), Google Gemini and AWS Bedrock Converse model call your process makes (the request, the reply, tool calls and results, timing, usage) and sends it in the background; General Context rebuilds each **conversation** from those calls. General Context reads conversations against the company's policies, repeated failures become **issues**, and each confirmed issue becomes a **task** built from the conversation, as can a conversation the agent handled well. A task has an environment and a grader: by default, an environment that answers with what the agent's tools returned plus a simulated person, and a rubric grader. The tasks together are the company's benchmark. Recording quality directly determines task quality: a tool result the agent never passed back to a model is not recorded, and a conversation split across ids, or missing the calls a worker made, cannot be judged or rebuilt.

## Prerequisites

- A General Context API key (`gck_...`), created on the workspace's **API keys** page (https://generalcontextlabs.com/platform). The raw key is shown **once** at creation. If you don't have it, ask the user to create one; never guess or reuse another workspace's key.
- Store it in the app's environment as `GENERALCONTEXT_API_KEY`; never hardcode it or commit it. The SDK reads it from the environment, so `initialize()` needs no key argument.
- Do not change the app's prompts, model calls or behavior. Recording must never block or fail a request; the SDK guarantees that as long as you only add the calls below.

## First, inventory the LLM call sites

Real codebases hide LLM calls outside the obvious client factory, and a missed call site produces no error, just a conversation with steps missing. Before wiring anything, enumerate every place the code talks to a model: official SDK clients (including ones pointed at OpenRouter, Azure OpenAI, vLLM or a proxy: they are recorded regardless of base URL, because the SDK patches the client, not the destination), raw HTTP calls, one-off side calls (titling, classification, summaries), and background workers, cron jobs and scripts (separate processes; each needs its own `initialize()`). Greps that find most: `OpenAI(`, `Anthropic(`, `chat.completions`, `responses.create`, `messages.create`, `chat/completions`, `/v1/messages`, `api.openai.com`, `api.anthropic.com`, `openrouter.ai`, `google.genai`, `@google/genai`, `generate_content`, `generateContent`, `bedrock-runtime`, `@aws-sdk/client-bedrock-runtime`, `converse(`, `ConverseCommand`, `ConverseStreamCommand`. Know the capture boundary: the official `openai` client (`chat.completions` and `responses`), `@anthropic-ai/sdk` / `anthropic` (`messages`), Google Gen AI `google-genai` / `@google/genai` (`models.generate_content` / `generateContent` and their streaming forms) and Bedrock Converse through `boto3` (`converse`, `converse_stream`) / `@aws-sdk/client-bedrock-runtime` (`ConverseCommand`, `ConverseStreamCommand`) are always recorded, sync or async, streamed or not. Bedrock `invoke_model` / `InvokeModelCommand` is not: switch it to Converse, or send it through OpenTelemetry. Voice: OpenAI Realtime connections (Python `client.realtime.connect()`, Node `OpenAIRealtimeWS` / `OpenAIRealtimeWebSocket`) are recorded too, one call per spoken reply, never the audio. Open the connection inside the session for the phone call, and turn on input audio transcription in the session or the caller's words are missing. A voice stack that runs its own speech-to-text and text-to-speech around an ordinary model call needs nothing extra: the model call is recorded, and the session ties a call's turns together (greps: `realtime.connect`, `OpenAIRealtimeWS`, `OpenAIRealtimeWebSocket`, `input_audio`, `transcription`). Raw HTTP is recorded only for `POST .../chat/completions`, and only when you opt in (`instrumentFetch: true` in Node, `instrument_http=True` in Python, with `requests` or `httpx`). Raw HTTP to Anthropic's `/v1/messages` or to `/responses`, Python `aiohttp`, LangChain or LiteLLM wrappers that bypass the official clients, and the Vercel AI SDK's own providers are NOT recorded: send those through OpenTelemetry instead (https://generalcontextlabs.com/docs/opentelemetry), or move the call to an official client.

## Node: `generalcontext`

```bash
npm install generalcontext
```

Three rules make recording reliable:

1. **`initialize()` once per process, before the agent makes model calls.** It patches the client classes, so clients created before it are recorded too, but calls made before it are not. Without `instrumentModules`, it finds `openai` and `@anthropic-ai/sdk` itself, both their CommonJS and ES module builds; the ES module ones load a moment later, so `await ready()` after `initialize()` when the first model call can come right away.
2. **Bundled apps (Next.js, esbuild, webpack, Vite SSR) must pass the client classes via `instrumentModules`.** A bundler can inline its own copy of `openai` that the auto-detection never sees; calls then vanish with no warning. In Next.js, initialize from `instrumentation.ts` and keep the SDKs external so every route shares the copy that was patched: `serverExternalPackages: ["openai", "@anthropic-ai/sdk", "generalcontext"]` in `next.config` (Next 15+).
3. **The Edge runtime is not supported.** Any route that makes model calls needs `export const runtime = "nodejs"`.

```ts
import OpenAI from "openai";
import Anthropic from "@anthropic-ai/sdk";
import { initialize } from "generalcontext";

initialize({ appName: "support-agent", instrumentModules: { openAI: OpenAI, anthropic: Anthropic } });
```

### Next.js

```ts
// instrumentation.ts, at the project root
export async function register() {
  if (process.env.NEXT_RUNTIME === "nodejs") {
    const { initialize } = await import("generalcontext");
    const { OpenAI } = await import("openai");
    const anthropic = await import("@anthropic-ai/sdk");
    initialize({ appName: "my-agent", instrumentModules: { openAI: OpenAI, anthropic } });
  }
}
```

### Sessions and named steps (required for whole conversations)

Wrap the handling of each conversation (a chat, a ticket, an SMS thread, an email chain, a call) in `withSession`, with the id the app already stores for it, so every model call in it is grouped. Optionally wrap distinct units of work in `task()` to name the step each call came from (these name steps inside the agent; they are not benchmark tasks):

```ts
import { withSession, task } from "generalcontext";

await withSession({ sessionId: conversation.id, userId: customer.id, channel: "sms" }, async () => {
  await task("triage", () => classify(message));
  await handle(message);
});
```

`channel` and `product` are free-form labels the console filters by. Without a session, calls are grouped by their first message, which every later call in the same conversation repeats; that works for a single process but breaks as soon as a conversation spans requests or workers.

### Serverless: flush before returning

Calls are sent in batches in the background. Serverless functions can freeze or exit before the batch is sent. `await flush()` at the end of any handler that makes model calls:

```ts
import { flush } from "generalcontext";

// ... model calls ...
await flush(); // never throws
return Response.json(result);
```

### Report outcomes and events (ground truth)

When the app knows whether a conversation actually worked (the order refunded, the tour booked, the customer came back angry, a person had to step in), report it. Reviewers see it next to the transcript, and a conversation that sounded fine but booked nothing is caught.

```ts
import { reportOutcome, logEvent } from "generalcontext";

await reportOutcome(conversation.id, tourBooked ? "success" : "fail", "no appointment created");
await logEvent({ sessionId: conversation.id, eventType: "order.refunded", eventId: `refund-${refund.id}`, properties: { amount: 42.5 } });
```

Both flush pending calls first, so the conversation exists before the outcome or event arrives. Both never throw and return `false` (with a warning) when not accepted. Pass a stable `eventId` so retries count once.

## Python: `generalcontext`

```bash
pip install generalcontext
```

Call `initialize()` once at process startup, before the agent makes model calls. There is no `instrument_modules`: Python has no bundler, so the installed `openai` and `anthropic` packages are always patched. Raw `requests`/`httpx` calls to `/chat/completions` are recorded only with `instrument_http=True`.

```python
import generalcontext

generalcontext.initialize(app_name="support-agent")

with generalcontext.session(conversation.id, user_id=customer.id, channel="sms"):
    with generalcontext.task("triage"):
        category = classify(message)
    reply = handle(message)

generalcontext.flush()  # before a short-lived process exits; returns True when everything was sent
generalcontext.report_outcome(conversation.id, "success" if booked else "fail", note="no appointment created")
generalcontext.log_event(conversation.id, "order.refunded", {"amount": 42.5}, event_id=f"refund-{refund.id}")
```

Both context managers follow `await`, so they work in `async def` bodies. They do not cross into a thread you start with `threading.Thread` or a thread pool: open `session(...)` again inside that work, or submit it with `contextvars.copy_context().run`. `flush()`, `report_outcome()` and `log_event()` never raise; failures log a warning at most once a minute.

**Background workers and forks:** `initialize()` is per process: call it at each worker's startup (Celery, RQ, Temporal workers, cron scripts, `multiprocessing` children). Forking servers (gunicorn, Celery prefork) are handled: the sender restarts in the child. The session context never crosses a process or queue boundary: pass the conversation id in the job payload and open `session(conversation_id)` again inside the worker; calls under the same id from both processes land in one conversation.

## Redaction and filtering

`redact=True` (`redact: true` in Node) replaces emails, phone numbers and card or SSN-like numbers with stable placeholders such as `<email:3f9a1c>` before anything leaves the process. Set `GENERALCONTEXT_REDACT_SALT` to one secret in every process so the same value gets the same placeholder everywhere; without it, placeholders differ between processes. Redaction also covers outcomes and events, so a redacted session id still matches its calls. For anything else, `before_send` / `beforeSend` receives each call before it is sent: return it (edited or not), or `None`/`null` to drop it.

## Configuration

| Node / Python | Default | Description |
|---|---|---|
| `apiKey` / `api_key` | `GENERALCONTEXT_API_KEY` | Your `gck_...` key |
| `baseUrl` / `base_url` | `GENERALCONTEXT_BASE_URL`, else `https://generalcontextlabs.com` | Must be `https://` (or `http://localhost`); any other URL is refused and nothing is sent. Redirects are never followed |
| `appName` / `app_name` | none | Your application name, sent with every batch |
| `redact` / `redact` | `false` | Replace contact and card numbers before sending; or pass your own `(text) => text` |
| `beforeSend` / `before_send` | none | Edit or drop each call before it is sent |
| `instrumentModules` (Node only) | installed `openai` and `@anthropic-ai/sdk` | Your imported client classes, e.g. `{ openAI: OpenAI, anthropic: Anthropic }`. Required in every bundled environment |
| `instrumentFetch` / `instrument_http` | `false` | Also record raw `fetch` / `requests` / `httpx` POSTs to `/chat/completions` |
| `disableBatch` / `disable_batch` | `false` | Send each call right away (local testing) |

| Environment | |
|---|---|
| `GENERALCONTEXT_API_KEY` | API key |
| `GENERALCONTEXT_BASE_URL` | Collector URL; the `baseUrl` option wins in Node |
| `GENERALCONTEXT_REDACT_SALT` | Keeps redaction placeholders stable across processes |
| `GENERALCONTEXT_DISABLED` | `1`, `true` or `yes` turns recording off without a code change |

## Verify the integration

1. Run one real conversation through the app inside a session, then `flush()`.
2. On the workspace's **Connect** page, step 4 (**Verify**) turns green within seconds of the first call, and the conversation appears under **Conversations**, grouped under the session id you passed.
3. Open it: the timeline shows the conversation's main thread, and **Every model call** lists each call in order, with side calls (a classifier, a summary, a worker) marked and named by their `task`. The agent's instructions and tools are under "What the agent was given". If a call you know happened is missing from that list, a call site was missed: go back to the inventory.
4. If the app reports outcomes, the conversation shows **Your outcome**; events show under **What happened in your systems**.

Check from the shell, with the session id you used (exit 0 when it arrived, 1 when not yet, 2 for a key or setup problem; it never prints conversation content):

```bash
generalcontext verify --session <id>        # Python (pip install generalcontext)
npx generalcontext verify --session <id>    # Node
```

Without `--session` it shows the workspace, what the key can do, and when traffic last arrived. Read errors, don't guess: `401` means the key is wrong, revoked, or from another workspace; `429` means too many requests for this key (spread out a backfill); `413` means a request over 8 MB.

## Run the benchmark in CI

Once the workspace has ready tasks, CI can run them with a key that can run the benchmark (on **API keys**, check **Run the benchmark**; send-only keys get `403`): `generalcontext benchmark pull --out benchmark.json`, then `generalcontext evals run benchmark.json --agent my_agent:handle --trials 3 --upload`. `--upload` sends the `generalcontext.runs/v1` results to your workspace, so each agent's scores show on **Tasks**.

Each task names an environment kind and a grader kind. `replay` and `rubric` are built in. A task of another kind (for example a coding or game-building agent's own workspace and play check) runs only with the company's code named on the command line: `--environment module:function` and `--grader module:function`; such a task is skipped with the reason otherwise. Before spending on a model, `generalcontext evals check benchmark.json` grades each task's reference runs (the production run, which must fail a task built from a failure and pass one built from a success) and exits 1 if a grader disagrees.

## Common issues

- **No conversations at all**: `GENERALCONTEXT_API_KEY` is unset in that process (the SDK warns `no API key` at startup), `GENERALCONTEXT_DISABLED` is set, `baseUrl` is not `https://` (warns `is not https; nothing will be sent`), or `initialize()` was never called in the process making the calls.
- **Calls missing in a bundled Node app** (Next.js, esbuild, Vite): the bundle has its own copy of `openai` that was not patched. Pass `instrumentModules` with the classes the app imports, and keep `openai`, `@anthropic-ai/sdk` and `generalcontext` external (`serverExternalPackages`). Both are needed.
- **Works locally, nothing from production serverless**: the function exited before the batch was sent. `await flush()` before returning.
- **One conversation shows up as several**: calls were made outside a session, or different processes used different ids. Wrap each conversation's handling in `withSession` / `session` with the id the app already stores, and pass that id into workers.
- **Worker or cron calls missing**: `initialize()` is per process; the worker never called it, so its calls were never recorded and `flush()` there does nothing.
- **Calls from a thread pool arrive without their conversation** (Python): the session did not follow into the thread. Open `session(...)` inside the thread's work, or run it with `contextvars.copy_context().run`.
- **Some steps of a conversation are missing**: those calls go through raw HTTP to `/v1/messages` or `/responses`, `aiohttp`, LangChain/LiteLLM, or the Vercel AI SDK's providers. Move them to the official client, turn on `instrumentFetch` / `instrument_http` for raw `/chat/completions`, or send them with OpenTelemetry.
- **Edge route records nothing**: the Edge runtime is not supported. Add `export const runtime = "nodejs"`.
- **`reportOutcome` / `report_outcome` returns false**: General Context has not seen the conversation yet. It flushes first; if the conversation's first call came from another process, report after that process has flushed, or later from a webhook or job.
- **Events counted twice**: no `eventId` / `event_id`, so each retry is a new event. Pass a stable id.
- **Redaction placeholders differ for the same value**: `GENERALCONTEXT_REDACT_SALT` is not set to the same secret in every process.
- **Behind a corporate proxy** (Node): Node's `fetch` ignores `HTTPS_PROXY`. On Node 24+ set `NODE_USE_ENV_PROXY=1`; otherwise install a global dispatcher. If the proxy inspects TLS, set `NODE_EXTRA_CA_CERTS`.
- **A warning about a dropped call**: a single call over 4 MB, or more than 5,000 calls (64 MB) waiting to send, is dropped rather than slowing the app. Check the app is not stuck in a loop, or that the collector is reachable.

Full docs: https://generalcontextlabs.com/docs
