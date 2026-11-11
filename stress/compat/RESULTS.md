# Layer 1 stress results: compatibility across real package versions and runtimes

Run on 2026-11-09 (macOS arm64) against generalcontext 0.3 (branch `stress-compat`), with the
fixes below applied. Every provider and the General Context collector were a local fake
(`python/fake.py`). No model API was called and nothing was sent to a real collector.

## What each cell checks

For each package version on each runtime, `check.py` / `check.mjs` runs every call style the
version offers twice: first without `initialize()`, then with it (clients created before
`initialize()` included). For each style:

1. **Recorded exactly once.** The fake collector gets one call per style, never 0 or 2, and
   nothing is recorded before `initialize()`.
2. **Agent result identical.** The parsed reply, streamed chunks or events, helper final
   message, and parsed structured output compare equal (as JSON) with and without the SDK.
3. **Recording complete.** The request has the user message, the model and the tools. The
   response has the reply text, the tool call and usage (4200 in / 42 out). The session tag is
   correct.
4. **Errors unchanged.** For HTTP 400, 429, 500 and a client timeout (1 s against a 2.5 s
   reply), plain, async or streamed, the agent gets the same exception type, message and
   status with and without the SDK, and exactly one call is recorded with that error and the
   full request.

Call styles covered:

- **Python openai**: chat create (sync, async, stream, async stream), `with_raw_response`,
  `chat.completions.stream` and `.parse` (sync and async, or under `beta.` where the version
  has them there), responses create (sync, async, stream, async stream), `responses.stream`
  and `.parse`, AzureOpenAI (sync and async).
- **Python anthropic**: messages create (sync, async, stream, async stream),
  `with_raw_response`, `messages.stream` (sync and async), `messages.parse`, beta create,
  stream and parse, and AnthropicBedrock and AnthropicVertex create and stream.
- **Python google-genai**: `generate_content` and `generate_content_stream` (sync and
  async), `chats.send_message`, and the Vertex client.
- **Python boto3**: `converse` and `converse_stream`.
- **Node** covers the same set for openai, @anthropic-ai/sdk, @google/genai and
  @aws-sdk/client-bedrock-runtime. Its streams are async only, and `withResponse()` stands in
  for `with_raw_response`.

`pass (N, M n/a)` means N checks passed and M did not apply to that version. A style is n/a
when the version does not have it (for example no Responses API in openai 1.0.0), or when the
package itself fails that style against the fake even without our SDK (see notes). Cells marked
n/a in the header columns are versions that do not install on that runtime.

## Python (uv venvs; Python 3.9.25, 3.10.21, 3.11.15, 3.12.14, 3.13.15)

| Package | Version | Python 3.9 | Python 3.10 | Python 3.11 | Python 3.12 | Python 3.13 |
| - | - | - | - | - | - | - |
| openai | 1.0.0 | pass (19, 10 n/a) | pass (19, 10 n/a) | pass (19, 10 n/a) | pass (19, 10 n/a) | pass (19, 10 n/a) |
| openai | 1.66.0 | pass (29) | pass (29) | pass (29) | pass (29) | pass (29) |
| openai | 2.20.0 | pass (29) | pass (29) | pass (29) | pass (29) | pass (29) |
| openai | 3.48.0 (latest) | n/a | pass (29) | pass (29) | pass (29) | pass (29) |
| anthropic | 0.20.0 | pass (19, 8 n/a) | pass (19, 8 n/a) | pass (19, 8 n/a) | pass (19, 8 n/a) | pass (19, 8 n/a) |
| anthropic | 0.60.0 | pass (25, 2 n/a) | pass (25, 2 n/a) | pass (25, 2 n/a) | pass (25, 2 n/a) | pass (25, 2 n/a) |
| anthropic | 1.8.0 | n/a | pass (27) | pass (27) | pass (27) | pass (27) |
| anthropic | 2.7.0 (latest) | n/a | pass (27) | pass (27) | pass (27) | pass (27) |
| google-genai | 1.0.0 | pass (19) | pass (19) | pass (19) | pass (19) | pass (19) |
| google-genai | 1.30.0 | pass (19) | pass (19) | pass (19) | pass (19) | pass (19) |
| google-genai | 2.15.0 | n/a | pass (19) | pass (19) | pass (19) | pass (19) |
| google-genai | 3.0.0 (latest) | n/a | n/a | pass (19) | pass (19) | pass (19) |
| boto3 | 1.34.116 | pass (10) | pass (10) | pass (10) | pass (10) | pass (10) |
| boto3 | 1.38.46 | pass (10) | pass (10) | pass (10) | pass (10) | pass (10) |
| boto3 | 1.42.97 | pass (10) | pass (10) | pass (10) | pass (10) | pass (10) |
| boto3 | 1.43.127 (latest) | n/a | pass (10) | pass (10) | pass (10) | pass (10) |

Notes:
- openai 1.0.0 and anthropic 0.20.0 do not accept `tools` (and openai 1.0.0 does not accept
  `stream_options`). On those versions the harness drops them and stops expecting tool calls.
  They also need `httpx<0.28`, as their users already pin (httpx 0.28 removed `proxies`).
  openai 1.66.0 is pinned the same way.
- anthropic 0.20.0's `messages.stream` helper fails against a reply with a `tool_use` block
  even without our SDK (an `AttributeError` inside the package). Its four stream-helper styles
  count as n/a, and the stream helper's error cases still pass.

## Node (fnm; Node 18.20.8, 20.20.2, 22.23.2, 24.14.1)

| Package | Version | Node 18 | Node 20 | Node 22 | Node 24 |
| - | - | - | - | - | - |
| openai | 4.0.0 | pass (11, 11 n/a) | pass (11, 11 n/a) | pass (11, 11 n/a) | pass (11, 11 n/a) |
| openai | 5.23.2 | pass (22) | pass (22) | pass (22) | pass (22) |
| openai | 6.49.0 | pass (22) | pass (22) | pass (22) | pass (22) |
| openai | 7.54.0 (latest, Node >= 22) | n/a | n/a | pass (22) | pass (22) |
| @anthropic-ai/sdk | 0.20.0 | pass (16, 4 n/a) | pass (16, 4 n/a) | pass (16, 4 n/a) | pass (16, 4 n/a) |
| @anthropic-ai/sdk | 0.60.0 | pass (18, 2 n/a) | pass (18, 2 n/a) | pass (18, 2 n/a) | pass (18, 2 n/a) |
| @anthropic-ai/sdk | 0.137.0 | pass (20) | pass (20) | pass (20) | pass (20) |
| @anthropic-ai/sdk | 1.7.0 (latest) | pass (20) | pass (20) | pass (20) | pass (20) |
| @google/genai | 1.0.0 | pass (11) | pass (11) | pass (11) | pass (11) |
| @google/genai | 1.52.0 | pass (11) | pass (11) | pass (11) | pass (11) |
| @google/genai | 2.32.0 (Node >= 20) | n/a | pass (11) | pass (11) | pass (11) |
| @google/genai | 3.0.0 (latest, Node >= 22) | n/a | n/a | pass (11) | pass (11) |
| @aws-sdk/client-bedrock-runtime | 3.587.0 | pass (10) | pass (10) | pass (10) | pass (10) |
| @aws-sdk/client-bedrock-runtime | 3.808.0 | pass (10) | pass (10) | pass (10) | pass (10) |
| @aws-sdk/client-bedrock-runtime | 3.1000.0 | pass (10) | pass (10) | pass (10) | pass (10) |
| @aws-sdk/client-bedrock-runtime | 3.1165.0 (latest) | pass (10) | pass (10) | pass (10) | pass (10) |

Notes:
- openai 4.0.0 has no AzureOpenAI, stream or parse helpers, or Responses API, so those are n/a.
- Newer Bedrock runtime clients default to HTTP/2. The harness passes `NodeHttpHandler` (HTTP/1.1)
  and a per-call `abortSignal` for the timeout case, the setup the unit tests already use.

## Minimum supported versions (from these results)

| Package | Minimum | Why |
| - | - | - |
| Python `openai` | 1.0.0 | Every 1.x/2.x/3.x call style present passes. Python 3.9 needs openai < 3 (3.x requires 3.10). |
| Python `anthropic` | 0.20.0 | Tested floor. `beta.messages` needs a release that has it (0.60.0 passes). |
| Python `google-genai` | 1.0.0 | First stable release. 2.x needs Python 3.10, 3.x needs 3.11. |
| Python `boto3` | 1.34.116 | First release with Bedrock `converse` (May 2024); 1.34.115 has none. |
| Node `openai` | 4.0.0 | Needs the stream fix below for streams before 4.15. 7.x needs Node 22. |
| Node `@anthropic-ai/sdk` | 0.20.0 | Tested floor. |
| Node `@google/genai` | 1.0.0 | 2.x needs Node 20, 3.x needs Node 22. |
| Node `@aws-sdk/client-bedrock-runtime` | 3.587.0 | First release with `ConverseCommand` (May 2024); 3.583.0 has none. |
| Python | 3.9 to 3.13 | |
| Node | 18 to 24 | |

## Bugs found and fixed (each with a regression test, separate commits)

1. **Python, `with_raw_response` recorded the wrapper, not the reply**
   (`d32f42f`). `client.chat.completions.with_raw_response.create(...)` (and the anthropic
   equivalent) recorded the string `"<APIResponse [200 OK] ...>"` as the response, so no reply
   text, tool calls or usage. The fix records `raw.parse()`, which the package caches, so the
   caller's own `parse()` returns the same object. Streamed and `with_streaming_response` bodies
   are left unread. Test: `test_with_raw_response_records_the_parsed_reply_not_the_wrapper`.
2. **Python, openai 1.40 to 1.92 `beta.chat.completions.parse` was not recorded at all**
   (`19ff2a4`). In those releases parse exists only under `beta.` and posts directly instead
   of calling `create`. The fix also patches `openai.resources.beta.chat.completions`. Test:
   `test_openai_beta_chat_completions_parse_is_recorded_once`, which runs where the beta
   module exists.
3. **Node, streams from openai before 4.15 recorded as `{}`** (`6a42fe4`). Those `Stream`
   objects have no `iterator` field (only `[Symbol.asyncIterator]`), so the SDK recorded the
   stream object at once, with no reply, tool calls or usage. The fix observes the instance's
   async iterator; chunks reach the caller unchanged. Test: `streams without an iterator
   field (openai before 4.15) are recorded assembled, chunks unchanged`.

## Not tested, and why

- **Python `openai` 0.x and Node `openai` 3.x.** The SDK patches the 1.x+/4.x+ client classes
  and does not record the old module-level APIs. They are outside the supported range.
- **Node `@anthropic-ai/bedrock-sdk` and `@anthropic-ai/vertex-sdk` across versions.** Both reuse
  the patched `@anthropic-ai/sdk` classes. The unit tests check this on the versions in
  `node/package.json`. The Python AnthropicBedrock and AnthropicVertex clients were run end to end.
- **Node `toolRunner`, `runTools` and beta threads across versions.** The unit tests cover them
  on current versions only. They post through `create`, which this matrix covers.
- **Python 3.14 and free-threaded builds, Node 16, Bun and Deno.** They are outside the
  supported range.
- **Linux.** This run was on macOS. The `compat` job in `.github/workflows/stress.yml` runs the
  same matrix on ubuntu-24.04, but it has not run yet (it starts on dispatch and weekly).

## Reproduce

    python3 stress/compat/python/run.py           # needs uv; ~3 min
    node stress/compat/node/run.mjs               # needs fnm with Node 18/20/22/24; ~2 min
    # one runtime only, no fnm (what CI does): node stress/compat/node/run.mjs --current-node

Environments go under `$STRESS_SCRATCH`, or the OS temp dir. At most
4 run at once. Each run writes `matrix-*.md` (committed) and `results-*.json` (ignored, also
uploaded by CI).
