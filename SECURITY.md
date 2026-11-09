# Security

This SDK runs inside your production processes. This page is written for the people who review
it before it gets there.

## Reporting a vulnerability

Please don't open a public issue. Email **research@generalcontextlabs.com** with "Security" in the
subject, the affected package and version, and steps to reproduce. We'll confirm receipt, keep
you updated, and credit you in the release notes unless you'd rather we didn't.

The latest minor release of each package receives security fixes.

## What the SDK does

- **Instrumentation.** `initialize()` wraps the `create` method of the OpenAI client (chat
  completions and responses) and the Anthropic client (messages), in this process only. The
  caller receives exactly what the client returns, including streams. Patching raw HTTP
  (`fetch`, `httpx`, `requests` calls to `/chat/completions`) is **off** unless you turn it on.
- **What is captured.** For each model call: the request your code passed (messages, tools,
  model, settings), the response, timestamps, token usage, an error message if the call failed,
  and the session and task tags you set. Headers, API keys of the model providers, environment
  variables and anything else in your process are never read.
- **Where it goes.** Batched in memory and sent over HTTPS, from a background thread (Python) or
  after the response (Node), to `https://generalcontextlabs.com/api/platform-collect/` or the
  `base_url` you set. The SDK refuses plain `http://` except to `localhost`, and does not follow
  redirects, so your General Context key is only ever sent to the host you configured.
- **Nothing is stored locally.** No files, no disk cache, no listening ports.
- **Bounded.** At most 5,000 calls and 64 MB are queued; larger calls or a fuller queue are
  dropped with a rate-limited warning. A single call over 4 MB is dropped.
- **Never in your way.** The SDK never raises or throws into your code and never blocks a model
  call. If General Context is slow or down, recording is dropped and logged; your agent is not
  affected.
- **No runtime dependencies.** Python uses the standard library only; the Node package has no
  `dependencies`.

## Controlling what leaves your process

| Control | Python | Node |
| --- | --- | --- |
| Turn the SDK off without a code change | `GENERALCONTEXT_DISABLED=1` | `GENERALCONTEXT_DISABLED=1` |
| Replace emails, phone numbers, card and SSN-like numbers with keyed placeholders | `initialize(redact=True)` | `initialize({ redact: true })` |
| Your own redaction | `redact=lambda text: ...` | `redact: text => ...` |
| Inspect, change or drop any call before it is sent | `before_send=fn` | `beforeSend: fn` |
| Record raw HTTP calls too | `instrument_http=True` | `instrumentFetch: true` |

Redaction and `before_send` run inside your process, before anything is sent. Placeholders are
keyed with `GENERALCONTEXT_REDACT_SALT` (HMAC-SHA256), so the same value reads the same across
your processes without being sent.

## Network

The SDK connects only to the configured base URL on port 443. Python honours `HTTPS_PROXY` and
`SSL_CERT_FILE`. Node's built-in `fetch` uses the system CA store plus `NODE_EXTRA_CA_CERTS`;
for an outbound proxy set `NODE_USE_ENV_PROXY=1` (Node 24+) or a global dispatcher.

## Supply chain

- Source is Apache-2.0, in this repository. Releases are built by GitHub Actions from a tagged
  commit, tested, and published from the exact files that were tested.
- Every release carries a signed build-provenance attestation. Verify a download with
  `gh attestation verify <file> --repo GeneralContext/sdk`. npm releases also carry npm
  provenance (`npm audit signatures`).
- PyPI publishing uses trusted publishing (no stored token). Publishing requires a second
  person's approval.
- Workflow actions are pinned to commit SHAs. CodeQL and Dependabot run on this repository.
