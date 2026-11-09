# Security

This SDK runs inside your production processes, so we take reports seriously.

**Please don't open a public issue for a vulnerability.** Email
[security@generalcontextlabs.com](mailto:security@generalcontextlabs.com) with a description,
the affected version, and steps to reproduce. You'll hear back within two business days, and
we'll tell you when a fix is released.

Supported versions: the latest minor release of each package receives security fixes.

What the SDK does, for your own review:

- It patches the `openai` and `anthropic` client classes (and, unless disabled, `fetch`,
  `httpx` and `requests` calls to `/chat/completions`) to copy each request and response.
- Copies are queued in memory (bounded) and sent over HTTPS to the configured `baseUrl` with
  your API key. Nothing is written to disk.
- With redaction on, matching values are replaced before anything is queued.
- It has no runtime dependencies, opens no listening ports, and never raises into your code.
