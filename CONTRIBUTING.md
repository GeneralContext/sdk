# Contributing

Thanks for helping. Issues and pull requests are welcome.

## Develop

```bash
# Python: standard library only; the provider clients are needed for the full test suite
cd python
python -m pip install openai anthropic httpx requests
python -m unittest test_sdk -v

# Node
cd node
npm ci
npm test
```

Tests run against local fake servers, so they need no API keys and no network.

## Rules the SDK keeps

- Never raise or throw into the host application, and never block a model call.
- No runtime dependencies.
- Don't change what the caller receives (including streams).
- Python and Node send the same wire format; change both together.

## Pull requests

Keep changes small and include a test. By submitting a contribution you agree it is licensed
under the Apache License 2.0, as stated in the license's section 5.
