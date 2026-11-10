# Layer 2 stress results: the customer's app is never hurt

Run on 2026-11-09, macOS arm64 (Apple Silicon). Python 3.12 with openai 3.48.0, anthropic 2.7.0.
Node 24.14.1 with openai 7.53.0, @anthropic-ai/sdk 1.6.0. Every model provider and the
collector were local fakes (`stress/robust/fakes.py`). No real model API or General Context
endpoint was contacted.

```
python stress/robust/python/run.py      # needs: pip install openai anthropic httpx pydantic
(cd node && npm ci && npm run build) && node stress/robust/node/run.mjs
```

Both accept `--quick`, which skips the memory run. Each scenario runs in its own process. The exit
code is 1 on any FAIL. CI runs both weekly and on demand: `.github/workflows/stress.yml`, job `robust`.

## How each scenario is checked

- **Collector modes**: dead (connection refused), blackhole (accepts, never answers), slow (30 s),
  500, 429, 401, garbage (non-HTTP bytes). Each mode runs 300 timed model calls against the real
  provider SDK before and after `initialize()`. Pass needs:
  - added p50 under 1 ms and zero exceptions reaching the agent
  - `flush(3)` returns within 3.5 s
  - the process exits at most 2.6 s after its last line, with calls still queued (2 s exit flush plus interpreter teardown)
- **Concurrency**: every call carries its session id in the prompt. The fake collector checks
  that every recorded call is in the session it names (cross-talk), with no duplicates, no missing
  replies and the exact count.
- **Memory**: 100k small calls, then 200 calls of 3 MB, with the collector black-holed. The same
  workload runs without the SDK in a separate process. The "SDK share" is the RSS growth beyond
  that baseline. Then the collector recovers, the queue must drain, and 50 new calls must arrive.
- **Bad payloads**: each case runs once without the SDK and once with it. Pass needs:
  - the agent gets the identical result or exception
  - the call takes no longer than 3x the baseline (2 s floor)
  - a normal call made right after it still arrives
  - the strict-JSON fake collector rejects nothing

## Python (all pass)

| Scenario | Result | Numbers |
|---|---|---|
| collector dead | pass | model p50 0.427 -> 0.467 ms (p99 0.537 -> 0.736), agent errors 0, flush(3) returned True in 0.506 s, exit after last call 0.14 s |
| collector blackhole | pass | model p50 0.422 -> 0.449 ms (p99 0.509 -> 2.068), agent errors 0, flush(3) returned False in 3.005 s, exit after last call 2.21 s |
| collector slow | pass | model p50 0.438 -> 0.445 ms (p99 0.578 -> 0.577), agent errors 0, flush(3) returned False in 3.002 s, exit after last call 2.16 s |
| collector 500 | pass | model p50 0.431 -> 0.468 ms (p99 0.526 -> 0.66), agent errors 0, flush(3) returned True in 0.506 s, exit after last call 0.11 s |
| collector 429 | pass | model p50 0.431 -> 0.463 ms (p99 0.531 -> 0.699), agent errors 0, flush(3) returned True in 0.504 s, exit after last call 0.21 s |
| collector 401 | pass | model p50 0.444 -> 0.464 ms (p99 0.884 -> 1.013), agent errors 0, flush(3) returned True in 0.502 s, exit after last call 0.15 s |
| collector garbage | pass | model p50 0.413 -> 0.459 ms (p99 0.528 -> 0.792), agent errors 0, flush(3) returned True in 0.506 s, exit after last call 0.17 s |
| concurrency threads | pass | 50 sessions, 1000/1000 calls recorded, cross-talk 0, duplicates 0, missing reply 0, agent errors 0, wall 1.33 s |
| concurrency asyncio | pass | 200 sessions, 1000/1000 calls recorded, cross-talk 0, duplicates 0, missing reply 0, agent errors 0, wall 1.7 s |
| memory, collector black-holed | pass | same workload without SDK: peak +265.9 MB. With SDK: RSS base 97.2 MB, after 100k small 103.4 MB, after 200x3MB 372.4 MB, peak +289.4 MB (SDK share +23.5 MB), queue at peak 5.8 MB / 5000 calls; after recovery: flushed True in 2.06 s, 5050 calls delivered, RSS 295.2 MB, new calls delivered 50/50 |
| bad payload oversized_5mb | pass | agent got ['ok', 'big zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz'] (without SDK ['ok', 'big zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz']), recorded 0, next call delivered True, collector 400s 0, call 0.031 -> 0.013 s |
| bad payload deeply_nested_2000 | pass | agent got ['ok', 'nested'] (without SDK ['ok', 'nested']), recorded 1, next call delivered True, collector 400s 0, call 0.016 -> 0.001 s |
| bad payload bytes_in_kwargs | pass | agent got ['ok', 'bytes'] (without SDK ['ok', 'bytes']), recorded 1, next call delivered True, collector 400s 0, call 0.017 -> 0.001 s |
| bad payload raw_bytes_extra_body | pass | agent got ['raised', 'TypeError'] (without SDK ['raised', 'TypeError']), recorded 1, next call delivered True, collector 400s 0, call 0.013 -> 0.001 s |
| bad payload surrogate_pairs | pass | agent got ['raised', 'UnicodeEncodeError'] (without SDK ['raised', 'UnicodeEncodeError']), recorded 1, next call delivered True, collector 400s 0, call 0.013 -> 0.001 s |
| bad payload lone_surrogate | pass | agent got ['raised', 'UnicodeEncodeError'] (without SDK ['raised', 'UnicodeEncodeError']), recorded 1, next call delivered True, collector 400s 0, call 0.012 -> 0.001 s |
| bad payload nan_infinity | pass | agent got ['raised', 'ValueError'] (without SDK ['raised', 'ValueError']), recorded 1, next call delivered True, collector 400s 0, call 0.012 -> 0.001 s |
| bad payload pydantic_class_parse | pass | agent got ['ok', '{"unit":"4B"}'] (without SDK ['ok', '{"unit":"4B"}']), recorded 1, next call delivered True, collector 400s 0, call 0.02 -> 0.002 s |
| bad payload pydantic_instance_kwarg | pass | agent got ['ok', 'inst'] (without SDK ['ok', 'inst']), recorded 1, next call delivered True, collector 400s 0, call 0.015 -> 0.001 s |
| bad payload circular_reference | pass | agent got ['raised', 'ValueError'] (without SDK ['raised', 'ValueError']), recorded 1, next call delivered True, collector 400s 0, call 0.012 -> 0.001 s |
| bad payload circular_wide | pass | agent got ['raised', 'ValueError'] (without SDK ['raised', 'ValueError']), recorded 1, next call delivered True, collector 400s 0, call 0.012 -> 0.001 s |
| fork (multiprocessing fork) | pass | child exit codes [0, 0, 0, 0], child calls [10, 10, 10, 10] of 10 each, parent 2/2 |
| short script (flush() then exit) | pass | 30/30 calls delivered, exit 0.18 s after last line |
| short script (exit, atexit flush only) | pass | 30/30 calls delivered, exit 0.71 s after last line |
| overhead (10k calls, instant fake) | pass | sync p50 0.249 -> 0.273 ms (+0.025), p99 0.62 -> 0.781; async p50 0.255 -> 0.28 ms (+0.025), p99 0.614 -> 0.642 |
| redaction under load (10k calls) | pass | 10000/10000 calls delivered in 500 batches, raw values found none |

Notes:
- Memory: the 100k small calls fill the 5000-call cap first, so later 3 MB calls are dropped. The
  queue held 5.8 MB. In a separate probe with only 3 MB calls, the queue held at the 64 MB byte cap
  (60.0 MB, 20 calls). RSS dropped to about 295 MB after recovery, not to the starting point. CPython
  keeps freed arenas, and the no-SDK baseline shows the same pattern.
- Overhead is measured with an in-process httpx MockTransport, so it shows the SDK's cost alone.
  Against the fake provider over loopback (the collector rows), p50 changed by less than 0.06 ms in
  every mode.
- Several "raised" bad-payload results (lone surrogates, raw bytes, NaN, circular references) are
  errors that openai or httpx raises itself. The SDK passes them through unchanged and still records
  the failed call.

## Node (all pass, 2 known)

| Scenario | Result | Numbers |
|---|---|---|
| collector dead | pass | model p50 0.236 -> 0.212 ms (p99 0.768 -> 1.154), agent errors 0, flush(3000) returned false in 0 s, exit after last call 0.01 s |
| collector blackhole | KNOWN | model p50 0.287 -> 0.244 ms (p99 1.832 -> 2.144), agent errors 0, flush(3000) returned false in 3.001 s, exit after last call 13.88 s |
| collector slow | KNOWN | model p50 0.238 -> 0.212 ms (p99 0.858 -> 1.037), agent errors 0, flush(3000) returned false in 3 s, exit after last call 13.92 s |
| collector 500 | pass | model p50 0.233 -> 0.21 ms (p99 0.839 -> 1.159), agent errors 0, flush(3000) returned true in 0 s, exit after last call 0.01 s |
| collector 429 | pass | model p50 0.232 -> 0.204 ms (p99 0.878 -> 1.395), agent errors 0, flush(3000) returned true in 0 s, exit after last call 0 s |
| collector 401 | pass | model p50 0.232 -> 0.203 ms (p99 0.863 -> 1.328), agent errors 0, flush(3000) returned true in 0 s, exit after last call 0.01 s |
| collector garbage | pass | model p50 0.236 -> 0.212 ms (p99 0.793 -> 1.19), agent errors 0, flush(3000) returned false in 0.001 s, exit after last call 0.01 s |
| concurrency (200 promises) | pass | 200 sessions, 1000/1000 calls recorded, cross-talk 0, duplicates 0, missing reply 0, agent errors 0, wall 1.19 s |
| memory, collector black-holed | pass | same workload without SDK: peak +291.6 MB. With SDK: SDK share +19.4 MB; RSS base 98.1 MB, after 100k small 287.7 MB, after 200x3MB 294.6 MB, peak +311 MB; after recovery: drained in 11.47 s, 5070 queued calls delivered, RSS 148.1 MB, new calls delivered 50/50, calls 3.7 s |
| bad payload oversized_5mb | pass | agent got ["ok","big zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz"] (without SDK ["ok","big zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz"]), recorded 0, next call delivered true, collector 400s 0, call 0.024 -> 0.015 s |
| bad payload deeply_nested_2000 | pass | agent got ["ok","nested"] (without SDK ["ok","nested"]), recorded 1, next call delivered true, collector 400s 0, call 0.016 -> 0.003 s |
| bad payload deeply_nested_100k | pass | agent got ["raised","RangeError"] (without SDK ["raised","RangeError"]), recorded 0, next call delivered true, collector 400s 0, call 0.01 -> 0.013 s |
| bad payload binary_uint8array | pass | agent got ["ok","bin"] (without SDK ["ok","bin"]), recorded 1, next call delivered true, collector 400s 0, call 0.012 -> 0.002 s |
| bad payload buffer | pass | agent got ["ok","buf"] (without SDK ["ok","buf"]), recorded 1, next call delivered true, collector 400s 0, call 0.014 -> 0.002 s |
| bad payload bigint | pass | agent got ["raised","TypeError"] (without SDK ["raised","TypeError"]), recorded 0, next call delivered true, collector 400s 0, call 0.001 -> 0.001 s |
| bad payload surrogate_pairs | pass | agent got ["ok","emoji 😀 and é"] (without SDK ["ok","emoji 😀 and é"]), recorded 1, next call delivered true, collector 400s 0, call 0.012 -> 0.002 s |
| bad payload lone_surrogate | pass | agent got ["ok","lone \ud83d here"] (without SDK ["ok","lone \ud83d here"]), recorded 1, next call delivered true, collector 400s 0, call 0.013 -> 0.002 s |
| bad payload nan_infinity | pass | agent got ["ok","nan"] (without SDK ["ok","nan"]), recorded 1, next call delivered true, collector 400s 0, call 0.011 -> 0.002 s |
| bad payload class_instance | pass | agent got ["ok","inst"] (without SDK ["ok","inst"]), recorded 1, next call delivered true, collector 400s 0, call 0.012 -> 0.001 s |
| bad payload to_json_throws | pass | agent got ["raised","Error"] (without SDK ["raised","Error"]), recorded 0, next call delivered true, collector 400s 0, call 0.001 -> 0.001 s |
| bad payload getter_throws | pass | agent got ["raised","Error"] (without SDK ["raised","Error"]), recorded 0, next call delivered true, collector 400s 0, call 0.001 -> 0.001 s |
| bad payload circular_reference | pass | agent got ["raised","TypeError"] (without SDK ["raised","TypeError"]), recorded 0, next call delivered true, collector 400s 0, call 0.001 -> 0.001 s |
| process.exit right after await flush() | pass | 30/30 calls delivered, exit 0.01 s after last line |
| script ends, beforeExit flush only | pass | 30/30 calls delivered, exit 0.01 s after last line |
| serverless handler (await flush() before return) | pass | 20/20 invocations each delivered before the handler returned, flush p50 0.18 ms p99 1.147 ms |
| overhead (10k calls, instant in-process fetch) | pass | p50 0.018 -> 0.02 ms (+0.003), p99 0.063 -> 0.048 ms |
| redaction under load (10k calls) | pass | 10000/10000 calls delivered in 500 batches, raw values found none, 0.64 s |

Notes:
- **KNOWN, not fixed: Node exit against a collector that accepts but never answers (blackhole and
  slow).** The in-flight `fetch` holds the event loop open, so `beforeExit` never fires and the 2 s
  exit cap never starts. The process ends when the 15 s send timeout aborts the request (measured:
  13.9 s after the last line). Without the circuit-breaker fix below, it stayed alive for minutes
  (270 s measured). The script prints KNOWN and fails if the delay goes past 16 s. A full fix means
  unref'ing the collector socket, which built-in `fetch` cannot do. That needs a design decision
  (node:http with `socket.unref()`, or a shorter send timeout), so I did not make it silently.
  Connection refused, 4xx/5xx and garbage replies all exit at once.
- The concurrency run caps requests on the wire at 64 while all 200 promises stay alive. Without
  that cap, the small threaded Python fake provider timed out on 73 of 1000 calls even with no SDK
  loaded.
- Memory: RSS fell from 409 MB at peak (98 MB + 311 MB) to 148 MB after recovery, and the queue drained in 11.5 s. Most of that wait is the in-flight black-holed request reaching its 15 s timeout.

## SDK bugs found and fixed

| Commit | SDK | Bug | Fix | Regression test |
|---|---|---|---|---|
| 0955443 | Python | A request kwarg holding a dict that refers to itself twice made `_plain` walk 2^40 paths **on the agent's thread**, so the model call hung forever (bad payload `circular_wide`: worker hung > 60 s). | Cycles become `"<circular>"`. A snapshot is capped at 200k values. | `test_shared_and_circular_references_are_recorded_quickly` |
| 0955443 | Python | NaN or Infinity in a request was sent as bare `NaN`, which is not JSON. The collector rejected the whole batch, so up to 19 unrelated calls were lost with it (`nan_infinity`: next call not delivered, collector 400). | Non-finite floats become strings. Encoding uses `allow_nan=False`, so a NaN added by before_send drops only that call. | `test_nan_and_infinity_in_a_request_never_poison_the_batch` |
| b8040b6 | Node | `await flush()` had no deadline. Against a black-holed collector it waited 15 s for every queued batch in turn: **240 s** for 341 queued calls, and a serverless handler waits that long. | `flush(timeoutMs = 10_000)` resolves `true` when the queue is sent and `false` at the deadline, matching Python's `flush(timeout=10)`. README updated. | `flush(timeoutMs) returns false at the deadline against a hung collector` |
| 797ad00 | Node | With the collector down, the sender posted every queued batch back to back. That hammered a struggling collector and kept a finished script alive for minutes (270 s measured). | After a batch gets no response, the sender stops, keeps the bounded queue, and retries after 30 s, on `flush()`, or at exit. | `an unreachable collector is tried once per pause, not once per queued batch` |

Python and Node unit suites both pass after the fixes (Python 58 tests, Node 49). Each new
regression test was confirmed to fail without its fix.

## Expected behavior seen, not bugs

- **Queue caps drop calls.** A burst of more than 5000 calls in less time than the sender needs to
  post them drops the excess, as the README documents. Unpaced, the Python redaction run delivered
  5300 of 10000 calls at about 0.25 ms per call. The harness now flushes every 2000 calls so every
  call gets scanned.
- **Calls over 4 MB are dropped** (5 MB case), and the agent's call is unaffected.
- **Node, nesting 100k deep**: openai itself throws a RangeError, and the SDK records nothing.
  Nesting 2000 deep is recorded.

## Not tested, and why

- Next.js `instrumentation.ts` (optional in the spec): skipped to keep the run self-contained and
  fast in CI. The serverless-handler and exit scenarios cover the same flush paths.
- The Python fork child calls `flush()` itself. The atexit flush in a forked child that exits with
  `os._exit` cannot run, by design (README: flush before `os._exit()`).
- Node worker_threads and cluster: not in the spec.
- One machine and one runtime version each (Python 3.12, Node 24). Layer 1 covers the version
  matrix. CI runs Python 3.12 and Node 22 on Linux.
- No end-to-end run against the dev platform collector. Layer 2 needs fault injection, which only
  the fake collector can do.
