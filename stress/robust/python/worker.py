"""One Layer 2 scenario per process (initialize() is once per process). Run by run.py.

    python worker.py SCENARIO [ARG]

Reads GC_PROVIDER (fake provider URL) and GC_COLLECTOR (fake collector base URL) and prints one
JSON line "RESULT {...}" with the measurements. Never talks to a real model API.
"""

import asyncio
import json
import logging
import multiprocessing
import os
import statistics
import subprocess
import sys
import threading
import time
import urllib.request

import httpx
import openai
import anthropic

import generalcontext

PROVIDER = os.environ["GC_PROVIDER"]
COLLECTOR = os.environ.get("GC_COLLECTOR", "")
CONTROL = os.environ.get("GC_CONTROL", PROVIDER)
os.environ.setdefault("GENERALCONTEXT_API_KEY", "gck_" + "f" * 48)
os.environ.setdefault("GENERALCONTEXT_REDACT_SALT", "stress-salt")
logging.basicConfig(level=logging.ERROR)


def result(**fields):
    print("RESULT " + json.dumps(fields), flush=True)


def control(path, body=None):
    data = None if body is None else json.dumps(body).encode()
    request = urllib.request.Request(CONTROL + path, data=data, headers={"content-type": "application/json"})
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read())


def oai(**kw):
    return openai.OpenAI(api_key="sk-fake", base_url=PROVIDER + "/v1", max_retries=0, **kw)


def ant(**kw):
    return anthropic.Anthropic(api_key="sk-ant-fake", base_url=PROVIDER, max_retries=0, **kw)


def pct(values, q):
    values = sorted(values)
    return values[min(len(values) - 1, int(q * len(values)))]


def ms(values):
    return {"p50": round(pct(values, 0.5) * 1000, 3), "p99": round(pct(values, 0.99) * 1000, 3)}


def init(**kw):
    generalcontext.initialize(base_url=COLLECTOR, app_name="stress", **kw)


# ---- Collector failure modes -----------------------------------------------------------------

def timed_calls(client, n, tag):
    times, errors = [], []
    for i in range(n):
        start = time.perf_counter()
        try:
            reply = client.chat.completions.create(model="m", messages=[{"role": "user", "content": f"{tag}-{i}"}])
            assert reply.choices[0].message.content == f"{tag}-{i}"
        except Exception as error:  # Anything here is an SDK fault reaching the agent.
            errors.append(repr(error))
        times.append(time.perf_counter() - start)
    return times, errors


def collector_mode(mode):
    client = oai()
    timed_calls(client, 50, "warm")
    before, _ = timed_calls(client, 300, "base")
    with generalcontext.session(f"mode-{mode}"):
        init()
        after, errors = timed_calls(client, 300, "with")
        s = client.chat.completions.create(model="m", stream=True, messages=[{"role": "user", "content": "s"}])
        "".join(c.choices[0].delta.content or "" for c in s if c.choices)
    start = time.perf_counter()
    flushed = generalcontext.flush(timeout=3)
    flush_s = time.perf_counter() - start
    # Leave calls queued, so the 2 s exit flush has work when the collector misbehaves.
    with generalcontext.session(f"mode-{mode}"):
        _, more = timed_calls(client, 40, "tail")
    errors += more
    result(mode=mode, before=ms(before), after=ms(after), errors=errors[:5], error_count=len(errors),
           flush_returned=flushed, flush_s=round(flush_s, 3), exit_mark=time.time())


# ---- Concurrency -----------------------------------------------------------------------------

def concurrency_threads():
    init()
    client, aclient = oai(), ant()
    errors = []

    def worker(t):
        sid = f"thread-{t}"
        with generalcontext.session(sid):
            for j in range(20):
                tag = f"{sid}/{j}"
                try:
                    kind = j % 4
                    if kind == 0:
                        text = client.chat.completions.create(model="m", messages=[{"role": "user", "content": tag}]).choices[0].message.content
                    elif kind == 1:
                        stream = client.chat.completions.create(model="m", stream=True, messages=[{"role": "user", "content": tag}])
                        text = "".join(c.choices[0].delta.content or "" for c in stream if c.choices)
                    elif kind == 2:
                        text = aclient.messages.create(model="m", max_tokens=5, messages=[{"role": "user", "content": tag}]).content[0].text
                    else:
                        with aclient.messages.stream(model="m", max_tokens=5, messages=[{"role": "user", "content": tag}]) as s:
                            text = "".join(s.text_stream)
                    if text != tag:
                        errors.append(f"wrong reply {text!r} for {tag}")
                except Exception as error:
                    errors.append(repr(error))

    threads = [threading.Thread(target=worker, args=(t,)) for t in range(50)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    flushed = generalcontext.flush(timeout=60)
    result(expected=50 * 20, flushed=flushed, errors=errors[:5], error_count=len(errors))


def concurrency_asyncio():
    init()

    async def main():
        client = openai.AsyncOpenAI(api_key="sk-fake", base_url=PROVIDER + "/v1", max_retries=0)
        aclient = anthropic.AsyncAnthropic(api_key="sk-ant-fake", base_url=PROVIDER, max_retries=0)
        errors = []
        gate = asyncio.Semaphore(64)  # The fake provider is a small threaded server.

        async def one(t):
            sid = f"task-{t}"
            with generalcontext.session(sid):
                for j in range(5):
                    tag = f"{sid}/{j}"
                    async with gate:
                        try:
                            kind = (t + j) % 4
                            if kind == 0:
                                text = (await client.chat.completions.create(model="m", messages=[{"role": "user", "content": tag}])).choices[0].message.content
                            elif kind == 1:
                                stream = await client.chat.completions.create(model="m", stream=True, messages=[{"role": "user", "content": tag}])
                                text = "".join([c.choices[0].delta.content or "" async for c in stream if c.choices])
                            elif kind == 2:
                                text = (await aclient.messages.create(model="m", max_tokens=5, messages=[{"role": "user", "content": tag}])).content[0].text
                            else:
                                async with aclient.messages.stream(model="m", max_tokens=5, messages=[{"role": "user", "content": tag}]) as s:
                                    text = "".join([x async for x in s.text_stream])
                            if text != tag:
                                errors.append(f"wrong reply {text!r} for {tag}")
                        except Exception as error:
                            errors.append(repr(error))
                    await asyncio.sleep(0)

        await asyncio.gather(*(one(t) for t in range(200)))
        return errors

    errors = asyncio.run(main())
    flushed = generalcontext.flush(timeout=60)
    result(expected=200 * 5, flushed=flushed, errors=errors[:5], error_count=len(errors))


# ---- Memory ----------------------------------------------------------------------------------

def rss_mb():
    out = subprocess.run(["ps", "-o", "rss=", "-p", str(os.getpid())], capture_output=True, text=True).stdout
    return round(int(out.strip()) / 1024, 1)


def mock_client(async_=False):
    def handler(request):
        body = json.loads(request.content)
        text = str(body["messages"][-1]["content"])[:200]
        return httpx.Response(200, json={"id": "c", "object": "chat.completion", "created": 1, "model": "fake",
                                         "choices": [{"index": 0, "finish_reason": "stop",
                                                      "message": {"role": "assistant", "content": text}}],
                                         "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}})

    if async_:
        async def ahandler(request):
            return handler(request)
        return openai.AsyncOpenAI(api_key="sk-fake", base_url="http://fake.invalid/v1", max_retries=0,
                                  http_client=httpx.AsyncClient(transport=httpx.MockTransport(ahandler)))
    return openai.OpenAI(api_key="sk-fake", base_url="http://fake.invalid/v1", max_retries=0,
                         http_client=httpx.Client(transport=httpx.MockTransport(handler)))


def memory(mode="sdk"):
    """mode "baseline" runs the same workload without initialize(), to separate the app's own memory."""
    client = mock_client()
    for i in range(200):
        client.chat.completions.create(model="m", messages=[{"role": "user", "content": f"warm {i}"}])
    base = rss_mb()
    if mode == "baseline":
        generalcontext._sender.bytes = 0
    else:
        control("/_mode", {"mode": "blackhole"})
        init()
    errors, peak, samples = 0, base, []
    t0 = time.perf_counter()
    for i in range(100_000):
        try:
            client.chat.completions.create(model="m", messages=[{"role": "user", "content": f"small call {i} " + "x" * 200}])
        except Exception:
            errors += 1
        if i % 10_000 == 0:
            peak = max(peak, rss_mb())
            samples.append(peak)
    after_small = rss_mb()
    big = "y" * 3_000_000
    for i in range(200):
        try:
            client.chat.completions.create(model="m", messages=[{"role": "user", "content": big[:-len(str(i))] + str(i)}])
        except Exception:
            errors += 1
        if i % 20 == 0:
            peak = max(peak, rss_mb())
    after_big = rss_mb()
    peak = max(peak, after_big)
    elapsed = time.perf_counter() - t0
    queued_bytes = generalcontext._sender.bytes
    queued_calls = generalcontext._sender.queue.qsize()
    if mode == "baseline":
        return result(base_mb=base, after_small_mb=after_small, after_big_mb=after_big, peak_mb=peak,
                      growth_peak_mb=round(peak - base, 1), errors=errors)
    control("/_reset")
    control("/_mode", {"mode": "healthy"})
    start = time.perf_counter()
    flushed = generalcontext.flush(timeout=120)
    recover_s = time.perf_counter() - start
    for i in range(50):
        with generalcontext.session("after-recovery"):
            client.chat.completions.create(model="m", messages=[{"role": "user", "content": f"after {i}"}])
    flushed_after = generalcontext.flush(timeout=60)
    import gc
    gc.collect()
    time.sleep(1)
    result(base_mb=base, after_small_mb=after_small, after_big_mb=after_big, peak_mb=peak,
           growth_peak_mb=round(peak - base, 1), queued_bytes_at_peak=queued_bytes, queued_calls_at_peak=queued_calls,
           queued_bytes_after=generalcontext._sender.bytes, recovered_flush=flushed, recover_s=round(recover_s, 2), flushed_after=flushed_after,
           after_recovery_mb=rss_mb(), errors=errors, elapsed_s=round(elapsed, 1))


# ---- Bad payloads ----------------------------------------------------------------------------

def bad_cases():
    import pydantic

    class Unit(pydantic.BaseModel):
        unit: str

    nested = leaf = {}
    for _ in range(2000):
        leaf["n"] = {}
        leaf = leaf["n"]
    circular = {"name": "loop"}
    circular["self"] = circular
    wide = {"name": "wide"}
    wide["a"] = wide
    wide["b"] = wide
    msg = lambda text: [{"role": "user", "content": text}]
    return {
        "oversized_5mb": lambda c: c.chat.completions.create(model="m", messages=msg("big " + "z" * 5_000_000)),
        "deeply_nested_2000": lambda c: c.chat.completions.create(model="m", messages=msg("nested"), extra_body={"deep": nested}),
        "bytes_in_kwargs": lambda c: c.chat.completions.create(model="m", messages=msg("bytes"), extra_headers={"x-b": "1"}, extra_query={"q": "1"}, user=b"\x00\xffraw".decode("latin-1")),
        "raw_bytes_extra_body": lambda c: c.chat.completions.create(model="m", messages=msg("rawbytes"), extra_body={"blob": b"\x00\xff\xfe"}),
        "surrogate_pairs": lambda c: c.chat.completions.create(model="m", messages=msg("emoji \U0001F600 and \u00e9 \ud83d\ude00")),
        "lone_surrogate": lambda c: c.chat.completions.create(model="m", messages=msg("lone \ud83d here")),
        "nan_infinity": lambda c: c.chat.completions.create(model="m", messages=msg("nan"), extra_body={"x": float("nan"), "y": float("inf")}),
        "pydantic_class_parse": lambda c: c.chat.completions.parse(model="m", messages=msg('{"unit":"4B"}'), response_format=Unit),
        "pydantic_instance_kwarg": lambda c: c.chat.completions.create(model="m", messages=msg("inst"), extra_body={"obj": Unit(unit="x")}),
        "circular_reference": lambda c: c.chat.completions.create(model="m", messages=msg("circ"), extra_body={"c": circular}),
        "circular_wide": lambda c: c.chat.completions.create(model="m", messages=msg("wide"), extra_body={"c": wide}),
    }


def outcome(fn, client):
    start = time.perf_counter()
    try:
        reply = fn(client)
        value = ("ok", reply.choices[0].message.content[:40])
    except Exception as error:
        value = ("raised", type(error).__name__)
    return value, time.perf_counter() - start


def bad_payload(name):
    """One case per process: a hang in one case must not hide the others."""
    fn = bad_cases()[name]
    client = oai()
    baseline, base_s = outcome(fn, client)
    control("/_reset")
    init()
    with generalcontext.session(f"bad-{name}"):
        got, got_s = outcome(fn, client)
        # A normal call right after, in the same batch: it must arrive.
        client.chat.completions.create(model="m", messages=[{"role": "user", "content": f"canary-{name}"}])
    flushed = generalcontext.flush(timeout=20)
    calls = control("/_calls")
    stats = control("/_stats")
    recorded = [c for c in calls if (c.get("conversation") or {}).get("id") == f"bad-{name}"]
    canary = any("canary" in json.dumps(c.get("request")) for c in recorded)
    result(case=name, baseline=baseline, with_sdk=got, same=baseline == got, base_s=round(base_s, 3),
           sdk_s=round(got_s, 3), recorded=len(recorded) - (1 if canary else 0), canary_arrived=canary,
           collector_rejected=stats["rejected"], flushed=flushed)


# ---- Process boundaries --------------------------------------------------------------------

def _fork_child(i):
    client = oai()
    with generalcontext.session(f"fork-{i}"):
        for j in range(10):
            client.chat.completions.create(model="m", messages=[{"role": "user", "content": f"fork-{i}/{j}"}])
    ok = generalcontext.flush(timeout=20)
    sys.exit(0 if ok else 3)


def fork():
    init()
    client = oai()
    with generalcontext.session("fork-parent"):
        client.chat.completions.create(model="m", messages=[{"role": "user", "content": "parent"}])
    # Fork while the parent's sender thread is alive and possibly mid-send.
    ctx = multiprocessing.get_context("fork")
    procs = [ctx.Process(target=_fork_child, args=(i,)) for i in range(4)]
    for p in procs:
        p.start()
    for p in procs:
        p.join(60)
    with generalcontext.session("fork-parent"):
        client.chat.completions.create(model="m", messages=[{"role": "user", "content": "parent-after"}])
    generalcontext.flush(timeout=20)
    result(exitcodes=[p.exitcode for p in procs])


def short_script(explicit_flush):
    init()
    client = oai()
    with generalcontext.session(f"short-{explicit_flush}"):
        for j in range(30):
            client.chat.completions.create(model="m", messages=[{"role": "user", "content": f"short/{j}"}])
    flushed = generalcontext.flush() if explicit_flush == "flush" else None
    result(flushed=flushed, exit_mark=time.time())


# ---- Overhead --------------------------------------------------------------------------------

def overhead():
    n = 10_000
    sync_client, async_client = mock_client(), mock_client(async_=True)

    def run_sync():
        out = []
        for i in range(n):
            start = time.perf_counter()
            sync_client.chat.completions.create(model="m", messages=[{"role": "user", "content": f"hello {i}"}])
            out.append(time.perf_counter() - start)
        return out

    async def run_async():
        out = []
        for i in range(n):
            start = time.perf_counter()
            await async_client.chat.completions.create(model="m", messages=[{"role": "user", "content": f"hello {i}"}])
            out.append(time.perf_counter() - start)
        return out

    def both():
        return run_sync(), asyncio.run(run_async())

    both()  # warm up
    base_sync, base_async = both()
    init()
    with generalcontext.session("overhead"):
        sdk_sync, sdk_async = both()
    flushed = generalcontext.flush(timeout=60)
    result(n=n, base_sync=ms(base_sync), sdk_sync=ms(sdk_sync), base_async=ms(base_async), sdk_async=ms(sdk_async),
           added_sync_p50_ms=round((statistics.median(sdk_sync) - statistics.median(base_sync)) * 1000, 3),
           added_async_p50_ms=round((statistics.median(sdk_async) - statistics.median(base_async)) * 1000, 3),
           flushed=flushed)


# ---- Redaction -------------------------------------------------------------------------------

def redaction():
    init(redact=True)
    client = mock_client()
    start = time.perf_counter()
    for i in range(10_000):
        if i and i % 2000 == 0:
            generalcontext.flush(timeout=60)  # Stay under the 5000-call queue cap; we want every call scanned.
        with generalcontext.session(f"redact-{i % 50}", user_id=f"owner{i}@example.com"):
            client.chat.completions.create(model="m", messages=[
                {"role": "system", "content": "Call the office at 415-555-0100."},
                {"role": "user", "content": f"I am user{i}@example.com, phone (415) 555-{i % 10000:04d} or +1 628 555 {i % 10000:04d}"}])
    flushed = generalcontext.flush(timeout=120)
    result(n=10_000, flushed=flushed, wall_s=round(time.perf_counter() - start, 2))


if __name__ == "__main__":
    scenario, arg = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else None)
    fn = globals()[scenario]
    fn(arg) if arg is not None else fn()
