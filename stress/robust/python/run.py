"""Layer 2 robustness stress test for the Python SDK: the customer's app is never hurt.

    python stress/robust/python/run.py            # from the repo root, needs openai anthropic httpx pydantic
    python stress/robust/python/run.py --quick    # fewer, shorter scenarios (smoke)

Starts the local fakes (stress/robust/fakes.py), runs every scenario in its own process
(worker.py), checks the results and prints a markdown table. Exit code 1 if any check fails.
Only local fake servers are contacted; no real model API, no real collector.
"""

import json
import os
import subprocess
import sys
import tempfile
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
FAKES = os.path.join(HERE, "..", "fakes.py")
QUICK = "--quick" in sys.argv
ROWS = []


def control(base, path, body=None):
    data = None if body is None else json.dumps(body).encode()
    request = urllib.request.Request(base + path, data=data, headers={"content-type": "application/json"})
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read())


def run_worker(env, *args, timeout=600):
    start = time.time()
    proc = subprocess.run([sys.executable, os.path.join(HERE, "worker.py"), *args], env=env, capture_output=True,
                          text=True, timeout=timeout)
    end = time.time()
    lines = [line for line in proc.stdout.splitlines() if line.startswith("RESULT ")]
    data = json.loads(lines[-1][7:]) if lines else {}
    data["_returncode"], data["_wall_s"], data["_end"] = proc.returncode, round(end - start, 2), end
    data["_stderr"] = proc.stderr[-1500:]
    return data


def row(scenario, ok, numbers):
    ROWS.append((scenario, "pass" if ok else "FAIL", numbers))
    print(f"[{'pass' if ok else 'FAIL'}] {scenario}: {numbers}", flush=True)


def main():
    portfile = tempfile.mktemp(suffix=".json")
    fakes = subprocess.Popen([sys.executable, FAKES, portfile], stdin=subprocess.PIPE)
    try:
        for _ in range(100):
            if os.path.exists(portfile) and os.path.getsize(portfile):
                break
            time.sleep(0.05)
        ports = json.load(open(portfile))
        provider = f"http://127.0.0.1:{ports['provider']}"
        collector = f"http://127.0.0.1:{ports['collector']}"
        env = dict(os.environ, GC_PROVIDER=provider, GC_COLLECTOR=collector,
                   PYTHONPATH=os.path.join(ROOT, "python") + os.pathsep + os.environ.get("PYTHONPATH", ""))
        env.pop("GENERALCONTEXT_DISABLED", None)
        scenarios(env, provider, collector, ports)
    finally:
        fakes.stdin.close()
        try:
            fakes.wait(5)
        except subprocess.TimeoutExpired:
            fakes.kill()
        if os.path.exists(portfile):
            os.unlink(portfile)
    print("\n| Scenario | Result | Numbers |\n|---|---|---|")
    for scenario, status, numbers in ROWS:
        print(f"| {scenario} | {status} | {numbers} |")
    out = os.environ.get("GC_RESULTS_JSON")
    if out:
        json.dump(ROWS, open(out, "w"), indent=1)
    return 0 if all(status == "pass" for _, status, _ in ROWS) else 1


def calls_by_session(provider):
    out = {}
    for call in control(provider, "/_calls"):
        out.setdefault((call.get("conversation") or {}).get("id"), []).append(call)
    return out


def request_text(call):
    messages = (call.get("request") or {}).get("messages") or [{}]
    return str(messages[-1].get("content"))


def scenarios(env, provider, collector, ports):
    # ---- Collector failure modes
    modes = ["dead", "blackhole", "slow", "500", "429", "401", "garbage"]
    for mode in modes:
        control(provider, "/_reset")
        mode_env = dict(env)
        if mode == "dead":
            mode_env["GC_COLLECTOR"] = f"http://127.0.0.1:{ports['dead']}"
            control(provider, "/_mode", {"mode": "healthy"})
        else:
            control(provider, "/_mode", {"mode": mode})
        r = run_worker(mode_env, "collector_mode", mode, timeout=120)
        exit_delay = round(r["_end"] - r.get("exit_mark", r["_end"]), 2)
        added = round(r["after"]["p50"] - r["before"]["p50"], 3) if "after" in r else None
        ok = (r["_returncode"] == 0 and r.get("error_count") == 0 and r["flush_s"] <= 3.5
              and exit_delay <= 2.6 and added is not None and added < 1.0)
        row(f"collector {mode}", ok, f"model p50 {r.get('before', {}).get('p50')} -> {r.get('after', {}).get('p50')} ms "
            f"(p99 {r.get('before', {}).get('p99')} -> {r.get('after', {}).get('p99')}), agent errors {r.get('error_count')}, "
            f"flush(3) returned {r.get('flush_returned')} in {r.get('flush_s')} s, exit after last call {exit_delay} s")
    control(provider, "/_mode", {"mode": "healthy"})

    # ---- Concurrency
    for scenario, prefix in (("concurrency_threads", "thread-"), ("concurrency_asyncio", "task-")):
        control(provider, "/_reset")
        r = run_worker(env, scenario, timeout=300)
        sessions = calls_by_session(provider)
        mine = {k: v for k, v in sessions.items() if k and k.startswith(prefix)}
        total = sum(len(v) for v in mine.values())
        crosstalk = sum(1 for sid, calls in mine.items() for c in calls if not request_text(c).startswith(sid + "/"))
        texts = [request_text(c) for v in mine.values() for c in v]
        dupes = len(texts) - len(set(texts))
        unreplied = sum(1 for v in mine.values() for c in v if not c.get("response") or c.get("error"))
        ok = (r["_returncode"] == 0 and r.get("error_count") == 0 and r.get("flushed") and total == r.get("expected")
              and crosstalk == 0 and dupes == 0 and unreplied == 0)
        row(scenario.replace("_", " "), ok, f"{len(mine)} sessions, {total}/{r.get('expected')} calls recorded, "
            f"cross-talk {crosstalk}, duplicates {dupes}, missing reply {unreplied}, agent errors {r.get('error_count')}, wall {r['_wall_s']} s")

    # ---- Memory
    if not QUICK:
        b = run_worker(env, "memory", "baseline", timeout=900)
        control(provider, "/_reset")
        r = run_worker(env, "memory", timeout=900)
        stats = control(provider, "/_stats")
        recovered = calls_by_session(provider).get("after-recovery", [])
        # The SDK's share is the growth beyond the same workload without the SDK. Bound: the 64 MB
        # queue plus room for one batch being encoded (4 MB) and allocator slack.
        extra = round(r.get("growth_peak_mb", 999) - b.get("growth_peak_mb", 0), 1)
        ok = (r["_returncode"] == 0 and r.get("errors") == 0 and extra < 64 * 1.5
              and r.get("queued_bytes_at_peak", 1e12) <= 64_000_000
              and r.get("recovered_flush") and r.get("flushed_after") and len(recovered) == 50
              and r.get("queued_bytes_after") == 0 and r.get("after_recovery_mb", 999) < r.get("peak_mb", 0))
        row("memory, collector black-holed", ok,
            f"same workload without SDK: peak +{b.get('growth_peak_mb')} MB. With SDK: RSS base {r.get('base_mb')} MB, "
            f"after 100k small {r.get('after_small_mb')} MB, after 200x3MB {r.get('after_big_mb')} MB, "
            f"peak +{r.get('growth_peak_mb')} MB (SDK share +{extra} MB), queue at peak {round(r.get('queued_bytes_at_peak', 0) / 1e6, 1)} MB / "
            f"{r.get('queued_calls_at_peak')} calls; after recovery: flushed {r.get('recovered_flush')} in {r.get('recover_s')} s, "
            f"{stats['calls']} calls delivered, RSS {r.get('after_recovery_mb')} MB, new calls delivered {len(recovered)}/50")

    # ---- Bad payloads
    cases = ["oversized_5mb", "deeply_nested_2000", "bytes_in_kwargs", "raw_bytes_extra_body", "surrogate_pairs",
             "lone_surrogate", "nan_infinity", "pydantic_class_parse", "pydantic_instance_kwarg",
             "circular_reference", "circular_wide"]
    for case in cases:
        control(provider, "/_reset")
        try:
            r = run_worker(env, "bad_payload", case, timeout=60)
        except subprocess.TimeoutExpired:
            row(f"bad payload {case}", False, "worker hung > 60 s")
            continue
        ok = (r["_returncode"] == 0 and r.get("same") and r.get("canary_arrived") and r.get("flushed")
              and r.get("collector_rejected") == 0 and r.get("sdk_s", 99) < max(2.0, r.get("base_s", 0) * 3))
        row(f"bad payload {case}", ok, f"agent got {r.get('with_sdk')} (without SDK {r.get('baseline')}), "
            f"recorded {r.get('recorded')}, next call delivered {r.get('canary_arrived')}, collector 400s {r.get('collector_rejected')}, "
            f"call {r.get('base_s')} -> {r.get('sdk_s')} s" + (f", stderr: {r['_stderr'][-300:]!r}" if r["_returncode"] else ""))

    # ---- Process boundaries
    control(provider, "/_reset")
    r = run_worker(env, "fork", timeout=180)
    sessions = calls_by_session(provider)
    child_counts = [len(sessions.get(f"fork-{i}", [])) for i in range(4)]
    parent = len(sessions.get("fork-parent", []))
    ok = r["_returncode"] == 0 and r.get("exitcodes") == [0, 0, 0, 0] and child_counts == [10] * 4 and parent == 2
    row("fork (multiprocessing fork)", ok, f"child exit codes {r.get('exitcodes')}, child calls {child_counts} of 10 each, parent {parent}/2")

    for mode in ("flush", "atexit"):
        control(provider, "/_reset")
        r = run_worker(env, "short_script", mode, timeout=60)
        got = len(calls_by_session(provider).get(f"short-{mode}", []))
        delay = round(r["_end"] - r.get("exit_mark", r["_end"]), 2)
        ok = r["_returncode"] == 0 and got == 30 and delay <= 2.6
        row(f"short script ({'flush() then exit' if mode == 'flush' else 'exit, atexit flush only'})", ok,
            f"{got}/30 calls delivered, exit {delay} s after last line")

    # ---- Overhead
    r = run_worker(env, "overhead", timeout=600)
    ok = r["_returncode"] == 0 and r.get("added_sync_p50_ms", 9) < 1.0 and r.get("added_async_p50_ms", 9) < 1.0
    row("overhead (10k calls, instant fake)", ok,
        f"sync p50 {r.get('base_sync', {}).get('p50')} -> {r.get('sdk_sync', {}).get('p50')} ms (+{r.get('added_sync_p50_ms')}), "
        f"p99 {r.get('base_sync', {}).get('p99')} -> {r.get('sdk_sync', {}).get('p99')}; "
        f"async p50 {r.get('base_async', {}).get('p50')} -> {r.get('sdk_async', {}).get('p50')} ms (+{r.get('added_async_p50_ms')}), "
        f"p99 {r.get('base_async', {}).get('p99')} -> {r.get('sdk_async', {}).get('p99')}")

    # ---- Redaction under load
    control(provider, "/_reset")
    r = run_worker(env, "redaction", timeout=600)
    needles = ["@example.com", "415-555-0100", "(415) 555-", "+1 628 555"] + [f"user{i}@" for i in range(0, 10_000, 997)]
    scan = control(provider, "/_scan", {"needles": needles})
    stats = control(provider, "/_stats")
    ok = r["_returncode"] == 0 and r.get("flushed") and stats["calls"] == 10_000 and not scan["hits"]
    row("redaction under load (10k calls)", ok, f"{stats['calls']}/10000 calls delivered in {scan['bodies']} batches, "
        f"raw values found {scan['hits'] or 'none'}")


if __name__ == "__main__":
    sys.exit(main())
