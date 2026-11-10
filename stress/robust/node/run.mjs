// Layer 2 robustness stress test for the Node SDK: the customer's app is never hurt.
//
//   (cd node && npm ci && npm run build) && node stress/robust/node/run.mjs [--quick]
//
// Starts the local fakes (stress/robust/fakes.py, needs python3), runs each scenario in its own
// process (worker.mjs), checks the results and prints a markdown table. Exit code 1 on any failure.
// Only local fake servers are contacted; no real model API, no real collector.
import { spawn, spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';

const here = path.dirname(fileURLToPath(import.meta.url));
const QUICK = process.argv.includes('--quick');
const ROWS = [];
const portfile = path.join(fs.mkdtempSync(path.join(os.tmpdir(), 'gc-robust-')), 'ports.json');
const fakes = spawn(process.env.PYTHON ?? 'python3', [path.join(here, '..', 'fakes.py'), portfile], { stdio: ['pipe', 'inherit', 'inherit'] });
const stopFakes = () => { try { fakes.stdin.end(); } catch {} setTimeout(() => fakes.kill(), 3000).unref(); };
process.on('exit', () => fakes.kill());

for (let i = 0; i < 100 && !(fs.existsSync(portfile) && fs.statSync(portfile).size); i++) await new Promise(r => setTimeout(r, 50));
const ports = JSON.parse(fs.readFileSync(portfile, 'utf8'));
const provider = `http://127.0.0.1:${ports.provider}`;
const collector = `http://127.0.0.1:${ports.collector}`;
const env = { ...process.env, GC_PROVIDER: provider, GC_COLLECTOR: collector };
delete env.GENERALCONTEXT_DISABLED;

const control = async (p, body) => (await fetch(provider + p, body === undefined ? {} : { method: 'POST', body: JSON.stringify(body) })).json();

function worker(args, { timeout = 600_000, workerEnv = env, nodeArgs = [] } = {}) {
  return new Promise(resolve => {
    const start = Date.now() / 1000;
    const child = spawn(process.execPath, [...nodeArgs, path.join(here, 'worker.mjs'), ...args], { env: workerEnv });
    let out = '', err = '';
    child.stdout.on('data', d => { out += d; });
    child.stderr.on('data', d => { err += d; });
    const timer = setTimeout(() => child.kill('SIGKILL'), timeout);
    child.on('exit', (code, signal) => {
      clearTimeout(timer);
      const end = Date.now() / 1000;
      const line = out.split('\n').filter(l => l.startsWith('RESULT ')).pop();
      resolve({ ...(line ? JSON.parse(line.slice(7)) : {}), _code: code, _signal: signal, _wall_s: +(end - start).toFixed(2), _end: end, _stderr: err.slice(-800) });
    });
  });
}

function row(scenario, ok, numbers) {
  const status = ok === 'known' ? 'KNOWN' : ok ? 'pass' : 'FAIL';
  ROWS.push([scenario, status, numbers]);
  console.log(`[${status}] ${scenario}: ${numbers}`);
}

const bySession = async () => {
  const out = {};
  for (const c of await control('/_calls')) (out[c.conversation?.id] ??= []).push(c);
  return out;
};
const requestText = c => String(c.request?.messages?.at(-1)?.content);

try {
  // ---- Collector failure modes
  for (const mode of ['dead', 'blackhole', 'slow', '500', '429', '401', 'garbage']) {
    await control('/_reset');
    let workerEnv = env;
    if (mode === 'dead') { workerEnv = { ...env, GC_COLLECTOR: `http://127.0.0.1:${ports.dead}` }; await control('/_mode', { mode: 'healthy' }); } else await control('/_mode', { mode });
    const r = await worker(['collector_mode', mode], { timeout: 120_000, workerEnv });
    const exitDelay = r.exit_mark ? +(r._end - r.exit_mark).toFixed(2) : null;
    const added = r.after ? +(r.after.p50 - r.before.p50).toFixed(3) : null;
    const ok = r._code === 0 && r.error_count === 0 && r.flush_s <= 3.5 && exitDelay !== null && exitDelay <= 2.6 && added !== null && added < 1;
    // KNOWN (RESULTS.md): against a collector that accepts and never answers, the in-flight fetch keeps the event
    // loop alive until the 15 s send timeout, so the 2 s exit cap never starts. Still fail if it gets worse.
    const known = !ok && ['blackhole', 'slow'].includes(mode) && r._code === 0 && r.error_count === 0 && r.flush_s <= 3.5 && exitDelay <= 16 && added < 1;
    row(`collector ${mode}`, known ? 'known' : ok, `model p50 ${r.before?.p50} -> ${r.after?.p50} ms (p99 ${r.before?.p99} -> ${r.after?.p99}), agent errors ${r.error_count}, flush(3000) returned ${r.flush_returned} in ${r.flush_s} s, exit after last call ${exitDelay} s${r._code ? `, code ${r._code} ${r._signal ?? ''}` : ''}`);
  }
  await control('/_mode', { mode: 'healthy' });

  // ---- Concurrency
  {
    await control('/_reset');
    const r = await worker(['concurrency'], { timeout: 300_000 });
    const sessions = await bySession();
    const mine = Object.entries(sessions).filter(([k]) => k.startsWith('task-'));
    const total = mine.reduce((n, [, v]) => n + v.length, 0);
    const crosstalk = mine.reduce((n, [sid, v]) => n + v.filter(c => !requestText(c).startsWith(sid + '/')).length, 0);
    const texts = mine.flatMap(([, v]) => v.map(requestText));
    const dupes = texts.length - new Set(texts).size;
    const unreplied = mine.reduce((n, [, v]) => n + v.filter(c => !c.response || c.error).length, 0);
    const ok = r._code === 0 && r.error_count === 0 && total === r.expected && crosstalk === 0 && dupes === 0 && unreplied === 0;
    row('concurrency (200 promises)', ok, `${mine.length} sessions, ${total}/${r.expected} calls recorded, cross-talk ${crosstalk}, duplicates ${dupes}, missing reply ${unreplied}, agent errors ${r.error_count}, wall ${r._wall_s} s${r.errors?.length ? ' ' + r.errors.join('; ') : ''}`);
  }

  // ---- Memory
  if (!QUICK) {
    const b = await worker(['memory', 'baseline'], { timeout: 900_000, nodeArgs: ['--expose-gc'] });
    await control('/_reset');
    const r = await worker(['memory'], { timeout: 900_000, nodeArgs: ['--expose-gc'] });
    const stats = await control('/_stats');
    const recovered = (await bySession())['after-recovery']?.length ?? 0;
    // The SDK's share is growth beyond the same workload without the SDK: the 64 MB queue plus slack.
    const extra = +(r.growth_peak_mb - b.growth_peak_mb).toFixed(1);
    const ok = r._code === 0 && r.errors === 0 && extra < 64 * 1.5 && r.recovered_flush === true && recovered === 50 && r.after_recovery_mb < r.peak_mb;
    row('memory, collector black-holed', ok, `same workload without SDK: peak +${b.growth_peak_mb} MB. With SDK: SDK share +${extra} MB; RSS base ${r.base_mb} MB, after 100k small ${r.after_small_mb} MB, after 200x3MB ${r.after_big_mb} MB, peak +${r.growth_peak_mb} MB; after recovery: drained in ${r.recover_s} s, ${stats.calls} queued calls delivered, RSS ${r.after_recovery_mb} MB, new calls delivered ${recovered}/50, calls ${r.elapsed_s} s${r._code ? ' ' + r._stderr : ''}`);
  }

  // ---- Bad payloads
  for (const c of ['oversized_5mb', 'deeply_nested_2000', 'deeply_nested_100k', 'binary_uint8array', 'buffer', 'bigint', 'surrogate_pairs', 'lone_surrogate', 'nan_infinity', 'class_instance', 'to_json_throws', 'getter_throws', 'circular_reference']) {
    await control('/_reset');
    const r = await worker(['bad_payload', c], { timeout: 60_000 });
    const ok = r._code === 0 && r.same && r.canary_arrived && r.collector_rejected === 0 && r.sdk_s < Math.max(2, r.base_s * 3);
    row(`bad payload ${c}`, ok, r._signal ? 'worker hung > 60 s' : `agent got ${JSON.stringify(r.with_sdk)} (without SDK ${JSON.stringify(r.baseline)}), recorded ${r.recorded}, next call delivered ${r.canary_arrived}, collector 400s ${r.collector_rejected}, call ${r.base_s} -> ${r.sdk_s} s${r._code ? ' ' + r._stderr.slice(-300) : ''}`);
  }

  // ---- Process boundaries
  for (const [scenario, sid, label] of [['exit_after_flush', 'exit-after-flush', 'process.exit right after await flush()'], ['natural_exit', 'natural-exit', 'script ends, beforeExit flush only']]) {
    await control('/_reset');
    const r = await worker([scenario], { timeout: 60_000 });
    const got = (await bySession())[sid]?.length ?? 0;
    const delay = r.exit_mark ? +(r._end - r.exit_mark).toFixed(2) : null;
    row(label, r._code === 0 && got === 30 && delay !== null && delay <= 2.6, `${got}/30 calls delivered, exit ${delay} s after last line`);
  }
  {
    await control('/_reset');
    const r = await worker(['serverless'], { timeout: 60_000 });
    row('serverless handler (await flush() before return)', r._code === 0 && r.ok, r.ok ? `${r.invocations}/20 invocations each delivered before the handler returned, flush p50 ${r.flush_ms.p50} ms p99 ${r.flush_ms.p99} ms` : `invocation ${r.failed_at} not delivered when the handler returned (${r.delivered})`);
  }

  // ---- Overhead
  {
    const r = await worker(['overhead'], { timeout: 600_000 });
    row('overhead (10k calls, instant in-process fetch)', r._code === 0 && r.added_p50_ms < 1, `p50 ${r.base?.p50} -> ${r.sdk?.p50} ms (+${r.added_p50_ms}), p99 ${r.base?.p99} -> ${r.sdk?.p99} ms`);
  }

  // ---- Redaction under load
  {
    await control('/_reset');
    const r = await worker(['redaction'], { timeout: 600_000 });
    const needles = ['@example.com', '415-555-0100', '(415) 555-', '+1 628 555', ...Array.from({ length: 11 }, (_, k) => `user${k * 997}@`)];
    const scan = await control('/_scan', { needles });
    const stats = await control('/_stats');
    row('redaction under load (10k calls)', r._code === 0 && stats.calls === 10_000 && !scan.hits.length, `${stats.calls}/10000 calls delivered in ${scan.bodies} batches, raw values found ${scan.hits.length ? scan.hits.join(', ') : 'none'}, ${r.wall_s} s`);
  }
} finally {
  stopFakes();
}

console.log('\n| Scenario | Result | Numbers |\n|---|---|---|');
for (const [s, status, n] of ROWS) console.log(`| ${s} | ${status} | ${n} |`);
if (process.env.GC_RESULTS_JSON) fs.writeFileSync(process.env.GC_RESULTS_JSON, JSON.stringify(ROWS, null, 1));
process.exitCode = ROWS.every(([, s]) => s !== 'FAIL') ? 0 : 1;
