// One Layer 2 scenario per process (initialize() is once per process). Run by run.mjs.
//
//   node worker.mjs SCENARIO [ARG]
//
// Uses the built SDK (node/dist) and the openai / @anthropic-ai/sdk packages installed in node/.
// GC_PROVIDER is the fake provider, GC_COLLECTOR the fake collector. Prints "RESULT {...}".
// Never contacts a real model API.
import { createRequire } from 'node:module';
import { fileURLToPath, pathToFileURL } from 'node:url';
import path from 'node:path';
import { performance } from 'node:perf_hooks';

const here = path.dirname(fileURLToPath(import.meta.url));
const nodeDir = path.resolve(here, '../../../node');
const req = createRequire(path.join(nodeDir, 'package.json'));
const load = async name => import(pathToFileURL(req.resolve(name)).href);
const OpenAI = (await load('openai')).default;
const Anthropic = (await load('@anthropic-ai/sdk')).default;
const gc = await import(pathToFileURL(path.join(nodeDir, 'dist/index.js')).href);

const PROVIDER = process.env.GC_PROVIDER;
const COLLECTOR = process.env.GC_COLLECTOR;
process.env.GENERALCONTEXT_API_KEY ??= 'gck_' + 'f'.repeat(48);
process.env.GENERALCONTEXT_REDACT_SALT ??= 'stress-salt';
console.warn = () => {}; // The SDK's rate-limited warnings are expected here.

const result = fields => process.stdout.write('RESULT ' + JSON.stringify(fields) + '\n');
const control = async (p, body) => (await fetch(PROVIDER + p, body === undefined ? {} : { method: 'POST', body: JSON.stringify(body) })).json();
const init = (extra = {}) => gc.initialize({ baseUrl: COLLECTOR, appName: 'stress', instrumentModules: { openAI: OpenAI, anthropic: Anthropic }, ...extra });
const oai = (opts = {}) => new OpenAI({ apiKey: 'sk-fake', baseURL: PROVIDER + '/v1', maxRetries: 0, ...opts });
const ant = () => new Anthropic({ apiKey: 'sk-ant-fake', baseURL: PROVIDER, maxRetries: 0 });
const pct = (values, q) => { const s = [...values].sort((a, b) => a - b); return s[Math.min(s.length - 1, Math.floor(q * s.length))]; };
const ms = values => ({ p50: +pct(values, 0.5).toFixed(3), p99: +pct(values, 0.99).toFixed(3) });
const median = values => pct(values, 0.5);
const rssMb = () => +(process.memoryUsage().rss / 1048576).toFixed(1);
const sleep = t => new Promise(r => setTimeout(r, t));
const timedFlush = async (ms) => { const t = performance.now(); const r = await gc.flush(ms); return { returned: r, s: +((performance.now() - t) / 1000).toFixed(3) }; };

// A fetch that answers like the provider, in-process, so overhead and memory measure the SDK only.
function instantFetch() {
  return async (url, init) => {
    const body = JSON.parse(init.body);
    const text = String(body.messages.at(-1).content).slice(0, 200);
    return new Response(JSON.stringify({ id: 'c', object: 'chat.completion', created: 1, model: 'fake', choices: [{ index: 0, finish_reason: 'stop', message: { role: 'assistant', content: text } }], usage: { prompt_tokens: 1, completion_tokens: 1, total_tokens: 2 } }), { status: 200, headers: { 'content-type': 'application/json' } });
  };
}

async function timedCalls(client, n, tag) {
  const times = [], errors = [];
  for (let i = 0; i < n; i++) {
    const t = performance.now();
    try {
      const reply = await client.chat.completions.create({ model: 'm', messages: [{ role: 'user', content: `${tag}-${i}` }] });
      if (reply.choices[0].message.content !== `${tag}-${i}`) errors.push('wrong reply');
    } catch (error) { errors.push(String(error)); }
    times.push(performance.now() - t);
  }
  return { times, errors };
}

const scenarios = {
  async collector_mode(mode) {
    const client = oai();
    await timedCalls(client, 50, 'warm');
    const before = await timedCalls(client, 300, 'base');
    init();
    const after = await gc.withSession(`mode-${mode}`, () => timedCalls(client, 300, 'with'));
    const stream = await gc.withSession(`mode-${mode}`, () => client.chat.completions.create({ model: 'm', stream: true, messages: [{ role: 'user', content: 's' }] }));
    for await (const _ of stream) { /* drain */ }
    const flushed = await timedFlush(3000);
    const tail = await gc.withSession(`mode-${mode}`, () => timedCalls(client, 40, 'tail'));
    const errors = [...after.errors, ...tail.errors];
    result({ mode, before: ms(before.times), after: ms(after.times), error_count: errors.length, errors: errors.slice(0, 5), flush_returned: flushed.returned ?? null, flush_s: flushed.s, exit_mark: Date.now() / 1000 });
  },

  async concurrency(mode) {
    if (mode !== 'baseline') init();
    const client = oai(), aclient = ant();
    const errors = [];
    // 200 promises, each its own session, all alive at once. At most 64 requests are on the wire at a time,
    // because the fake provider is a small threaded Python server (without the cap it times out even without the SDK).
    let free = 64; const waiting = [];
    const acquire = () => free > 0 ? (free--, Promise.resolve()) : new Promise(r => waiting.push(r));
    const release = () => { const next = waiting.shift(); if (next) next(); else free++; };
    const one = async t => gc.withSession(`task-${t}`, async () => {
      for (let j = 0; j < 5; j++) {
        const tag = `task-${t}/${j}`;
        await acquire();
        try {
          let text;
          const kind = (t + j) % 4;
          if (kind === 0) text = (await client.chat.completions.create({ model: 'm', messages: [{ role: 'user', content: tag }] })).choices[0].message.content;
          else if (kind === 1) {
            text = '';
            for await (const c of await client.chat.completions.create({ model: 'm', stream: true, messages: [{ role: 'user', content: tag }] })) text += c.choices[0]?.delta?.content ?? '';
          } else if (kind === 2) text = (await aclient.messages.create({ model: 'm', max_tokens: 5, messages: [{ role: 'user', content: tag }] })).content[0].text;
          else text = await aclient.messages.stream({ model: 'm', max_tokens: 5, messages: [{ role: 'user', content: tag }] }).finalText();
          if (text !== tag) errors.push(`wrong reply ${text} for ${tag}`);
        } catch (error) { errors.push(String(error)); } finally { release(); }
        await new Promise(r => setImmediate(r));
      }
    });
    await Promise.all(Array.from({ length: 200 }, (_, t) => one(t)));
    const flushed = await timedFlush();
    result({ expected: 1000, flushed: flushed.returned ?? true, error_count: errors.length, errors: errors.slice(0, 5) });
  },

  async memory(mode) {
    const client = oai({ fetch: instantFetch() });
    for (let i = 0; i < 200; i++) await client.chat.completions.create({ model: 'm', messages: [{ role: 'user', content: `warm ${i}` }] });
    global.gc?.();
    const base = rssMb();
    if (mode !== 'baseline') { await control('/_mode', { mode: 'blackhole' }); init(); }
    let errors = 0, peak = base;
    const t0 = performance.now();
    for (let i = 0; i < 100_000; i++) {
      try { await client.chat.completions.create({ model: 'm', messages: [{ role: 'user', content: `small call ${i} ` + 'x'.repeat(200) }] }); } catch { errors++; }
      if (i % 10_000 === 0) peak = Math.max(peak, rssMb());
    }
    const afterSmall = rssMb();
    const big = 'y'.repeat(3_000_000);
    for (let i = 0; i < 200; i++) {
      try { await client.chat.completions.create({ model: 'm', messages: [{ role: 'user', content: big.slice(String(i).length) + i }] }); } catch { errors++; }
      if (i % 20 === 0) { peak = Math.max(peak, rssMb()); }
    }
    global.gc?.();
    const afterBig = rssMb();
    peak = Math.max(peak, afterBig);
    const elapsed = (performance.now() - t0) / 1000;
    if (mode === 'baseline') return result({ base_mb: base, after_small_mb: afterSmall, after_big_mb: afterBig, peak_mb: peak, growth_peak_mb: +(peak - base).toFixed(1), errors });
    await control('/_reset');
    await control('/_mode', { mode: 'healthy' });
    // Requests already black-holed time out after the SDK's 15 s send timeout; then the queue drains.
    const t1 = performance.now();
    let flushed = await gc.flush(30_000);
    for (let k = 0; k < 10 && !flushed; k++) flushed = await gc.flush(30_000);
    const recoverS = (performance.now() - t1) / 1000;
    if (process.env.GC_DEBUG) process.stderr.write(`recover flushed=${flushed} stats=${JSON.stringify(await control('/_stats'))} t=${recoverS}\n`);
    await gc.withSession('after-recovery', async () => {
      for (let i = 0; i < 50; i++) await client.chat.completions.create({ model: 'm', messages: [{ role: 'user', content: `after ${i}` }] });
    });
    const flushedAfter = await gc.flush(30_000);
    if (process.env.GC_DEBUG) process.stderr.write(`after flushed=${flushedAfter} stats=${JSON.stringify(await control('/_stats'))}\n`);
    global.gc?.();
    await sleep(500);
    global.gc?.();
    result({ base_mb: base, after_small_mb: afterSmall, after_big_mb: afterBig, peak_mb: peak, growth_peak_mb: +(peak - base).toFixed(1), recovered_flush: flushed ?? true, recover_s: +recoverS.toFixed(2), flushed_after: flushedAfter ?? true, after_recovery_mb: rssMb(), errors, elapsed_s: +elapsed.toFixed(1) });
  },

  async bad_payload(name) {
    const msg = text => [{ role: 'user', content: text }];
    let nested = {}; const root = nested;
    for (let i = 0; i < 2000; i++) { nested.n = {}; nested = nested.n; }
    const circular = { name: 'loop' }; circular.self = circular;
    const cases = {
      oversized_5mb: c => c.chat.completions.create({ model: 'm', messages: msg('big ' + 'z'.repeat(5_000_000)) }),
      deeply_nested_2000: c => c.chat.completions.create({ model: 'm', messages: msg('nested'), deep: root }),
      deeply_nested_100k: c => { let d = {}; const r = d; for (let i = 0; i < 100_000; i++) { d.n = {}; d = d.n; } return c.chat.completions.create({ model: 'm', messages: msg('nested100k'), metadata: r }); },
      binary_uint8array: c => c.chat.completions.create({ model: 'm', messages: msg('bin'), blob: new Uint8Array([0, 255, 254]) }),
      buffer: c => c.chat.completions.create({ model: 'm', messages: msg('buf'), blob: Buffer.from([0, 255]) }),
      bigint: c => c.chat.completions.create({ model: 'm', messages: msg('bigint'), seed: 10n }),
      surrogate_pairs: c => c.chat.completions.create({ model: 'm', messages: msg('emoji \u{1F600} and \u00e9') }),
      lone_surrogate: c => c.chat.completions.create({ model: 'm', messages: msg('lone \ud83d here') }),
      nan_infinity: c => c.chat.completions.create({ model: 'm', messages: msg('nan'), temperature: NaN, top_p: Infinity }),
      class_instance: c => c.chat.completions.create({ model: 'm', messages: msg('inst'), meta: new (class Unit { constructor() { this.unit = '4B'; } })(), when: new Date(0), re: /x/g, map: new Map([[1, 2]]) }),
      to_json_throws: c => c.chat.completions.create({ model: 'm', messages: msg('tojson'), meta: { toJSON() { throw new Error('nope'); } } }),
      getter_throws: c => c.chat.completions.create({ model: 'm', messages: msg('getter'), meta: Object.defineProperty({}, 'x', { enumerable: true, get() { throw new Error('getter'); } }) }),
      circular_reference: c => c.chat.completions.create({ model: 'm', messages: msg('circ'), meta: circular }),
    };
    const fn = cases[name];
    const client = oai();
    const outcome = async () => {
      const t = performance.now();
      let value;
      try { const r = await fn(client); value = ['ok', String(r.choices[0].message.content).slice(0, 40)]; } catch (error) { value = ['raised', error?.constructor?.name ?? typeof error]; }
      return [value, (performance.now() - t) / 1000];
    };
    const [baseline, baseS] = await outcome();
    await control('/_reset');
    init();
    const [got, gotS] = await gc.withSession(`bad-${name}`, outcome);
    await gc.withSession(`bad-${name}`, () => client.chat.completions.create({ model: 'm', messages: msg(`canary-${name}`) }));
    await gc.flush();
    const calls = (await control('/_calls')).filter(c => c.conversation?.id === `bad-${name}`);
    const canary = calls.some(c => JSON.stringify(c.request).includes('canary'));
    const stats = await control('/_stats');
    result({ case: name, baseline, with_sdk: got, same: JSON.stringify(baseline) === JSON.stringify(got), base_s: +baseS.toFixed(3), sdk_s: +gotS.toFixed(3), recorded: calls.length - (canary ? 1 : 0), canary_arrived: canary, collector_rejected: stats.rejected, flushed: true });
  },

  // A process that exits right after `await flush()`, with process.exit.
  async exit_after_flush() {
    init();
    const client = oai();
    await gc.withSession('exit-after-flush', async () => {
      for (let j = 0; j < 30; j++) await client.chat.completions.create({ model: 'm', messages: [{ role: 'user', content: `short/${j}` }] });
    });
    await gc.flush();
    result({ exit_mark: Date.now() / 1000 });
    process.exit(0);
  },

  // A script that just ends: the beforeExit hook is the only flush.
  async natural_exit() {
    init();
    const client = oai();
    await gc.withSession('natural-exit', async () => {
      for (let j = 0; j < 30; j++) await client.chat.completions.create({ model: 'm', messages: [{ role: 'user', content: `short/${j}` }] });
    });
    result({ exit_mark: Date.now() / 1000 });
  },

  // Serverless style: each invocation awaits flush() before returning; the runtime may freeze right after.
  async serverless() {
    init();
    const client = oai();
    const handler = async event => gc.withSession(`lambda-${event.id}`, async () => {
      const r = await client.chat.completions.create({ model: 'm', messages: [{ role: 'user', content: `invoke-${event.id}` }] });
      const t = performance.now();
      await gc.flush();
      return { body: r.choices[0].message.content, flushMs: performance.now() - t };
    });
    const flushes = [];
    for (let i = 0; i < 20; i++) {
      const out = await handler({ id: i });
      flushes.push(out.flushMs);
      const delivered = (await control('/_calls')).filter(c => c.conversation?.id === `lambda-${i}`).length;
      if (delivered !== 1) { result({ ok: false, failed_at: i, delivered }); process.exit(0); }
    }
    result({ ok: true, invocations: 20, flush_ms: ms(flushes) });
    process.exit(0);
  },

  async overhead() {
    const n = 10_000;
    const client = oai({ fetch: instantFetch() });
    const run = async () => {
      const out = [];
      for (let i = 0; i < n; i++) {
        const t = performance.now();
        await client.chat.completions.create({ model: 'm', messages: [{ role: 'user', content: `hello ${i}` }] });
        out.push(performance.now() - t);
      }
      return out;
    };
    await run();
    const base = await run();
    init();
    const withSdk = await gc.withSession('overhead', run);
    await gc.flush();
    result({ n, base: ms(base), sdk: ms(withSdk), added_p50_ms: +(median(withSdk) - median(base)).toFixed(3) });
  },

  async redaction() {
    init({ redact: true });
    const client = oai({ fetch: instantFetch() });
    const t = performance.now();
    for (let i = 0; i < 10_000; i++) {
      if (i && i % 2000 === 0) await gc.flush();
      await gc.withSession({ sessionId: `redact-${i % 50}`, userId: `owner${i}@example.com` }, () => client.chat.completions.create({ model: 'm', messages: [
        { role: 'system', content: 'Call the office at 415-555-0100.' },
        { role: 'user', content: `I am user${i}@example.com, phone (415) 555-${String(i % 10000).padStart(4, '0')} or +1 628 555 ${String(i % 10000).padStart(4, '0')}` }] }));
    }
    await gc.flush();
    result({ n: 10_000, wall_s: +((performance.now() - t) / 1000).toFixed(2) });
  },
};

const [scenario, arg] = process.argv.slice(2);
await scenarios[scenario](arg);
