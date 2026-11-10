// Runs the compiled CLI against a local fake collector. No network beyond 127.0.0.1.
import { test, before, after } from 'node:test';
import assert from 'node:assert/strict';
import http from 'node:http';
import { spawn } from 'node:child_process';
import { mkdtempSync, readFileSync, statSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';

const cli = fileURLToPath(new URL('../dist/cli.js', import.meta.url));
const seen = [];
const benchmark = { format: 'generalcontext.benchmark/v1', tasks: [{ id: 't1', split: 'train' }, { id: 't2', split: 'train' }, { id: 't3', split: 'held_out' }] };
let server, base;
before(async () => {
  server = http.createServer((req, res) => {
    seen.push({ url: req.url, auth: req.headers.authorization });
    const reply = (status, obj) => { res.writeHead(status, { 'content-type': 'application/json' }); res.end(JSON.stringify(obj)); };
    const url = new URL(req.url, 'http://x');
    const key = req.headers.authorization;
    if (key === 'Bearer bad') return reply(401, { error: 'invalid key' });
    if (url.pathname === '/api/platform-collect/status') {
      const id = url.searchParams.get('session');
      const session = id == null ? null : id === 'conv-1'
        ? { id, found: true, calls: 3, models: ['gpt-x', 'claude-x'], received_at: '2026-11-10T00:00:00Z' }
        : { id, found: false, calls: 0, models: [], received_at: null };
      return reply(200, { workspace: 'elise', key: { label: 'prod', scopes: ['ingest'] }, last_call_at: '2026-11-10T00:00:00Z', last_source: 'sdk', session });
    }
    if (url.pathname === '/api/platform-collect/benchmark') {
      if (key !== 'Bearer evals') return reply(403, { error: 'This key cannot download the benchmark. On API keys, create a key that can run the benchmark.' });
      return reply(200, benchmark);
    }
    reply(404, {});
  });
  await new Promise(r => server.listen(0, '127.0.0.1', r));
  base = `http://127.0.0.1:${server.address().port}`;
});
after(() => server.close());

function run(args, env = {}) {
  return new Promise(resolve => {
    const p = spawn(process.execPath, [cli, ...args], { env: { ...process.env, GENERALCONTEXT_API_KEY: 'k', GENERALCONTEXT_BASE_URL: base, ...env } });
    let stdout = '', stderr = '';
    p.stdout.on('data', d => { stdout += d; });
    p.stderr.on('data', d => { stderr += d; });
    p.on('close', code => resolve({ code, stdout, stderr }));
  });
}

test('dist/cli.js starts with a node shebang', () => {
  assert.match(readFileSync(cli, 'utf8'), /^#!\/usr\/bin\/env node\n/);
});

test('verify prints workspace, scopes and last call', async () => {
  const out = await run(['verify']);
  assert.equal(out.code, 0, out.stderr);
  assert.match(out.stdout, /Workspace elise\. Key "prod" sends traffic\./);
  assert.match(out.stdout, /Last call arrived 2026-11-10T00:00:00Z from sdk\./);
  assert.equal(seen.at(-1).auth, 'Bearer k');
});

test('verify --session found exits 0, not found exits 1', async () => {
  let out = await run(['verify', '--session', 'conv-1']);
  assert.equal(out.code, 0, out.stderr);
  assert.match(out.stdout, /Conversation conv-1: arrived, 3 calls \(gpt-x, claude-x\)\./);
  out = await run(['verify', '--session=nope']);
  assert.equal(out.code, 1);
  assert.match(out.stdout, /Conversation nope: not found yet\./);
  assert.equal(seen.at(-1).url, '/api/platform-collect/status?session=nope');
});

test('verify --json prints the status body', async () => {
  const out = await run(['verify', '--session', 'nope', '--json']);
  assert.equal(out.code, 1);
  assert.equal(JSON.parse(out.stdout).session.found, false);
});

test('benchmark pull writes the file and prints counts by split', async () => {
  const file = join(mkdtempSync(join(tmpdir(), 'gc-cli-')), 'b.json');
  const out = await run(['benchmark', 'pull', '--out', file], { GENERALCONTEXT_API_KEY: 'evals' });
  assert.equal(out.code, 0, out.stderr);
  assert.deepEqual(JSON.parse(readFileSync(file, 'utf8')), benchmark);
  assert.match(out.stdout, /Wrote 3 tasks to .*b\.json \(train 2, held_out 1\)\./);
});

test('benchmark pull with an ingest-only key prints the 403 message and exits 1', async () => {
  const file = join(mkdtempSync(join(tmpdir(), 'gc-cli-')), 'b.json');
  const out = await run(['benchmark', 'pull', '--out', file]);
  assert.equal(out.code, 1);
  assert.match(out.stderr, /This key cannot download the benchmark/);
  assert.throws(() => statSync(file));
});

test('a rejected key exits 1 with a clear message', async () => {
  const out = await run(['verify'], { GENERALCONTEXT_API_KEY: 'bad' });
  assert.equal(out.code, 1);
  assert.match(out.stderr, /API key was not accepted/);
});

test('configuration and usage errors exit 2', async () => {
  for (const [args, env, pattern] of [
    [['verify'], { GENERALCONTEXT_API_KEY: '' }, /GENERALCONTEXT_API_KEY is not set/],
    [['verify'], { GENERALCONTEXT_BASE_URL: 'http://collector.example.com' }, /not https/],
    [['verify', '--bogus'], {}, /Unknown argument --bogus/],
    [['verify', '--session'], {}, /--session needs a value/],
    [['benchmark'], {}, /Usage/],
    [[], {}, /Usage/],
    [['evals', 'run', 'x'], {}, /pip install generalcontext && generalcontext evals run/],
    [['improve', 'x'], {}, /Usage/],
  ]) {
    const n = seen.length;
    const out = await run(args, env);
    assert.equal(out.code, 2, `${args.join(' ')}: ${out.stderr}`);
    assert.match(out.stderr, pattern);
    assert.equal(seen.length, n, 'nothing sent');
  }
});

test('an unreachable collector exits 1', async () => {
  const out = await run(['verify'], { GENERALCONTEXT_BASE_URL: 'http://127.0.0.1:1' });
  assert.equal(out.code, 1);
  assert.match(out.stderr, /Could not reach/);
});
