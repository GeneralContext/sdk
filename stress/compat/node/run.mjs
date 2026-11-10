#!/usr/bin/env node
// Run the layer 1 compatibility matrix for the Node SDK. One command (from the repo root):
//
//   node stress/compat/node/run.mjs            # full matrix
//   node stress/compat/node/run.mjs --quick    # latest version of each package, current Node only
//   node stress/compat/node/run.mjs --current-node   # full package matrix on the running Node, no fnm (CI)
//
// Needs `fnm` (Node 18, 20, 22, 24) and python3 (the fake provider is stress/compat/python/fake.py,
// shared with the Python harness). Builds and packs the SDK once, then installs each package version
// into its own directory under $STRESS_SCRATCH (default $JCODE_SCRATCH_DIR or the OS temp dir),
// never in the repository or the home directory. At most 4 environments run at once.
// Writes results-node.json and matrix-node.md next to this file; exits 1 on any failure.
import { spawn, spawnSync } from 'node:child_process';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const REPO = path.resolve(HERE, '../../..');
const args = process.argv.slice(2);
const quick = args.includes('--quick');
const only = (flag) => args.flatMap((a, i) => (a === flag ? [args[i + 1]] : []));
// --current-node: use the node running this script for every cell (CI sets one up per job) instead of fnm.
const CURRENT = args.includes('--current-node');
const NODES = CURRENT ? [process.versions.node.split('.')[0]] : only('--node').length ? only('--node') : ['18', '20', '22', '24'];
const nodeCmd = (node, cmd) => (CURRENT ? cmd : ['fnm', 'exec', `--using=${node}`, '--', ...cmd]);
const JOBS = 4;

// Oldest supported, two in the middle, latest (checked on npm 2026-11-09). version -> lowest Node major.
const MATRIX = {
  openai: { '4.0.0': 18, '5.23.2': 18, '6.49.0': 18, '7.54.0': 22 },
  '@anthropic-ai/sdk': { '0.20.0': 18, '0.60.0': 18, '0.137.0': 18, '1.7.0': 18 },
  '@google/genai': { '1.0.0': 18, '1.52.0': 18, '2.32.0': 20, '3.0.0': 22 },
  '@aws-sdk/client-bedrock-runtime': { '3.587.0': 18, '3.808.0': 18, '3.1000.0': 18, '3.1165.0': 18 },
};
const key = v => v.split('.').map(Number);
const cmp = (a, b) => { const x = key(a), y = key(b); for (let i = 0; i < 3; i++) if (x[i] !== y[i]) return x[i] - y[i]; return 0; };

// A chosen scratch root is used as is (CI and local runs reuse it); otherwise a private temp dir.
const ROOT = process.env.STRESS_SCRATCH || process.env.JCODE_SCRATCH_DIR;
const SCRATCH = ROOT ? path.join(ROOT, 'gc-compat-node') : fs.mkdtempSync(path.join(os.tmpdir(), 'gc-compat-node-'));
fs.mkdirSync(SCRATCH, { recursive: true, mode: 0o700 });
const NPM_ENV = { ...process.env, npm_config_cache: path.join(SCRATCH, 'npm-cache'), npm_config_audit: 'false', npm_config_fund: 'false', npm_config_update_notifier: 'false' };

function fnmExec(node, cmd, cwd, timeout = 600_000) {
  const [bin, ...rest] = nodeCmd(node, cmd);
  return spawnSync(bin, rest, { cwd, env: NPM_ENV, encoding: 'utf8', timeout });
}

// Build and pack the SDK once. No shell: the scratch path is an argument, never part of a command line.
const build = spawnSync('npm', ['run', 'build'], { cwd: path.join(REPO, 'node'), env: NPM_ENV, encoding: 'utf8' });
if (build.status) { console.error(build.stdout, build.stderr); process.exit(2); }
const pack = spawnSync('npm', ['pack', '--silent', '--pack-destination', SCRATCH], { cwd: path.join(REPO, 'node'), env: NPM_ENV, encoding: 'utf8' });
if (pack.status) { console.error(pack.stdout, pack.stderr); process.exit(2); }
const TARBALL = path.join(SCRATCH, pack.stdout.trim().split('\n').pop());

const installed = CURRENT ? new Set(NODES) : new Set(spawnSync('fnm', ['list'], { encoding: 'utf8' }).stdout.match(/v(\d+)\./g)?.map(v => v.slice(1, -1)) ?? []);

// One fake provider shared by all cells; each cell drains /__calls, so cells run one package per fake.
function startFake() {
  return new Promise((resolve, reject) => {
    const child = spawn('python3', [path.join(HERE, '..', 'python', 'fake.py')], { stdio: ['pipe', 'pipe', 'inherit'] });
    child.stdout.once('data', d => resolve({ url: d.toString().trim(), stop: () => child.kill() }));
    child.once('error', reject);
  });
}

async function cell(pkg, version, node) {
  if (!installed.has(node)) return { status: 'runtime', detail: `Node ${node} is not installed (fnm install ${node})` };
  const dir = path.join(SCRATCH, `${pkg.replace(/[@/]/g, '_')}-${version}-node${node}`);
  fs.mkdirSync(dir, { recursive: true });
  fs.writeFileSync(path.join(dir, 'package.json'), JSON.stringify({ name: 'compat-cell', private: true, type: 'module' }));
  const install = fnmExec(node, ['npm', 'install', '--no-save', '--ignore-scripts', '--no-package-lock', `${pkg}@${version}`, TARBALL], dir);
  if (install.status) return { status: 'install', detail: (install.stderr || install.stdout).slice(-800) };
  fs.copyFileSync(path.join(HERE, 'check.mjs'), path.join(dir, 'check.mjs'));
  const fake = await startFake();
  try {
    const done = await new Promise(resolve => {
      const [bin, ...rest] = nodeCmd(node, [CURRENT ? process.execPath : 'node', 'check.mjs', pkg, fake.url]);
      const child = spawn(bin, rest, { cwd: dir, env: process.env });
      let out = '';
      child.stdout.on('data', d => { out += d; });
      child.stderr.on('data', d => { out += d; });
      const timer = setTimeout(() => child.kill('SIGKILL'), 300_000);
      child.on('close', () => { clearTimeout(timer); resolve(out); });
    });
    const line = done.split('\n').find(l => l.startsWith('RESULT '));
    if (!line) return { status: 'crash', detail: done.slice(-1500) };
    const out = JSON.parse(line.slice(7));
    return out.crash ? { status: 'crash', detail: out.crash } : { status: 'ran', results: out.results };
  } finally { fake.stop(); }
}

const summary = o => {
  if (o.status !== 'ran') return [o.status, []];
  const failed = Object.entries(o.results).filter(([, r]) => r.status === 'fail').map(([n]) => n);
  return [failed.length ? 'fail' : 'pass', failed];
};

const work = [];
for (const [pkg, versions] of Object.entries(MATRIX)) {
  if (only('--package').length && !only('--package').includes(pkg)) continue;
  const chosen = quick ? [Object.keys(versions).sort(cmp).pop()] : Object.keys(versions).sort(cmp);
  for (const version of chosen) for (const node of quick && !CURRENT ? [process.versions.node.split('.')[0]] : NODES) {
    if (Number(node) >= versions[version]) work.push([pkg, version, node]);
  }
}

const rows = new Map();
let next = 0;
await Promise.all(Array.from({ length: JOBS }, async () => {
  while (next < work.length) {
    const item = work[next++];
    let outcome;
    try { outcome = await cell(...item); } catch (e) { outcome = { status: 'crash', detail: String(e?.stack ?? e) }; }
    rows.set(item.join('|'), outcome);
    const [status, failed] = summary(outcome);
    console.log(`${item[0].padStart(32)} ${item[1].padStart(9)} node${item[2]} ${status}${failed.length ? ': ' + failed.join(', ') : ''}`);
  }
}));

const lines = [`| Package | Version | ${NODES.map(n => `Node ${n}`).join(' | ')} |`, `| - | - | ${NODES.map(() => '-').join(' | ')} |`];
const notes = [];
for (const [pkg, versions] of Object.entries(MATRIX)) {
  for (const version of Object.keys(versions).sort(cmp)) {
    const cells = NODES.map(node => {
      const o = rows.get([pkg, version, node].join('|'));
      if (!o) return 'n/a';
      const [status, failed] = summary(o);
      if (status === 'pass') {
        const r = Object.values(o.results);
        const na = r.filter(x => x.status === 'n/a' || x.status === 'env').length;
        return `pass (${r.filter(x => x.status === 'pass').length}${na ? `, ${na} n/a` : ''})`;
      }
      if (status === 'fail') return `FAIL: ${failed.join(', ')}`;
      notes.push(`- ${pkg} ${version} on Node ${node}: ${status}: ${String(o.detail).slice(0, 300).replace(/\n/g, ' ')}`);
      return status.toUpperCase();
    });
    if (cells.some(c => c !== 'n/a')) lines.push(`| ${pkg} | ${version} | ${cells.join(' | ')} |`);
  }
}
const table = [...lines, ...(notes.length ? ['', ...notes] : [])].join('\n') + '\n';
const suffix = CURRENT ? `-${NODES[0]}` : '';
fs.writeFileSync(path.join(HERE, `results-node${suffix}.json`), JSON.stringify([...rows].map(([k, o]) => { const [p, v, n] = k.split('|'); return { package: p, version: v, node: n, ...o }; }), null, 1));
fs.writeFileSync(path.join(HERE, `matrix-node${suffix}.md`), table);
console.log(table);
process.exit([...rows.values()].some(o => summary(o)[0] !== 'pass') ? 1 : 0);
