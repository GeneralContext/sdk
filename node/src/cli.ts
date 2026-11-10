#!/usr/bin/env node
// Copyright 2026 General Context Labs, Inc.
// SPDX-License-Identifier: Apache-2.0

// generalcontext verify [--session ID] [--json]
// generalcontext benchmark pull [--out FILE]
// Exit 0 on success, 1 on a failed check, 2 on a usage or configuration error. No runtime dependencies.

import { writeFileSync } from 'node:fs';

const DEFAULT_BASE_URL = 'https://generalcontextlabs.com';
const TIMEOUT_MS = 30_000;
const USAGE = `Usage:
  generalcontext verify [--session ID] [--json]
  generalcontext benchmark pull [--out FILE]
Uses GENERALCONTEXT_API_KEY and GENERALCONTEXT_BASE_URL.`;
const PYTHON_HINT = 'Run the benchmark with Python: pip install generalcontext && generalcontext evals run ...';

class Exit extends Error { constructor(public code: number, message: string) { super(message); } }

const trimSlashes = (url: string) => { let end = url.length; while (end > 0 && url[end - 1] === '/') end--; return url.slice(0, end); };

// Same rule as the SDK: https, or plain http to this machine only.
function secureUrl(url: string): boolean {
  try {
    const u = new URL(url);
    return u.protocol === 'https:' || (u.protocol === 'http:' && ['localhost', '127.0.0.1', '[::1]'].includes(u.hostname));
  } catch { return false; }
}

function settings() {
  const key = (process.env.GENERALCONTEXT_API_KEY ?? '').trim();
  if (!key) throw new Exit(2, 'GENERALCONTEXT_API_KEY is not set. Create a key on the API keys page and export it.');
  const base = trimSlashes((process.env.GENERALCONTEXT_BASE_URL ?? '').trim() || DEFAULT_BASE_URL);
  if (!secureUrl(base)) throw new Exit(2, `GENERALCONTEXT_BASE_URL ${base} is not https. Use https, or http://localhost for local development.`);
  return { key, base };
}

async function get(path: string): Promise<{ status: number; body: any }> {
  const { key, base } = settings();
  let response: Response;
  try {
    response = await fetch(`${base}/api/platform-collect${path}`, {
      headers: { Authorization: `Bearer ${key}`, Accept: 'application/json' },
      redirect: 'error', // never forward the key to another host
      signal: AbortSignal.timeout(TIMEOUT_MS),
    });
  } catch (error) {
    throw new Exit(1, `Could not reach ${base}: ${(error as Error).message}`);
  }
  const text = await response.text();
  let body: any = null;
  try { body = text ? JSON.parse(text) : null; } catch { /* not JSON */ }
  return { status: response.status, body };
}

const errorText = (status: number, body: any) => typeof body?.error === 'string' ? body.error : `HTTP ${status}`;

function failed(status: number, body: any): never {
  if (status === 401) throw new Exit(1, 'The API key was not accepted. Check GENERALCONTEXT_API_KEY.');
  throw new Exit(1, `Request failed: ${errorText(status, body)}`);
}

function parseFlags(args: string[], known: Record<string, 'value' | 'bool'>): Record<string, string | boolean> {
  const out: Record<string, string | boolean> = {};
  for (let i = 0; i < args.length; i++) {
    let arg = args[i], value: string | undefined;
    const eq = arg.indexOf('=');
    if (arg.startsWith('--') && eq > 0) { value = arg.slice(eq + 1); arg = arg.slice(0, eq); }
    const kind = known[arg];
    if (!kind) throw new Exit(2, `Unknown argument ${arg}.\n${USAGE}`);
    if (kind === 'bool') { out[arg] = true; continue; }
    value ??= args[++i];
    if (value === undefined || value === '') throw new Exit(2, `${arg} needs a value.\n${USAGE}`);
    out[arg] = value;
  }
  return out;
}

const scopeWords: Record<string, string> = { ingest: 'sends traffic', evals: 'runs the benchmark' };

async function verify(args: string[]): Promise<number> {
  const flags = parseFlags(args, { '--session': 'value', '--json': 'bool' });
  const session = flags['--session'] as string | undefined;
  const { status, body } = await get(`/status${session ? `?session=${encodeURIComponent(session)}` : ''}`);
  if (status !== 200 || !body) failed(status, body);
  const found = session ? Boolean(body.session?.found) : true;
  if (flags['--json']) {
    console.log(JSON.stringify(body, null, 2));
    return found ? 0 : 1;
  }
  const scopes: string[] = Array.isArray(body.key?.scopes) ? body.key.scopes : [];
  const label = body.key?.label ? ` "${body.key.label}"` : '';
  console.log(`Workspace ${body.workspace}. Key${label} ${scopes.length ? scopes.map(s => scopeWords[s] ?? s).join(' and ') : 'has no scopes'}.`);
  console.log(body.last_call_at ? `Last call arrived ${body.last_call_at} from ${body.last_source ?? 'unknown'}.` : 'No calls have arrived yet.');
  if (session) {
    const s = body.session ?? {};
    if (!found) { console.log(`Conversation ${session}: not found yet.`); return 1; }
    const models: string[] = Array.isArray(s.models) ? s.models : [];
    console.log(`Conversation ${session}: arrived, ${s.calls ?? 0} call${s.calls === 1 ? '' : 's'}${models.length ? ` (${models.join(', ')})` : ''}.`);
  }
  return 0;
}

async function benchmark(args: string[]): Promise<number> {
  const [sub, ...rest] = args;
  if (sub !== 'pull') throw new Exit(2, USAGE);
  const flags = parseFlags(rest, { '--out': 'value' });
  const out = (flags['--out'] as string | undefined) ?? 'benchmark.json';
  const { status, body } = await get('/benchmark');
  if (status === 403) throw new Exit(1, errorText(status, body) === `HTTP 403`
    ? 'This key cannot download the benchmark. On API keys, create a key that can run the benchmark.' : errorText(status, body));
  if (status !== 200 || !body) failed(status, body);
  try { writeFileSync(out, JSON.stringify(body, null, 2) + '\n'); } catch (error) {
    throw new Exit(1, `Could not write ${out}: ${(error as Error).message}`);
  }
  const tasks: any[] = Array.isArray(body.tasks) ? body.tasks : [];
  const splits = new Map<string, number>();
  for (const t of tasks) { const s = typeof t?.split === 'string' && t.split ? t.split : 'unsplit'; splits.set(s, (splits.get(s) ?? 0) + 1); }
  const bySplit = [...splits].map(([s, n]) => `${s} ${n}`).join(', ');
  console.log(`Wrote ${tasks.length} task${tasks.length === 1 ? '' : 's'} to ${out}${bySplit ? ` (${bySplit})` : ''}.`);
  return 0;
}

async function main(argv: string[]): Promise<number> {
  const [command, ...args] = argv;
  try {
    if (command === 'verify') return await verify(args);
    if (command === 'benchmark') return await benchmark(args);
    if (command === 'evals') { console.error(PYTHON_HINT); return 2; }
    if (command === '-h' || command === '--help' || command === 'help') { console.log(USAGE); return 0; }
    console.error(USAGE);
    return 2;
  } catch (error) {
    if (error instanceof Exit) { (error.code === 0 ? console.log : console.error)(error.message); return error.code; }
    console.error(`generalcontext: ${(error as Error)?.message ?? error}`);
    return 1;
  }
}

main(process.argv.slice(2)).then(code => { process.exitCode = code; });
