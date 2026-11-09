// Copyright 2026 General Context Labs, Inc.
// SPDX-License-Identifier: Apache-2.0

// Record your agent's model calls for General Context, without changing how it runs.
//
//   import OpenAI from 'openai';
//   import { initialize, withSession } from '@generalcontext/sdk';
//
//   initialize({ instrumentModules: { openAI: OpenAI } });
//   await withSession(conversationId, async () => { ... });
//
// Every model call (history sent, tools offered, reply and tool calls) is queued and sent in
// the background. Your agent never waits on General Context, and a failure on our side is
// logged and dropped, never thrown. No runtime dependencies.

import { AsyncLocalStorage } from 'node:async_hooks';
import { createHmac, randomBytes, randomUUID } from 'node:crypto';
import { createRequire } from 'node:module';

export const VERSION = '0.2.0';
const BATCH = 20;
const MAX_QUEUE = 5000;

type Metadata = Record<string, string | number | boolean>;
export type Session = { sessionId: string; userId?: string; channel?: string; product?: string; metadata?: Metadata };
type Tags = { id: string; user_id?: string; channel?: string; product?: string; metadata?: Metadata };
type Api = 'openai-chat' | 'openai-responses' | 'anthropic';
type Call = {
  api: Api; conversation: Tags | null; task: string | null; request: unknown; response: unknown;
  started_at: string; ended_at: string; error: string | null;
};
type Scrub = (text: string) => string;
type Scope = { tags?: Tags; task?: string; inSdkCall?: boolean };

export type InitializeOptions = {
  apiKey?: string;
  baseUrl?: string;
  appName?: string;
  /** true for built-in email/phone/card/SSN placeholders, or your own (text) => text. */
  redact?: boolean | Scrub;
  /** SDK classes or modules to patch, e.g. { openAI: OpenAI, anthropic: Anthropic }. */
  instrumentModules?: { openAI?: unknown; anthropic?: unknown };
  /** Record raw fetch POSTs to any URL ending /chat/completions. Default true. */
  instrumentFetch?: boolean;
  /** Send each call right away instead of batching. */
  disableBatch?: boolean;
};

const config = {
  apiKey: undefined as string | undefined,
  baseUrl: 'https://generalcontextlabs.com',
  appName: undefined as string | undefined,
  scrub: null as Scrub | null,
  disableBatch: false,
};
const scope = new AsyncLocalStorage<Scope>();
const queue: Call[] = [];
let initialized = false;
let sending: Promise<void> | null = null;
let timer: ReturnType<typeof setTimeout> | null = null;
let originalFetch: typeof fetch = globalThis.fetch;

const apiKey = () => config.apiKey ?? process.env.GENERALCONTEXT_API_KEY;
const baseUrl = () => (process.env.GENERALCONTEXT_BASE_URL ?? config.baseUrl).replace(/\/+$/, '');
const warn = (message: string) => console.warn(`[generalcontext] ${message}`);

/** Configure recording once per process, before your agent makes model calls. */
export function initialize(options: InitializeOptions = {}): void {
  if (initialized) { warn('initialize was already called; ignoring the second call'); return; }
  initialized = true;
  config.apiKey = options.apiKey ?? process.env.GENERALCONTEXT_API_KEY;
  config.baseUrl = options.baseUrl ?? process.env.GENERALCONTEXT_BASE_URL ?? config.baseUrl;
  config.appName = options.appName;
  config.scrub = redactor(options.redact);
  config.disableBatch = options.disableBatch ?? false;
  if (!config.apiKey) warn('no API key; set GENERALCONTEXT_API_KEY or pass apiKey');

  const modules = options.instrumentModules ?? autoload();
  if (modules.openAI) patchOpenAI(modules.openAI);
  if (modules.anthropic) patchAnthropic(modules.anthropic);
  if (options.instrumentFetch ?? true) patchFetch();
}

// Load installed SDKs when the caller did not pass them. Bundlers can hide these; pass instrumentModules there.
function autoload(): { openAI?: unknown; anthropic?: unknown } {
  const require = createRequire(import.meta.url);
  const load = (name: string) => { try { return require(name); } catch { return undefined; } };
  return { openAI: load('openai'), anthropic: load('@anthropic-ai/sdk') };
}

// Accept a module namespace, its default export, or the client class itself.
function clientClass(value: unknown, name: string): Record<string, any> | undefined {
  const v = value as any;
  if (typeof v === 'function' && (v.Messages || v.Chat)) return v;
  return v?.[name] ?? v?.default?.[name] ?? (typeof v === 'function' ? v : typeof v?.default === 'function' ? v.default : undefined);
}

function patchOpenAI(value: unknown) {
  const OpenAI = clientClass(value, 'OpenAI');
  patchMethod(OpenAI?.Chat?.Completions?.prototype, 'create', 'openai-chat');
  patchMethod(OpenAI?.Responses?.prototype, 'create', 'openai-responses');
}

function patchAnthropic(value: unknown) {
  const Anthropic = clientClass(value, 'Anthropic');
  // messages.stream() calls messages.create({ stream: true }) internally, so it is covered too.
  patchMethod(Anthropic?.Messages?.prototype, 'create', 'anthropic');
}

const MARK = Symbol.for('generalcontext.recorded');

function patchMethod(owner: Record<string, any> | undefined, name: string, api: Api) {
  const original = owner?.[name];
  if (typeof original !== 'function' || original[MARK]) return;
  owner![name] = recording(original, api);
}

/** Tag every call made inside `run` with one session. `conversation` is the same function. */
export function withSession<T>(session: Session | string, run: () => T): T {
  const s = typeof session === 'string' ? { sessionId: session } : session;
  const tags: Tags = { id: String(s.sessionId) };
  if (s.userId) tags.user_id = s.userId;
  if (s.channel) tags.channel = s.channel;
  if (s.product) tags.product = s.product;
  if (s.metadata) tags.metadata = s.metadata;
  return scope.run({ ...scope.getStore(), tags }, run);
}

/** Older name for withSession. Accepts { id } as well as { sessionId }. */
export function conversation<T>(tags: Session | { id: string; channel?: string; product?: string; metadata?: Metadata } | string, run: () => T): T {
  if (typeof tags === 'object' && 'id' in tags) {
    const { id, ...rest } = tags;
    return withSession({ ...rest, sessionId: id }, run);
  }
  return withSession(tags, run);
}

/** Tag every call made inside `run` with a task name, such as one step of your agent loop. */
export function task<T>(name: string, run: () => T): T {
  return scope.run({ ...scope.getStore(), task: name }, run);
}

function record(call: Omit<Call, 'conversation' | 'task'>, at: Scope | undefined) {
  try {
    const full: Call = { ...call, conversation: at?.tags ?? null, task: at?.task ?? null };
    enqueue(scrubbed(JSON.parse(JSON.stringify(full)), config.scrub) as Call);
  } catch (error) { warn(`could not record a model call: ${(error as Error).message}`); }
}

function enqueue(call: Call) {
  if (!apiKey()) { warn('GENERALCONTEXT_API_KEY is not set; a model call was not recorded'); return; }
  if (queue.length >= MAX_QUEUE) { warn('queue is full; a model call was dropped'); return; }
  queue.push(call);
  if (config.disableBatch || queue.length >= BATCH) void send();
  else if (!timer) { timer = setTimeout(() => { timer = null; void send(); }, 500); timer.unref?.(); }
}

async function post(path: string, body: unknown): Promise<Response | null> {
  try {
    return await originalFetch(`${baseUrl()}${path}`, {
      method: 'POST',
      headers: { Authorization: `Bearer ${apiKey()}`, 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
      signal: AbortSignal.timeout(30_000),
    });
  } catch (error) {
    warn(`could not reach ${path}: ${(error as Error).message}`);
    return null;
  }
}

function send(): Promise<void> {
  if (sending) return sending;
  sending = (async () => {
    while (queue.length) {
      const batch = queue.splice(0, BATCH);
      const response = await post('/api/platform-collect/sdk', { calls: batch, sdk: `node/${VERSION}`, app: config.appName });
      if (response && !response.ok) warn(`${batch.length} calls not accepted: ${response.status}`);
    }
  })().catch(() => {}).finally(() => { sending = null; });
  return sending;
}

/** Wait until recorded calls are sent. Call before a serverless function or script exits. Never throws. */
export async function flush(): Promise<void> {
  if (timer) { clearTimeout(timer); timer = null; }
  try { await send(); } catch { /* send already logs */ }
}

/** Mark how a session ended. Returns true when the platform accepted it. Never throws. */
export async function reportOutcome(sessionId: string, outcome: 'success' | 'fail', note?: string): Promise<boolean> {
  await flush();
  if (!apiKey()) { warn('GENERALCONTEXT_API_KEY is not set; outcome not reported'); return false; }
  const response = await post('/api/platform-collect/outcomes', { session_id: sessionId, outcome, note: note ?? null });
  return Boolean(response?.ok);
}

export type AppEvent = { sessionId: string; eventType: string; properties?: Metadata; eventId?: string; timestamp?: string | Date };

/** Record something that happened in your systems, such as 'order.refunded'. Pass a stable eventId so retries dedupe. Never throws. */
export async function logEvent(event: AppEvent): Promise<boolean> {
  await flush();
  if (!apiKey()) { warn('GENERALCONTEXT_API_KEY is not set; event not logged'); return false; }
  const timestamp = event.timestamp instanceof Date ? event.timestamp.toISOString() : event.timestamp ?? new Date().toISOString();
  const response = await post('/api/platform-collect/events', { events: [{
    event_id: event.eventId ?? randomUUID(), event_type: event.eventType, timestamp,
    session_id: event.sessionId, properties: event.properties ?? {},
  }] });
  return Boolean(response?.ok);
}

// Wrap an SDK create method. The SDK's own promise is returned untouched; recording happens on a side branch.
function recording(create: (...args: unknown[]) => unknown, api: Api) {
  const wrapped = function (this: unknown, ...args: unknown[]): unknown {
    const started = new Date().toISOString();
    const at = scope.getStore();
    const request = args[0] as { stream?: boolean } | undefined;
    // Mark the call so the fetch patch does not record it a second time.
    const reply = scope.run({ ...at, inSdkCall: true }, () => create.apply(this, args));
    if (!reply || typeof (reply as { then?: unknown }).then !== 'function') return reply;
    (reply as Promise<unknown>).then(response => {
      const done = (result: unknown) => record({ api, request, response: result, started_at: started, ended_at: new Date().toISOString(), error: null }, at);
      if (request?.stream && isStream(response)) observeStream(response, api, done);
      else done(response);
    }, error => {
      record({ api, request, response: null, started_at: started, ended_at: new Date().toISOString(), error: String((error as Error)?.message ?? error) }, at);
    });
    return reply;
  };
  (wrapped as any)[MARK] = true;
  return wrapped;
}

type SdkStream = { iterator: () => AsyncIterator<unknown> };
const isStream = (value: unknown): value is SdkStream => typeof (value as SdkStream)?.iterator === 'function';

// Both SDKs' Stream classes read chunks through `this.iterator`, including tee() and toReadableStream().
// Replacing it keeps the object, its type and its methods, and lets us see each chunk once.
function observeStream(stream: SdkStream, api: Api, done: (response: unknown) => void) {
  const source = stream.iterator.bind(stream);
  stream.iterator = async function* () {
    const assemble = api === 'anthropic' ? anthropicAssembler() : openAIAssembler();
    let finished = false;
    try {
      for await (const chunk of { [Symbol.asyncIterator]: source }) {
        try { assemble.add(chunk); } catch { /* never break the caller's stream */ }
        yield chunk;
      }
      finished = true;
    } finally {
      if (finished) done(assemble.result());
    }
  };
}

type Assembler = { add: (chunk: any) => void; result: () => unknown };

function openAIAssembler(): Assembler {
  let model: string | undefined, id: string | undefined, usage: unknown = null, final: unknown = null;
  const choices: { index: number; message: { role: string; content: string | null; tool_calls?: any[] }; finish_reason: string | null }[] = [];
  return {
    add(chunk) {
      // Responses API events: the final response arrives whole.
      if (chunk.type) { if (chunk.type === 'response.completed') final = chunk.response; return; }
      model ??= chunk.model; id ??= chunk.id;
      if (chunk.usage) usage = chunk.usage;
      for (const c of chunk.choices ?? []) {
        const choice = choices[c.index] ??= { index: c.index, message: { role: 'assistant', content: null }, finish_reason: null };
        const d = c.delta ?? {};
        if (d.role) choice.message.role = d.role;
        if (d.content) choice.message.content = (choice.message.content ?? '') + d.content;
        for (const t of d.tool_calls ?? []) {
          const calls = choice.message.tool_calls ??= [];
          const call = calls[t.index] ??= { id: t.id, type: 'function', function: { name: '', arguments: '' } };
          if (t.id) call.id = t.id;
          if (t.function?.name) call.function.name += t.function.name;
          if (t.function?.arguments) call.function.arguments += t.function.arguments;
        }
        if (c.finish_reason) choice.finish_reason = c.finish_reason;
      }
    },
    result: () => final ?? ({ id, object: 'chat.completion', model, choices: choices.filter(Boolean), usage }),
  };
}

function anthropicAssembler(): Assembler {
  let message: any = null;
  const json: string[] = [];
  return {
    add(event) {
      if (event.type === 'message_start') message = { ...event.message, content: [] };
      else if (!message) return;
      else if (event.type === 'content_block_start') { message.content[event.index] = { ...event.content_block }; json[event.index] = ''; }
      else if (event.type === 'content_block_delta') {
        const block = message.content[event.index], d = event.delta;
        if (d.type === 'text_delta') block.text = (block.text ?? '') + d.text;
        else if (d.type === 'input_json_delta') json[event.index] += d.partial_json;
        else if (d.type === 'thinking_delta') block.thinking = (block.thinking ?? '') + d.thinking;
      } else if (event.type === 'content_block_stop') {
        const block = message.content[event.index];
        if (json[event.index]) { try { block.input = JSON.parse(json[event.index]); } catch { block.input = json[event.index]; } }
      } else if (event.type === 'message_delta') {
        Object.assign(message, event.delta);
        message.usage = { ...message.usage, ...event.usage };
      }
    },
    result: () => message,
  };
}

// Record raw fetch POSTs to chat-completions endpoints made without a patched SDK.
function patchFetch() {
  const current = globalThis.fetch;
  if (typeof current !== 'function' || (current as any)[MARK]) return;
  originalFetch = current;
  const patched = async function (input: RequestInfo | URL, init?: RequestInit): Promise<Response> {
    const url = typeof input === 'string' ? input : input instanceof URL ? input.href : input.url;
    const method = (init?.method ?? (input instanceof Request ? input.method : 'GET')).toUpperCase();
    const at = scope.getStore();
    const watch = method === 'POST' && /\/chat\/completions\/?(\?.*)?$/.test(url) && !at?.inSdkCall && !url.startsWith(`${baseUrl()}/api/platform-collect/`);
    if (!watch) return current(input, init);
    const started = new Date().toISOString();
    let request: unknown = null;
    try { request = typeof init?.body === 'string' ? JSON.parse(init.body) : input instanceof Request ? await input.clone().json() : null; } catch { /* not JSON */ }
    const base = { api: 'openai-chat' as const, request, started_at: started };
    let response: Response;
    try { response = await current(input, init); } catch (error) {
      record({ ...base, response: null, ended_at: new Date().toISOString(), error: String((error as Error)?.message ?? error) }, at);
      throw error;
    }
    try {
      const copy = response.clone();
      void (async () => {
        try {
          const text = await copy.text();
          const ended = new Date().toISOString();
          if (!response.ok) return record({ ...base, response: null, ended_at: ended, error: `HTTP ${response.status}: ${text.slice(0, 500)}` }, at);
          const sse = (response.headers.get('content-type') ?? '').includes('text/event-stream');
          record({ ...base, response: sse ? assembleSSE(text) : JSON.parse(text), ended_at: ended, error: null }, at);
        } catch (error) { warn(`could not record a fetch call: ${(error as Error).message}`); }
      })();
    } catch { /* body already used */ }
    return response;
  };
  (patched as any)[MARK] = true;
  globalThis.fetch = patched as typeof fetch;
}

function assembleSSE(text: string): unknown {
  const assemble = openAIAssembler();
  for (const line of text.split('\n')) {
    const data = line.startsWith('data:') ? line.slice(5).trim() : '';
    if (data && data !== '[DONE]') { try { assemble.add(JSON.parse(data)); } catch { /* skip bad event */ } }
  }
  return assemble.result();
}

const PATTERNS: [string, RegExp][] = [
  ['email', /[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}/g],
  ['card', /\b(?:\d[ -]?){13,16}\b/g],
  ['ssn', /\b\d{3}-\d{2}-\d{4}\b/g],
  ['phone', /(?<![\w])\+?1?[\s.-]?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}(?![\w])/g],
];

function redactor(redact: boolean | Scrub | undefined): Scrub | null {
  if (typeof redact === 'function') return redact;
  if (!redact) return null;
  let salt = process.env.GENERALCONTEXT_REDACT_SALT;
  if (!salt) {
    warn('GENERALCONTEXT_REDACT_SALT is not set; placeholders will differ between processes');
    salt = randomBytes(16).toString('hex');
  }
  const placeholder = (kind: string, value: string) => {
    const digits = value.replace(/\D/g, '');
    const normal = kind === 'email' ? value.toLowerCase() : kind === 'phone' ? digits.slice(-10) : digits;
    return `<${kind}:${createHmac('sha256', salt as string).update(normal).digest('hex').slice(0, 6)}>`;
  };
  return text => PATTERNS.reduce((result, [kind, pattern]) => result.replace(pattern, match => placeholder(kind, match)), text);
}

function scrubbed(value: unknown, scrub: Scrub | null): unknown {
  if (!scrub) return value;
  if (typeof value === 'string') return scrub(value);
  if (Array.isArray(value)) return value.map(item => scrubbed(item, scrub));
  if (value && typeof value === 'object') return Object.fromEntries(Object.entries(value).map(([key, item]) => [key, scrubbed(item, scrub)]));
  return value;
}

/** Record one client instance only, without initialize(). Returns the same client. */
export function wrap<C extends object>(client: C, options: { redact?: boolean | Scrub } = {}): C {
  if (options.redact !== undefined) config.scrub = redactor(options.redact);
  const target = client as Record<string, any>;
  const owners: [Record<string, any>, Api][] = [];
  if (target.chat?.completions?.create) owners.push([target.chat.completions, 'openai-chat']);
  if (target.responses?.create) owners.push([target.responses, 'openai-responses']);
  if (!target.chat && target.messages?.create) owners.push([target.messages, 'anthropic']);
  if (!owners.length) throw new TypeError('wrap expects an OpenAI or Anthropic client');
  for (const [owner, api] of owners) {
    if (!owner.create[MARK]) owner.create = recording(owner.create.bind(owner), api);
  }
  return client;
}
