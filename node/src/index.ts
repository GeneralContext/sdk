// Copyright 2026 General Context Labs, Inc.
// SPDX-License-Identifier: Apache-2.0

// Record your agent's model calls for General Context, without changing how it runs.
//
//   import OpenAI from 'openai';
//   import { initialize, withSession } from 'generalcontext';
//
//   initialize({ instrumentModules: { openAI: OpenAI } });
//   await withSession(conversationId, async () => { ... });
//
// Every SDK model call (history sent, tools offered, reply and tool calls) is queued and sent in
// the background. Raw fetch capture is opt-in with instrumentFetch: true. Your agent never waits on General Context, and a failure on our side is
// logged and dropped, never thrown. No runtime dependencies.

import { AsyncLocalStorage } from 'node:async_hooks';
import { createHash, createHmac, randomBytes, randomUUID } from 'node:crypto';
import { request as httpRequest } from 'node:http';
import { request as httpsRequest } from 'node:https';
import { createRequire } from 'node:module';

export const VERSION = '0.2.0';
const BATCH = 20;
const MAX_QUEUE = 5000;
const MAX_CALL_BYTES = 4_000_000; // the server rejects larger calls
const MAX_QUEUE_BYTES = 64_000_000;
const MAX_BATCH_BYTES = 4_000_000;
const SEND_TIMEOUT_MS = 15_000;
const DIRECT_TIMEOUT_MS = 10_000; // reportOutcome and logEvent
const EXIT_FLUSH_MS = 2_000;
const FLUSH_TIMEOUT_MS = 10_000; // flush() default, like the Python SDK
const RETRY_PAUSE_MS = 30_000; // after the collector fails to answer, wait this long before sending again

type Metadata = Record<string, string | number | boolean>;
export type Session = { sessionId: string; userId?: string; channel?: string; product?: string; metadata?: Metadata };
type Tags = { id: string; user_id?: string; channel?: string; product?: string; metadata?: Metadata };
type Api = 'openai-chat' | 'openai-responses' | 'anthropic' | 'gemini' | 'bedrock-converse' | 'openai-realtime';
/** Voice timing on 'openai-realtime' calls. Timestamps are ISO strings, null when not seen. */
export type VoiceTiming = {
  speech_stopped_at: string | null; first_audio_at: string | null; interrupted: boolean; truncated_at_ms?: number;
};
export type Call = {
  api: Api; conversation: Tags | null; task: string | null; request: unknown; response: unknown;
  started_at: string; ended_at: string; error: string | null; voice?: VoiceTiming;
};
type Scrub = (text: string) => string;
type BeforeSend = (call: Call) => Call | null | undefined;
type Scope = { tags?: Tags; task?: string; inSdkCall?: boolean };
// A queued call: its JSON snapshot, and the scrub to apply at send time (undefined = the global one).
type Entry = { json: string; bytes: number; scrub?: Scrub | null };

export type InitializeOptions = {
  apiKey?: string;
  /** Collector URL. Must be https://, except http://localhost, 127.0.0.1 or [::1]. */
  baseUrl?: string;
  appName?: string;
  /** true for built-in email/phone/card/SSN placeholders, or your own (text) => text. */
  redact?: boolean | Scrub;
  /** Runs at send time on each call, before redaction. Return the call to send, or null to drop it. */
  beforeSend?: BeforeSend;
  /**
   * SDK classes or modules to patch, e.g. { openAI: OpenAI, anthropic: Anthropic }. Also accepts
   * googleGenAI (the '@google/genai' module) and bedrockRuntime (the '@aws-sdk/client-bedrock-runtime' module).
   */
  instrumentModules?: Modules;
  /** Also record raw fetch POSTs to any URL ending /chat/completions. Default false. */
  instrumentFetch?: boolean;
  /** Send each call right away instead of batching. */
  disableBatch?: boolean;
};

type Modules = { openAI?: unknown; anthropic?: unknown; googleGenAI?: unknown; bedrockRuntime?: unknown };

const DEFAULT_BASE_URL = 'https://generalcontextlabs.com';
const config = {
  apiKey: undefined as string | undefined,
  baseUrl: undefined as string | undefined,
  appName: undefined as string | undefined,
  scrub: null as Scrub | null,
  beforeSend: null as BeforeSend | null,
  disableBatch: false,
};
const scope = new AsyncLocalStorage<Scope>();
const queue: Entry[] = [];
let queuedBytes = 0;
let initialized = false;
let sending: Promise<void> | null = null;
let timer: ReturnType<typeof setTimeout> | null = null;
let originalFetch: typeof fetch = globalThis.fetch;
let exitHooked = false;
let stopped = false; // set when the exit flush runs out of time
let pausedUntil = 0; // background sends wait until then after a failed send; flush() still tries
// Prepared calls the collector refused for now (busy, failing, unreachable). Sent before the queue.
const retry: string[] = [];
let retryBytes = 0;
const inflight = new Set<AbortController>();

const apiKey = () => config.apiKey ?? process.env.GENERALCONTEXT_API_KEY;
// Explicit option, then env var, then default. Trailing slashes are trimmed without a regex.
const trimSlashes = (url: string) => { let end = url.length; while (end > 0 && url[end - 1] === '/') end--; return url.slice(0, end); };
const baseUrl = () => trimSlashes(config.baseUrl ?? process.env.GENERALCONTEXT_BASE_URL ?? DEFAULT_BASE_URL);
const warn = (message: string) => console.warn(`[generalcontext] ${message}`);

// Warn at most once per minute for each key, so a broken collector cannot flood logs.
const lastWarned = new Map<string, number>();
function warnLimited(key: string, message: string) {
  const now = Date.now();
  if (now - (lastWarned.get(key) ?? -Infinity) < 60_000) return;
  lastWarned.set(key, now);
  warn(message);
}

// Only send the API key over https, or plain http to this machine for local development.
function secureUrl(url: string): boolean {
  try {
    const u = new URL(url);
    return u.protocol === 'https:' || (u.protocol === 'http:' && ['localhost', '127.0.0.1', '[::1]'].includes(u.hostname));
  } catch { return false; }
}

const disabledByEnv = () => /^(1|true|yes)$/i.test((process.env.GENERALCONTEXT_DISABLED ?? '').trim());

/** Configure recording once per process, before your agent makes model calls. */
export function initialize(options: InitializeOptions = {}): void {
  if (initialized) { warn('initialize was already called; ignoring the second call'); return; }
  initialized = true;
  if (disabledByEnv()) { warn('GENERALCONTEXT_DISABLED is set; recording is off'); return; }
  config.apiKey = options.apiKey ?? process.env.GENERALCONTEXT_API_KEY;
  config.baseUrl = options.baseUrl;
  config.appName = options.appName;
  config.scrub = redactor(options.redact);
  config.beforeSend = options.beforeSend ?? null;
  config.disableBatch = options.disableBatch ?? false;
  if (!config.apiKey) warn('no API key; set GENERALCONTEXT_API_KEY or pass apiKey');
  if (!secureUrl(baseUrl())) warn(`baseUrl ${baseUrl()} is not https; nothing will be sent`);

  const modules = options.instrumentModules ?? autoload();
  // openai, @anthropic-ai/sdk and @google/genai ship separate CommonJS and ES module builds with their own classes.
  // require() above found the CommonJS ones, which is what a CommonJS app uses. An ES module app (import OpenAI
  // from 'openai') gets the other build, so patch that too. patchMethod skips a method already patched.
  // The imports settle a few ticks after initialize() returns, before an app's own `await import(...)` of the
  // same module resumes. A call made synchronously right after initialize(), before any await, can run
  // unrecorded: await ready() first in that case.
  // OpenAI Realtime (voice) connection classes live in their own entry points. openai/realtime/ws needs the
  // optional `ws` package, so either may be missing.
  if (!options.instrumentModules || options.instrumentModules.openAI) {
    const require = createRequire(import.meta.url);
    for (const name of REALTIME_MODULES) { try { patchRealtime(require(name)); } catch { /* not installed */ } }
    const realtime = Promise.all(REALTIME_MODULES.map(name => import(name).then(patchRealtime, () => {}))).then(() => {});
    pending = pending.then(() => realtime);
  }
  if (!options.instrumentModules) {
    const esm: [string, (m: unknown) => void][] = [['openai', patchOpenAI], ['@anthropic-ai/sdk', patchAnthropic]];
    if (modules.googleGenAI) esm.push(['@google/genai', patchGemini]);
    // @aws-sdk/client-bedrock-runtime resolves to one class from both require() and import(), so it needs nothing here.
    let found = Object.values(modules).some(Boolean);
    const realtime = pending;
    pending = Promise.all(esm.map(([name, patch]) => import(name as string).then(m => { found = true; patch(m); }, () => {})))
      .then(() => { if (!found) warn(NO_SDK); }).then(() => realtime);
  }
  if (modules.openAI) patchOpenAI(modules.openAI);
  if (modules.anthropic) patchAnthropic(modules.anthropic);
  if (modules.googleGenAI) patchGemini(modules.googleGenAI);
  if (modules.bedrockRuntime) patchBedrock(modules.bedrockRuntime);
  if (options.instrumentFetch ?? false) patchFetch();
}

let pending: Promise<void> = Promise.resolve();
// Resolution is from this file's location, so an SDK installed in another project's node_modules is not found.
const NO_SDK = 'found no model SDK to record (openai, @anthropic-ai/sdk, @google/genai, @aws-sdk/client-bedrock-runtime). '
  + 'Install generalcontext in the same project as your model SDK, or pass instrumentModules.';

/**
 * Resolves once initialize() has patched every SDK build it found, including the ES module builds it loads
 * asynchronously. Await it when a model call may run before the app's next await after initialize(). Never rejects.
 */
export function ready(): Promise<void> {
  return pending;
}

// Load installed SDKs when the caller did not pass them. Bundlers can hide these; pass instrumentModules there.
function autoload(): Modules {
  const require = createRequire(import.meta.url);
  const load = (name: string) => { try { return require(name); } catch { return undefined; } };
  return {
    openAI: load('openai'), anthropic: load('@anthropic-ai/sdk'),
    googleGenAI: load('@google/genai'), bedrockRuntime: load('@aws-sdk/client-bedrock-runtime'),
  };
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
  // client.beta.messages: create, parse, stream, toolRunner and threads all post through beta.messages.create.
  patchMethod(Anthropic?.Beta?.Messages?.prototype, 'create', 'anthropic');
}

const exported = (value: unknown, name: string): any => {
  const v = value as any;
  return v?.[name] ?? v?.default?.[name] ?? (typeof v === 'function' && v.name === name ? v : undefined);
};

// @google/genai assigns generateContent and generateContentStream as instance properties in the Models
// constructor. A setter on the prototype sees that assignment and stores the recording version instead.
function patchGemini(value: unknown) {
  const proto = exported(value, 'Models')?.prototype;
  if (!proto) return;
  const wrappers: Record<string, (fn: any) => any> = {
    generateContent: fn => recording(fn, 'gemini'),
    generateContentStream: fn => geminiStreamRecording(fn),
  };
  for (const [name, wrapFn] of Object.entries(wrappers)) {
    const existing = Object.getOwnPropertyDescriptor(proto, name);
    if (existing?.set && (existing.set as any)[MARK]) continue;
    if (typeof existing?.value === 'function') { patchMethod(proto, name, 'gemini'); continue; }
    const set = function (this: object, fn: unknown) {
      const v = typeof fn === 'function' && !(fn as any)[MARK] ? wrapFn(fn) : fn;
      Object.defineProperty(this, name, { value: v, writable: true, configurable: true, enumerable: true });
    };
    (set as any)[MARK] = true;
    Object.defineProperty(proto, name, { get: () => undefined, set, configurable: true });
  }
}

function patchBedrock(value: unknown) {
  const proto = exported(value, 'BedrockRuntimeClient')?.prototype;
  const original = proto?.send;
  if (typeof original !== 'function' || original[MARK]) return;
  proto.send = bedrockRecording(original);
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

const CLIP_OVER = 200_000; // strings longer than this many characters are clipped
const CLIP_HEAD = 100_000;
const CLIP_TAIL = 20_000;
const BINARY_MIN = 256; // shorter base64 payloads are left alone
const BASE64 = /^[A-Za-z0-9+/_-]+={0,2}$/;
const DATA_URL = /^data:([^,;]*)((?:;[^,;]*)*);base64,/;
const omitted = (payload: string) => `[omitted ${payload.length} bytes sha256:${createHash('sha256').update(payload).digest('hex').slice(0, 12)}]`;

/**
 * The copy of a call that is recorded. Never changes `value`, and never touches what the caller sends to the provider.
 * Agents resend the whole run on every call, so without this, later calls pass the per-call limit and are lost.
 *
 * - Inline binary becomes a placeholder that keeps the shape: a base64 data URL keeps its prefix
 *   (`data:image/png;base64,[omitted N bytes sha256:ab12cd34ef56]`), and a base64 `data`, `result` or
 *   `b64_json` field (Anthropic `source.data`, Gemini `inlineData.data`, OpenAI `input_audio.data`, Responses
 *   `image_generation_call.result`) becomes `[omitted N bytes sha256:...]`. N is the length of the base64 text
 *   and the hash is the first 12 hex digits of the sha256 of that text, so the same image gets the same placeholder
 *   in every call. Media types stay. Binary is never recorded: it does not help review and it is a privacy risk.
 * - Any string over 200,000 characters keeps its first 100,000 and last 20,000 characters around
 *   `\n[... clipped N characters ...]\n`.
 *
 * Runs before beforeSend and redaction. A call still over the limit afterwards is dropped.
 */
export function compactForRecording(value: unknown, key?: string, depth = 0): unknown {
  if (typeof value === 'string') {
    if (value.length > BINARY_MIN) {
      const url = DATA_URL.exec(value.slice(0, 512));
      if (url) return value.slice(0, url[0].length) + omitted(value.slice(url[0].length));
      if ((key === 'data' || key === 'result' || key === 'b64_json') && BASE64.test(value)) return omitted(value);
    }
    if (value.length > CLIP_OVER) return `${value.slice(0, CLIP_HEAD)}\n[... clipped ${value.length - CLIP_HEAD - CLIP_TAIL} characters ...]\n${value.slice(-CLIP_TAIL)}`;
    return value;
  }
  if (value === null || typeof value !== 'object' || depth > 64) return value;
  if (typeof (value as { toJSON?: unknown }).toJSON === 'function') value = (value as { toJSON: () => unknown }).toJSON();
  if (value === null || typeof value !== 'object') return compactForRecording(value, key, depth + 1);
  if (Array.isArray(value)) return value.map(item => compactForRecording(item, undefined, depth + 1));
  const out: Record<string, unknown> = {};
  for (const [k, v] of Object.entries(value)) out[k] = compactForRecording(v, k, depth + 1);
  return out;
}

// Runs before the caller's await continues, so only take the snapshot here. Redaction and beforeSend run at send time.
function record(call: Omit<Call, 'conversation' | 'task'>, at: Scope | undefined, scrub?: Scrub | null) {
  try {
    if (!apiKey()) { warnLimited('no-key', 'GENERALCONTEXT_API_KEY is not set; a model call was not recorded'); return; }
    const json = JSON.stringify(compactForRecording({ ...call, request: withoutRequestOptions(call.request), conversation: at?.tags ?? null, task: at?.task ?? null }));
    const bytes = Buffer.byteLength(json);
    if (bytes > MAX_CALL_BYTES) { warnLimited('oversized', `a model call of ${bytes} bytes is over the ${MAX_CALL_BYTES} byte limit and was dropped`); return; }
    if (queue.length >= MAX_QUEUE || queuedBytes + bytes > MAX_QUEUE_BYTES) { warnLimited('queue-full', 'queue is full; a model call was dropped'); return; }
    queue.push({ json, bytes, scrub });
    queuedBytes += bytes;
    hookExit();
    if (Date.now() < pausedUntil) return; // the collector just failed; a timer resumes sending
    if (config.disableBatch || queue.length >= BATCH) void send();
    else if (!timer) { timer = setTimeout(() => { timer = null; void send(); }, 500); timer.unref?.(); }
  } catch (error) { warnLimited('record', `could not record a model call: ${(error as Error).message}`); }
}

// Flush once when the event loop empties, capped at EXIT_FLUSH_MS. Pending requests are aborted at the cap.
function hookExit() {
  if (exitHooked) return;
  exitHooked = true;
  process.once('beforeExit', () => {
    const cap = setTimeout(() => { stopped = true; for (const c of inflight) c.abort(); }, EXIT_FLUSH_MS);
    cap.unref?.();
    void flush().finally(() => clearTimeout(cap));
  });
}

// Credentials are never sent, whatever `redact` says. They reach a recording when an app pastes a key into a
// prompt, a tool returns one, or a request carries one in its options. Each pattern needs the provider's
// fixed prefix, so ordinary text is never touched. The key name stays, so a reader knows one was there.
const SECRETS: [string, RegExp][] = [
  ['anthropic-key', /\bsk-ant-[A-Za-z0-9_-]{20,}/g],
  ['openai-key', /\bsk-(?:proj-|svcacct-|admin-)?[A-Za-z0-9_-]{20,}/g],
  ['stripe-key', /\b[rsp]k_(?:live|test)_[A-Za-z0-9]{16,}/g],
  ['google-key', /\bAIza[A-Za-z0-9_-]{35}\b/g],
  ['aws-key', /\b(?:AKIA|ASIA)[A-Z0-9]{16}\b/g],
  ['github-token', /\b(?:gh[pousr]_[A-Za-z0-9]{36,}|github_pat_[A-Za-z0-9_]{40,})/g],
  ['slack-token', /\bxox[abposr]-[A-Za-z0-9-]{10,}/g],
  ['generalcontext-key', /\bgck_[a-f0-9]{48}\b/g],
  ['jwt', /\beyJ[A-Za-z0-9_-]{8,}\.eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}/g],
  ['bearer-token', /(?<=\bBearer\s)[A-Za-z0-9._~+/-]{20,}=*/g],
  // A password inside a connection string or URL: postgres://user:PASSWORD@host. Only the password goes.
  ['password', /(?<=\b[a-z][a-z0-9+.-]{1,20}:\/\/[^\s:/@]{1,64}:)[^\s@/]{1,128}(?=@)/g],
  ['private-key', /-----BEGIN [A-Z ]{0,20}PRIVATE KEY-----(?:[A-Za-z0-9+/=\s:,]|-(?!-)){1,16384}?-----END [A-Z ]{0,20}PRIVATE KEY-----/g],
];
export function maskSecrets(text: string): string {
  if (text.length < 16) return text;
  return SECRETS.reduce((out, [kind, pattern]) => out.replace(pattern, `<${kind}>`), text);
}
// Request options that carry credentials or transport settings, not what the model was asked. Dropped from the
// recorded request: Gemini's httpOptions (per-request headers) and anything spelled like a header map.
const REQUEST_OPTIONS = new Set(['httpOptions', 'http_options', 'headers', 'extra_headers', 'extraHeaders', 'extra_query', 'extraQuery', 'abortSignal', 'signal']);
function withoutRequestOptions(value: unknown, depth = 0): unknown {
  if (!value || typeof value !== 'object' || depth > 3) return value;
  if (Array.isArray(value)) return value;
  const out: Record<string, unknown> = {};
  for (const [k, v] of Object.entries(value)) if (!REQUEST_OPTIONS.has(k)) out[k] = k === 'config' ? withoutRequestOptions(v, depth + 1) : v;
  return out;
}

// Apply beforeSend and redaction. Returns the JSON to send, or null to drop.
function prepare(entry: Entry): string | null {
  let call = JSON.parse(entry.json) as Call | null | undefined;
  if (config.beforeSend) {
    try { call = config.beforeSend(call as Call); } catch (error) {
      warnLimited('before-send', `beforeSend threw; a model call was dropped: ${(error as Error).message}`);
      return null;
    }
    if (call == null) return null;
  }
  const scrub = entry.scrub === undefined ? config.scrub : entry.scrub;
  return JSON.stringify(scrubbed(call, scrub ? (text: string) => scrub(maskSecrets(text)) : maskSecrets));
}

type Reply = { ok: boolean; status: number; retryAfterMs?: number };
// Answers worth retrying: the collector is busy (429) or failing (5xx). Anything else (a bad key, a body too
// large) will fail again the same way, so that batch is dropped with a warning.
const retryable = (status: number) => status === 429 || status === 408 || status >= 500;
// Retry-After in seconds (or an HTTP date), kept between 1 s and 5 minutes.
function retryAfterMs(value: string | string[] | undefined): number | undefined {
  const raw = Array.isArray(value) ? value[0] : value;
  if (!raw) return undefined;
  const seconds = /^\d+$/.test(raw.trim()) ? Number(raw) * 1000 : Date.parse(raw) - Date.now();
  return Number.isFinite(seconds) ? Math.min(300_000, Math.max(1000, seconds)) : undefined;
}

// While a caller awaits flush(), reportOutcome() or logEvent(), our sockets keep the process alive
// so the await can finish. Otherwise they are unref'd: background sending never keeps a finished
// process alive, even against a collector that accepts and never answers.
let waiters = 0;
const sockets = new Set<{ ref(): unknown; unref(): unknown }>();
function holdOpen<T>(work: Promise<T>): Promise<T> {
  if (waiters++ === 0) for (const socket of sockets) socket.ref();
  return work.finally(() => { if (--waiters === 0) for (const socket of sockets) socket.unref(); });
}

/**
 * POST to the collector with node:http(s), not fetch, so the socket can be unref'd (see above).
 * Redirects are never followed (node:http does not follow them), so the key never reaches another host.
 */
function postRaw(url: string, body: string, signal: AbortSignal): Promise<Reply> {
  return new Promise((resolve, reject) => {
    const target = new URL(url);
    const send = target.protocol === 'https:' ? httpsRequest : httpRequest;
    const req = send(target, {
      method: 'POST', signal, agent: false,
      headers: { Authorization: `Bearer ${apiKey()}`, 'Content-Type': 'application/json', 'Content-Length': Buffer.byteLength(body) },
    }, res => {
      res.resume(); // drain, so the socket is freed
      res.on('end', () => resolve({ ok: (res.statusCode ?? 0) >= 200 && (res.statusCode ?? 0) < 300, status: res.statusCode ?? 0,
        retryAfterMs: retryAfterMs(res.headers['retry-after']) }));
      res.on('error', reject);
    });
    req.on('socket', socket => {
      sockets.add(socket);
      socket.once('close', () => sockets.delete(socket));
      if (waiters === 0) socket.unref();
    });
    req.on('error', reject);
    req.end(body);
  });
}

async function post(path: string, body: string, timeoutMs: number): Promise<Reply | null> {
  const url = `${baseUrl()}${path}`;
  if (!secureUrl(url)) { warnLimited('insecure', `refusing to send to ${baseUrl()}: baseUrl must be https`); return null; }
  if (stopped) return null;
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), timeoutMs);
  timeout.unref?.();
  inflight.add(controller);
  try {
    return await postRaw(url, body, controller.signal);
  } catch (error) {
    warnLimited(`send:${path}`, `could not reach ${path}: ${(error as Error).message}`);
    return null;
  } finally {
    clearTimeout(timeout);
    inflight.delete(controller);
  }
}

function send(): Promise<void> {
  if (sending) return sending;
  sending = (async () => {
    const envelope = JSON.stringify({ sdk: `node/${VERSION}`, app: config.appName });
    while ((queue.length || retry.length) && !stopped) {
      // Up to BATCH calls and MAX_BATCH_BYTES per request. Calls refused earlier go first, already prepared.
      const parts: string[] = retry.splice(0, BATCH);
      retryBytes = retry.reduce((sum, part) => sum + part.length, 0);
      let size = envelope.length + 16 + parts.reduce((sum, part) => sum + part.length + 1, 0);
      while (queue.length && parts.length < BATCH) {
        const entry = queue[0];
        let json: string | null;
        try { json = prepare(entry); } catch (error) {
          warnLimited('prepare', `could not prepare a model call: ${(error as Error).message}`);
          json = null;
        }
        const bytes = json === null ? 0 : Buffer.byteLength(json);
        if (json !== null && bytes > MAX_CALL_BYTES) {
          warnLimited('oversized', `a model call of ${bytes} bytes is over the ${MAX_CALL_BYTES} byte limit and was dropped`);
          json = null;
        }
        if (json !== null && parts.length && size + bytes + 1 > MAX_BATCH_BYTES) break; // send what we have; retry this one next
        queue.shift();
        queuedBytes -= entry.bytes;
        if (json === null) continue;
        parts.push(json);
        size += bytes + 1;
      }
      if (!parts.length) continue;
      const body = `{"calls":[${parts.join(',')}],${envelope.slice(1)}`;
      const response = await post('/api/platform-collect/sdk', body, SEND_TIMEOUT_MS);
      if (response && !response.ok && !retryable(response.status)) warnLimited('rejected', `${parts.length} calls not accepted: ${response.status}`);
      if ((!response || (!response.ok && retryable(response.status))) && !stopped) {
        // Busy, failing, unreachable, hung or timed out: put this batch back at the front, already prepared, and
        // try again after a pause. Nothing recorded is lost to a deploy or a brief outage.
        retry.unshift(...parts);
        retryBytes += parts.reduce((sum, part) => sum + part.length, 0);
        if (retryBytes > MAX_QUEUE_BYTES) { const dropped = retry.splice(BATCH * 10); retryBytes = retry.reduce((sum, part) => sum + part.length, 0); warnLimited('queue-full', `${dropped.length} calls dropped while the collector was unavailable`); }
        const pause = response?.retryAfterMs ?? RETRY_PAUSE_MS;
        // Unreachable, hung or timed out. Stop here instead of trying every queued batch in turn (15 s each),
        // which kept the process alive for minutes after the app finished. The queue stays bounded and is
        // retried after a pause, on flush(), or at exit.
        pausedUntil = Date.now() + pause;
        const resume = setTimeout(() => { if (queue.length || retry.length) void send(); }, pause);
        resume.unref?.();
        break;
      }
    }
  })().catch(() => {}).finally(() => { sending = null; });
  return sending;
}

/**
 * Wait until recorded calls are sent, for at most `timeoutMs` (default 10 s). Call before a serverless
 * function or script exits. Resolves true when the queue was sent, false at the deadline. Never throws.
 * Sending continues in the background after the deadline.
 */
export async function flush(timeoutMs = FLUSH_TIMEOUT_MS): Promise<boolean> {
  if (timer) { clearTimeout(timer); timer = null; }
  let deadline: ReturnType<typeof setTimeout> | undefined;
  try {
    const sent = send().then(() => queue.length === 0 && retry.length === 0, () => false);
    // The deadline timer stays referenced: an awaited flush() must be able to finish and report.
    const late = new Promise<boolean>(resolve => { deadline = setTimeout(() => resolve(false), Math.max(0, timeoutMs)); });
    return await holdOpen(Promise.race([sent, late]));
  } catch { return false; } finally { clearTimeout(deadline); }
}

const scrubText = (value: string) => (config.scrub ? config.scrub(maskSecrets(value)) : maskSecrets(value));

/** Mark how a session ended. Returns true when the platform accepted it. Never throws. */
export async function reportOutcome(sessionId: string, outcome: 'success' | 'fail', note?: string): Promise<boolean> {
  try {
    if (!apiKey()) { warnLimited('no-key', 'GENERALCONTEXT_API_KEY is not set; outcome not reported'); return false; }
    await flush();
    const body = { session_id: scrubText(String(sessionId)), outcome, note: note == null ? null : scrubText(note) };
    const response = await holdOpen(post('/api/platform-collect/outcomes', JSON.stringify(body), DIRECT_TIMEOUT_MS));
    return Boolean(response?.ok);
  } catch { return false; }
}

export type AppEvent = { sessionId: string; eventType: string; properties?: Metadata; eventId?: string; timestamp?: string | Date };

/** Record something that happened in your systems, such as 'order.refunded'. Pass a stable eventId so retries dedupe. Never throws. */
export async function logEvent(event: AppEvent): Promise<boolean> {
  try {
    if (!apiKey()) { warnLimited('no-key', 'GENERALCONTEXT_API_KEY is not set; event not logged'); return false; }
    await flush();
    const timestamp = event.timestamp instanceof Date ? event.timestamp.toISOString() : event.timestamp ?? new Date().toISOString();
    const sessionId = scrubText(String(event.sessionId));
    const properties = Object.fromEntries(Object.entries(event.properties ?? {}).map(([k, v]) => [k, typeof v === 'string' ? scrubText(v) : v]));
    const body = { events: [{
      event_id: event.eventId ?? randomUUID(), event_type: scrubText(event.eventType), timestamp,
      session_id: sessionId, properties,
    }] };
    const response = await holdOpen(post('/api/platform-collect/events', JSON.stringify(body), DIRECT_TIMEOUT_MS));
    return Boolean(response?.ok);
  } catch { return false; }
}

// Wrap an SDK create method. The SDK's own promise is returned untouched; recording happens on a side branch.
function recording(create: (...args: unknown[]) => unknown, api: Api, scrub?: Scrub | null) {
  const wrapped = function (this: unknown, ...args: unknown[]): unknown {
    const started = new Date().toISOString();
    const at = scope.getStore();
    const request = args[0] as { stream?: boolean } | undefined;
    // Mark the call so the fetch patch does not record it a second time.
    const reply = scope.run({ ...at, inSdkCall: true }, () => create.apply(this, args));
    if (!reply || typeof (reply as { then?: unknown }).then !== 'function') return reply;
    // The SDKs' APIPromise reads the body inside parseResponse. Helpers such as parse() build a new promise that
    // calls it again, so keep the first result: the body is read once and both sides get the same object.
    const api_ = reply as { parseResponse?: (...a: any[]) => any; _thenUnwrap?: (t: (...a: any[]) => any) => any };
    if (typeof api_.parseResponse === 'function') {
      try {
        const parseOnce = api_.parseResponse;
        const seen = new WeakMap<object, unknown>();
        const parse = function (this: unknown, ...a: unknown[]) {
          const key = a[1];
          if (!key || typeof key !== 'object') return parseOnce.apply(this, a);
          if (!seen.has(key)) seen.set(key, parseOnce.apply(this, a));
          return seen.get(key);
        };
        api_.parseResponse = parse;
        const thenUnwrap = api_._thenUnwrap;
        if (typeof thenUnwrap === 'function') {
          api_._thenUnwrap = function (this: unknown, transform) {
            const next = thenUnwrap.call(this, transform);
            if (next && typeof next.parseResponse === 'function') {
              next.parseResponse = async (client: unknown, props: any) => {
                const parsed = await parse.call(reply, client, props);
                const out = transform(parsed, props);
                const id = (parsed as any)?._request_id;
                if (id && out && typeof out === 'object' && !('_request_id' in out)) Object.defineProperty(out, '_request_id', { value: id, enumerable: false });
                return out;
              };
            }
            return next;
          };
        }
      } catch { /* leave the SDK promise as it was */ }
    }
    (reply as Promise<unknown>).then(response => {
      const done = (result: unknown, error: string | null = null) => record({ api, request, response: api === 'gemini' ? plainGemini(result) : result, started_at: started, ended_at: new Date().toISOString(), error }, at, scrub);
      if (request?.stream && (isStream(response) || isLegacyStream(response))) observeStream(response, api, done);
      else done(response);
    }, error => {
      record({ api, request, response: null, started_at: started, ended_at: new Date().toISOString(), error: String((error as Error)?.message ?? error) }, at, scrub);
    });
    return reply;
  };
  (wrapped as any)[MARK] = true;
  return wrapped;
}

type SdkStream = { iterator: () => AsyncIterator<unknown> };
type LegacyStream = { [Symbol.asyncIterator]: () => AsyncIterator<unknown> };
const isStream = (value: unknown): value is SdkStream => typeof (value as SdkStream)?.iterator === 'function';
// openai before 4.15 has no `iterator` field: the Stream class only defines [Symbol.asyncIterator].
const isLegacyStream = (value: unknown): value is LegacyStream =>
  !!value && typeof value === 'object' && typeof (value as LegacyStream)[Symbol.asyncIterator] === 'function';

// Both SDKs' Stream classes read chunks through `this.iterator`, including tee() and toReadableStream().
// Replacing it keeps the object, its type and its methods, and lets us see each chunk once.
// A stream the caller stops early (break, return or a throw inside for await, which aborts the request) or
// that fails is still recorded, with what was assembled so far and an error. The caller's exception is untouched.
function observeStream(stream: SdkStream | LegacyStream, api: Api, done: (response: unknown, error?: string | null) => void) {
  const legacy = !isStream(stream);
  const source = legacy ? (stream as LegacyStream)[Symbol.asyncIterator].bind(stream) : (stream as SdkStream).iterator.bind(stream);
  const observed = async function* () {
    const assemble = api === 'anthropic' ? anthropicAssembler() : openAIAssembler();
    let error: string | null = 'stream ended early: the caller stopped reading before the response completed';
    try {
      for await (const chunk of { [Symbol.asyncIterator]: source }) {
        try { assemble.add(chunk); } catch { /* never break the caller's stream */ }
        yield chunk;
      }
      error = null;
    } catch (failure) {
      error = String((failure as Error)?.message ?? failure);
      throw failure;
    } finally {
      let result: unknown = null;
      try { result = assemble.result(); } catch { /* record the error alone */ }
      done(result, error);
    }
  };
  // An own property shadows the class method on this instance only.
  if (legacy) Object.defineProperty(stream, Symbol.asyncIterator, { value: observed, writable: true, configurable: true });
  else (stream as SdkStream).iterator = observed;
}

type Assembler = { add: (chunk: any) => void; result: () => unknown };

function openAIAssembler(): Assembler {
  let model: string | undefined, id: string | undefined, usage: unknown = null, final: unknown = null, partial: any = null;
  const items: unknown[] = [];
  const choices: { index: number; message: { role: string; content: string | null; tool_calls?: any[] }; finish_reason: string | null }[] = [];
  return {
    add(chunk) {
      // Responses API events: the final response arrives whole.
      // Keep the latest snapshot and finished output items too, so a stream stopped early still records what arrived.
      if (chunk.type) {
        if (chunk.type === 'response.completed') final = chunk.response;
        else if (chunk.response) partial = chunk.response;
        else if (chunk.type === 'response.output_item.done' && chunk.item) items[chunk.output_index ?? items.length] = chunk.item;
        return;
      }
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
    result: () => final ?? (partial ? { ...partial, output: items.filter(Boolean) } : { id, object: 'chat.completion', model, choices: choices.filter(Boolean), usage }),
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

// generateContentStream resolves to an async generator. Hand the caller a generator that yields the same chunks.
function geminiStreamRecording(create: (...args: unknown[]) => unknown, scrub?: Scrub | null) {
  const wrapped = function (this: unknown, ...args: unknown[]): unknown {
    const started = new Date().toISOString();
    const at = scope.getStore();
    const request = args[0];
    const reply = scope.run({ ...at, inSdkCall: true }, () => create.apply(this, args));
    if (!reply || typeof (reply as Promise<unknown>).then !== 'function') return reply;
    const fail = (error: unknown) => record({ api: 'gemini', request, response: null, started_at: started, ended_at: new Date().toISOString(), error: String((error as Error)?.message ?? error) }, at, scrub);
    return (reply as Promise<AsyncIterable<unknown>>).then(source => {
      if (!source || typeof (source as any)[Symbol.asyncIterator] !== 'function') return source;
      return (async function* () {
        const assemble = geminiAssembler();
        let finished = false;
        try {
          for await (const chunk of source) {
            try { assemble.add(chunk); } catch { /* never break the caller's stream */ }
            yield chunk;
          }
          finished = true;
        } catch (error) { fail(error); throw error; } finally {
          if (finished) record({ api: 'gemini', request, response: assemble.result(), started_at: started, ended_at: new Date().toISOString(), error: null }, at, scrub);
        }
      })();
    }, error => { fail(error); throw error; });
  };
  (wrapped as any)[MARK] = true;
  return wrapped;
}

// The response as plain JSON, without the raw HTTP headers the SDK attaches.
function plainGemini(response: unknown): unknown {
  if (!response || typeof response !== 'object') return response;
  const { sdkHttpResponse, ...rest } = response as Record<string, unknown>;
  return rest;
}

function geminiAssembler(): Assembler {
  const out: any = { candidates: [] };
  const parts: any[] = [];
  let candidate: any = null;
  return {
    add(chunk) {
      const c = plainGemini(chunk) as any;
      if (c.usageMetadata) out.usageMetadata = c.usageMetadata;
      if (c.modelVersion) out.modelVersion = c.modelVersion;
      if (c.responseId) out.responseId = c.responseId;
      const first = c.candidates?.[0];
      if (!first) return;
      candidate ??= { content: { role: 'model', parts } };
      if (first.content?.role) candidate.content.role = first.content.role;
      if (first.finishReason) candidate.finishReason = first.finishReason;
      for (const part of first.content?.parts ?? []) {
        const last = parts[parts.length - 1];
        if (typeof part.text === 'string' && last && typeof last.text === 'string' && Boolean(last.thought) === Boolean(part.thought) && Object.keys(part).every(k => k === 'text' || k === 'thought')) last.text += part.text;
        else parts.push({ ...part });
      }
    },
    result: () => ({ ...out, candidates: candidate ? [candidate] : [] }),
  };
}

// Images and documents arrive as bytes. Send a marker in their place, never the bytes.
function omitBytes(value: unknown, depth = 0): unknown {
  if (value instanceof Uint8Array || value instanceof ArrayBuffer || ArrayBuffer.isView(value)) return { omitted: 'bytes' };
  if (depth > 32 || value === null || typeof value !== 'object') return value;
  if (Array.isArray(value)) return value.map(item => omitBytes(item, depth + 1));
  return Object.fromEntries(Object.entries(value).map(([k, v]) => [k, omitBytes(v, depth + 1)]));
}

const CONVERSE = new Set(['ConverseCommand', 'ConverseStreamCommand']);

function bedrockRecording(send: (...args: unknown[]) => unknown, scrub?: Scrub | null) {
  const wrapped = function (this: unknown, ...args: unknown[]): unknown {
    const command = args[0] as { input?: any; constructor?: { name?: string } } | undefined;
    const name = command?.constructor?.name ?? '';
    const input = command?.input;
    // By class name, or by input shape when a bundler renamed the classes. InvokeModel has a body, not messages.
    const converse = CONVERSE.has(name) || (typeof input?.modelId === 'string' && Array.isArray(input?.messages) && !('body' in input));
    if (!converse) return send.apply(this, args);
    const started = new Date().toISOString();
    const at = scope.getStore();
    let request: unknown = null;
    try { request = omitBytes(command?.input); } catch { /* record without the request */ }
    const reply = scope.run({ ...at, inSdkCall: true }, () => send.apply(this, args));
    if (!reply || typeof (reply as Promise<unknown>).then !== 'function') return reply;
    const done = (response: unknown, error: string | null) => record({ api: 'bedrock-converse', request, response, started_at: started, ended_at: new Date().toISOString(), error }, at, scrub);
    (reply as Promise<any>).then(output => {
      try {
        const { $metadata, ...rest } = output ?? {};
        const stream = rest.stream;
        if (stream && typeof stream[Symbol.asyncIterator] === 'function') {
          output.stream = (async function* () {
            const assemble = converseAssembler();
            let finished = false;
            try {
              for await (const event of stream) {
                try { assemble.add(event); } catch { /* never break the caller's stream */ }
                yield event;
              }
              finished = true;
            } catch (error) { done(null, String((error as Error)?.message ?? error)); throw error; } finally {
              if (finished) done(assemble.result(), null);
            }
          })();
        } else done(omitBytes(rest), null);
      } catch (error) { warnLimited('record', `could not record a model call: ${(error as Error).message}`); }
    }, error => done(null, String((error as Error)?.message ?? error)));
    return reply;
  };
  (wrapped as any)[MARK] = true;
  return wrapped;
}

function converseAssembler(): Assembler {
  const content: any[] = [];
  const json: string[] = [];
  const out: any = { output: { message: { role: 'assistant', content } }, stopReason: null };
  return {
    add(event) {
      if (event.messageStart?.role) out.output.message.role = event.messageStart.role;
      const start = event.contentBlockStart;
      if (start) {
        const tool = start.start?.toolUse;
        content[start.contentBlockIndex ?? content.length] = tool ? { toolUse: { toolUseId: tool.toolUseId, name: tool.name, input: {} } } : {};
      }
      const delta = event.contentBlockDelta;
      if (delta) {
        const i = delta.contentBlockIndex ?? 0, d = delta.delta ?? {};
        const block = content[i] ??= {};
        if (typeof d.text === 'string') block.text = (block.text ?? '') + d.text;
        if (d.toolUse) { block.toolUse ??= { input: {} }; json[i] = (json[i] ?? '') + (d.toolUse.input ?? ''); }
        if (d.reasoningContent) {
          const r = block.reasoningContent ??= { reasoningText: { text: '' } };
          if (d.reasoningContent.text) r.reasoningText.text += d.reasoningContent.text;
          if (d.reasoningContent.signature) r.reasoningText.signature = d.reasoningContent.signature;
        }
      }
      const stop = event.contentBlockStop;
      if (stop && json[stop.contentBlockIndex] && content[stop.contentBlockIndex]?.toolUse) {
        try { content[stop.contentBlockIndex].toolUse.input = JSON.parse(json[stop.contentBlockIndex]); } catch { content[stop.contentBlockIndex].toolUse.input = json[stop.contentBlockIndex]; }
      }
      if (event.messageStop) out.stopReason = event.messageStop.stopReason ?? null;
      if (event.metadata?.usage) out.usage = event.metadata.usage;
      if (event.metadata?.metrics) out.metrics = event.metadata.metrics;
    },
    result: () => ({ ...out, output: { message: { ...out.output.message, content: content.filter(Boolean) } } }),
  };
}

// ---- OpenAI Realtime (voice) ----------------------------------------------------------------
//
// A realtime session is one WebSocket carrying many responses. Each response.done becomes one 'openai-realtime'
// call shaped like a Responses call: request { model, instructions, tools, input } and response { id, output, usage },
// so the platform folds a whole phone call into one conversation. Only text is kept: transcripts, text, tool calls
// and tool results. Audio bytes are never read into this state. Events are observed, never changed or delayed.

const REALTIME_MODULES = ['openai/realtime/ws', 'openai/realtime/websocket'];
const REALTIME_MAX_ITEMS = 200;
// A finished response is recorded once the next response starts, the socket closes, or this long after
// response.done, so an interruption that arrives while its audio is still playing is on the call.
const REALTIME_HOLD_MS = 60_000;

type RtItem = Record<string, unknown>;
type RtEntry = { id: string | null; item: RtItem | null; seq: number; response?: string };
type RtResponse = {
  id: string; seq: number; started_at: string; ended_at?: string; usage?: unknown; error: string | null;
  instructions?: unknown; tools?: unknown; voice: VoiceTiming; output_ids: Set<string>; timer?: ReturnType<typeof setTimeout>;
  firstAudioMs?: number; audioBytes: number;
};
type RtState = {
  at: Scope | undefined; scrub?: Scrub | null; model?: string; instructions?: unknown; tools?: unknown;
  next?: { instructions?: unknown; tools?: unknown }; entries: RtEntry[]; seq: number; speechStoppedAt: string | null;
  current: RtResponse | null; held: RtResponse | null; closed: boolean; bytesPerMs: number;
  // The id of the last response recorded, and the highest item seq it covered. Later responses send only
  // newer items, chained by previous_response_id, so a long call costs one item per item, not the square.
  lastId?: string; sentSeq: number;
};

const RT_STATE = Symbol.for('generalcontext.realtime');
const rtLive = new Set<RtState>();
let rtExitHooked = false;

function patchRealtime(value: unknown) {
  const v = value as any;
  for (const cls of [v?.OpenAIRealtimeWS, v?.OpenAIRealtimeWebSocket, v?.default?.OpenAIRealtimeWS, v?.default?.OpenAIRealtimeWebSocket]) {
    const proto = cls?.prototype;
    if (!proto || typeof proto.send !== 'function' || typeof proto._emit !== 'function' || Object.prototype.hasOwnProperty.call(proto, MARK)) continue;
    Object.defineProperty(proto, MARK, { value: true });
    patchRealtimeMethods(proto, undefined);
    // The constructor assigns this.socket in the app's async scope, so that is where the session id is read.
    // Automatic reconnects assign it again; the state (and its session) stays the same.
    const SOCKET = Symbol('socket');
    Object.defineProperty(proto, 'socket', {
      configurable: true,
      get(this: any) { return this[SOCKET]; },
      set(this: any, socket: unknown) {
        Object.defineProperty(this, SOCKET, { value: socket, writable: true, configurable: true });
        try { watchSocketClose(this, rtState(this, undefined), socket); } catch { /* never break the app */ }
      },
    });
  }
}

function patchRealtimeMethods(owner: any, scrub: Scrub | null | undefined) {
  const emit = owner._emit, send = owner.send, sendRef = owner.sendResponseCreate;
  if (typeof emit === 'function' && !emit[MARK]) {
    owner._emit = function (this: any, name: string, ...args: unknown[]) {
      try {
        if (name === 'event') rtServerEvent(rtState(this, scrub), args[0] as any);
        else if (name === 'close') rtClose(rtState(this, scrub)); // connections with automatic reconnects
      } catch { /* never break the app */ }
      return emit.call(this, name, ...args);
    };
    owner._emit[MARK] = true;
  }
  for (const [key, original] of [['send', send], ['sendResponseCreate', sendRef]] as const) {
    if (typeof original !== 'function' || original[MARK]) continue;
    owner[key] = function (this: any, event: unknown, ...rest: unknown[]) {
      try { rtClientEvent(rtState(this, scrub), event as any); } catch { /* never break the app */ }
      return original.call(this, event, ...rest);
    };
    owner[key][MARK] = true;
  }
}

function rtState(conn: any, scrub: Scrub | null | undefined): RtState {
  let state: RtState = conn[RT_STATE];
  if (!state) {
    state = { at: scope.getStore(), scrub, entries: [], seq: 0, speechStoppedAt: null, current: null, held: null, closed: false, bytesPerMs: 48, sentSeq: 0 };
    try { state.model = new URL(String(conn.url)).searchParams.get('model') ?? undefined; } catch { /* no url yet */ }
    Object.defineProperty(conn, RT_STATE, { value: state, configurable: true });
  } else if (scrub !== undefined && state.scrub === undefined) state.scrub = scrub;
  return state;
}

const rtWatched = new WeakSet<object>();
function watchSocketClose(conn: any, state: RtState, socket: any) {
  if (!socket || typeof socket !== 'object' || rtWatched.has(socket)) return;
  rtWatched.add(socket);
  const onClose = () => { if (conn.socket === socket || conn.socket === undefined) rtClose(state); };
  if (typeof socket.on === 'function') socket.on('close', onClose);
  else if (typeof socket.addEventListener === 'function') socket.addEventListener('close', onClose);
}

const rtText = (item: any): string | null => {
  const parts: string[] = [];
  for (const part of Array.isArray(item?.content) ? item.content : []) {
    const text = typeof part?.text === 'string' ? part.text : typeof part?.transcript === 'string' ? part.transcript : null;
    if (text) parts.push(text);
  }
  return parts.length ? parts.join('') : null;
};
const rtHasAudio = (item: any) => (Array.isArray(item?.content) ? item.content : []).some((p: any) => /audio/.test(String(p?.type)));

// A realtime item as a Responses item, text only. Returns null when there is no text yet (audio awaiting its transcript).
function rtConvert(item: any): RtItem | null {
  if (!item || typeof item !== 'object') return null;
  if (item.type === 'function_call') return { type: 'function_call', call_id: item.call_id, name: item.name, arguments: item.arguments ?? '' };
  if (item.type === 'function_call_output') return { type: 'function_call_output', call_id: item.call_id, output: item.output };
  if (item.type !== 'message') return null;
  const audio = rtHasAudio(item);
  // Caller audio with no transcript (input transcription is off, or it has not arrived yet) is kept as an
  // empty spoken turn, so the call still shows that the caller spoke. A transcript fills it in later.
  const text = rtText(item) ?? (audio && item.role !== 'assistant' ? '' : null);
  if (text == null) return null;
  if (item.role === 'assistant') return { type: 'message', role: 'assistant', content: [{ type: 'output_text', text }], ...(audio ? { audio: true } : {}) };
  return { role: item.role ?? 'user', content: [{ type: 'input_text', text }], ...(audio ? { audio: true } : {}) };
}

function rtEntry(state: RtState, id: string | null | undefined, match?: (e: RtEntry) => boolean): RtEntry {
  let entry = id ? state.entries.find(e => e.id === id) : undefined;
  if (!entry && match) entry = state.entries.find(e => !e.id && match(e));
  if (entry) { if (id) entry.id = id; return entry; }
  entry = { id: id ?? null, item: null, seq: ++state.seq };
  state.entries.push(entry);
  if (state.entries.length > REALTIME_MAX_ITEMS) state.entries.splice(0, state.entries.length - REALTIME_MAX_ITEMS);
  return entry;
}

function rtUpsert(state: RtState, raw: any, responseId?: string) {
  const id = typeof raw?.id === 'string' ? raw.id : null;
  const converted = rtConvert(raw);
  // The app's own conversation.item.create usually has no id. Match the server's echo of it by content.
  const same = converted && JSON.stringify(converted);
  const entry = rtEntry(state, id, e => !!same && !!e.item && JSON.stringify(e.item) === same);
  // An untranscribed echo of an item never erases a transcript that already arrived.
  const blank = converted && 'role' in converted && converted.role !== 'assistant' && !(converted as any).content?.[0]?.text;
  if (converted && !(blank && entry.item)) entry.item = { ...entry.item, ...converted };
  if (responseId) entry.response = responseId;
}

function rtServerEvent(state: RtState, event: any) {
  const type = typeof event?.type === 'string' ? event.type : '';
  const now = new Date().toISOString();
  switch (type) {
    case 'session.created': case 'session.updated': {
      const s = event.session ?? {};
      if (typeof s.model === 'string') state.model = s.model;
      if ('instructions' in s) state.instructions = s.instructions;
      if ('tools' in s) state.tools = s.tools;
      rtFormat(state, s);
      break;
    }
    case 'conversation.item.added': case 'conversation.item.created': case 'conversation.item.done': {
      const r = state.current;
      const own = r && typeof event.item?.id === 'string' && r.output_ids.has(event.item.id) ? r.id : undefined;
      rtUpsert(state, event.item, own);
      break;
    }
    case 'conversation.item.input_audio_transcription.completed': {
      const entry = rtEntry(state, event.item_id);
      entry.item = { role: 'user', content: [{ type: 'input_text', text: String(event.transcript ?? '') }], audio: true, at: now };
      break;
    }
    // Every user turn starts with speech_started. It interrupts the response being generated, or the last one
    // while its audio is still playing (audio length estimated from the delta sizes, the bytes are not kept).
    case 'input_audio_buffer.speech_started': {
      const r = state.current ?? state.held;
      if (r && (r === state.current || Date.now() < rtPlaybackEnd(state, r))) r.voice.interrupted = true;
      break;
    }
    case 'input_audio_buffer.speech_stopped': state.speechStoppedAt = now; break;
    case 'response.created': {
      rtRelease(state);
      if (state.current) rtFinish(state, 'a new response started before response.done');
      state.current = rtStart(state, event.response?.id, now);
      break;
    }
    case 'response.output_item.added': case 'response.output_item.done':
      rtOwn(state, event.response_id, event.item?.id);
      rtUpsert(state, event.item, rtResponseFor(state, event.response_id)?.id);
      break;
    case 'response.output_audio.delta': case 'response.audio.delta': {
      const r = rtResponseFor(state, event.response_id);
      if (r && !r.voice.first_audio_at) {
        r.voice.first_audio_at = now; r.firstAudioMs = Date.now();
        // The caller has been waiting since they stopped speaking, through any silent response before
        // this one (a tool call): the wait ends at the first audio they hear.
        if (state.speechStoppedAt) r.voice.speech_stopped_at = state.speechStoppedAt;
        state.speechStoppedAt = null;
      }
      if (r && typeof event.delta === 'string') r.audioBytes += Math.floor(event.delta.length * 3 / 4);
      break;
    }
    case 'response.output_audio_transcript.done': case 'response.audio_transcript.done': case 'response.output_text.done': case 'response.text.done': {
      const r = rtResponseFor(state, event.response_id);
      rtOwn(state, event.response_id, event.item_id);
      const entry = rtEntry(state, event.item_id);
      const text = String(event.transcript ?? event.text ?? '');
      const audio = type.includes('audio');
      entry.item = { type: 'message', role: 'assistant', content: [{ type: 'output_text', text }], ...(audio ? { audio: true } : {}) };
      if (r) entry.response = r.id;
      break;
    }
    case 'response.function_call_arguments.done': {
      rtOwn(state, event.response_id, event.item_id);
      const entry = rtEntry(state, event.item_id);
      entry.item = { type: 'function_call', call_id: event.call_id, name: event.name, arguments: event.arguments ?? '' };
      const r = rtResponseFor(state, event.response_id);
      if (r) entry.response = r.id;
      break;
    }
    case 'conversation.item.truncated': rtTruncate(state, event.item_id, event.audio_end_ms); break;
    case 'error': {
      const r = state.current;
      if (r) r.error = `realtime error: ${String(event.error?.message ?? 'unknown error')}${event.error?.code ? ` (${event.error.code})` : ''}`;
      break;
    }
    case 'response.done': {
      const res = event.response ?? {};
      let r = rtResponseFor(state, res.id);
      if (!r) r = state.current = rtStart(state, res.id, now);
      for (const item of Array.isArray(res.output) ? res.output : []) { rtOwn(state, r.id, item?.id); rtUpsert(state, item, r.id); }
      r.ended_at = now;
      r.usage = res.usage ?? null;
      if (res.status === 'failed' && !r.error) r.error = `response failed: ${JSON.stringify(res.status_details ?? null)}`;
      if (res.status === 'cancelled') r.voice.interrupted = true;
      if (state.current === r) state.current = null;
      rtHold(state, r);
      break;
    }
  }
}

function rtClientEvent(state: RtState, event: any) {
  switch (event?.type) {
    case 'session.update': {
      const s = event.session ?? {};
      if (typeof s.model === 'string') state.model = s.model;
      if ('instructions' in s) state.instructions = s.instructions;
      if ('tools' in s) state.tools = s.tools;
      rtFormat(state, s);
      break;
    }
    case 'conversation.item.create': rtUpsert(state, event.item); break;
    case 'response.create': {
      const r = event.response ?? {};
      if ('instructions' in r || 'tools' in r) state.next = { instructions: r.instructions, tools: r.tools };
      break;
    }
    case 'response.cancel': rtInterrupt(state); break;
    case 'conversation.item.truncate': rtTruncate(state, event.item_id, event.audio_end_ms); break;
  }
}

function rtStart(state: RtState, id: unknown, now: string): RtResponse {
  const r: RtResponse = {
    id: typeof id === 'string' ? id : `resp_${randomUUID()}`, seq: ++state.seq, started_at: now, error: null, output_ids: new Set(), audioBytes: 0,
    voice: { speech_stopped_at: state.speechStoppedAt, first_audio_at: null, interrupted: false },
  };
  if (state.next) { r.instructions = state.next.instructions; r.tools = state.next.tools; state.next = undefined; }
  rtLive.add(state);
  return r;
}

// Output audio is pcm16 at 24 kHz (48 bytes per ms) unless the session picked G.711 (8 bytes per ms).
function rtFormat(state: RtState, session: any) {
  const format = session?.audio?.output?.format?.type ?? session?.output_audio_format;
  if (typeof format === 'string') state.bytesPerMs = /pcmu|pcma|g711/.test(format) ? 8 : 48;
}

const rtPlaybackEnd = (state: RtState, r: RtResponse) =>
  r.firstAudioMs === undefined ? 0 : r.firstAudioMs + r.audioBytes / state.bytesPerMs;

const rtResponseFor = (state: RtState, id: unknown): RtResponse | null =>
  [state.current, state.held].find(r => r && (!id || r.id === id)) ?? null;

function rtOwn(state: RtState, responseId: unknown, itemId: unknown) {
  const r = rtResponseFor(state, responseId);
  if (r && typeof itemId === 'string') r.output_ids.add(itemId);
}

// The response the caller is hearing: the one in flight, else the last finished one still playing.
function rtInterrupt(state: RtState) {
  const r = state.current ?? state.held;
  if (r) r.voice.interrupted = true;
}

function rtTruncate(state: RtState, itemId: unknown, ms: unknown) {
  const r = [state.current, state.held].find(x => x && typeof itemId === 'string' && x.output_ids.has(itemId)) ?? state.current ?? state.held;
  if (!r) return;
  r.voice.interrupted = true;
  if (typeof ms === 'number') r.voice.truncated_at_ms = ms;
}

function rtHold(state: RtState, r: RtResponse) {
  rtRelease(state);
  state.held = r;
  r.timer = setTimeout(() => { if (state.held === r) rtRelease(state); }, REALTIME_HOLD_MS);
  r.timer.unref?.();
  if (!rtExitHooked) {
    rtExitHooked = true;
    process.once('beforeExit', () => {
      for (const s of rtLive) { rtRelease(s); }
      if (queue.length) void flush(EXIT_FLUSH_MS);
    });
  }
}

function rtRelease(state: RtState) {
  const r = state.held;
  if (!r) return;
  state.held = null;
  if (r.timer) clearTimeout(r.timer);
  rtRecord(state, r);
  if (!state.current) rtLive.delete(state);
}

function rtFinish(state: RtState, error: string) {
  const r = state.current;
  if (!r) return;
  state.current = null;
  r.ended_at = new Date().toISOString();
  r.error ??= error;
  rtRecord(state, r);
}

function rtClose(state: RtState) {
  if (state.closed) return;
  state.closed = true;
  rtRelease(state);
  rtFinish(state, 'connection closed before response.done');
  rtLive.delete(state);
}

function rtRecord(state: RtState, r: RtResponse) {
  try {
    const input: RtItem[] = [], output: RtItem[] = [];
    // Timing goes on the stored assistant item too, so every later call's input still carries it.
    const stopped = r.voice.speech_stopped_at ? Date.parse(r.voice.speech_stopped_at) : NaN;
    const audio = r.voice.first_audio_at ? Date.parse(r.voice.first_audio_at) : NaN;
    const timing: Record<string, unknown> = {
      latency_ms: Number.isFinite(stopped) && Number.isFinite(audio) ? audio - stopped : null, interrupted: r.voice.interrupted,
    };
    if (r.voice.truncated_at_ms !== undefined) timing.truncated_at_ms = r.voice.truncated_at_ms;
    for (const e of state.entries) {
      if (!e.item) continue;
      const mine = e.response === r.id || (e.id && r.output_ids.has(e.id));
      if (mine && e.item.role === 'assistant') e.item = { ...e.item, voice: timing };
      if (mine) output.push(e.item);
      else if (e.seq < r.seq && e.seq > state.sentSeq) input.push(e.item);
    }
    const request: Record<string, unknown> = {
      model: state.model ?? null,
      instructions: r.instructions ?? state.instructions ?? null,
      tools: r.tools ?? state.tools ?? [],
      input,
    };
    // After the first reply, only what is new since the last one, chained to it, as the Responses API does.
    if (state.lastId) request.previous_response_id = state.lastId;
    state.lastId = r.id;
    state.sentSeq = Math.max(state.sentSeq, ...state.entries.filter(e => e.item && (e.seq < r.seq || e.response === r.id || (e.id && r.output_ids.has(e.id)))).map(e => e.seq));
    record({
      api: 'openai-realtime', request, response: { id: r.id, output, usage: r.usage ?? null },
      started_at: r.started_at, ended_at: r.ended_at ?? new Date().toISOString(), error: r.error, voice: { ...r.voice },
    }, state.at?.tags && !state.at.tags.channel ? { ...state.at, tags: { ...state.at.tags, channel: 'voice' } } : state.at, state.scrub);
  } catch (error) { warnLimited('record', `could not record a realtime response: ${(error as Error).message}`); }
}

const PATTERNS: [string, RegExp][] = [
  // Bounded repeats and a token-start lookbehind keep this linear on adversarial input.
  ['email', /(?<![A-Za-z0-9._%+-])[A-Za-z0-9._%+-]{1,64}@[A-Za-z0-9-]{1,63}(?:\.[A-Za-z0-9-]{1,63}){0,8}\.[A-Za-z]{2,24}/g],
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

/**
 * Record one client instance only, without initialize(). Returns the same client.
 * `redact` applies to this client only. If initialize() already patched the SDK, its settings apply instead.
 */
export function wrap<C extends object>(client: C, options: { redact?: boolean | Scrub } = {}): C {
  if (disabledByEnv()) return client;
  const scrub = options.redact === undefined ? undefined : redactor(options.redact);
  const target = client as Record<string, any>;
  const owners: [Record<string, any>, Api][] = [];
  if (target.chat?.completions?.create) owners.push([target.chat.completions, 'openai-chat']);
  if (target.responses?.create) owners.push([target.responses, 'openai-responses']);
  if (!target.chat && target.messages?.create) owners.push([target.messages, 'anthropic']);
  if (!target.chat && target.beta?.messages?.create) owners.push([target.beta.messages, 'anthropic']);
  const models = target.models;
  const gemini = typeof models?.generateContent === 'function';
  const bedrock = typeof target.send === 'function' && target.config && !gemini && !owners.length;
  // An OpenAIRealtimeWS or OpenAIRealtimeWebSocket connection. Its session is the one wrap() runs in.
  const realtime = typeof target.send === 'function' && typeof target._emit === 'function' && 'socket' in target;
  if (realtime) {
    patchRealtimeMethods(target, scrub);
    try { watchSocketClose(target, rtState(target, scrub), target.socket); } catch { /* never break the app */ }
    return client;
  }
  if (!owners.length && !gemini && !bedrock) throw new TypeError('wrap expects an OpenAI, Anthropic, Google Gen AI or Bedrock Runtime client, or an OpenAI Realtime connection');
  for (const [owner, api] of owners) {
    if (!owner.create[MARK]) owner.create = recording(owner.create.bind(owner), api, scrub);
  }
  if (gemini && !models.generateContent[MARK]) models.generateContent = recording(models.generateContent.bind(models), 'gemini', scrub);
  if (gemini && typeof models.generateContentStream === 'function' && !models.generateContentStream[MARK]) models.generateContentStream = geminiStreamRecording(models.generateContentStream.bind(models), scrub);
  if (bedrock && !target.send[MARK]) target.send = bedrockRecording(target.send.bind(target), scrub);
  return client;
}
