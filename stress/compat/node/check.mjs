// Layer 1 compatibility check for one installed provider package. Prints one RESULT JSON line.
//
//   node check.mjs openai|@anthropic-ai/sdk|@google/genai|@aws-sdk/client-bedrock-runtime <fake-url>
//
// Same contract as python/check.py: run every call style once without initialize() and once with
// it, against the local fake. Assert identical agent results, exactly one recording per call,
// complete recorded request/response, and provider errors (400, 429, 500, timeout) unchanged.
import * as gc from 'generalcontext';

const [pkg, URL_] = process.argv.slice(2);
const USER = 'MARKER-user-asks-about-unit-4B';
const SYSTEM = 'MARKER-system-be-brief';

class NA extends Error {}
const need = (ok, why) => { if (!ok) throw new NA(why); };

// Agent-visible result as comparable JSON, minus per-request noise (dates, request ids, headers).
const NOISE = new Set(['sdkHttpResponse', '$metadata', 'headers', '_request_id', 'requestId']);
function plain(value) {
  return JSON.parse(JSON.stringify(value ?? null, (k, v) => (NOISE.has(k) ? undefined : typeof v === 'bigint' ? String(v) : v)));
}
async function collect(iterable) { const out = []; for await (const x of iterable) out.push(plain(x)); return out; }
const errorSig = e => ({ type: e?.constructor?.name, message: String(e?.message ?? e).replace(/\d+ ?ms/g, 'Nms'), status: e?.status ?? e?.$metadata?.httpStatusCode ?? e?.code });

const TOOLS_OPENAI = [{ type: 'function', function: { name: 'get_unit', description: 'Look up a unit', parameters: { type: 'object', properties: { unit: { type: 'string' } } } } }];
const TOOLS_RESPONSES = [{ type: 'function', name: 'get_unit', description: 'Look up a unit', parameters: { type: 'object', properties: { unit: { type: 'string' } } }, strict: false }];
const TOOLS_ANTHROPIC = [{ name: 'get_unit', description: 'Look up a unit', input_schema: { type: 'object', properties: { unit: { type: 'string' } } } }];
const UNIT_SCHEMA = { type: 'object', properties: { unit: { type: 'string' }, open: { type: 'boolean' } }, required: ['unit', 'open'], additionalProperties: false };

async function openaiStyles() {
  const mod = await import('openai');
  const OpenAI = mod.default ?? mod.OpenAI;
  const client = () => new OpenAI({ apiKey: 'sk-fake', baseURL: `${URL_}/v1`, maxRetries: 0, timeout: 1000 });
  const messages = [{ role: 'system', content: SYSTEM }, { role: 'user', content: USER }];
  const chat = { model: 'gpt-fake', messages, tools: TOOLS_OPENAI };
  const helper = (c, name) => {
    const m = c.chat.completions[name] ? c.chat.completions : c.beta?.chat?.completions;
    need(typeof m?.[name] === 'function', `no chat.completions.${name}`);
    return (...a) => m[name](...a);
  };
  const responses = c => { need(c.responses?.create, 'no responses API'); return c.responses; };
  const rsp = { model: 'gpt-fake', input: USER, instructions: SYSTEM, tools: TOOLS_RESPONSES };
  const jsonFormat = { type: 'json_schema', json_schema: { name: 'Unit', schema: UNIT_SCHEMA, strict: true } };
  const azure = () => { need(mod.AzureOpenAI, 'no AzureOpenAI'); return new mod.AzureOpenAI({ apiKey: 'sk-fake', apiVersion: '2024-10-21', endpoint: URL_, maxRetries: 0, timeout: 1000 }); };
  const styles = [
    ['chat.create', async () => plain(await client().chat.completions.create(chat)), { tools: true }],
    ['chat.create stream', async () => collect(await client().chat.completions.create({ ...chat, stream: true, stream_options: { include_usage: true } })), { tools: true }],
    ['chat.create withResponse', async () => { const { data, response } = await client().chat.completions.create(chat).withResponse(); return [response.status, plain(data)]; }, { tools: true }],
    ['chat.stream helper', async () => { const s = helper(client(), 'stream')({ ...chat, stream_options: { include_usage: true } }); const ev = await collect(s); return [ev, plain(await s.finalChatCompletion())]; }, { tools: true }],
    ['chat.parse', async () => { const r = await helper(client(), 'parse')({ model: 'gpt-fake', messages, response_format: jsonFormat }); return [plain(r), plain(r.choices[0].message.parsed)]; }, { structured: true }],
    ['responses.create', async () => plain(await responses(client()).create(rsp)), { tools: true }],
    ['responses.create stream', async () => collect(await responses(client()).create({ ...rsp, stream: true })), { tools: true }],
    ['responses.stream helper', async () => { const r = responses(client()); need(r.stream, 'no responses.stream'); const s = r.stream(rsp); const ev = await collect(s); return [ev, plain(await s.finalResponse())]; }, { tools: true }],
    ['responses.parse', async () => { const r = responses(client()); need(r.parse, 'no responses.parse'); const out = await r.parse({ model: 'gpt-fake', input: USER, instructions: SYSTEM, text: { format: { type: 'json_schema', name: 'Unit', schema: UNIT_SCHEMA, strict: true } } }); return [plain(out), plain(out.output_parsed)]; }, { structured: true }],
    ['azure chat.create', async () => plain(await azure().chat.completions.create(chat)), { tools: true }],
  ];
  const errors = [
    ['chat.create', model => async () => client().chat.completions.create({ model, messages })],
    ['chat.create stream', model => async () => collect(await client().chat.completions.create({ model, messages, stream: true }))],
    ['chat.stream helper', model => async () => { const s = helper(client(), 'stream')({ model, messages }); return collect(s); }],
  ];
  return { styles, errors, modules: { openAI: mod } };
}

async function anthropicStyles() {
  const mod = await import('@anthropic-ai/sdk');
  const Anthropic = mod.default ?? mod.Anthropic;
  const client = () => new Anthropic({ apiKey: 'sk-fake', baseURL: URL_, maxRetries: 0, timeout: 1000 });
  const req = { model: 'claude-fake', max_tokens: 64, system: SYSTEM, messages: [{ role: 'user', content: USER }], tools: TOOLS_ANTHROPIC };
  const beta = c => { need(c.beta?.messages?.create, 'no beta.messages'); return c.beta.messages; };
  const helper = async owner => { need(owner.stream, 'no stream helper'); const s = owner.stream(req); const ev = await collect(s); return [ev, plain(await s.finalMessage())]; };
  const parse = async owner => {
    need(typeof owner.parse === 'function', 'no parse');
    const r = await owner.parse({ model: 'claude-fake', max_tokens: 64, messages: [{ role: 'user', content: USER }], output_format: { type: 'json_schema', schema: UNIT_SCHEMA } });
    return [plain(r), plain(r.parsed_output)];
  };
  const styles = [
    ['messages.create', async () => plain(await client().messages.create(req)), { tools: true }],
    ['messages.create stream', async () => collect(await client().messages.create({ ...req, stream: true })), { tools: true }],
    ['messages.create withResponse', async () => { const { data, response } = await client().messages.create(req).withResponse(); return [response.status, plain(data)]; }, { tools: true }],
    ['messages.stream helper', async () => helper(client().messages), { tools: true }],
    ['messages.parse', async () => parse(client().messages), { structured: true }],
    ['beta.messages.create', async () => plain(await beta(client()).create(req)), { tools: true }],
    ['beta.messages.stream helper', async () => helper(beta(client())), { tools: true }],
    ['beta.messages.parse', async () => parse(beta(client())), { structured: true }],
  ];
  const errors = [
    // Per-call timeout too: old releases ignore the client's timeout for messages.
    ['messages.create', model => async () => client().messages.create({ ...req, model }, { timeout: 1000 })],
    ['messages.create stream', model => async () => collect(await client().messages.create({ ...req, model, stream: true }, { timeout: 1000 }))],
    ['messages.stream helper', model => async () => { const s = client().messages.stream({ ...req, model }, { timeout: 1000 }); await collect(s); return s.finalMessage(); }],
  ];
  return { styles, errors, modules: { anthropic: mod } };
}

async function genaiStyles() {
  const mod = await import('@google/genai');
  const client = () => new mod.GoogleGenAI({ apiKey: 'fake', httpOptions: { baseUrl: URL_, apiVersion: 'v1beta', timeout: 1000 } });
  const config = { systemInstruction: SYSTEM, tools: [{ functionDeclarations: [{ name: 'get_unit', description: 'Look up a unit', parameters: { type: 'OBJECT', properties: { unit: { type: 'STRING' } } } }] }] };
  const req = { model: 'gemini-fake', contents: USER, config };
  const styles = [
    ['models.generateContent', async () => plain(await client().models.generateContent(req)), { tools: true }],
    ['models.generateContentStream', async () => collect(await client().models.generateContentStream(req)), { tools: true }],
    ['chats.sendMessage', async () => { const c = client(); need(c.chats?.create, 'no chats'); return plain(await c.chats.create({ model: 'gemini-fake', config }).sendMessage({ message: USER })); }, { tools: true }],
  ];
  const errors = [
    ['models.generateContent', model => async () => client().models.generateContent({ ...req, model })],
    ['models.generateContentStream', model => async () => collect(await client().models.generateContentStream({ ...req, model }))],
  ];
  return { styles, errors, modules: { googleGenAI: mod } };
}

async function bedrockStyles() {
  const mod = await import('@aws-sdk/client-bedrock-runtime');
  need(mod.ConverseCommand, 'no ConverseCommand');
  // Newer clients default to HTTP/2; the fake (like most proxies) speaks HTTP/1.1.
  const { createRequire } = await import('node:module');
  const fromClient = createRequire(createRequire(import.meta.url).resolve('@aws-sdk/client-bedrock-runtime'));
  const { NodeHttpHandler } = fromClient('@smithy/node-http-handler');
  const client = () => new mod.BedrockRuntimeClient({ region: 'us-east-1', endpoint: URL_, maxAttempts: 1, requestHandler: new NodeHttpHandler({ requestTimeout: 1000 }), credentials: { accessKeyId: 'AKIDFAKE', secretAccessKey: 'fake' } });
  const req = { modelId: 'anthropic.claude-fake', system: [{ text: SYSTEM }], messages: [{ role: 'user', content: [{ text: USER }] }], toolConfig: { tools: [{ toolSpec: { name: 'get_unit', description: 'Look up a unit', inputSchema: { json: { type: 'object', properties: { unit: { type: 'string' } } } } } }] } };
  const styles = [
    ['ConverseCommand', async () => plain(await client().send(new mod.ConverseCommand(req))), { tools: true }],
    ['ConverseStreamCommand', async () => { const out = await client().send(new mod.ConverseStreamCommand(req)); return collect(out.stream); }, { tools: true }],
  ];
  const errors = [
    // A per-call abort signal is the timeout that behaves the same on every client version.
    ['ConverseCommand', model => async () => client().send(new mod.ConverseCommand({ ...req, modelId: model }), { abortSignal: AbortSignal.timeout(1000) })],
    ['ConverseStreamCommand', model => async () => { const out = await client().send(new mod.ConverseStreamCommand({ ...req, modelId: model }), { abortSignal: AbortSignal.timeout(1000) }); return collect(out.stream); }],
  ];
  return { styles, errors, modules: { bedrockRuntime: mod } };
}

const PACKAGES = { openai: openaiStyles, '@anthropic-ai/sdk': anthropicStyles, '@google/genai': genaiStyles, '@aws-sdk/client-bedrock-runtime': bedrockStyles };

async function run(fn) {
  try { return ['ok', await fn()]; } catch (e) { return e instanceof NA ? ['na', e.message] : ['error', e]; }
}

async function drain(initialized) {
  await new Promise(r => setTimeout(r, 60));
  if (initialized) await gc.flush();
  return (await fetch(`${URL_}/__calls`)).json();
}

function complete(call, expect, session) {
  const problems = [];
  const request = JSON.stringify(call.request ?? null), response = JSON.stringify(call.response ?? null);
  if (call.error) problems.push(`error recorded on success: ${call.error.slice(0, 120)}`);
  if (!request.includes(USER)) problems.push('request lacks the user message');
  if (expect.tools && !request.includes('get_unit')) problems.push('request lacks the tools');
  if (!request.includes('-fake')) problems.push('request lacks the model');
  if (call.response == null) problems.push('no response');
  else {
    if (expect.structured ? !/\\?"unit\\?": ?\\?"4B/.test(response) : !response.includes('Unit 4B')) problems.push('response lacks the reply text');
    if (expect.tools && !response.includes('get_unit')) problems.push('response lacks the tool call');
    if (!response.includes('4200') || !response.replace(/4200/g, '').includes('42')) problems.push('response lacks usage');
  }
  if (call.conversation?.id !== session) problems.push('wrong or missing session');
  return problems;
}

const ERROR_MODELS = { 400: 'err-400', 429: 'err-429', 500: 'err-500', timeout: 'err-timeout' };

async function main() {
  const { styles, errors, modules } = await PACKAGES[pkg]();
  const errorCases = errors.flatMap(([name, make]) => Object.entries(ERROR_MODELS).map(([label, model]) => [`${name} ${label}`, make(model)]));
  const baseline = {}, baselineErrors = {};
  for (const [name, fn] of styles) baseline[name] = await run(fn);
  for (const [name, fn] of errorCases) baselineErrors[name] = await run(fn);
  const stray = await drain(false);

  gc.initialize({ apiKey: 'gck_' + 'f'.repeat(48), baseUrl: URL_, appName: 'compat', disableBatch: true, instrumentModules: modules });
  const results = {};
  for (const [name, fn, expect] of styles) {
    const session = `s-${name}`;
    const [status, value] = await gc.withSession(session, () => run(fn));
    const calls = await drain(true);
    const [bs, bv] = baseline[name];
    if (bs === 'na') { results[name] = { status: 'n/a', detail: bv }; continue; }
    if (bs === 'error') { results[name] = { status: 'env', detail: `fails without the SDK: ${bv?.constructor?.name}: ${String(bv?.message).slice(0, 200)}` }; continue; }
    const problems = [];
    if (status !== bs) problems.push(`with the SDK: ${status}: ${status === 'error' ? `${value?.constructor?.name}: ${value?.message}`.slice(0, 300) : ''}`);
    else if (JSON.stringify(value) !== JSON.stringify(bv)) problems.push('agent result differs with initialize()');
    if (calls.length !== 1) problems.push(`recorded ${calls.length} times`);
    if (calls[0]) problems.push(...complete(calls[0], expect, session));
    results[name] = { status: problems.length ? 'fail' : 'pass', detail: problems.join('; ') };
  }
  for (const [name, fn] of errorCases) {
    const session = `s-${name}`;
    const [status, value] = await gc.withSession(session, () => run(fn));
    const calls = await drain(true);
    const [bs, bv] = baselineErrors[name];
    if (bs === 'na') { results[name] = { status: 'n/a', detail: bv }; continue; }
    const problems = [];
    if (bs !== 'error') problems.push(`fake did not fail without the SDK (${bs})`);
    else if (status !== 'error') problems.push(`with the SDK the agent got ${status}, not the error`);
    else if (JSON.stringify(errorSig(value)) !== JSON.stringify(errorSig(bv))) problems.push(`error changed: ${JSON.stringify(errorSig(value))} vs ${JSON.stringify(errorSig(bv))}`);
    if (calls.length !== 1) problems.push(`recorded ${calls.length} times`);
    else if (!calls[0].error) problems.push('no error recorded');
    else if (!JSON.stringify(calls[0].request ?? null).includes(USER)) problems.push('request of failed call incomplete');
    results[name] = { status: problems.length ? 'fail' : 'pass', detail: problems.join('; ') };
  }
  if (stray.length) results['no recording before initialize'] = { status: 'fail', detail: `${stray.length} calls recorded` };
  return results;
}

let out;
try { out = { package: pkg, results: await main() }; } catch (e) { out = { package: pkg, crash: String(e?.stack ?? e).slice(-2000) }; }
console.log('RESULT ' + JSON.stringify(out));
process.exit(0);
