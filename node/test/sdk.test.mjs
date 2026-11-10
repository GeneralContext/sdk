// Runs the compiled SDK against real openai and @anthropic-ai/sdk clients pointed at a local fake server.
import { test, before, after } from 'node:test';
import assert from 'node:assert/strict';
import http from 'node:http';
import { spawn } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import OpenAI from 'openai';
import Anthropic from '@anthropic-ai/sdk';
import { AzureOpenAI } from 'openai';

const optional = async name => { try { return await import(name); } catch { return null; } };
const genai = await optional('@google/genai');
const bedrock = await optional('@aws-sdk/client-bedrock-runtime');
const anthropicBedrock = await optional('@anthropic-ai/bedrock-sdk');
const anthropicVertex = await optional('@anthropic-ai/vertex-sdk');

const childPosts = new Map(); // keyed by request path
const collected = { calls: [], outcomes: [], events: [], batches: [] };
const sse = (res, events) => {
  res.writeHead(200, { 'content-type': 'text/event-stream' });
  for (const e of events) res.write(e);
  res.end();
};
const data = obj => `data: ${JSON.stringify(obj)}\n\n`;
const event = obj => `event: ${obj.type}\ndata: ${JSON.stringify(obj)}\n\n`;

const chatJSON = { id: 'c1', object: 'chat.completion', model: 'gpt-x', choices: [{ index: 0, message: { role: 'assistant', content: 'hi' }, finish_reason: 'stop' }], usage: { prompt_tokens: 1, completion_tokens: 1, total_tokens: 2 } };
const chatSSE = [
  data({ id: 'c2', model: 'gpt-x', choices: [{ index: 0, delta: { role: 'assistant', content: 'Let me ' } }] }),
  data({ id: 'c2', model: 'gpt-x', choices: [{ index: 0, delta: { content: 'check.' } }] }),
  data({ id: 'c2', model: 'gpt-x', choices: [{ index: 0, delta: { tool_calls: [{ index: 0, id: 'call_1', type: 'function', function: { name: 'get_unit', arguments: '{"un' } }] } }] }),
  data({ id: 'c2', model: 'gpt-x', choices: [{ index: 0, delta: { tool_calls: [{ index: 0, function: { arguments: 'it":"4B"}' } }] }, finish_reason: 'tool_calls' }] }),
  data({ id: 'c2', model: 'gpt-x', choices: [], usage: { prompt_tokens: 5, completion_tokens: 7, total_tokens: 12 } }),
  'data: [DONE]\n\n',
];
const anthropicJSON = { id: 'm1', type: 'message', role: 'assistant', model: 'claude-x', content: [{ type: 'text', text: 'hello' }], stop_reason: 'end_turn', usage: { input_tokens: 3, output_tokens: 1 } };
const anthropicSSE = [
  event({ type: 'message_start', message: { id: 'm2', type: 'message', role: 'assistant', model: 'claude-x', content: [], usage: { input_tokens: 9, output_tokens: 0 } } }),
  event({ type: 'content_block_start', index: 0, content_block: { type: 'text', text: '' } }),
  event({ type: 'content_block_delta', index: 0, delta: { type: 'text_delta', text: 'Checking' } }),
  event({ type: 'content_block_stop', index: 0 }),
  event({ type: 'content_block_start', index: 1, content_block: { type: 'tool_use', id: 'tu_1', name: 'get_unit', input: {} } }),
  event({ type: 'content_block_delta', index: 1, delta: { type: 'input_json_delta', partial_json: '{"unit":' } }),
  event({ type: 'content_block_delta', index: 1, delta: { type: 'input_json_delta', partial_json: '"4B"}' } }),
  event({ type: 'content_block_stop', index: 1 }),
  event({ type: 'message_delta', delta: { stop_reason: 'tool_use' }, usage: { output_tokens: 11 } }),
  event({ type: 'message_stop' }),
];

const responsesJSON = { id: 'r1', object: 'response', created_at: 1, model: 'gpt-x', status: 'completed', output: [{ type: 'message', id: 'msg1', role: 'assistant', status: 'completed', content: [{ type: 'output_text', text: '{"unit":"4B"}', annotations: [] }] }], usage: { input_tokens: 2, output_tokens: 3, total_tokens: 5 } };
const responsesSSE = [
  event({ type: 'response.created', sequence_number: 0, response: { ...responsesJSON, status: 'in_progress', output: [] } }),
  event({ type: 'response.output_item.added', sequence_number: 1, output_index: 0, item: { type: 'message', id: 'msg1', role: 'assistant', status: 'in_progress', content: [] } }),
  event({ type: 'response.content_part.added', sequence_number: 2, output_index: 0, item_id: 'msg1', content_index: 0, part: { type: 'output_text', text: '', annotations: [] } }),
  event({ type: 'response.output_text.delta', sequence_number: 3, output_index: 0, item_id: 'msg1', content_index: 0, delta: '{"unit":"4B"}' }),
  event({ type: 'response.output_text.done', sequence_number: 4, output_index: 0, item_id: 'msg1', content_index: 0, text: '{"unit":"4B"}' }),
  event({ type: 'response.content_part.done', sequence_number: 5, output_index: 0, item_id: 'msg1', content_index: 0, part: { type: 'output_text', text: '{"unit":"4B"}', annotations: [] } }),
  event({ type: 'response.output_item.done', sequence_number: 6, output_index: 0, item: responsesJSON.output[0] }),
  event({ type: 'response.completed', sequence_number: 7, response: responsesJSON }),
];
const geminiJSON = { candidates: [{ content: { role: 'model', parts: [{ text: 'hello' }] }, finishReason: 'STOP', index: 0 }], usageMetadata: { promptTokenCount: 4, candidatesTokenCount: 1, totalTokenCount: 5 }, modelVersion: 'gemini-x' };
const geminiSSE = [
  data({ candidates: [{ content: { role: 'model', parts: [{ text: 'Let me ' }] }, index: 0 }], modelVersion: 'gemini-x' }),
  data({ candidates: [{ content: { role: 'model', parts: [{ text: 'check.' }] }, index: 0 }], modelVersion: 'gemini-x' }),
  data({ candidates: [{ content: { role: 'model', parts: [{ functionCall: { name: 'get_unit', args: { unit: '4B' } } }] }, finishReason: 'STOP', index: 0 }], usageMetadata: { promptTokenCount: 6, candidatesTokenCount: 8, totalTokenCount: 14 }, modelVersion: 'gemini-x' }),
];
const converseJSON = { output: { message: { role: 'assistant', content: [{ text: 'hello' }] } }, stopReason: 'end_turn', usage: { inputTokens: 3, outputTokens: 1, totalTokens: 4 }, metrics: { latencyMs: 5 } };
const converseEvents = [
  ['messageStart', { role: 'assistant' }],
  ['contentBlockDelta', { contentBlockIndex: 0, delta: { text: 'Checking' } }],
  ['contentBlockStop', { contentBlockIndex: 0 }],
  ['contentBlockStart', { contentBlockIndex: 1, start: { toolUse: { toolUseId: 'tu_1', name: 'get_unit' } } }],
  ['contentBlockDelta', { contentBlockIndex: 1, delta: { toolUse: { input: '{"unit":' } } }],
  ['contentBlockDelta', { contentBlockIndex: 1, delta: { toolUse: { input: '"4B"}' } } }],
  ['contentBlockStop', { contentBlockIndex: 1 }],
  ['messageStop', { stopReason: 'tool_use' }],
  ['metadata', { usage: { inputTokens: 9, outputTokens: 11, totalTokens: 20 }, metrics: { latencyMs: 7 } }],
];
// AWS event stream framing: prelude (total, headers length, crc), headers, payload, message crc.
const CRC = Array.from({ length: 256 }, (_, n) => { let c = n; for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1; return c >>> 0; });
const crc32 = buf => { let c = 0xffffffff; for (const b of buf) c = CRC[(c ^ b) & 0xff] ^ (c >>> 8); return (c ^ 0xffffffff) >>> 0; };
function awsEvent(type, payload) {
  const header = (name, value) => { const n = Buffer.from(name), v = Buffer.from(value); const h = Buffer.alloc(1 + n.length + 1 + 2 + v.length); h[0] = n.length; n.copy(h, 1); h[1 + n.length] = 7; h.writeUInt16BE(v.length, 2 + n.length); v.copy(h, 4 + n.length); return h; };
  const headers = Buffer.concat([header(':event-type', type), header(':content-type', 'application/json'), header(':message-type', 'event')]);
  const body = Buffer.from(JSON.stringify(payload));
  const total = 12 + headers.length + body.length + 4;
  const prelude = Buffer.alloc(8); prelude.writeUInt32BE(total, 0); prelude.writeUInt32BE(headers.length, 4);
  const pc = Buffer.alloc(4); pc.writeUInt32BE(crc32(prelude), 0);
  const msg = Buffer.concat([prelude, pc, headers, body]);
  const mc = Buffer.alloc(4); mc.writeUInt32BE(crc32(msg), 0);
  return Buffer.concat([msg, mc]);
}
const lastRequests = [];

let server, base;
before(async () => {
  server = http.createServer(async (req, res) => {
    let body = '';
    for await (const chunk of req) body += chunk;
    const json = body ? JSON.parse(body) : {};
    const reply = (status, obj) => { res.writeHead(status, { 'content-type': 'application/json' }); res.end(JSON.stringify(obj)); };
    lastRequests.push({ url: req.url, body: json });
    if (req.url.startsWith('/child/')) {
      if (!childPosts.has(req.url)) childPosts.set(req.url, []);
      childPosts.get(req.url).push({ auth: req.headers.authorization, body: json });
      return reply(200, { ok: true });
    }
    if (req.url === '/api/platform-collect/sdk') { collected.batches.push(json); collected.calls.push(...json.calls); return reply(200, { ok: true }); }
    if (req.url === '/api/platform-collect/outcomes') { collected.outcomes.push({ auth: req.headers.authorization, ...json }); return reply(200, { ok: true }); }
    if (req.url === '/api/platform-collect/events') { collected.events.push(json); return reply(200, { ingested: 1, skipped: 0, failures: [] }); }
    if (req.url.split('?')[0].endsWith('/chat/completions')) {
      if (json.model === 'bad') return reply(400, { error: { message: 'bad model' } });
      return json.stream ? sse(res, chatSSE) : reply(200, chatJSON);
    }
    if (req.url.endsWith('/responses')) return json.stream ? sse(res, responsesSSE) : reply(200, responsesJSON);
    if (req.url.includes(':generateContent')) return json.contents?.[0]?.parts?.[0]?.text === 'bad' ? reply(400, { error: { code: 400, message: 'bad request', status: 'INVALID_ARGUMENT' } }) : reply(200, geminiJSON);
    if (req.url.includes(':streamGenerateContent')) return sse(res, geminiSSE);
    if (req.url.endsWith('/converse')) return reply(200, converseJSON);
    if (req.url.endsWith('/converse-stream')) {
      res.writeHead(200, { 'content-type': 'application/vnd.amazon.eventstream' });
      for (const [type, payload] of converseEvents) res.write(awsEvent(type, payload));
      return res.end();
    }
    if (req.url.endsWith('/invoke')) return reply(200, { ok: 1 });
    if (req.url.endsWith('/messages') || req.url.includes('/messages?')) return json.stream ? sse(res, anthropicSSE) : reply(200, anthropicJSON);
    reply(404, {});
  });
  await new Promise(r => server.listen(0, '127.0.0.1', r));
  base = `http://127.0.0.1:${server.address().port}`;
});
after(() => server.close());

const sdk = await import('../dist/index.js');
const openai = () => new OpenAI({ apiKey: 'k', baseURL: `${base}/v1`, maxRetries: 0 });
const anthropic = () => new Anthropic({ apiKey: 'k', baseURL: base, maxRetries: 0 });
const early = { openai: null, anthropic: null };

async function drain() {
  await new Promise(r => setTimeout(r, 30)); // let side-branch recorders run
  await sdk.flush();
  return collected.calls.splice(0);
}

test('initialize patches instances created before and after', async () => {
  early.openai = openai();
  early.anthropic = anthropic();
  process.env.GENERALCONTEXT_REDACT_SALT = 'salt';
  sdk.initialize({ apiKey: 'gc_key', baseUrl: base, appName: 'test-app', redact: true, instrumentFetch: true, instrumentModules: { openAI: OpenAI, anthropic: Anthropic, googleGenAI: genai ?? undefined, bedrockRuntime: bedrock ?? undefined } });
  const warnings = [];
  const warn = console.warn; console.warn = m => warnings.push(m);
  sdk.initialize({ apiKey: 'other' });
  console.warn = warn;
  assert.match(warnings[0], /already called/);

  await early.openai.chat.completions.create({ model: 'gpt-x', messages: [{ role: 'user', content: 'a' }] });
  await openai().chat.completions.create({ model: 'gpt-x', messages: [{ role: 'user', content: 'b' }] });
  await early.anthropic.messages.create({ model: 'claude-x', max_tokens: 5, messages: [{ role: 'user', content: 'c' }] });
  await anthropic().messages.create({ model: 'claude-x', max_tokens: 5, messages: [{ role: 'user', content: 'd' }] });
  const calls = await drain();
  assert.deepEqual(calls.map(c => c.api), ['openai-chat', 'openai-chat', 'anthropic', 'anthropic']);
  assert.equal(calls[0].response.choices[0].message.content, 'hi');
  assert.equal(calls[2].response.content[0].text, 'hello');
  assert.equal(collected.batches.at(-1).sdk, `node/${sdk.VERSION}`);
  assert.equal(collected.batches.at(-1).app, 'test-app');
  for (const c of calls) assert.deepEqual(Object.keys(c).sort(), ['api', 'conversation', 'ended_at', 'error', 'request', 'response', 'started_at', 'task']);
});

test('OpenAI stream passes chunks through and records reassembled tool calls', async () => {
  const stream = await openai().chat.completions.create({ model: 'gpt-x', stream: true, stream_options: { include_usage: true }, messages: [{ role: 'user', content: 'x' }] });
  assert.equal(typeof stream.tee, 'function');
  assert.equal(typeof stream.toReadableStream, 'function');
  const chunks = [];
  for await (const chunk of stream) chunks.push(chunk);
  assert.equal(chunks.length, 5);
  const [call] = await drain();
  const message = call.response.choices[0].message;
  assert.equal(message.content, 'Let me check.');
  assert.deepEqual(message.tool_calls, [{ id: 'call_1', type: 'function', function: { name: 'get_unit', arguments: '{"unit":"4B"}' } }]);
  assert.equal(call.response.usage.total_tokens, 12);
});

test('OpenAI stream tee() still works and records once', async () => {
  const stream = await openai().chat.completions.create({ model: 'gpt-x', stream: true, messages: [{ role: 'user', content: 'x' }] });
  const [left, right] = stream.tee();
  let a = 0, b = 0;
  for await (const _ of left) a++;
  for await (const _ of right) b++;
  assert.equal(a, b);
  assert.equal((await drain()).length, 1);
});

test('Anthropic stream and messages.stream() record tool_use with parsed input', async () => {
  const stream = await early.anthropic.messages.create({ model: 'claude-x', max_tokens: 5, stream: true, messages: [{ role: 'user', content: 'x' }] });
  let events = 0;
  for await (const _ of stream) events++;
  assert.equal(events, 10);
  const final = await anthropic().messages.stream({ model: 'claude-x', max_tokens: 5, messages: [{ role: 'user', content: 'y' }] }).finalMessage();
  assert.deepEqual(final.content[1].input, { unit: '4B' });
  const calls = await drain();
  assert.equal(calls.length, 2);
  for (const call of calls) {
    assert.equal(call.response.content[0].text, 'Checking');
    assert.deepEqual(call.response.content[1], { type: 'tool_use', id: 'tu_1', name: 'get_unit', input: { unit: '4B' } });
    assert.equal(call.response.usage.output_tokens, 11);
    assert.equal(call.response.stop_reason, 'tool_use');
  }
});

test('raw fetch to /chat/completions is captured without consuming the body', async () => {
  const res = await fetch(`${base}/v1/chat/completions`, { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ model: 'gpt-x', messages: [] }) });
  assert.equal((await res.json()).choices[0].message.content, 'hi');
  const streamed = await fetch(`${base}/v1/chat/completions`, { method: 'POST', body: JSON.stringify({ model: 'gpt-x', stream: true, messages: [] }) });
  assert.match(await streamed.text(), /\[DONE\]/);
  const calls = await drain();
  assert.equal(calls.length, 2);
  assert.equal(calls[0].api, 'openai-chat');
  assert.equal(calls[0].response.choices[0].message.content, 'hi');
  assert.equal(calls[1].response.choices[0].message.tool_calls[0].function.arguments, '{"unit":"4B"}');
});

test('session and task tags survive awaits', async () => {
  await sdk.withSession({ sessionId: 's1', userId: 'u1', channel: 'sms' }, async () => {
    await new Promise(r => setTimeout(r, 5));
    await sdk.task('plan', async () => {
      await new Promise(r => setTimeout(r, 5));
      await openai().chat.completions.create({ model: 'gpt-x', messages: [] });
    });
    await openai().chat.completions.create({ model: 'gpt-x', messages: [] });
  });
  await sdk.conversation({ id: 'legacy' }, () => anthropic().messages.create({ model: 'claude-x', max_tokens: 5, messages: [] }));
  const calls = await drain();
  assert.deepEqual(calls[0].conversation, { id: 's1', user_id: 'u1', channel: 'sms' });
  assert.equal(calls[0].task, 'plan');
  assert.equal(calls[1].task, null);
  assert.equal(calls[1].conversation.id, 's1');
  assert.equal(calls[2].conversation.id, 'legacy');
});

test('rejected calls are recorded and rethrown unchanged', async () => {
  await assert.rejects(openai().chat.completions.create({ model: 'bad', messages: [] }), err => err instanceof OpenAI.BadRequestError);
  const [call] = await drain();
  assert.equal(call.response, null);
  assert.match(call.error, /bad model/);
});

test('redaction replaces personal details with stable placeholders', async () => {
  await openai().chat.completions.create({ model: 'gpt-x', messages: [{ role: 'user', content: 'mail jo@x.com or 612-555-0177' }] });
  const [call] = await drain();
  const text = call.request.messages[0].content;
  assert.doesNotMatch(text, /jo@x\.com|555/);
  assert.match(text, /<email:[0-9a-f]{6}> or <phone:[0-9a-f]{6}>/);
});

test('reportOutcome and logEvent post to the collector', async () => {
  assert.equal(await sdk.reportOutcome('s1', 'success', 'booked'), true);
  assert.deepEqual(collected.outcomes.at(-1), { auth: 'Bearer gc_key', session_id: 's1', outcome: 'success', note: 'booked' });
  assert.equal(await sdk.logEvent({ sessionId: 's1', eventType: 'tour.booked', eventId: 'e1', properties: { unit: '4B' }, timestamp: '2026-01-01T00:00:00.000Z' }), true);
  assert.deepEqual(collected.events.at(-1), { events: [{ event_id: 'e1', event_type: 'tour.booked', timestamp: '2026-01-01T00:00:00.000Z', session_id: 's1', properties: { unit: '4B' } }] });
});

test('reportOutcome and logEvent apply redaction so scrubbed session ids match', async () => {
  assert.equal(await sdk.reportOutcome('jo@x.com', 'fail', 'call 612-555-0177'), true);
  const outcome = collected.outcomes.at(-1);
  assert.match(outcome.session_id, /^<email:[0-9a-f]{6}>$/);
  assert.match(outcome.note, /<phone:/);
  await sdk.withSession('jo@x.com', () => openai().chat.completions.create({ model: 'gpt-x', messages: [] }));
  const [call] = await drain();
  assert.equal(call.conversation.id, outcome.session_id);
  assert.equal(await sdk.logEvent({ sessionId: 'jo@x.com', eventType: 'mail.jo@x.com', properties: { to: 'jo@x.com', n: 1 } }), true);
  const ev = collected.events.at(-1).events[0];
  assert.equal(ev.session_id, outcome.session_id);
  assert.doesNotMatch(JSON.stringify(ev), /jo@x\.com/);
  assert.equal(ev.properties.n, 1);
});

test('email redaction is linear on a 1 MB adversarial string', async () => {
  const evil = 'a'.repeat(500_000) + '@' + 'a.'.repeat(250_000);
  const started = Date.now();
  await openai().chat.completions.create({ model: 'gpt-x', messages: [{ role: 'user', content: evil }] });
  const [call] = await drain();
  assert.ok(Date.now() - started < 1000, `took ${Date.now() - started} ms`);
  assert.ok(call);
});

test('SDK calls are recorded but raw fetch is not without instrumentFetch', async () => {
  const out = await child(`
    initialize({ apiKey: 'k', baseUrl: BASE + '/child/nofetch', instrumentModules: {} });
    await fetch(BASE + '/v1/chat/completions', { method: 'POST', body: JSON.stringify({ model: 'gpt-x', messages: [] }) });
    await new Promise(r => setTimeout(r, 30)); await flush();`);
  assert.equal(out.code, 0);
  assert.equal(childPosts.get('/child/nofetch/api/platform-collect/sdk'), undefined);
});

// Run a short ES module against the built SDK in a fresh process, with BASE pointing at the fake server.
const dist = fileURLToPath(new URL('../dist/index.js', import.meta.url));
function child(code, env = {}) {
  const src = `import { initialize, flush, reportOutcome, logEvent, wrap } from ${JSON.stringify(dist)};
    import OpenAI from 'openai';
    const BASE = ${JSON.stringify(base)};
    const record = (path) => { const o = new OpenAI({ apiKey: 'k', baseURL: BASE + '/v1', maxRetries: 0 }); return o; };
    ${code}`;
  return new Promise(resolve => {
    const p = spawn(process.execPath, ['--input-type=module', '-e', src], {
      cwd: fileURLToPath(new URL('..', import.meta.url)),
      env: { ...process.env, GENERALCONTEXT_API_KEY: '', GENERALCONTEXT_BASE_URL: '', GENERALCONTEXT_DISABLED: '', ...env },
    });
    let stderr = '', stdout = '';
    p.stderr.on('data', d => { stderr += d; });
    p.stdout.on('data', d => { stdout += d; });
    p.on('close', code => resolve({ code, stderr, stdout }));
  });
}
const callOnce = `{ const o = record(); await o.chat.completions.create({ model: 'gpt-x', messages: [{ role: 'user', content: 'hello' }] }); }`;
const recorded = path => (childPosts.get(`/child/${path}/api/platform-collect/sdk`) ?? []).flatMap(p => p.body.calls);

test('collector outage never throws', async () => {
  const out = await child(`
    initialize({ apiKey: 'k', baseUrl: 'http://127.0.0.1:1', instrumentModules: { openAI: OpenAI } });
    ${callOnce}
    await new Promise(r => setTimeout(r, 30)); await flush();
    if (await reportOutcome('s1', 'fail') !== false) process.exit(2);
    if (await logEvent({ sessionId: 's1', eventType: 'x' }) !== false) process.exit(3);`);
  assert.equal(out.code, 0, out.stderr);
  assert.equal((out.stderr.match(/could not reach \/api\/platform-collect\/sdk/g) ?? []).length, 1);
});

test('explicit baseUrl wins over GENERALCONTEXT_BASE_URL, which wins over the default', async () => {
  let out = await child(`initialize({ apiKey: 'k', baseUrl: BASE + '/child/opt', instrumentModules: { openAI: OpenAI } }); ${callOnce} await new Promise(r => setTimeout(r, 30)); await flush();`,
    { GENERALCONTEXT_BASE_URL: `${base}/child/env` });
  assert.equal(out.code, 0, out.stderr);
  assert.equal(recorded('opt').length, 1);
  assert.equal(recorded('env').length, 0);
  out = await child(`initialize({ apiKey: 'k', instrumentModules: { openAI: OpenAI } }); ${callOnce} await new Promise(r => setTimeout(r, 30)); await flush();`,
    { GENERALCONTEXT_BASE_URL: `${base}/child/env` });
  assert.equal(recorded('env').length, 1);
});

test('refuses to send the key over plain http to a non-local host', async () => {
  const out = await child(`
    globalThis.fetch = new Proxy(globalThis.fetch, { apply(t, s, a) { if (String(a[0]).includes('example.com')) process.exit(9); return Reflect.apply(t, s, a); } });
    initialize({ apiKey: 'k', baseUrl: 'http://collector.example.com', instrumentModules: { openAI: OpenAI } });
    ${callOnce} ${callOnce}
    await new Promise(r => setTimeout(r, 30)); await flush();
    if (await reportOutcome('s', 'fail') !== false) process.exit(2);`);
  assert.equal(out.code, 0, out.stderr);
  assert.equal((out.stderr.match(/refusing to send/g) ?? []).length, 1);
});

test('redirects are not followed and the key never reaches the redirect target', async () => {
  let leaked = [];
  const target = http.createServer((req, res) => { leaked.push(req.headers.authorization); res.end('{}'); });
  await new Promise(r => target.listen(0, '127.0.0.1', r));
  const redirector = http.createServer((req, res) => { res.writeHead(302, { location: `http://127.0.0.1:${target.address().port}${req.url}` }); res.end(); });
  await new Promise(r => redirector.listen(0, '127.0.0.1', r));
  try {
    const out = await child(`
      initialize({ apiKey: 'secret', baseUrl: 'http://127.0.0.1:${redirector.address().port}', instrumentModules: { openAI: OpenAI } });
      ${callOnce} await new Promise(r => setTimeout(r, 30)); await flush();
      if (await reportOutcome('s', 'fail') !== false) process.exit(2);`);
    assert.equal(out.code, 0, out.stderr);
    assert.deepEqual(leaked, []);
  } finally { target.close(); redirector.close(); }
});

test('oversized calls are dropped with one warning', async () => {
  const out = await child(`
    initialize({ apiKey: 'k', baseUrl: BASE + '/child/big', instrumentModules: { openAI: OpenAI } });
    const o = record();
    for (let i = 0; i < 2; i++) await o.chat.completions.create({ model: 'gpt-x', messages: [{ role: 'user', content: 'x'.repeat(4_100_000) }] });
    ${callOnce} await new Promise(r => setTimeout(r, 30)); await flush();`);
  assert.equal(out.code, 0, out.stderr);
  assert.equal((out.stderr.match(/byte limit/g) ?? []).length, 1);
  assert.deepEqual(recorded('big').map(c => c.request.messages[0].content), ['hello']);
});

test('batches split by bytes as well as count', async () => {
  const out = await child(`
    initialize({ apiKey: 'k', baseUrl: BASE + '/child/split', instrumentModules: { openAI: OpenAI } });
    const o = record();
    for (let i = 0; i < 5; i++) await o.chat.completions.create({ model: 'gpt-x', messages: [{ role: 'user', content: String(i) + 'x'.repeat(1_500_000) }] });
    await new Promise(r => setTimeout(r, 30)); await flush();`);
  assert.equal(out.code, 0, out.stderr);
  const posts = childPosts.get('/child/split/api/platform-collect/sdk');
  assert.deepEqual(posts.map(p => p.body.calls.length), [2, 2, 1]);
  assert.deepEqual(recorded('split').map(c => c.request.messages[0].content[0]), ['0', '1', '2', '3', '4']);
});

test('beforeSend can modify, drop, and throw', async () => {
  const out = await child(`
    let n = 0;
    initialize({ apiKey: 'k', baseUrl: BASE + '/child/hook', redact: true, instrumentModules: { openAI: OpenAI },
      beforeSend: call => {
        n++;
        if (!call.request.messages[0].content.includes('jo@x.com')) process.exit(4); // sees unredacted data
        if (n === 2) return null;
        if (n === 3) throw new Error('boom');
        call.task = 'edited'; return call;
      } });
    const o = record();
    for (let i = 0; i < 3; i++) await o.chat.completions.create({ model: 'gpt-x', messages: [{ role: 'user', content: 'jo@x.com ' + i }] });
    await new Promise(r => setTimeout(r, 30)); await flush();`, { GENERALCONTEXT_REDACT_SALT: 's' });
  assert.equal(out.code, 0, out.stderr);
  const calls = recorded('hook');
  assert.equal(calls.length, 1);
  assert.equal(calls[0].task, 'edited');
  assert.match(calls[0].request.messages[0].content, /^<email:[0-9a-f]{6}> 0$/);
  assert.match(out.stderr, /beforeSend threw/);
});

test('GENERALCONTEXT_DISABLED turns initialize into a no-op', async () => {
  const out = await child(`initialize({ apiKey: 'k', baseUrl: BASE + '/child/off', instrumentModules: { openAI: OpenAI } }); ${callOnce} await new Promise(r => setTimeout(r, 30)); await flush();`,
    { GENERALCONTEXT_DISABLED: 'TRUE' });
  assert.equal(out.code, 0, out.stderr);
  assert.match(out.stderr, /GENERALCONTEXT_DISABLED/);
  assert.equal(recorded('off').length, 0);
});

test('pending calls are flushed on exit without flush()', async () => {
  const out = await child(`initialize({ apiKey: 'k', baseUrl: BASE + '/child/exit', instrumentModules: { openAI: OpenAI } }); ${callOnce}`);
  assert.equal(out.code, 0, out.stderr);
  assert.equal(recorded('exit').length, 1);
});

test('exit flush gives up after 2 s against a hung collector', async () => {
  const hung = http.createServer(() => {});
  await new Promise(r => hung.listen(0, '127.0.0.1', r));
  try {
    const started = Date.now();
    const out = await child(`initialize({ apiKey: 'k', baseUrl: 'http://127.0.0.1:${hung.address().port}', instrumentModules: { openAI: OpenAI } }); ${callOnce}`);
    assert.equal(out.code, 0, out.stderr);
    assert.ok(Date.now() - started < 5000, `took ${Date.now() - started} ms`);
  } finally { hung.closeAllConnections?.(); hung.close(); }
});

test('flush(timeoutMs) returns false at the deadline against a hung collector', async () => {
  // flush() used to wait for every queued batch, 15 s each, so a serverless handler hung for minutes.
  const hung = http.createServer(() => {});
  await new Promise(r => hung.listen(0, '127.0.0.1', r));
  try {
    const out = await child(`initialize({ apiKey: 'k', baseUrl: 'http://127.0.0.1:${hung.address().port}', instrumentModules: { openAI: OpenAI } });
      for (let i = 0; i < 45; i++) ${callOnce}
      const t = Date.now(); const ok = await flush(300); const waited = Date.now() - t;
      if (ok !== false || waited > 1500) { console.error('flush', ok, waited); process.exit(5); }
      process.exit(0);`);
    assert.equal(out.code, 0, out.stderr);
  } finally { hung.closeAllConnections?.(); hung.close(); }
});

test('an unreachable collector is tried once per pause, not once per queued batch', async () => {
  // Each failed batch used to be followed by the next, 15 s apiece against a hung collector, so a finished
  // script stayed alive for minutes. Now the sender stops at the first failure and keeps the rest queued.
  let connections = 0;
  const reset = http.createServer();
  reset.on('connection', socket => { connections++; socket.destroy(); });
  await new Promise(r => reset.listen(0, '127.0.0.1', r));
  try {
    const out = await child(`initialize({ apiKey: 'k', baseUrl: 'http://127.0.0.1:${reset.address().port}', instrumentModules: { openAI: OpenAI } });
      for (let i = 0; i < 65; i++) ${callOnce}
      await new Promise(r => setTimeout(r, 300));
      if (await flush(1000) !== false) process.exit(6);
      process.exit(0);`);
    assert.equal(out.code, 0, out.stderr);
    assert.ok(connections <= 2, `${connections} connections`);
  } finally { reset.close(); }
});

test('wrap redact applies to that client only', async () => {
  const out = await child(`
    process.env.GENERALCONTEXT_API_KEY = 'k'; process.env.GENERALCONTEXT_BASE_URL = BASE + '/child/wrap';
    const a = wrap(record(), { redact: true }); const b = wrap(record());
    await a.chat.completions.create({ model: 'gpt-x', messages: [{ role: 'user', content: 'jo@x.com' }] });
    await b.chat.completions.create({ model: 'gpt-x', messages: [{ role: 'user', content: 'jo@x.com' }] });
    await new Promise(r => setTimeout(r, 30)); await flush();`, { GENERALCONTEXT_REDACT_SALT: 's' });
  assert.equal(out.code, 0, out.stderr);
  assert.deepEqual(recorded('wrap').map(c => c.request.messages[0].content.startsWith('<email:')), [true, false]);
});

// ---- 0.3.0 coverage ----

test('Anthropic beta messages: create, parse, stream and threads record each call once', async () => {
  const client = anthropic();
  await client.beta.messages.create({ model: 'claude-x', max_tokens: 5, messages: [{ role: 'user', content: 'a' }] });
  await client.beta.messages.parse({ model: 'claude-x', max_tokens: 5, messages: [{ role: 'user', content: 'b' }] });
  const final = await client.beta.messages.stream({ model: 'claude-x', max_tokens: 5, messages: [{ role: 'user', content: 'c' }] }).finalMessage();
  assert.deepEqual(final.content[1].input, { unit: '4B' });
  const calls = await drain();
  assert.deepEqual(calls.map(c => c.api), ['anthropic', 'anthropic', 'anthropic']);
  assert.deepEqual(calls.map(c => c.request.messages[0].content), ['a', 'b', 'c']);
  assert.equal(calls[0].response.content[0].text, 'hello');
  assert.deepEqual(calls[2].response.content[1].input, { unit: '4B' });
});

test('Anthropic beta toolRunner records each turn once', async () => {
  const client = anthropic();
  const runner = client.beta.messages.toolRunner({ model: 'claude-x', max_tokens: 5, max_iterations: 1, messages: [{ role: 'user', content: 'go' }], tools: [] , thread: false });
  let turns = 0;
  for await (const _ of runner) turns++;
  const calls = await drain();
  assert.ok(turns >= 1);
  assert.equal(calls.length, turns);
  assert.ok(calls.every(c => c.api === 'anthropic'));
});

test('AnthropicBedrock and AnthropicVertex share the patched Messages class', { skip: (!anthropicBedrock || !anthropicVertex) && 'bedrock/vertex sdk not installed' }, () => {
  const b = new anthropicBedrock.AnthropicBedrock({ awsRegion: 'us-east-1', awsAccessKey: 'x', awsSecretKey: 'y' });
  const v = new anthropicVertex.AnthropicVertex({ region: 'us-east5', projectId: 'p', accessToken: 't' });
  assert.ok(b.messages.create[Symbol.for('generalcontext.recorded')]);
  assert.ok(v.messages.create[Symbol.for('generalcontext.recorded')]);
});

test('OpenAI chat.completions.parse and stream record once', async () => {
  const client = openai();
  const parsed = await client.chat.completions.parse({ model: 'gpt-x', messages: [{ role: 'user', content: 'p' }] });
  assert.equal(parsed.choices[0].message.content, 'hi');
  const stream = client.chat.completions.stream({ model: 'gpt-x', messages: [{ role: 'user', content: 's' }] });
  for await (const _ of stream) { /* drain */ }
  await stream.finalChatCompletion();
  const calls = await drain();
  assert.deepEqual(calls.map(c => [c.api, c.request.messages[0].content]), [['openai-chat', 'p'], ['openai-chat', 's']]);
  assert.equal(calls[1].response.choices[0].message.content, 'Let me check.');
});

test('OpenAI responses.create, parse and stream record once each', async () => {
  const client = openai();
  await client.responses.create({ model: 'gpt-x', input: 'c' });
  const parsed = await client.responses.parse({ model: 'gpt-x', input: 'p' });
  assert.equal(parsed.output_text, '{"unit":"4B"}');
  const stream = client.responses.stream({ model: 'gpt-x', input: 's' });
  for await (const _ of stream) { /* drain */ }
  await stream.finalResponse();
  const calls = await drain();
  assert.deepEqual(calls.map(c => [c.api, c.request.input]), [['openai-responses', 'c'], ['openai-responses', 'p'], ['openai-responses', 's']]);
  assert.equal(calls[2].response.id, 'r1');
});

test('AzureOpenAI clients are recorded', async () => {
  const az = new AzureOpenAI({ apiKey: 'k', endpoint: base, apiVersion: '2024-10-21', deployment: 'dep', maxRetries: 0 });
  const out = await az.chat.completions.create({ model: 'gpt-x', messages: [{ role: 'user', content: 'azure' }] });
  assert.equal(out.choices[0].message.content, 'hi');
  const calls = await drain();
  assert.equal(calls.length, 1);
  assert.equal(calls[0].api, 'openai-chat');
  assert.ok(lastRequests.some(r => r.url.startsWith('/openai/deployments/dep/chat/completions')));
});

const gemini = () => new genai.GoogleGenAI({ apiKey: 'k', httpOptions: { baseUrl: base } });

test('Gemini generateContent records request and plain response', { skip: !genai && '@google/genai not installed' }, async () => {
  const config = { systemInstruction: 'be brief', tools: [{ functionDeclarations: [{ name: 'get_unit', description: 'd', parameters: { type: 'OBJECT', properties: {} } }] }] };
  const res = await gemini().models.generateContent({ model: 'gemini-x', contents: 'hello there', config });
  assert.equal(res.text, 'hello');
  const [call] = await drain();
  assert.equal(call.api, 'gemini');
  assert.deepEqual(call.request, { model: 'gemini-x', contents: 'hello there', config });
  assert.equal(call.response.candidates[0].content.parts[0].text, 'hello');
  assert.equal(call.response.usageMetadata.promptTokenCount, 4);
  assert.equal(call.response.modelVersion, 'gemini-x');
  assert.equal(call.response.sdkHttpResponse, undefined);
});

test('Gemini errors are recorded and rethrown', { skip: !genai && '@google/genai not installed' }, async () => {
  await assert.rejects(gemini().models.generateContent({ model: 'gemini-x', contents: [{ role: 'user', parts: [{ text: 'bad' }] }] }));
  const [call] = await drain();
  assert.equal(call.response, null);
  assert.match(call.error, /bad request|400/);
});

test('Gemini generateContentStream passes chunks through and records one assembled response', { skip: !genai && '@google/genai not installed' }, async () => {
  const stream = await gemini().models.generateContentStream({ model: 'gemini-x', contents: 'go' });
  const texts = [];
  for await (const chunk of stream) texts.push(chunk.text ?? null);
  assert.equal(texts.length, 3);
  const calls = await drain();
  assert.equal(calls.length, 1);
  const r = calls[0].response;
  assert.deepEqual(r.candidates[0].content.parts, [{ text: 'Let me check.' }, { functionCall: { name: 'get_unit', args: { unit: '4B' } } }]);
  assert.equal(r.candidates[0].finishReason, 'STOP');
  assert.equal(r.usageMetadata.candidatesTokenCount, 8);
});

const { NodeHttpHandler } = await optional('@smithy/node-http-handler') ?? {};
const bedrockClient = () => new bedrock.BedrockRuntimeClient({ region: 'us-east-1', endpoint: base, requestHandler: new NodeHttpHandler(), credentials: { accessKeyId: 'AKIDFAKE', secretAccessKey: 'fake' }, maxAttempts: 1 });

test('Bedrock Converse records input with bytes omitted and output without $metadata', { skip: !bedrock && '@aws-sdk/client-bedrock-runtime not installed' }, async () => {
  const input = { modelId: 'anthropic.claude-x', system: [{ text: 'be brief' }], messages: [{ role: 'user', content: [{ text: 'hi' }, { image: { format: 'png', source: { bytes: new Uint8Array([1, 2, 3]) } } }] }], inferenceConfig: { maxTokens: 5 } };
  const out = await bedrockClient().send(new bedrock.ConverseCommand(input));
  assert.equal(out.output.message.content[0].text, 'hello');
  assert.ok(out.$metadata);
  const [call] = await drain();
  assert.equal(call.api, 'bedrock-converse');
  assert.deepEqual(call.request.messages[0].content[1].image.source.bytes, { omitted: 'bytes' });
  assert.equal(call.request.modelId, 'anthropic.claude-x');
  assert.equal(call.response.$metadata, undefined);
  assert.equal(call.response.stopReason, 'end_turn');
  assert.equal(call.response.usage.inputTokens, 3);
});

test('Bedrock ConverseStream passes events through and records one assembled response', { skip: !bedrock && '@aws-sdk/client-bedrock-runtime not installed' }, async () => {
  const out = await bedrockClient().send(new bedrock.ConverseStreamCommand({ modelId: 'm', messages: [{ role: 'user', content: [{ text: 'go' }] }] }));
  const seen = [];
  for await (const e of out.stream) seen.push(Object.keys(e)[0]);
  assert.deepEqual(seen, converseEvents.map(([t]) => t));
  const calls = await drain();
  assert.equal(calls.length, 1);
  const r = calls[0].response;
  assert.deepEqual(r.output.message, { role: 'assistant', content: [{ text: 'Checking' }, { toolUse: { toolUseId: 'tu_1', name: 'get_unit', input: { unit: '4B' } } }] });
  assert.equal(r.stopReason, 'tool_use');
  assert.equal(r.usage.outputTokens, 11);
});

test('Bedrock InvokeModel and other commands are not recorded', { skip: !bedrock && '@aws-sdk/client-bedrock-runtime not installed' }, async () => {
  await bedrockClient().send(new bedrock.InvokeModelCommand({ modelId: 'm', body: new TextEncoder().encode('{}'), contentType: 'application/json' }));
  assert.equal((await drain()).length, 0);
});

test('wrap supports Gemini and Bedrock clients', { skip: (!genai || !bedrock) && 'optional SDKs not installed' }, async () => {
  const out = await child(`
    process.env.GENERALCONTEXT_API_KEY = 'k'; process.env.GENERALCONTEXT_BASE_URL = BASE + '/child/wrap2';
    const { GoogleGenAI } = await import('@google/genai');
    const br = await import('@aws-sdk/client-bedrock-runtime');
    const g = wrap(new GoogleGenAI({ apiKey: 'k', httpOptions: { baseUrl: BASE } }));
    await g.models.generateContent({ model: 'gemini-x', contents: 'hi' });
    const b = wrap(new br.BedrockRuntimeClient({ region: 'us-east-1', endpoint: BASE, requestHandler: new (await import('@smithy/node-http-handler')).NodeHttpHandler(), credentials: { accessKeyId: 'a', secretAccessKey: 'b' } }));
    await b.send(new br.ConverseCommand({ modelId: 'm', messages: [{ role: 'user', content: [{ text: 'x' }] }] }));
    await new Promise(r => setTimeout(r, 30)); await flush();`);
  assert.equal(out.code, 0, out.stderr);
  assert.deepEqual(recorded('wrap2').map(c => c.api), ['gemini', 'bedrock-converse']);
});

test('autoload finds @google/genai and the Bedrock runtime', { skip: (!genai || !bedrock) && 'optional SDKs not installed' }, async () => {
  const out = await child(`
    initialize({ apiKey: 'k', baseUrl: BASE + '/child/auto' });
    const { GoogleGenAI } = await import('@google/genai');
    const br = await import('@aws-sdk/client-bedrock-runtime');
    await new GoogleGenAI({ apiKey: 'k', httpOptions: { baseUrl: BASE } }).models.generateContent({ model: 'gemini-x', contents: 'hi' });
    await new br.BedrockRuntimeClient({ region: 'us-east-1', endpoint: BASE, requestHandler: new (await import('@smithy/node-http-handler')).NodeHttpHandler(), credentials: { accessKeyId: 'a', secretAccessKey: 'b' } }).send(new br.ConverseCommand({ modelId: 'm', messages: [] }));
    await new Promise(r => setTimeout(r, 30)); await flush();`);
  assert.equal(out.code, 0, out.stderr);
  assert.deepEqual(recorded('auto').map(c => c.api), ['gemini', 'bedrock-converse']);
});
