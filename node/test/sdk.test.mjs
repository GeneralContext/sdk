// Runs the compiled SDK against real openai and @anthropic-ai/sdk clients pointed at a local fake server.
import { test, before, after } from 'node:test';
import assert from 'node:assert/strict';
import http from 'node:http';
import OpenAI from 'openai';
import Anthropic from '@anthropic-ai/sdk';

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

let server, base;
before(async () => {
  server = http.createServer(async (req, res) => {
    let body = '';
    for await (const chunk of req) body += chunk;
    const json = body ? JSON.parse(body) : {};
    const reply = (status, obj) => { res.writeHead(status, { 'content-type': 'application/json' }); res.end(JSON.stringify(obj)); };
    if (req.url === '/api/platform-collect/sdk') { collected.batches.push(json); collected.calls.push(...json.calls); return reply(200, { ok: true }); }
    if (req.url === '/api/platform-collect/outcomes') { collected.outcomes.push({ auth: req.headers.authorization, ...json }); return reply(200, { ok: true }); }
    if (req.url === '/api/platform-collect/events') { collected.events.push(json); return reply(200, { ingested: 1, skipped: 0, failures: [] }); }
    if (req.url.endsWith('/chat/completions')) {
      if (json.model === 'bad') return reply(400, { error: { message: 'bad model' } });
      return json.stream ? sse(res, chatSSE) : reply(200, chatJSON);
    }
    if (req.url.endsWith('/messages')) return json.stream ? sse(res, anthropicSSE) : reply(200, anthropicJSON);
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
  sdk.initialize({ apiKey: 'gc_key', baseUrl: base, appName: 'test-app', redact: true, instrumentModules: { openAI: OpenAI, anthropic: Anthropic } });
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

test('collector outage never throws', async () => {
  process.env.GENERALCONTEXT_BASE_URL = 'http://127.0.0.1:1';
  const warn = console.warn; console.warn = () => {};
  try {
    await openai().chat.completions.create({ model: 'gpt-x', messages: [] });
    await new Promise(r => setTimeout(r, 30));
    await sdk.flush();
    assert.equal(await sdk.reportOutcome('s1', 'fail'), false);
    assert.equal(await sdk.logEvent({ sessionId: 's1', eventType: 'handoff.to_human' }), false);
  } finally {
    console.warn = warn;
    delete process.env.GENERALCONTEXT_BASE_URL;
  }
});
