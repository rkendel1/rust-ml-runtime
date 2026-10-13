'use strict';

// Exercises the real native addon against an in-process OpenAI-compatible mock.
// Opt-in: set ML_RUNTIME_NODE_ADDON to a built addon (see docs/node.md). No
// network, API key, or model service is used.

const assert = require('node:assert/strict');
const http = require('node:http');
const test = require('node:test');

const addon = process.env.ML_RUNTIME_NODE_ADDON;
const skip = addon ? false : 'set ML_RUNTIME_NODE_ADDON to a built addon to run native tests';
const SECRET = 'sk-node-SECRET-789';

const api = skip ? {} : require('..');
const { GenerativeML, GenerationError, DecisionCancellation } = api;

function mock(handler) {
  const requests = [];
  const sockets = new Set();
  const state = { closedStreams: 0 };
  const server = http.createServer((req, res) => {
    const chunks = [];
    req.on('data', (c) => chunks.push(c));
    req.on('end', () => {
      const body = chunks.length ? JSON.parse(Buffer.concat(chunks)) : undefined;
      requests.push({ method: req.method, url: req.url, headers: req.headers, body });
      res.on('close', () => { if (!res.writableEnded) state.closedStreams += 1; });
      handler(req, res, body);
    });
  });
  server.on('connection', (s) => { sockets.add(s); s.on('close', () => sockets.delete(s)); });
  return new Promise((resolve) => server.listen(0, '127.0.0.1', () => resolve({
    url: `http://127.0.0.1:${server.address().port}/v1`,
    requests,
    state,
    close: () => { for (const s of sockets) s.destroy(); server.close(); },
  })));
}

const completion = (content) => JSON.stringify({
  id: 'c1', model: 'reported',
  choices: [{ index: 0, message: { role: 'assistant', content }, finish_reason: 'stop' }],
  usage: { prompt_tokens: 1, completion_tokens: 2, total_tokens: 3 },
});
const sse = (obj) => `data: ${typeof obj === 'string' ? obj : JSON.stringify(obj)}\n\n`;
const request = (extra = {}) => ({ model: 'm', messages: [{ role: 'system', content: 's' }, { role: 'user', content: 'u' }], ...extra });
const eventually = async (predicate) => {
  for (let i = 0; i < 200 && !predicate(); i += 1) await new Promise((r) => setTimeout(r, 10));
  return predicate();
};

test('chat returns the typed response with honest metadata', { skip }, async () => {
  const server = await mock((req, res) => { res.setHeader('content-type', 'application/json'); res.end(completion('hello')); });
  try {
    const ml = new GenerativeML({ endpoint: server.url });
    const response = await ml.chat(request({ options: { max_tokens: 5, stop: ['x'] } }));
    assert.equal(response.message.content, 'hello');
    assert.equal(response.finish_reason, 'stop');
    assert.equal(response.usage.total_tokens, 3);
    assert.equal(response.metadata.execution, 'local');
    assert.equal(response.metadata.execution_inferred, true);
    assert.equal(response.metadata.transport, 'http');
    assert.equal(response.metadata.requested_model, 'm');
    assert.equal(response.structured, null);
    const sent = server.requests[0];
    assert.equal(sent.url, '/v1/chat/completions');
    assert.deepEqual(sent.body.messages.map((m) => m.role), ['system', 'user']);
    assert.equal(sent.body.max_tokens, 5);
    assert.equal(sent.headers.authorization, undefined);
  } finally { server.close(); }
});

test('credentials are sent as a bearer token and never surface', { skip }, async () => {
  const server = await mock((req, res) => { res.statusCode = 401; res.end(JSON.stringify({ error: { message: 'no' } })); });
  try {
    process.env.NODE_TEST_KEY = SECRET;
    for (const options of [{ apiKey: SECRET }, { apiKeyEnv: 'NODE_TEST_KEY' }]) {
      const ml = new GenerativeML({ endpoint: server.url, ...options });
      await assert.rejects(ml.chat(request()), (error) => {
        assert.ok(error instanceof GenerationError);
        assert.equal(error.code, 'provider_unavailable');
        assert.ok(!`${error.message}${error.stack}`.includes(SECRET));
        return true;
      });
      assert.ok(!JSON.stringify(ml).includes(SECRET));
      assert.ok(!require('node:util').inspect(ml, { showHidden: true, depth: 5 }).includes(SECRET));
    }
    assert.ok(server.requests.every((r) => r.headers.authorization === `Bearer ${SECRET}`));
    assert.throws(() => new GenerativeML({ endpoint: server.url, apiKey: 'a', apiKeyEnv: 'NODE_TEST_KEY' }), GenerationError);
    assert.throws(() => new GenerativeML({ endpoint: server.url, apiKeyEnv: 'NODE_TEST_UNSET_VAR' }), /NODE_TEST_UNSET_VAR/);
    // A key must not be sent in clear text to a non-loopback host.
    assert.throws(() => new GenerativeML({ endpoint: 'http://example.com/v1', apiKey: SECRET }), (e) => e.code === 'invalid_request' && !e.message.includes(SECRET));
  } finally { server.close(); delete process.env.NODE_TEST_KEY; }
});

test('provider, transport, timeout, and request errors carry distinct codes', { skip }, async () => {
  const cases = [[400, '{"error":{"message":"bad"}}', 'invalid_request'], [503, '{"error":"busy"}', 'provider_unavailable'], [200, 'garbage', 'protocol']];
  for (const [status, body, code] of cases) {
    const server = await mock((req, res) => { res.statusCode = status; res.end(body); });
    try {
      await assert.rejects(new GenerativeML({ endpoint: server.url }).chat(request()), (e) => e.code === code);
    } finally { server.close(); }
  }
  await assert.rejects(new GenerativeML({ endpoint: 'http://127.0.0.1:1/v1' }).chat(request()), (e) => e.code === 'transport');

  const stalled = await mock(() => {});
  try {
    await assert.rejects(new GenerativeML({ endpoint: stalled.url, requestTimeoutMs: 100 }).chat(request()), (e) => e.code === 'timeout');
  } finally { stalled.close(); }

  const ml = new GenerativeML({ endpoint: 'http://127.0.0.1:1/v1' });
  assert.throws(() => ml.chatStream({ model: '', messages: [] }), TypeError);
  await assert.rejects(ml.chat({ model: 'm', messages: [{ role: 'system', content: 'only system' }] }), (e) => e.code === 'invalid_request');
});

test('cancellation rejects with code cancelled and closes the connection', { skip }, async () => {
  const server = await mock(() => {});
  try {
    const cancellation = new DecisionCancellation();
    const pending = new GenerativeML({ endpoint: server.url }).chat(request(), { cancellation });
    setTimeout(() => cancellation.cancel(), 50);
    await assert.rejects(pending, (e) => e.code === 'cancelled');
    assert.ok(await eventually(() => server.state.closedStreams === 1), 'upstream request must be aborted');
  } finally { server.close(); }
});

function streamingHandler({ hold = false } = {}) {
  return (req, res) => {
    res.writeHead(200, { 'content-type': 'text/event-stream' });
    res.write(sse({ id: 'c', model: 'reported', choices: [{ delta: { content: 'Hel' } }] }));
    setTimeout(() => {
      res.write(sse({ choices: [{ delta: { content: 'lo' } }] }));
      res.write(sse({ choices: [{ delta: {}, finish_reason: 'stop' }] }));
      res.write(sse({ choices: [], usage: { prompt_tokens: 1, completion_tokens: 2, total_tokens: 3 } }));
      if (!hold) { res.write(sse('[DONE]')); res.end(); }
    }, 20);
  };
}

test('streaming yields started, deltas, and one completed event', { skip }, async () => {
  const server = await mock(streamingHandler());
  try {
    const events = [];
    for await (const event of new GenerativeML({ endpoint: server.url }).chatStream(request())) events.push(event);
    assert.equal(events[0].type, 'started');
    assert.equal(events.filter((e) => e.type === 'delta').map((e) => e.content).join(''), 'Hello');
    assert.equal(events.at(-1).type, 'completed');
    assert.equal(events.filter((e) => e.type === 'completed').length, 1);
    assert.equal(events.at(-1).usage.total_tokens, 3);
    assert.equal(events.at(-1).metadata.streamed, true);
    assert.equal(server.requests[0].body.stream, true);
  } finally { server.close(); }
});

test('breaking out of a stream releases the upstream connection', { skip }, async () => {
  const server = await mock(streamingHandler({ hold: true }));
  try {
    const stream = new GenerativeML({ endpoint: server.url }).chatStream(request());
    for await (const event of stream) { if (event.type === 'delta') break; }
    assert.ok(await eventually(() => server.state.closedStreams === 1), 'break must close the connection');
    await assert.rejects(async () => { for await (const _ of stream) { /* second iteration */ } }, TypeError);
  } finally { server.close(); }
});

test('close() ends an idle stream and an explicit cancellation errors it', { skip }, async () => {
  const server = await mock(streamingHandler({ hold: true }));
  try {
    const ml = new GenerativeML({ endpoint: server.url });
    const closed = ml.chatStream(request());
    const iterator = closed[Symbol.asyncIterator]();
    await iterator.next(); // started
    closed.close();
    const after = await iterator.next();
    assert.ok(after.done || after.value, 'closing must not hang');
    assert.ok(await eventually(() => server.state.closedStreams >= 1));

    const cancellation = new DecisionCancellation();
    const events = [];
    await assert.rejects(async () => {
      for await (const event of ml.chatStream(request(), { cancellation })) {
        events.push(event.type);
        if (event.type === 'delta') cancellation.cancel();
      }
    }, (e) => e.code === 'cancelled');
    assert.ok(!events.includes('completed'));
    assert.ok(await eventually(() => server.state.closedStreams >= 2));
  } finally { server.close(); }
});

test('a truncated stream is an error rather than a completion', { skip }, async () => {
  const server = await mock((req, res) => {
    res.writeHead(200, { 'content-type': 'text/event-stream' });
    res.write(sse({ choices: [{ delta: { content: 'par' } }] }));
    res.end();
  });
  try {
    const seen = [];
    await assert.rejects(async () => {
      for await (const event of new GenerativeML({ endpoint: server.url }).chatStream(request())) seen.push(event.type);
    }, (e) => e.code === 'protocol');
    assert.ok(!seen.includes('completed'));
  } finally { server.close(); }
});

test('structured output is validated natively and reported explicitly', { skip }, async () => {
  const schema = { type: 'object', properties: { n: { type: 'integer' } }, required: ['n'] };
  const structured = request({ options: { response_format: { type: 'json_schema', name: 'out', schema, strict: true } } });
  const outcomes = [['{"n":1}', 'validated'], ['nope', 'invalid_json'], ['{"n":"x"}', 'schema_violation']];
  for (const [text, status] of outcomes) {
    const server = await mock((req, res) => { res.end(completion(text)); });
    try {
      const ml = new GenerativeML({ endpoint: server.url, capabilities: { jsonSchema: true } });
      const response = await ml.chat(structured);
      assert.equal(response.structured.status, status);
      assert.equal(response.message.content, text);
      assert.equal(server.requests[0].body.response_format.type, 'json_schema');
    } finally { server.close(); }
  }
  // Not declared as supported: rejected before any request is made.
  const server = await mock((req, res) => res.end(completion('{}')));
  try {
    await assert.rejects(new GenerativeML({ endpoint: server.url }).chat(structured), (e) => e.code === 'capability_mismatch');
    assert.equal(server.requests.length, 0);
  } finally { server.close(); }
});

test('listModels reports discovery and its absence', { skip }, async () => {
  const ok = await mock((req, res) => res.end(JSON.stringify({ data: [{ id: 'a' }, { id: 'b', owned_by: 'x' }] })));
  try {
    const models = await new GenerativeML({ endpoint: ok.url }).listModels();
    assert.deepEqual(models.map((m) => m.id), ['a', 'b']);
  } finally { ok.close(); }
  const missing = await mock((req, res) => { res.statusCode = 404; res.end('{}'); });
  try {
    await assert.rejects(new GenerativeML({ endpoint: missing.url }).listModels(), (e) => e.code === 'capability_mismatch');
  } finally { missing.close(); }
});

test('many abandoned streams do not leak connections', { skip }, async () => {
  const server = await mock(streamingHandler({ hold: true }));
  try {
    const ml = new GenerativeML({ endpoint: server.url });
    for (let i = 0; i < 20; i += 1) {
      const stream = ml.chatStream(request());
      const iterator = stream[Symbol.asyncIterator]();
      await iterator.next();
      await iterator.return();
    }
    assert.ok(await eventually(() => server.state.closedStreams === 20), `closed ${server.state.closedStreams}/20`);
  } finally { server.close(); }
});
