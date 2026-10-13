'use strict';

// JavaScript-layer behavior with an injected fake addon, so it runs on any CI
// runner without building the native library. Native behavior (transport, SSE,
// validation, cancellation) is covered by generation.native.test.cjs and the
// Rust test suites.

const assert = require('node:assert/strict');
const { mkdtempSync, writeFileSync } = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const test = require('node:test');

const directory = mkdtempSync(path.join(os.tmpdir(), 'rust-ml-gen-'));
const addon = path.join(directory, 'addon.cjs');
writeFileSync(addon, `
const log = globalThis.__genLog = { configs: [], closed: 0, requests: [] };
class GenerationClient {
  constructor(config) { log.configs.push(config); if (JSON.parse(config).endpoint.includes('bad')) throw new Error('[invalid_request] endpoint is not a valid URL'); }
  async chatJson(request) { log.requests.push(request); if (request.includes('fail')) throw new Error('[timeout] timeout while chat completion'); return JSON.stringify({ message: { role: 'assistant', content: 'ok' } }); }
  async listModelsJson() { return '[]'; }
  openChatStream(request) {
    const events = [{ type: 'started' }, { type: 'delta', content: 'a' }, { type: 'delta', content: 'b' }, request.includes('boom') ? null : { type: 'completed' }];
    let i = 0;
    return {
      async next() {
        const e = events[i++];
        if (e === null) throw new Error('[protocol] protocol error while chat stream: boom');
        return e ? JSON.stringify(e) : null;
      },
      close() { log.closed += 1; },
    };
  }
}
class LocalDecisionModel {}
class DecisionCancellation {}
module.exports = { GenerationClient, LocalDecisionModel, DecisionCancellation, runtimeSmokeTest() { return '{}'; } };
`);
process.env.ML_RUNTIME_NODE_ADDON = addon;
const originalError = console.error;
console.error = () => {};
const { GenerativeML, GenerationError } = require('..');
console.error = originalError;
const log = globalThis.__genLog;
const request = { model: 'm', messages: [{ role: 'user', content: 'x' }] };

test('requests are validated before reaching native code', async () => {
  const ml = new GenerativeML({ endpoint: 'http://127.0.0.1:1/v1' });
  await assert.rejects(ml.chat({ messages: request.messages }), TypeError);
  await assert.rejects(ml.chat({ model: ' ', messages: request.messages }), TypeError);
  await assert.rejects(ml.chat({ model: 'm', messages: [] }), TypeError);
  assert.throws(() => ml.chatStream({ model: 'm' }), TypeError);
  assert.throws(() => new GenerativeML({}), TypeError);
  assert.equal(log.requests.length, 0);
});

test('native error prefixes become GenerationError codes', async () => {
  assert.throws(() => new GenerativeML({ endpoint: 'bad' }), (e) => e instanceof GenerationError && e.code === 'invalid_request' && !e.message.startsWith('['));
  const ml = new GenerativeML({ endpoint: 'http://127.0.0.1:1/v1' });
  assert.equal((await ml.chat(request)).message.content, 'ok');
  await assert.rejects(ml.chat({ ...request, model: 'fail' }), (e) => e.code === 'timeout' && e.name === 'GenerationError');
});

test('streams are single-use async iterables that always close the native handle', async () => {
  const ml = new GenerativeML({ endpoint: 'http://127.0.0.1:1/v1' });
  const before = log.closed;
  const types = [];
  const stream = ml.chatStream(request);
  for await (const event of stream) types.push(event.type);
  assert.deepEqual(types, ['started', 'delta', 'delta', 'completed']);
  assert.equal(log.closed, before + 1);
  await assert.rejects(async () => { for await (const _ of stream) { /* again */ } }, TypeError);

  for await (const event of ml.chatStream(request)) { if (event.type === 'delta') break; }
  assert.equal(log.closed, before + 2, 'early break must close');

  await assert.rejects(async () => { for await (const _ of ml.chatStream({ ...request, model: 'boom' })) { /* drain */ } }, (e) => e.code === 'protocol');
  assert.equal(log.closed, before + 3, 'errors must close');
});

test('instances never expose their configuration', () => {
  const ml = new GenerativeML({ endpoint: 'http://127.0.0.1:1/v1', apiKey: 'sk-secret-xyz' });
  assert.ok(!JSON.stringify(ml).includes('sk-secret-xyz'));
  assert.ok(!require('node:util').inspect(ml, { showHidden: true, depth: 6 }).includes('sk-secret-xyz'));
});
