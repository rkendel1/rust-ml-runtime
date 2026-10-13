'use strict';
// Loads the built addon and exercises it. Never downloads a model.
//   ML_RUNTIME_LLAMA_ADDON=/path/to/ml_runtime_node_llama.node  (default: ../ml_runtime_node_llama.node)
//   ML_RUNTIME_TEST_GGUF=/path/model.gguf [ML_RUNTIME_TEST_GGUF_SHA256=<hex>]  enables the real-model cases
const test = require('node:test');
const assert = require('node:assert/strict');
const path = require('node:path');
const fs = require('node:fs');
const os = require('node:os');

const addonPath = process.env.ML_RUNTIME_LLAMA_ADDON ?? path.join(__dirname, '..', 'ml_runtime_node_llama.node');
const modelPath = process.env.ML_RUNTIME_TEST_GGUF;
const digest = process.env.ML_RUNTIME_TEST_GGUF_SHA256;

test('a missing addon file fails with a diagnosable error, not a crash', () => {
  assert.throws(
    () => require(path.join(os.tmpdir(), 'does-not-exist-ml-runtime.node')),
    (error) => error.code === 'MODULE_NOT_FOUND' || /cannot open shared object|Cannot find module/.test(error.message),
  );
});

test('a file that is not a native addon fails to load with a clear error', () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'llama-addon-'));
  const bogus = path.join(dir, 'bogus.node');
  fs.writeFileSync(bogus, 'not an ELF or Mach-O file');
  assert.throws(() => require(bogus), /invalid ELF header|not a mach-o|slice is not valid|file too short|invalid|Unable to load/i);
});

const addon = require(addonPath);

test('the addon exports only the local-generation surface', () => {
  assert.deepEqual(Object.keys(addon).sort(), ['LocalCancellation', 'LocalGenerationClient']);
});

test('configuration and missing-model failures are typed and never download', () => {
  assert.throws(() => new addon.LocalGenerationClient('{'), /\[invalid_request\]/);
  assert.throws(() => new addon.LocalGenerationClient('{}'), /\[invalid_request\].*modelPath/);
  assert.throws(
    () => new addon.LocalGenerationClient(JSON.stringify({ modelPath: '/nope/absent.gguf', modelId: 'm' })),
    /\[model_unavailable\].*never downloaded/s,
  );
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'llama-addon-'));
  const text = path.join(dir, 'notes.gguf');
  fs.writeFileSync(text, 'hello');
  assert.throws(
    () => new addon.LocalGenerationClient(JSON.stringify({ modelPath: text, modelId: 'm' })),
    /\[unsupported_model\]/,
  );
});

const real = { skip: modelPath ? false : 'set ML_RUNTIME_TEST_GGUF to run against a real model' };
const request = (overrides = {}) => JSON.stringify({
  model: 'local',
  messages: [{ role: 'user', content: 'What is the capital of France? Answer in one word.' }],
  options: { max_tokens: 24, temperature: 0 },
  ...overrides,
});
const open = (extra = {}) => new addon.LocalGenerationClient(JSON.stringify({
  modelPath, modelId: 'local', sha256: digest, contextTokens: 2048, maxOutputTokens: 256, ...extra,
}));

test('real model: generates through the existing ChatRequest/ChatResponse JSON', real, async () => {
  const client = open();
  const response = JSON.parse(await client.chatJson(request()));
  assert.match(response.message.content, /paris/i);
  assert.equal(response.metadata.provider, 'llama-cpp');
  assert.equal(response.metadata.transport, 'in-process');
  assert.equal(response.metadata.requested_model, 'local');
});

test('real model: a different model id fails instead of substituting', real, async () => {
  const client = open();
  await assert.rejects(client.chatJson(request({ model: 'other' })), /\[model_not_found\]/);
});

test('real model: a wrong digest fails before any inference', real, () => {
  assert.throws(() => open({ sha256: '0'.repeat(64) }), /\[model_integrity\]/);
});

test('real model: schema-constrained output is valid and validated', real, async () => {
  const client = open();
  const body = JSON.parse(request());
  body.options.max_tokens = 120;
  body.options.response_format = {
    type: 'json_schema', name: 'answer', strict: true,
    schema: { type: 'object', properties: { city: { type: 'string' } }, required: ['city'], additionalProperties: false },
  };
  const response = JSON.parse(await client.chatJson(JSON.stringify(body)));
  assert.equal(typeof JSON.parse(response.message.content).city, 'string');
  assert.ok(response.structured, 'runtime validation result missing');
});

test('real model: cancellation rejects promptly', real, async () => {
  const client = open();
  const cancellation = new addon.LocalCancellation();
  const body = JSON.parse(request());
  body.messages[0].content = 'Write a very long story about a dragon.';
  body.options.max_tokens = 256;
  const pending = client.chatJson(JSON.stringify(body), cancellation);
  setTimeout(() => cancellation.cancel(), 100);
  const started = Date.now();
  await assert.rejects(pending, /\[cancelled\]/);
  assert.ok(Date.now() - started < 2000);
});
