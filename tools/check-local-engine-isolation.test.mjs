import assert from 'node:assert/strict';
import test from 'node:test';
import { engineMatches, manifestProblems, npmFileProblems } from './check-local-engine-isolation.mjs';

test('detects the engine in a dependency tree', () => {
  assert.equal(engineMatches('a v1\n└── llama-cpp-sys-2 v0.1.159\n').length, 1);
  assert.equal(engineMatches('a v1\n└── ml-runtime-llama-provider v0.2.3\n').length, 1);
  assert.deepEqual(engineMatches('a v1\n└── serde v1\n'), []);
});

test('flags weights and local-engine artifacts in an npm package', () => {
  assert.deepEqual(npmFileProblems(['index.cjs', 'index.d.ts', 'package.json']), []);
  assert.deepEqual(npmFileProblems(['model.gguf', 'ml_runtime_node_llama.node']), ['model.gguf', 'ml_runtime_node_llama.node']);
});

test('flags a local-engine dependency in a manifest', () => {
  assert.deepEqual(manifestProblems({ optionalDependencies: { '@rust-ml-runtime/node-linux-x64-gnu': '1' } }), []);
  assert.deepEqual(manifestProblems({ optionalDependencies: { '@rust-ml-runtime/node-llama-linux-arm64-gnu': '1' } }), [
    '@rust-ml-runtime/node-llama-linux-arm64-gnu',
  ]);
});
