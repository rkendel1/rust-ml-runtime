'use strict';

const assert = require('node:assert/strict');
const { mkdirSync, mkdtempSync, writeFileSync } = require('node:fs');
const { spawnSync } = require('node:child_process');
const os = require('node:os');
const path = require('node:path');
const test = require('node:test');

const loader = path.resolve(__dirname, '..', 'index.cjs');

function run(script, env = {}) {
  return spawnSync(process.execPath, ['-e', script], {
    encoding: 'utf8',
    env: { ...process.env, ...env },
  });
}

test('loads an injected addon and reports a successful self-test', () => {
  const directory = mkdtempSync(path.join(os.tmpdir(), 'rust-ml-node-'));
  const addon = path.join(directory, 'addon.cjs');
  writeFileSync(addon, `
    class LocalDecisionModel {}
    module.exports = { LocalDecisionModel };
  `);
  const result = run(`
    const runtime = require(${JSON.stringify(loader)});
    const diagnostics = runtime.diagnoseNative();
    if (!diagnostics.available || diagnostics.load !== 'success') process.exit(1);
    if (!runtime.LocalML.selfTest().available) process.exit(2);
  `, { ML_RUNTIME_NODE_ADDON: addon });
  assert.equal(result.status, 0, result.stderr);
});

test('distinguishes a missing optional package', () => {
  const result = run(`
    try { require(${JSON.stringify(loader)}); process.exit(1); }
    catch (error) {
      if (error.code !== 'PACKAGE_MISSING') process.exit(2);
      if (error.diagnostics.packageInstalled !== false) process.exit(3);
    }
  `, { ML_RUNTIME_NODE_ADDON: '' });
  assert.equal(result.status, 0, result.stderr);
});

test('distinguishes a package without its native artifact', () => {
  const directory = mkdtempSync(path.join(os.tmpdir(), 'rust-ml-node-'));
  const packageDirectory = path.join(directory, '@rust-ml-runtime', 'node-linux-x64-gnu');
  mkdirSync(packageDirectory, { recursive: true });
  writeFileSync(path.join(packageDirectory, 'package.json'), JSON.stringify({
    name: '@rust-ml-runtime/node-linux-x64-gnu',
    main: 'ml_runtime_node.node',
  }));
  const result = run(`
    try { require(${JSON.stringify(loader)}); process.exit(1); }
    catch (error) {
      if (error.code !== 'NATIVE_BINARY_MISSING') process.exit(2);
      if (!error.diagnostics.packageInstalled) process.exit(3);
      if (error.diagnostics.nativeBinaryPresent) process.exit(4);
    }
  `, { ML_RUNTIME_NODE_ADDON: '', NODE_PATH: directory });
  assert.equal(result.status, 0, result.stderr);
});

test('preserves native load failures after resolving the package', () => {
  const directory = mkdtempSync(path.join(os.tmpdir(), 'rust-ml-node-'));
  const packageDirectory = path.join(directory, '@rust-ml-runtime', 'node-linux-x64-gnu');
  mkdirSync(packageDirectory, { recursive: true });
  writeFileSync(path.join(packageDirectory, 'package.json'), JSON.stringify({
    name: '@rust-ml-runtime/node-linux-x64-gnu',
    main: 'ml_runtime_node.js',
  }));
  writeFileSync(path.join(packageDirectory, 'ml_runtime_node.js'), 'throw new Error("ABI mismatch");');
  const result = run(`
    try { require(${JSON.stringify(loader)}); process.exit(1); }
    catch (error) {
      if (error.code !== 'NATIVE_LOAD_FAILED') process.exit(2);
      if (!error.cause || !error.cause.message.includes('ABI mismatch')) process.exit(3);
    }
  `, { ML_RUNTIME_NODE_ADDON: '', NODE_PATH: directory });
  assert.equal(result.status, 0, result.stderr);
});
