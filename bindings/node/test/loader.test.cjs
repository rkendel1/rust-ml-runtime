'use strict';

const assert = require('node:assert/strict');
const { copyFileSync, mkdirSync, mkdtempSync, readFileSync, writeFileSync } = require('node:fs');
const { spawnSync } = require('node:child_process');
const os = require('node:os');
const path = require('node:path');
const test = require('node:test');

const loader = path.resolve(__dirname, '..', 'index.cjs');
const rootPackageDirectory = path.resolve(__dirname, '..');
const nativePlatform = {
  'darwin-arm64': 'darwin-arm64',
  'darwin-x64': 'darwin-x64',
  'linux-arm64': 'linux-arm64-gnu',
  'linux-x64': 'linux-x64-gnu',
  'win32-x64': 'win32-x64-msvc',
}[`${process.platform}-${process.arch}`];
const nativePackageName = `@rust-ml-runtime/node-${nativePlatform}`;
const nativePackageDirectory = path.resolve(__dirname, '..', 'npm', nativePlatform);

function run(script, env = {}) {
  return spawnSync(process.execPath, ['-e', script], {
    encoding: 'utf8',
    env: { ...process.env, ...env },
  });
}

function pack(directory, destination) {
  const result = spawnSync('npm', ['pack', '--pack-destination', destination], {
    cwd: directory,
    encoding: 'utf8',
  });
  assert.equal(result.status, 0, result.stderr);
  return path.join(destination, result.stdout.trim().split(/\r?\n/).at(-1));
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

test('exposes capability-driven async decisions without changing the sync API', () => {
  const directory = mkdtempSync(path.join(os.tmpdir(), 'rust-ml-node-'));
  const addon = path.join(directory, 'addon.cjs');
  writeFileSync(addon, `
    class DecisionCancellation {
      constructor() { this.isCancelled = false; }
      cancel() { this.isCancelled = true; }
    }
    class LocalDecisionModel {
      capabilitiesJson() { return JSON.stringify({ backend: 'fixture', device: 'cpu' }); }
      explainDecisionJson() {
        return JSON.stringify({ strategy: { type: 'batched', batch_size: 4 }, reason: 'backend_batching', stages: [] });
      }
      decideJson(request) { return JSON.stringify({ mode: 'sync', request: JSON.parse(request) }); }
      async decideAsyncJson(request) { return JSON.stringify({ mode: 'async', request: JSON.parse(request) }); }
      async executeGraphJson(request) { return JSON.stringify({ mode: 'graph', request: JSON.parse(request) }); }
    }
    module.exports = { DecisionCancellation, LocalDecisionModel };
  `);
  const result = run(`
    (async () => {
      const runtime = require(${JSON.stringify(loader)});
      const local = await runtime.LocalML.create();
      const question = { name: 'intent', instructions: 'classify', kind: { type: 'noul' } };
      if (local.decide({ model: 'fixture', input: {}, decisions: [question] }).mode !== 'sync') process.exit(1);
      if ((await local.decideAsync({ model: 'fixture', input: {}, decisions: [question] })).mode !== 'async') process.exit(2);
      if (local.capabilities('fixture').backend !== 'fixture') process.exit(3);
      const graph = { model: 'fixture', input: {}, nodes: [{ question }] };
      if (local.explainDecision(graph).reason !== 'backend_batching') process.exit(4);
      if ((await local.executeGraph(graph)).mode !== 'graph') process.exit(5);
      const cancellation = new runtime.DecisionCancellation();
      cancellation.cancel();
      if (!cancellation.isCancelled) process.exit(6);
    })().catch((error) => { console.error(error); process.exit(7); });
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
  const packageDirectory = path.join(directory, ...nativePackageName.split('/'));
  mkdirSync(packageDirectory, { recursive: true });
  writeFileSync(path.join(packageDirectory, 'package.json'), JSON.stringify({
    name: nativePackageName,
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
  const packageDirectory = path.join(directory, ...nativePackageName.split('/'));
  mkdirSync(packageDirectory, { recursive: true });
  writeFileSync(path.join(packageDirectory, 'package.json'), JSON.stringify({
    name: nativePackageName,
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

test('installs the packed root and native tarballs offline from absolute paths', () => {
  const directory = mkdtempSync(path.join(os.tmpdir(), 'rust-ml-node-pack-'));
  const rootSource = path.join(directory, 'root');
  const nativeSource = path.join(directory, 'native');
  const tarballs = path.join(directory, 'tarballs');
  mkdirSync(rootSource, { recursive: true });
  mkdirSync(nativeSource, { recursive: true });
  mkdirSync(tarballs, { recursive: true });

  for (const file of ['package.json', 'index.cjs', 'index.d.ts']) {
    copyFileSync(path.join(rootPackageDirectory, file), path.join(rootSource, file));
  }

  const nativeManifest = JSON.parse(readFileSync(path.join(nativePackageDirectory, 'package.json'), 'utf8'));
  nativeManifest.main = 'ml_runtime_node.cjs';
  nativeManifest.files = ['ml_runtime_node.cjs'];
  writeFileSync(path.join(nativeSource, 'package.json'), JSON.stringify(nativeManifest, null, 2));
  writeFileSync(path.join(nativeSource, 'ml_runtime_node.cjs'), `
    class LocalDecisionModel {}
    module.exports = { LocalDecisionModel };
  `);

  const rootTarball = pack(rootSource, tarballs);
  const nativeTarball = pack(nativeSource, tarballs);

  const project = mkdtempSync(path.join(os.tmpdir(), 'rust-ml-node-project-'));
  const install = spawnSync('npm', [
    'install',
    '--offline',
    '--include=optional',
    '--ignore-scripts',
    rootTarball,
    nativeTarball,
  ], {
    cwd: project,
    encoding: 'utf8',
  });
  assert.equal(install.status, 0, install.stderr);
  const installedManifest = JSON.parse(readFileSync(
    path.join(project, 'node_modules', '@rust-ml-runtime', 'node', 'package.json'),
    'utf8',
  ));
  assert.equal(installedManifest.optionalDependencies[nativePackageName], nativeManifest.version);
  assert.ok(!installedManifest.optionalDependencies[nativePackageName].startsWith('file:'));

  const result = run(`
    const runtime = require('@rust-ml-runtime/node');
    const diagnostics = runtime.diagnoseNative();
    if (!diagnostics.available) process.exit(1);
    if (diagnostics.packageName !== ${JSON.stringify(nativePackageName)}) process.exit(2);
  `, { NODE_PATH: path.join(project, 'node_modules') });
  assert.equal(result.status, 0, result.stderr);
});
