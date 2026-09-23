import assert from 'node:assert/strict';
import { copyFileSync, mkdirSync, mkdtempSync, writeFileSync } from 'node:fs';
import { spawnSync } from 'node:child_process';
import os from 'node:os';
import path from 'node:path';
import test from 'node:test';
import { fileURLToPath, pathToFileURL } from 'node:url';

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const publishScript = path.join(repoRoot, 'tools', 'publish-packages.mjs');
const publishScriptUrl = pathToFileURL(publishScript).href;

function pack(directory, destination) {
  const result = spawnSync('npm', ['pack'], {
    cwd: directory,
    encoding: 'utf8',
  });
  assert.equal(result.status, 0, result.stderr);
  const name = result.stdout.trim().split(/\r?\n/).at(-1);
  const source = path.join(directory, name);
  const target = path.join(destination, name);
  copyFileSync(source, target);
  return target;
}

function makePackage(directory, manifest, files) {
  mkdirSync(directory, { recursive: true });
  writeFileSync(path.join(directory, 'package.json'), JSON.stringify(manifest, null, 2));
  for (const [name, content] of Object.entries(files)) {
    writeFileSync(path.join(directory, name), content);
  }
}

function createArtifacts(version) {
  const temp = mkdtempSync(path.join(os.tmpdir(), 'rust-ml-publish-'));
  const sources = path.join(temp, 'sources');
  const tarballs = path.join(temp, 'tarballs');
  const artifacts = path.join(temp, 'artifacts');
  mkdirSync(sources, { recursive: true });
  mkdirSync(tarballs, { recursive: true });
  mkdirSync(artifacts, { recursive: true });

  const optionalDependencies = {
    '@rust-ml-runtime/node-darwin-arm64': version,
    '@rust-ml-runtime/node-darwin-x64': version,
    '@rust-ml-runtime/node-linux-arm64-gnu': version,
    '@rust-ml-runtime/node-linux-x64-gnu': version,
    '@rust-ml-runtime/node-win32-x64-msvc': version,
  };

  makePackage(path.join(sources, 'root'), {
    name: '@rust-ml-runtime/node',
    version,
    main: 'index.cjs',
    types: 'index.d.ts',
    files: ['index.cjs', 'index.d.ts'],
    optionalDependencies,
  }, {
    'index.cjs': 'module.exports = {};\n',
    'index.d.ts': 'export {};\n',
  });
  mkdirSync(path.join(artifacts, 'release-metadata'), { recursive: true });
  copyFileSync(pack(path.join(sources, 'root'), tarballs), path.join(artifacts, 'release-metadata', `rust-ml-runtime-node-${version}.tgz`));

  const natives = [
    ['darwin-arm64', { os: ['darwin'], cpu: ['arm64'] }],
    ['darwin-x64', { os: ['darwin'], cpu: ['x64'] }],
    ['linux-arm64-gnu', { os: ['linux'], cpu: ['arm64'], libc: ['glibc'] }],
    ['linux-x64-gnu', { os: ['linux'], cpu: ['x64'], libc: ['glibc'] }],
    ['win32-x64-msvc', { os: ['win32'], cpu: ['x64'] }],
  ];
  for (const [platform, extra] of natives) {
    const directory = path.join(sources, platform);
    makePackage(directory, {
      name: `@rust-ml-runtime/node-${platform}`,
      version,
      main: 'ml_runtime_node.node',
      files: ['ml_runtime_node.node'],
      ...extra,
    }, {
      'ml_runtime_node.node': 'placeholder binary\n',
    });
    mkdirSync(path.join(artifacts, platform), { recursive: true });
    copyFileSync(pack(directory, tarballs), path.join(artifacts, platform, `rust-ml-runtime-node-${platform}-${version}.tgz`));
  }

  return { artifacts };
}

test('npm-only publish dry-run discovers the root tarball without a macOS-specific path', () => {
  const { artifacts } = createArtifacts('0.2.0');
  const result = spawnSync(process.execPath, [
    publishScript,
    '--dry-run',
    '--npm-only',
    '--version', '0.2.0',
    '--artifacts', artifacts,
  ], {
    cwd: repoRoot,
    encoding: 'utf8',
  });

  assert.equal(result.status, 0, result.stderr);
  assert.match(result.stdout, /Discovered 6 npm tarball\(s\):/);
  assert.match(result.stdout, /Native npm publish order:/);
  assert.match(result.stdout, /release-metadata\/rust-ml-runtime-node-0\.2\.0\.tgz/);
});

test('npm-only publish reports an artifact/version mismatch clearly', () => {
  const { artifacts } = createArtifacts('0.1.0');
  const result = spawnSync(process.execPath, [
    publishScript,
    '--dry-run',
    '--npm-only',
    '--version', '0.2.0',
    '--artifacts', artifacts,
  ], {
    cwd: repoRoot,
    encoding: 'utf8',
  });

  assert.notEqual(result.status, 0);
  assert.match(result.stdout, /rust-ml-runtime-node-0\.1\.0\.tgz/);
  assert.match(result.stderr, /Expected one root npm package tarball for 0\.2\.0; found 0/);
  assert.match(result.stderr, /Check that --artifacts points at the release run for version 0\.2\.0/);
});

test('npm-only publish reports when tar is unavailable', () => {
  const { artifacts } = createArtifacts('0.2.0');
  const result = spawnSync(process.execPath, [
    publishScript,
    '--dry-run',
    '--npm-only',
    '--version', '0.2.0',
    '--artifacts', artifacts,
  ], {
    cwd: repoRoot,
    encoding: 'utf8',
    env: { ...process.env, PATH: '' },
  });

  assert.notEqual(result.status, 0);
  assert.match(result.stderr, /tar is required to inspect release artifacts on this runner/);
});

test('npm-only publish reports when fetch is unavailable', () => {
  const { artifacts } = createArtifacts('0.2.0');
  const bootstrap = `
    globalThis.fetch = undefined;
    process.argv = [
      'node',
      ${JSON.stringify(publishScript)},
      '--npm-only',
      '--version',
      '0.2.0',
      '--artifacts',
      ${JSON.stringify(artifacts)},
    ];
    await import(${JSON.stringify(publishScriptUrl)});
  `;
  const result = spawnSync(process.execPath, ['--input-type=module', '-e', bootstrap], {
    cwd: repoRoot,
    encoding: 'utf8',
    env: { ...process.env, NODE_AUTH_TOKEN: 'test-token' },
  });

  assert.notEqual(result.status, 0);
  assert.match(result.stderr, /Global fetch is required for registry preflight; run this script with Node\.js 18 or newer/);
});
