#!/usr/bin/env node
import { execFileSync } from 'node:child_process';
import { existsSync, readdirSync, statSync } from 'node:fs';
import { join } from 'node:path';

const [artifactRoot = 'artifacts', version] = process.argv.slice(2);
if (!version || !/^\d+\.\d+\.\d+$/.test(version)) {
  throw new Error('usage: validate-node-packages.mjs ARTIFACT_ROOT VERSION');
}

const packages = [];
function collect(directory) {
  for (const entry of readdirSync(directory)) {
    const file = join(directory, entry);
    if (statSync(file).isDirectory()) collect(file);
    else if (file.endsWith('.tgz') && file.includes('rust-ml-runtime-node-')) packages.push(file);
  }
}
if (!existsSync(artifactRoot)) throw new Error(`artifact directory not found: ${artifactRoot}`);
collect(artifactRoot);

const readTar = (archive, file) => execFileSync('tar', ['-xOf', archive, `package/${file}`], { encoding: 'utf8' });
const listTar = archive => execFileSync('tar', ['-tzf', archive], { encoding: 'utf8' })
  .trim().split('\n').filter(Boolean).map(entry => entry.replace(/^package\//, '')).filter(Boolean);
const manifests = packages.map(archive => ({
  archive,
  files: listTar(archive),
  manifest: JSON.parse(readTar(archive, 'package.json')),
}));

const native = manifests.filter(({ manifest }) => manifest.name !== '@rust-ml-runtime/node');
const root = manifests.filter(({ manifest }) => manifest.name === '@rust-ml-runtime/node');
if (native.length !== 5 || root.length !== 1) {
  throw new Error(`expected one root package and five native packages, found ${root.length} and ${native.length}`);
}

const rootManifest = root[0].manifest;
if (rootManifest.version !== version) throw new Error('root package version mismatch');
const expectedOptional = Object.keys(rootManifest.optionalDependencies ?? {}).sort();
const actualNative = native.map(({ manifest }) => manifest.name).sort();
if (expectedOptional.length !== 5 || JSON.stringify(expectedOptional) !== JSON.stringify(actualNative)) {
  throw new Error('root optionalDependencies do not match the five native packages');
}
for (const packageInfo of native) {
  const { manifest, files } = packageInfo;
  if (manifest.version !== version) throw new Error(`${manifest.name} version mismatch`);
  if (!files.includes('package.json') || !files.includes(manifest.main ?? '')) {
    throw new Error(`${manifest.name} is missing package metadata or ${manifest.main}`);
  }
  if (!manifest.main?.endsWith('.node')) throw new Error(`${manifest.name} does not point to a .node artifact`);
  const platform = manifest.name.replace('@rust-ml-runtime/node-', '');
  const expected = {
    'darwin-arm64': ['darwin', 'arm64'],
    'darwin-x64': ['darwin', 'x64'],
    'linux-arm64-gnu': ['linux', 'arm64'],
    'linux-x64-gnu': ['linux', 'x64'],
    'win32-x64-msvc': ['win32', 'x64'],
  }[platform];
  if (!expected || manifest.os?.[0] !== expected[0] || manifest.cpu?.[0] !== expected[1]) {
    throw new Error(`${manifest.name} has incorrect platform metadata`);
  }
  if (platform.endsWith('-gnu') && manifest.libc?.[0] !== 'glibc') {
    throw new Error(`${manifest.name} is missing glibc metadata`);
  }
  if (rootManifest.optionalDependencies[manifest.name] !== version) {
    throw new Error(`${manifest.name} is not aligned with the root package version`);
  }
}
const rootFiles = root[0].files;
for (const expected of ['package.json', 'index.cjs', 'index.d.ts']) {
  if (!rootFiles.includes(expected)) throw new Error(`root package is missing ${expected}`);
}
if (rootFiles.some(file => !['package.json', 'index.cjs', 'index.d.ts'].includes(file))) {
  throw new Error('root package contains unexpected files');
}
console.log(`validated ${rootFiles.length} root files and ${native.length} native npm packages`);
