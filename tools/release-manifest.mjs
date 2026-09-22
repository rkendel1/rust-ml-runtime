#!/usr/bin/env node
import { createHash } from 'node:crypto';
import { readdir, readFile, writeFile } from 'node:fs/promises';
import path from 'node:path';

const [version, artifactRoot, output] = process.argv.slice(2);
if (!version || !artifactRoot || !output) {
  throw new Error('usage: release-manifest.mjs VERSION ARTIFACT_ROOT OUTPUT');
}

async function filesUnder(directory) {
  const entries = await readdir(directory, { withFileTypes: true });
  const nested = await Promise.all(entries.map((entry) => {
    const file = path.join(directory, entry.name);
    return entry.isDirectory() ? filesUnder(file) : [file];
  }));
  return nested.flat();
}

function identity(filename) {
  const cli = filename.match(/^ml-runtime-v.+-(macos|linux|windows)-(aarch64|x86_64)\.tar\.gz$/);
  if (cli) return { product: 'cli', platform: cli[1], architecture: cli[2] };
  if (filename === `rust-ml-runtime-node-${version}.tgz`) {
    return { product: 'node', platform: 'any', architecture: 'any' };
  }
  const node = filename.match(/^rust-ml-runtime-node-(darwin|linux|win32)-(arm64|x64)(?:-(?:gnu|msvc))?-.+\.tgz$/);
  if (node) {
    const platforms = { darwin: 'macos', linux: 'linux', win32: 'windows' };
    const architectures = { arm64: 'aarch64', x64: 'x86_64' };
    return {
      product: 'node-native',
      platform: platforms[node[1]],
      architecture: architectures[node[2]],
    };
  }
  return undefined;
}

const artifacts = [];
for (const file of (await filesUnder(artifactRoot)).sort()) {
  const artifact = path.basename(file);
  const parsed = identity(artifact);
  if (!parsed) continue;
  const contents = await readFile(file);
  artifacts.push({
    artifact,
    version,
    ...parsed,
    sha256: createHash('sha256').update(contents).digest('hex'),
  });
}
if (artifacts.length === 0) throw new Error(`no release artifacts found under ${artifactRoot}`);
await writeFile(output, `${JSON.stringify({ schema_version: 1, version, artifacts }, null, 2)}\n`);
