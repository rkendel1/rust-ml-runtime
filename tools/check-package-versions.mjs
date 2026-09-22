#!/usr/bin/env node
import { readdir, readFile } from 'node:fs/promises';

const root = JSON.parse(await readFile('bindings/node/package.json', 'utf8'));
const expected = process.argv[2];
if (expected && root.version !== expected.replace(/^v/, '')) {
  throw new Error(`@rust-ml-runtime/node is ${root.version}; expected ${expected}`);
}
for (const directory of await readdir('bindings/node/npm')) {
  const manifest = JSON.parse(await readFile(`bindings/node/npm/${directory}/package.json`, 'utf8'));
  if (manifest.version !== root.version) {
    throw new Error(`${manifest.name} is ${manifest.version}; expected ${root.version}`);
  }
  if (root.optionalDependencies[manifest.name] !== root.version) {
    throw new Error(`${manifest.name} is not pinned to ${root.version}`);
  }
}
