#!/usr/bin/env node
import { execFileSync } from 'node:child_process';
import { readdir, readFile } from 'node:fs/promises';

const root = JSON.parse(await readFile('bindings/node/package.json', 'utf8'));
const expected = process.argv[2];
if (expected && root.version !== expected.replace(/^v/, '')) {
  throw new Error(`@rust-ml-runtime/node is ${root.version}; expected ${expected}`);
}
const metadata = JSON.parse(execFileSync('cargo', ['metadata', '--no-deps', '--format-version', '1'], { encoding: 'utf8' }));
const rustPackage = metadata.packages.find((pkg) => pkg.name === 'rust-ml-runtime');
if (!rustPackage) throw new Error('workspace does not contain the rust-ml-runtime package');
if (rustPackage.version !== root.version) {
  throw new Error(`rust-ml-runtime is ${rustPackage.version}; @rust-ml-runtime/node is ${root.version}`);
}
if (expected && rustPackage.version !== expected.replace(/^v/, '')) {
  throw new Error(`rust-ml-runtime is ${rustPackage.version}; expected ${expected}`);
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
