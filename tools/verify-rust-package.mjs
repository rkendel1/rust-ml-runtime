#!/usr/bin/env node
import { execFileSync } from 'node:child_process';
import { cpSync, lstatSync, mkdtempSync, mkdirSync, readFileSync, readdirSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { basename, join } from 'node:path';

const version = (process.argv[2] ?? '').replace(/^v/, '');
const archiveDirectory = process.argv[3] ?? 'target/package';
if (!version) throw new Error('usage: verify-rust-package.mjs VERSION [PACKAGE_DIRECTORY]');
const root = mkdtempSync(join(tmpdir(), 'rust-ml-runtime-consumer-'));
const packages = join(root, 'packages');
const consumer = join(root, 'consumer');
mkdirSync(packages);
cpSync('examples/rust-consumer', consumer, { recursive: true });

const collectArchives = (directory) => {
  const archives = [];
  for (const entry of readdirSync(directory)) {
    const path = join(directory, entry);
    const stat = lstatSync(path);
    if (stat.isDirectory()) archives.push(...collectArchives(path));
    else if (stat.isFile() && entry.endsWith(`-${version}.crate`)) archives.push(path);
  }
  return archives;
};
const crateNames = [];
const archives = [...new Map(collectArchives(archiveDirectory).sort().map((path) => [basename(path), path])).values()];
if (archives.length === 0) throw new Error(`no packaged crates found for version ${version} in ${archiveDirectory}`);
for (const archive of archives) {
  execFileSync('tar', ['-xzf', archive, '-C', packages]);
  const name = basename(archive);
  crateNames.push(name.slice(0, -`.crate`.length).slice(0, -(`-${version}`.length)));
}
if (!crateNames.includes('rust-ml-runtime')) throw new Error(`missing packaged rust-ml-runtime-${version}.crate`);
const assertNoSymlinks = (directory) => {
  for (const entry of readdirSync(directory)) {
    const path = join(directory, entry);
    const stat = lstatSync(path);
    if (stat.isSymbolicLink()) throw new Error(`packaged consumer input contains symlink ${path}`);
    if (stat.isDirectory()) assertNoSymlinks(path);
  }
};
assertNoSymlinks(packages);

let manifest = readFileSync(join(consumer, 'Cargo.toml'), 'utf8');
const runtimePackagePath = join(packages, `rust-ml-runtime-${version}`);
const runtimeDependency = `rust-ml-runtime = { version = ${JSON.stringify(`=${version}`)}, path = ${JSON.stringify(runtimePackagePath)} }`;
const updatedManifest = manifest.replace(/^rust-ml-runtime\s*=.*$/m, runtimeDependency);
if (updatedManifest === manifest) throw new Error('expected rust-ml-runtime dependency in external consumer manifest');
manifest = updatedManifest;
manifest += '\n[patch.crates-io]\n';
for (const name of crateNames.sort()) {
  manifest += `${JSON.stringify(name)} = { path = ${JSON.stringify(join(packages, `${name}-${version}`))} }\n`;
}
writeFileSync(join(consumer, 'Cargo.toml'), manifest);
const target = join(root, 'target');
execFileSync('cargo', ['build', '--manifest-path', join(consumer, 'Cargo.toml')], {
  stdio: 'inherit',
  env: { ...process.env, CARGO_TARGET_DIR: target },
});
const metadata = JSON.parse(execFileSync('cargo', ['metadata', '--format-version', '1', '--manifest-path', join(consumer, 'Cargo.toml')], {
  encoding: 'utf8',
  maxBuffer: 64 * 1024 * 1024,
}));
for (const pkg of metadata.packages.filter((candidate) => crateNames.includes(candidate.name))) {
  if (!pkg.manifest_path.startsWith(packages)) throw new Error(`${pkg.name} resolved outside packaged archives`);
}
const emptyPath = join(root, 'empty-path');
const home = join(root, 'home');
mkdirSync(emptyPath);
mkdirSync(home);
const executable = join(target, 'debug', process.platform === 'win32' ? 'rust-ml-runtime-external-consumer.exe' : 'rust-ml-runtime-external-consumer');
const output = execFileSync(executable, [], {
  encoding: 'utf8',
  env: {
    HOME: home,
    PATH: emptyPath,
    TMPDIR: root,
    HTTP_PROXY: 'http://127.0.0.1:9',
    HTTPS_PROXY: 'http://127.0.0.1:9',
    ALL_PROXY: 'http://127.0.0.1:9',
  },
});
if (!output.includes('model=external-linear provider=local backend=cpu target=local')) {
  throw new Error(`unexpected external consumer result: ${output}`);
}
console.log('Python unavailable: packaged Rust consumer executed typed local inference with provenance');
