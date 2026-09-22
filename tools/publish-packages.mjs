#!/usr/bin/env node
import { execFileSync } from 'node:child_process';
import { existsSync, readdirSync, readFileSync, statSync } from 'node:fs';
import { join } from 'node:path';

const crateOrder = [
  'ml-runtime-common',
  'ml-runtime-model',
  'ml-runtime-inference',
  'ml-runtime-backend',
  'ml-runtime-provider',
  'ml-runtime-protocol',
  'ml-runtime-cpu-backend',
  'ml-runtime-onnx-backend',
  'ml-runtime-coreml-backend',
  'ml-runtime-http-provider',
  'rust-ml-runtime',
];

const args = process.argv.slice(2);
const dryRun = args.includes('--dry-run');
const valueAfter = (flag, fallback) => {
  const index = args.indexOf(flag);
  return index === -1 ? fallback : args[index + 1];
};
const releasePackage = JSON.parse(readFileSync('package.json', 'utf8'));
const requested = valueAfter('--version', process.env.RELEASE_VERSION ?? releasePackage.version);
const version = requested.split('/').at(-1).replace(/^v/, '');
const artifacts = valueAfter('--artifacts', 'artifacts');
if (!/^\d+\.\d+\.\d+$/.test(version)) throw new Error('Pass --version X.Y.Z or set RELEASE_VERSION');

const run = (command, commandArgs, options = {}) => {
  console.log(`+ ${command} ${commandArgs.join(' ')}`);
  execFileSync(command, commandArgs, { stdio: 'inherit', ...options });
};
const metadata = JSON.parse(execFileSync('cargo', ['metadata', '--no-deps', '--format-version', '1'], { encoding: 'utf8' }));
for (const name of crateOrder) {
  const pkg = metadata.packages.find((candidate) => candidate.name === name);
  if (!pkg) throw new Error(`Missing publishable crate ${name}`);
  if (pkg.version !== version) throw new Error(`${name} is ${pkg.version}; expected ${version}`);
}
const nodeRoot = JSON.parse(readFileSync('bindings/node/package.json', 'utf8'));
if (nodeRoot.version !== version) throw new Error(`${nodeRoot.name} is ${nodeRoot.version}; expected ${version}`);

if (!dryRun) {
  if (!process.env.CARGO_REGISTRY_TOKEN) throw new Error('CARGO_REGISTRY_TOKEN is required for crates.io publication');
  if (!process.env.NODE_AUTH_TOKEN) throw new Error('NODE_AUTH_TOKEN is required');
}

const collect = (directory, predicate, found = []) => {
  for (const entry of readdirSync(directory)) {
    const path = join(directory, entry);
    if (statSync(path).isDirectory()) collect(path, predicate, found);
    else if (predicate(path)) found.push(path);
  }
  return found;
};

if (!existsSync(artifacts)) {
  throw new Error(`Release artifact directory not found: ${artifacts}. Pass --artifacts PATH after downloading the release build artifacts`);
}
const platformPackages = collect(artifacts, (path) => /rust-ml-runtime-node-(?:darwin|linux|win32)-.+\.tgz$/.test(path)).sort();
const rootPackages = collect(artifacts, (path) => /rust-ml-runtime-node-\d+\.\d+\.\d+\.tgz$/.test(path)).sort();
if (platformPackages.length !== 5 || rootPackages.length !== 1) {
  throw new Error(`Expected five native npm packages and one root package in ${artifacts}`);
}

if (dryRun) {
  run('cargo', ['publish', '--dry-run', ...crateOrder.flatMap((name) => ['-p', name])]);
  for (const path of [...platformPackages, ...rootPackages]) run('npm', ['publish', '--dry-run', '--access', 'public', path]);
  process.exit(0);
}

const userAgent = 'rust-ml-runtime-release/0.1 (github.com/rkendel1/rust-ml-runtime)';
const exists = async (url) => {
  const response = await fetch(url, { headers: { 'user-agent': userAgent } });
  if (response.status === 404) return false;
  if (!response.ok) throw new Error(`Registry preflight failed (${response.status}) for ${url}`);
  return true;
};
for (const name of crateOrder) {
  if (await exists(`https://crates.io/api/v1/crates/${name}/${version}`)) {
    throw new Error(`${name}@${version} already exists on crates.io; versions are immutable`);
  }
}
for (const name of [nodeRoot.name, ...Object.keys(nodeRoot.optionalDependencies)]) {
  const encoded = encodeURIComponent(name).replace('%2F', '%2f');
  if (await exists(`https://registry.npmjs.org/${encoded}/${version}`)) {
    throw new Error(`${name}@${version} already exists on npm; refusing a duplicate coordinated release`);
  }
}

const cargoSelection = crateOrder.flatMap((name) => ['-p', name]);
run('cargo', ['publish', '--dry-run', ...cargoSelection]);
run('cargo', ['publish', ...cargoSelection]);
for (const path of platformPackages) run('npm', ['publish', '--access', 'public', path]);
run('npm', ['publish', '--access', 'public', rootPackages[0]]);
