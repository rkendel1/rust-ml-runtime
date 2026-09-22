#!/usr/bin/env node
import { execFileSync } from 'node:child_process';
import { readdirSync, readFileSync, statSync } from 'node:fs';
import { isAbsolute, join } from 'node:path';

const expectedVersion = (process.argv[2] ?? '').replace(/^v/, '');
const packageDirectory = process.argv[3] ?? 'target/package';
if (!expectedVersion) throw new Error('usage: validate-crates.mjs VERSION [PACKAGE_DIRECTORY]');

const publishable = new Set([
  'ml-runtime-common', 'ml-runtime-model', 'ml-runtime-inference', 'ml-runtime-backend',
  'ml-runtime-provider', 'ml-runtime-protocol', 'ml-runtime-cpu-backend',
  'ml-runtime-onnx-backend', 'ml-runtime-coreml-backend', 'ml-runtime-http-provider',
  'rust-ml-runtime',
]);
const privatePackages = new Set([
  'ml-runtime-cli', 'ml-runtime-cuda-backend', 'ml-runtime-local-provider',
  'ml-runtime-server-provider', 'ml-runtime-webgpu-backend', 'ml-runtime-node',
  'ml-runtime-benchmarks',
]);
const metadata = JSON.parse(execFileSync('cargo', ['metadata', '--no-deps', '--format-version', '1'], { encoding: 'utf8' }));
for (const pkg of metadata.packages) {
  if (pkg.version !== expectedVersion) throw new Error(`${pkg.name} is ${pkg.version}; expected ${expectedVersion}`);
  const canPublish = pkg.publish === null || pkg.publish.includes('crates-io');
  if (publishable.has(pkg.name) !== canPublish) throw new Error(`${pkg.name} has an incorrect Cargo publish boundary`);
  if (!publishable.has(pkg.name) && !privatePackages.has(pkg.name)) throw new Error(`Unclassified workspace crate ${pkg.name}`);
  for (const dependency of pkg.dependencies.filter((item) => item.path)) {
    if (isAbsolute(dependency.path) && !dependency.path.startsWith(metadata.workspace_root)) {
      throw new Error(`${pkg.name} uses an external absolute dependency path: ${dependency.path}`);
    }
    if (publishable.has(pkg.name) && publishable.has(dependency.name) && dependency.req !== `^${expectedVersion}`) {
      throw new Error(`${pkg.name} does not pin ${dependency.name} to ${expectedVersion}`);
    }
  }
}

const runtime = metadata.packages.find((pkg) => pkg.name === 'rust-ml-runtime');
const required = {
  description: 'Native, model-neutral runtime for running local ML models',
  homepage: 'https://github.com/rkendel1/rust-ml-runtime',
  repository: 'https://github.com/rkendel1/rust-ml-runtime',
  documentation: 'https://docs.rs/rust-ml-runtime',
  license: 'MIT',
};
for (const [field, value] of Object.entries(required)) {
  if (runtime[field] !== value) throw new Error(`rust-ml-runtime ${field} is ${runtime[field]}; expected ${value}`);
}
if (runtime.targets.some((target) => target.kind.includes('bin'))) throw new Error('rust-ml-runtime package unexpectedly contains a CLI binary');

const archives = readdirSync(packageDirectory).filter((name) => name.endsWith('.crate')).sort();
const expectedArchives = [...publishable].map((name) => `${name}-${expectedVersion}.crate`).sort();
if (JSON.stringify(archives) !== JSON.stringify(expectedArchives)) {
  throw new Error(`Unexpected crate archives: ${archives.join(', ')}`);
}
for (const archive of archives) {
  const path = join(packageDirectory, archive);
  if (!statSync(path).isFile()) throw new Error(`${path} is not a regular file`);
  const entries = execFileSync('tar', ['-tzf', path], { encoding: 'utf8' }).trim().split('\n');
  if (entries.some((entry) => entry.startsWith('/') || entry.split('/').includes('..'))) {
    throw new Error(`${archive} contains an unsafe path`);
  }
  if (entries.some((entry) => /(?:^|\/)(?:models|target|\.git|\.env(?:\.|$))/.test(entry))) {
    throw new Error(`${archive} contains repository, model, build, or environment state`);
  }
}
const runtimeEntries = execFileSync('tar', ['-tzf', join(packageDirectory, `rust-ml-runtime-${expectedVersion}.crate`)], { encoding: 'utf8' });
for (const requiredEntry of ['README.md', 'src/lib.rs', 'src/capability.rs']) {
  if (!runtimeEntries.includes(`/` + requiredEntry)) throw new Error(`rust-ml-runtime archive lacks ${requiredEntry}`);
}

const workflow = readFileSync('.github/workflows/release.yml', 'utf8');
if (!workflow.includes('CARGO_REGISTRY_TOKEN')) throw new Error('release workflow lacks CARGO_REGISTRY_TOKEN');
if (!workflow.includes('npm run publish:packages')) throw new Error('release workflow does not use npm run publish:packages');
console.log(`validated ${archives.length} crates.io archives at version ${expectedVersion}`);
