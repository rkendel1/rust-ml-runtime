#!/usr/bin/env node
import { execFileSync } from 'node:child_process';
import { existsSync, readdirSync, readFileSync, statSync } from 'node:fs';
import { basename, join } from 'node:path';

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
const npmOnly = args.includes('--npm-only');
const valueAfter = (flag, fallback) => {
  const index = args.indexOf(flag);
  return index === -1 ? fallback : args[index + 1];
};
const run = (command, commandArgs, options = {}) => {
  console.log(`+ ${command} ${commandArgs.join(' ')}`);
  execFileSync(command, commandArgs, { stdio: 'inherit', ...options });
};
const collect = (directory, predicate, found = []) => {
  for (const entry of readdirSync(directory)) {
    const path = join(directory, entry);
    if (statSync(path).isDirectory()) collect(path, predicate, found);
    else if (predicate(path)) found.push(path);
  }
  return found;
};
const formatPaths = paths => paths.length ? `\n - ${paths.join('\n - ')}` : '\n - (none)';
const escapeRegExp = value => value.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
const readTarJson = (archive, file) => {
  try {
    return JSON.parse(execFileSync('tar', ['-xOf', archive, `package/${file}`], { encoding: 'utf8' }));
  } catch (error) {
    if (error?.code === 'ENOENT' || error?.cause?.code === 'ENOENT') {
      throw new Error('tar is required to inspect release artifacts on this runner');
    }
    throw error;
  }
};

async function main() {
  const releasePackage = JSON.parse(readFileSync('package.json', 'utf8'));
  const requested = valueAfter('--version', process.env.RELEASE_VERSION ?? releasePackage.version);
  const version = requested.split('/').at(-1).replace(/^v/, '');
  const artifacts = valueAfter('--artifacts', 'artifacts');
  if (!/^\d+\.\d+\.\d+$/.test(version)) throw new Error('Pass --version X.Y.Z or set RELEASE_VERSION');

  if (!npmOnly) {
    const metadata = JSON.parse(execFileSync('cargo', ['metadata', '--no-deps', '--format-version', '1'], { encoding: 'utf8' }));
    for (const name of crateOrder) {
      const pkg = metadata.packages.find((candidate) => candidate.name === name);
      if (!pkg) throw new Error(`Missing publishable crate ${name}`);
      if (pkg.version !== version) throw new Error(`${name} is ${pkg.version}; expected ${version}`);
    }
  }

  if (!dryRun) {
    if (!npmOnly && !process.env.CARGO_REGISTRY_TOKEN) {
      throw new Error('CARGO_REGISTRY_TOKEN is required for crates.io publication');
    }
    if (!process.env.NODE_AUTH_TOKEN) throw new Error('NODE_AUTH_TOKEN is required');
  }

  if (!existsSync(artifacts)) {
    throw new Error(`Release artifact directory not found: ${artifacts}. Pass --artifacts PATH after downloading the release build artifacts`);
  }

  const npmTarballs = collect(
    artifacts,
    (path) => path.endsWith('.tgz') && basename(path).startsWith('rust-ml-runtime-node-'),
  ).sort();
  const versionPattern = escapeRegExp(version);
  const rootPattern = new RegExp(`rust-ml-runtime-node-${versionPattern}\\.tgz$`);
  const rootPackages = npmTarballs.filter((path) => rootPattern.test(path));
  console.log(`Using release artifacts from ${artifacts}`);
  console.log(`Discovered ${npmTarballs.length} npm tarball(s):${formatPaths(npmTarballs)}`);
  let nodeRoot;
  if (npmOnly) {
    if (rootPackages.length !== 1) {
      throw new Error(
        `Expected one root npm package tarball for ${version}; found ${rootPackages.length}. ` +
        `Check that --artifacts points at the release run for version ${version}.`
      );
    }
    nodeRoot = readTarJson(rootPackages[0], 'package.json');
  } else {
    nodeRoot = JSON.parse(readFileSync('bindings/node/package.json', 'utf8'));
  }
  if (nodeRoot.version !== version) throw new Error(`${nodeRoot.name} is ${nodeRoot.version}; expected ${version}`);
  const optionalDependencies = nodeRoot.optionalDependencies ?? {};
  const expectedPlatformPackages = Object.keys(optionalDependencies).map((name) => {
    if (!name.startsWith(`${nodeRoot.name}-`)) {
      throw new Error(`Unexpected optional dependency ${name}; expected ${nodeRoot.name}-<platform>`);
    }
    return `rust-ml-runtime-node-${name.slice(`${nodeRoot.name}-`.length)}-${version}.tgz`;
  }).sort();
  const expectedPlatformPackageSet = new Set(expectedPlatformPackages);
  const platformPackages = npmTarballs.filter((path) => expectedPlatformPackageSet.has(basename(path)));
  const discoveredPlatformPackages = platformPackages.map((path) => basename(path)).sort();
  if (
    rootPackages.length !== 1 ||
    JSON.stringify(discoveredPlatformPackages) !== JSON.stringify(expectedPlatformPackages)
  ) {
    throw new Error(
      `Expected native npm tarballs ${expectedPlatformPackages.join(', ')} and one root package for ${version}; ` +
      `found native tarballs ${discoveredPlatformPackages.join(', ') || '(none)'} and ${rootPackages.length} root package(s). ` +
      `Check that --artifacts points at the release run for version ${version}.`
    );
  }
  console.log(`Native npm publish order:${formatPaths(platformPackages)}`);
  console.log(`Root npm package:\n - ${rootPackages[0]}`);

  if (dryRun) {
    if (!npmOnly) run('cargo', ['publish', '--dry-run', ...crateOrder.flatMap((name) => ['-p', name])]);
    for (const path of [...platformPackages, ...rootPackages]) run('npm', ['publish', '--dry-run', '--access', 'public', path]);
    return;
  }

  if (typeof fetch !== 'function') {
    throw new Error('Global fetch is required for registry preflight; run this script with Node.js 18 or newer');
  }
  const userAgent = 'rust-ml-runtime-release (github.com/rkendel1/rust-ml-runtime)';
  const registryHeaders = { 'user-agent': userAgent };
  const npmRegistryHeaders = { ...registryHeaders };
  if (process.env.NODE_AUTH_TOKEN) npmRegistryHeaders.authorization = 'Bearer ' + process.env.NODE_AUTH_TOKEN;
  const exists = async (url, headers = registryHeaders) => {
    const response = await fetch(url, { headers });
    if (response.status === 404) return false;
    if (!response.ok) throw new Error(`Registry preflight failed (${response.status}) for ${url}`);
    return true;
  };
  if (!npmOnly) {
    for (const name of crateOrder) {
      if (await exists(`https://crates.io/api/v1/crates/${name}/${version}`)) {
        throw new Error(`${name}@${version} already exists on crates.io; versions are immutable`);
      }
    }
  }
  for (const name of [nodeRoot.name, ...Object.keys(optionalDependencies)]) {
    const encoded = encodeURIComponent(name).replace(/%2F/gi, '%2f');
    if (await exists(`https://registry.npmjs.org/${encoded}/${version}`, npmRegistryHeaders)) {
      throw new Error(`${name}@${version} already exists on npm; refusing a duplicate coordinated release`);
    }
  }

  if (!npmOnly) {
    const cargoSelection = crateOrder.flatMap((name) => ['-p', name]);
    run('cargo', ['publish', '--dry-run', ...cargoSelection]);
    for (const [index, name] of crateOrder.entries()) {
      run('cargo', ['publish', '-p', name]);
      if (index < crateOrder.length - 1) {
        // crates.io needs time to make each newly published dependency visible.
        run('sleep', ['15']);
      }
    }
  }
  for (const path of platformPackages) run('npm', ['publish', '--access', 'public', path]);
  run('npm', ['publish', '--access', 'public', rootPackages[0]]);
}

main().catch((error) => {
  console.error(error.message);
  process.exit(1);
});
