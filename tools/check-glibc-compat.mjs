#!/usr/bin/env node
import { execFileSync } from 'node:child_process';
import { existsSync, mkdtempSync, realpathSync } from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

export const LINUX_GLIBC_BASELINE = '2.36';

export function compareVersions(left, right) {
  const normalize = value => value.split('.').map(part => Number.parseInt(part, 10));
  const a = normalize(left);
  const b = normalize(right);
  for (let index = 0; index < Math.max(a.length, b.length); index += 1) {
    const difference = (a[index] ?? 0) - (b[index] ?? 0);
    if (difference !== 0) return Math.sign(difference);
  }
  return 0;
}

export function requiredGlibcVersions(readelfOutput) {
  const needs = readelfOutput.match(
    /Version needs section[\s\S]*?(?=\nVersion (?:definition|symbols) section|$)/,
  )?.[0] ?? '';
  return [...new Set([...needs.matchAll(/\bName:\s+GLIBC_(\d+(?:\.\d+)+)\b/g)]
    .map(match => match[1]))].sort(compareVersions);
}

export function assertGlibcCompatible(versions, baseline = LINUX_GLIBC_BASELINE, label = 'ELF artifact') {
  const unsupported = versions.filter(version => compareVersions(version, baseline) > 0);
  if (unsupported.length > 0) {
    throw new Error(
      `${label} requires unsupported glibc symbol versions: ` +
      `${unsupported.map(version => `GLIBC_${version}`).join(', ')} (baseline: GLIBC_${baseline})`,
    );
  }
}

function readelfVersions(file) {
  let output;
  try {
    output = execFileSync('readelf', ['--version-info', '--wide', file], { encoding: 'utf8' });
  } catch (error) {
    if (error?.code === 'ENOENT') throw new Error('readelf is required to inspect Linux native artifacts');
    throw error;
  }
  return requiredGlibcVersions(output);
}

function linkedDependencies(file) {
  let output;
  try {
    output = execFileSync('ldd', [file], { encoding: 'utf8' });
  } catch (error) {
    if (error?.code === 'ENOENT') throw new Error('ldd is required for dependency-graph inspection');
    throw error;
  }
  if (/=>\s+not found\b/.test(output)) {
    throw new Error(`unresolved native dependency for ${file}:\n${output.trim()}`);
  }
  const files = [];
  for (const line of output.split('\n')) {
    const resolved = line.match(/=>\s+(\/\S+)\s+\(/)?.[1];
    const loader = line.match(/^\s*(\/\S+)\s+\(/)?.[1];
    const candidate = resolved ?? loader;
    if (candidate && existsSync(candidate)) files.push(realpathSync(candidate));
  }
  return files;
}

function inspectArtifact(file, includeDependencies) {
  const root = realpathSync(file);
  const pending = [root];
  const inspected = new Map();
  while (pending.length > 0) {
    const current = pending.shift();
    if (inspected.has(current)) continue;
    inspected.set(current, readelfVersions(current));
    if (includeDependencies) pending.push(...linkedDependencies(current));
  }
  return inspected;
}

function extractArchive(archive) {
  const directory = mkdtempSync(path.join(tmpdir(), 'rust-ml-glibc-'));
  try {
    execFileSync('tar', ['-xzf', archive, '-C', directory]);
  } catch (error) {
    if (error?.code === 'ENOENT') throw new Error('tar is required to inspect packed npm artifacts');
    throw error;
  }
  const artifact = path.join(directory, 'package', 'ml_runtime_node.node');
  if (!existsSync(artifact)) throw new Error(`${archive} does not contain package/ml_runtime_node.node`);
  return artifact;
}

export function checkGlibcCompatibility(input, options = {}) {
  const baseline = options.baseline ?? LINUX_GLIBC_BASELINE;
  const artifact = input.endsWith('.tgz') ? extractArchive(input) : input;
  if (!existsSync(artifact)) throw new Error(`native artifact not found: ${artifact}`);
  const inspected = inspectArtifact(artifact, options.dependencies ?? false);
  for (const [file, versions] of inspected) assertGlibcCompatible(versions, baseline, file);
  return { artifact, baseline, inspected };
}

function main() {
  const args = process.argv.slice(2);
  const dependencies = args.includes('--dependencies');
  const baselineIndex = args.indexOf('--baseline');
  const baseline = baselineIndex === -1 ? LINUX_GLIBC_BASELINE : args[baselineIndex + 1];
  const input = args.find((argument, index) =>
    !argument.startsWith('--') && (baselineIndex === -1 || index !== baselineIndex + 1));
  if (!input || !baseline || !/^\d+\.\d+(?:\.\d+)?$/.test(baseline)) {
    throw new Error('usage: check-glibc-compat.mjs [--dependencies] [--baseline 2.36] ARTIFACT_OR_TGZ');
  }
  const result = checkGlibcCompatibility(input, { baseline, dependencies });
  for (const [file, versions] of result.inspected) {
    const maximum = versions.at(-1);
    console.log(`${file}: maximum required GLIBC_${maximum ?? 'none'}`);
  }
  console.log(`glibc compatibility passed (baseline: GLIBC_${baseline})`);
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  try {
    main();
  } catch (error) {
    console.error(error.message);
    process.exit(1);
  }
}
