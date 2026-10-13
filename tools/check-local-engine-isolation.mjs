#!/usr/bin/env node
// The embedded local-inference engine (llama.cpp) is opt-in: nothing the
// repository publishes by default may compile, link, or ship it.
import { execFileSync } from 'node:child_process';
import { readFile } from 'node:fs/promises';

const ENGINE = /llama-cpp|llama_cpp|ml-runtime-llama|node-llama|node_llama/i;

export function engineMatches(text) {
  return text.split('\n').filter(line => ENGINE.test(line));
}

export function npmFileProblems(files) {
  return files.filter(file => /\.gguf$|node-llama|ml_runtime_node_llama/i.test(file));
}

export function manifestProblems(manifest) {
  const names = [
    ...Object.keys(manifest.dependencies ?? {}),
    ...Object.keys(manifest.optionalDependencies ?? {}),
    ...Object.keys(manifest.peerDependencies ?? {}),
  ];
  return names.filter(name => ENGINE.test(name));
}

const cargoTree = (...args) => execFileSync('cargo', ['tree', '--locked', '-e', 'normal,build', ...args], {
  encoding: 'utf8',
  maxBuffer: 64 * 1024 * 1024,
});

function main() {
  const failures = [];
  for (const [label, args] of [
    ['workspace (default features)', ['--workspace']],
    ['workspace (--all-features)', ['--workspace', '--all-features']],
    ['@rust-ml-runtime/node addon', ['-p', 'ml-runtime-node']],
  ]) {
    const hits = engineMatches(cargoTree(...args));
    if (hits.length > 0) failures.push(`${label} depends on the local engine:\n  ${hits.slice(0, 5).join('\n  ')}`);
  }
  return Promise.all([
    readFile('bindings/node/package.json', 'utf8'),
    Promise.resolve(execFileSync('npm', ['pack', '--dry-run', '--json', '--ignore-scripts'], {
      cwd: 'bindings/node',
      encoding: 'utf8',
    })),
  ]).then(([manifestText, packText]) => {
    const manifest = JSON.parse(manifestText);
    const bad = manifestProblems(manifest);
    if (bad.length > 0) failures.push(`@rust-ml-runtime/node declares local-engine packages: ${bad.join(', ')}`);
    const files = JSON.parse(packText).flatMap(entry => entry.files.map(file => file.path));
    const shipped = npmFileProblems(files);
    if (shipped.length > 0) failures.push(`npm package contains local-engine artifacts or model weights: ${shipped.join(', ')}`);
    if (failures.length > 0) {
      console.error(failures.join('\n'));
      process.exit(1);
    }
    console.log('local engine isolation: ok (no llama.cpp in default or --all-features builds, the node addon, or the npm package)');
  });
}

if (import.meta.url === `file://${process.argv[1]}`) await main();
