#!/usr/bin/env node
import { execFileSync } from 'node:child_process';
import { cpSync, existsSync, lstatSync, mkdtempSync, mkdirSync, readFileSync, readdirSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { basename, join } from 'node:path';

const version = (process.argv[2] ?? '').replace(/^v/, '');
const archiveDirectory = process.argv[3] ?? 'target/package';
if (!version) throw new Error('usage: verify-rust-package.mjs VERSION [PACKAGE_DIRECTORY]');
const archiveSuffix = `-${version}.crate`;
const root = mkdtempSync(join(tmpdir(), 'rust-ml-runtime-consumer-'));
const packages = join(root, 'packages');
const consumer = join(root, 'consumer');
mkdirSync(packages);
cpSync('examples/rust-consumer', consumer, { recursive: true });

const collectArchives = (directory) => {
  const archives = [];
  for (const entry of readdirSync(directory, { withFileTypes: true })) {
    const path = join(directory, entry.name);
    if (entry.isDirectory()) archives.push(...collectArchives(path));
    else if (entry.isFile() && entry.name.endsWith(archiveSuffix)) archives.push(path);
  }
  return archives;
};
const crateNameFromArchive = (archive) => {
  const name = basename(archive);
  if (!name.endsWith(archiveSuffix)) throw new Error(`unexpected crate archive ${name}`);
  return name.slice(0, -archiveSuffix.length);
};
const listArchives = (directory) => readdirSync(directory, { withFileTypes: true })
  .filter((entry) => entry.isFile() && entry.name.endsWith(archiveSuffix))
  .map((entry) => join(directory, entry.name))
  .sort();
const crateNames = [];
let archives = listArchives(archiveDirectory);
if (archives.length === 0) {
  const packagedArchives = join(archiveDirectory, 'tmp-crate');
  try {
    archives = listArchives(packagedArchives);
  } catch {
    archives = [];
  }
}
if (archives.length === 0) {
  const seen = new Map();
  for (const path of collectArchives(archiveDirectory).sort()) {
    const name = basename(path);
    if (seen.has(name)) throw new Error(`multiple packaged crates found for ${name}`);
    seen.set(name, path);
  }
  archives = [...seen.values()];
}
if (archives.length === 0) throw new Error(`no packaged crates found for version ${version} in ${archiveDirectory}`);
for (const archive of archives) {
  execFileSync('tar', ['-xzf', archive, '-C', packages]);
  crateNames.push(crateNameFromArchive(archive));
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
if (!existsSync(runtimePackagePath) || !lstatSync(runtimePackagePath).isDirectory()) {
  throw new Error(`missing extracted package directory ${runtimePackagePath}`);
}
const runtimeDependency = `rust-ml-runtime = { version = ${JSON.stringify(`=${version}`)}, path = ${JSON.stringify(runtimePackagePath)} }`;
const createRuntimeDependencyTable = (header) => [
  header,
  `version = ${JSON.stringify(`=${version}`)}`,
  `path = ${JSON.stringify(runtimePackagePath)}`,
].join('\n');
const dependencySections = ['build-dependencies', 'dependencies', 'dev-dependencies'];
const getTableHeader = (line) => line.match(/^\s*\[([^\]]+)\]\s*$/)?.[1] ?? null;
const isTableHeader = (line) => getTableHeader(line) !== null;
const isTableEntry = (line) => {
  const trimmed = line.trim();
  return trimmed !== '' && !trimmed.startsWith('#') && !isTableHeader(line);
};
const isCommentOrBlank = (line) => {
  const trimmed = line.trim();
  return trimmed === '' || trimmed.startsWith('#');
};
const isDependenciesSection = (line) => {
  const header = getTableHeader(line);
  return header !== null && dependencySections.some((section) => header === section || (header.startsWith('target.') && header.endsWith(`.${section}`)));
};
const isRuntimeDependencyTable = (line) => {
  const header = getTableHeader(line);
  return header !== null && dependencySections.some((section) => header === `${section}.rust-ml-runtime` || (header.startsWith('target.') && header.endsWith(`.${section}.rust-ml-runtime`)));
};
const runtimeDependencyEntryPattern = /^(\s*)((?:"rust-ml-runtime"|rust-ml-runtime))(\s*)=/;
const runtimeDependencyStringPattern = /^(\s*)((?:"rust-ml-runtime"|rust-ml-runtime))(\s*)=\s*(?:"(?:[^"\\]|\\.)*"|'[^']*')(\s*(?:#.*)?)$/;
const isOverriddenDependencySetting = (line) => /^\s*(?:"(?:path|version)"|path|version)\s*=/.test(line);
const isCommentStart = (previousChar, char) => char === '#' && (previousChar === '' || /\s/.test(previousChar));
const extractTrailingComment = (line) => {
  let inSingleQuote = false;
  let inDoubleQuote = false;
  let escaped = false;
  let previousChar = '';
  for (let index = 0; index < line.length; index += 1) {
    const char = line[index];
    if (escaped) {
      escaped = false;
      previousChar = char;
      continue;
    }
    if (inDoubleQuote) {
      if (char === '\\') escaped = true;
      else if (char === '"') inDoubleQuote = false;
      previousChar = char;
      continue;
    }
    if (inSingleQuote) {
      if (char === '\'') inSingleQuote = false;
      previousChar = char;
      continue;
    }
    if (isCommentStart(previousChar, char)) return line.slice(index);
    if (char === '"') inDoubleQuote = true;
    else if (char === '\'') inSingleQuote = true;
    previousChar = char;
  }
  return '';
};
const splitInlineTableEntries = (content) => {
  const entries = [];
  let current = '';
  let bracketDepth = 0;
  let braceDepth = 0;
  let inComment = false;
  let inSingleQuote = false;
  let inDoubleQuote = false;
  let escaped = false;
  for (const char of content) {
    if (inComment) {
      if (char === '\n') inComment = false;
      continue;
    }
    if (escaped) {
      current += char;
      escaped = false;
      continue;
    }
    if (inDoubleQuote) {
      current += char;
      if (char === '\\') escaped = true;
      else if (char === '"') inDoubleQuote = false;
      continue;
    }
    if (inSingleQuote) {
      current += char;
      if (char === '\'') inSingleQuote = false;
      continue;
    }
    if (char === '"') inDoubleQuote = true;
    else if (char === '\'') inSingleQuote = true;
    else if (isCommentStart(current.at(-1) ?? '', char)) {
      inComment = true;
      continue;
    }
    else if (char === '[') bracketDepth += 1;
    else if (char === ']') bracketDepth -= 1;
    else if (char === '{') braceDepth += 1;
    else if (char === '}') braceDepth -= 1;
    else if (char === ',' && bracketDepth === 0 && braceDepth === 0) {
      if (current.trim()) entries.push(current.trim());
      current = '';
      continue;
    }
    current += char;
  }
  if (current.trim()) entries.push(current.trim());
  return entries;
};
const braceDelta = (line) => {
  let delta = 0;
  let inSingleQuote = false;
  let inDoubleQuote = false;
  let escaped = false;
  let previousChar = '';
  for (const char of line) {
    if (escaped) {
      escaped = false;
      previousChar = char;
      continue;
    }
    if (inDoubleQuote) {
      if (char === '\\') escaped = true;
      else if (char === '"') inDoubleQuote = false;
      previousChar = char;
      continue;
    }
    if (inSingleQuote) {
      if (char === '\'') inSingleQuote = false;
      previousChar = char;
      continue;
    }
    if (isCommentStart(previousChar, char)) break;
    if (char === '"') inDoubleQuote = true;
    else if (char === '\'') inSingleQuote = true;
    else if (char === '{') delta += 1;
    else if (char === '}') delta -= 1;
    previousChar = char;
  }
  return delta;
};
const lines = manifest.split('\n');
const updatedLines = [];
let inDependencies = false;
let replaced = false;
for (let index = 0; index < lines.length; index += 1) {
  const line = lines[index];
  const trimmedLine = line.trim();
  if (isRuntimeDependencyTable(line)) {
    const bodyLines = [];
    while (index + 1 < lines.length && !isTableHeader(lines[index + 1])) {
      index += 1;
      bodyLines.push(lines[index]);
    }
    let trailingIndex = bodyLines.length;
    while (trailingIndex > 0 && isCommentOrBlank(bodyLines[trailingIndex - 1])) trailingIndex -= 1;
    const preservedEntries = [];
    for (const bodyLine of bodyLines.slice(0, trailingIndex)) {
      if (!isOverriddenDependencySetting(bodyLine)) {
        preservedEntries.push(bodyLine);
        continue;
      }
      const trailingComment = extractTrailingComment(bodyLine);
      if (trailingComment) preservedEntries.push(`${bodyLine.match(/^\s*/)?.[0] ?? ''}${trailingComment}`);
    }
    const trailingLines = bodyLines.slice(trailingIndex);
    updatedLines.push(createRuntimeDependencyTable(line.trimEnd()));
    updatedLines.push(...preservedEntries);
    updatedLines.push(...trailingLines);
    replaced = true;
    continue;
  }
  if (isTableHeader(line)) inDependencies = isDependenciesSection(line);
  const dependencyMatch = runtimeDependencyEntryPattern.exec(line);
  if (inDependencies && dependencyMatch) {
    const dependencyLines = [line];
    replaced = true;
    let braceDepth = braceDelta(line);
    while (braceDepth > 0 && index + 1 < lines.length) {
      index += 1;
      dependencyLines.push(lines[index]);
      braceDepth += braceDelta(lines[index]);
    }
    const inlineTableStart = dependencyLines[0].indexOf('{');
    if (inlineTableStart === -1) {
      const stringDependencyMatch = runtimeDependencyStringPattern.exec(dependencyLines[0]);
      if (!stringDependencyMatch) throw new Error(`unsupported rust-ml-runtime dependency declaration: ${dependencyLines[0]}`);
      updatedLines.push(`${dependencyMatch[1]}${dependencyMatch[2]}${dependencyMatch[3]}= { version = ${JSON.stringify(`=${version}`)}, path = ${JSON.stringify(runtimePackagePath)} }${stringDependencyMatch[4]}`);
    } else {
      const dependencyContent = dependencyLines.join('\n');
      const inlineTableEnd = dependencyContent.lastIndexOf('}');
      const inlineTableSuffix = dependencyContent.slice(inlineTableEnd + 1);
      const preservedEntries = splitInlineTableEntries(dependencyContent.slice(inlineTableStart + 1, inlineTableEnd))
        .filter((entry) => !isOverriddenDependencySetting(entry));
      const entries = [
        `version = ${JSON.stringify(`=${version}`)}`,
        `path = ${JSON.stringify(runtimePackagePath)}`,
        ...preservedEntries,
      ];
      updatedLines.push(`${dependencyMatch[1]}${dependencyMatch[2]}${dependencyMatch[3]}= { ${entries.join(', ')} }${inlineTableSuffix}`);
    }
    continue;
  }
  updatedLines.push(line);
}
if (!replaced) throw new Error('expected rust-ml-runtime dependency in external consumer manifest');
manifest = updatedLines.join('\n');
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
