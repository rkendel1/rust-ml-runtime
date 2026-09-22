#!/usr/bin/env node
import { readFile, readdir } from 'node:fs/promises';

const workflow = await readFile('.github/workflows/release.yml', 'utf8');
const forbiddenWorkflowPatterns = [
  [/\brepository\s*:/i, 'external repository checkout'],
  [/flow_db|feltdb/i, 'FeltDB workspace reference'],
  [/JEV_ROOT|healthcare-jev|verify-jev/i, 'Jev source-tree integration'],
];
for (const [pattern, description] of forbiddenWorkflowPatterns) {
  if (pattern.test(workflow)) throw new Error(`release workflow contains ${description}`);
}

for (const name of await readdir('examples/consumer')) {
  if (!/\.(?:cjs|mjs|json)$/.test(name)) continue;
  const contents = await readFile(`examples/consumer/${name}`, 'utf8');
  if (/\b(?:file|workspace):|flow_db|JEV_ROOT|@feltdb\//i.test(contents)) {
    throw new Error(`external consumer ${name} depends on a workspace or FeltDB source`);
  }
}
