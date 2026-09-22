import { copyFile, mkdir } from 'node:fs/promises';
import { spawnSync } from 'node:child_process';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const directory = path.dirname(fileURLToPath(import.meta.url));
const workspace = path.resolve(directory, '../..');
const target = process.env.ML_RUNTIME_NODE_TARGET;
const cargoArguments = ['build', '--release', '-p', 'ml-runtime-node'];
if (target) cargoArguments.push('--target', target);
const built = spawnSync('cargo', cargoArguments, {
  cwd: workspace,
  stdio: 'inherit',
});
if (built.status !== 0) process.exit(built.status ?? 1);

const library = process.platform === 'darwin'
  ? 'libml_runtime_node.dylib'
  : process.platform === 'win32'
    ? 'ml_runtime_node.dll'
    : 'libml_runtime_node.so';
const platform = `${process.platform}-${process.arch}`;
const output = path.join(directory, 'native', platform, 'ml_runtime_node.node');
await mkdir(path.dirname(output), { recursive: true });
await copyFile(path.join(workspace, 'target', ...(target ? [target] : []), 'release', library), output);
console.log(output);
