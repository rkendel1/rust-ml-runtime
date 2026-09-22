import { copyFile, mkdir } from 'node:fs/promises';
import { spawnSync } from 'node:child_process';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const directory = path.dirname(fileURLToPath(import.meta.url));
const workspace = path.resolve(directory, '../..');
const built = spawnSync('cargo', ['build', '-p', 'ml-runtime-node'], {
  cwd: workspace,
  stdio: 'inherit',
});
if (built.status !== 0) process.exit(built.status ?? 1);

const library = process.platform === 'darwin'
  ? 'libml_runtime_node.dylib'
  : process.platform === 'win32'
    ? 'ml_runtime_node.dll'
    : 'libml_runtime_node.so';
const output = path.join(directory, 'native', 'ml_runtime_node.node');
await mkdir(path.dirname(output), { recursive: true });
await copyFile(path.join(workspace, 'target', 'debug', library), output);
console.log(output);
