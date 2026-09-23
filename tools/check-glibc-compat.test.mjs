import assert from 'node:assert/strict';
import test from 'node:test';
import {
  assertGlibcCompatible,
  compareVersions,
  requiredGlibcVersions,
} from './check-glibc-compat.mjs';

const versionNeeds = versions => `
Version symbols section '.gnu.version' contains 4 entries:
  000:   0 (*local*)       2 (GLIBC_2.2.5)   3 (GLIBC_2.39)
Version needs section '.gnu.version_r' contains 1 entry:
 Addr: 0x0000000000000000  Offset: 0x000000  Link: 4 (.dynstr)
  000000: Version: 1  File: libc.so.6  Cnt: ${versions.length}
${versions.map((version, index) => `  0x${index}: Name: GLIBC_${version}  Flags: none  Version: ${index + 2}`).join('\n')}
`;

test('the published v0.2.0 requirement is rejected by the Linux ABI baseline', () => {
  const versions = requiredGlibcVersions(versionNeeds(['2.2.5', '2.39']));
  assert.deepEqual(versions, ['2.2.5', '2.39']);
  assert.throws(
    () => assertGlibcCompatible(versions, '2.36', 'v0.2.0 ml_runtime_node.node'),
    /GLIBC_2\.39.*baseline: GLIBC_2\.36/,
  );
});

test('Bookworm-compatible symbol requirements pass and sort numerically', () => {
  const versions = requiredGlibcVersions(versionNeeds(['2.17', '2.3', '2.36', '2.9']));
  assert.deepEqual(versions, ['2.3', '2.9', '2.17', '2.36']);
  assert.doesNotThrow(() => assertGlibcCompatible(versions, '2.36'));
  assert.equal(compareVersions('2.39', '2.36'), 1);
});

test('GLIBC names outside the version-needs section are ignored', () => {
  const output = `${versionNeeds(['2.36'])}\nVersion definition section '.gnu.version_d':\n  Name: GLIBC_2.39`;
  assert.deepEqual(requiredGlibcVersions(output), ['2.36']);
});
