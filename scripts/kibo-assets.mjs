/** Validate the external source/toolchain and every generated WASM asset. */
import { createHash } from 'node:crypto';
import { execFileSync } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

export const projectRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
export const assetDirectory = path.join(projectRoot, '.cache/kibo');
export const toolchains = JSON.parse(fs.readFileSync(path.join(projectRoot, 'integration/kibo/toolchains.json'), 'utf8'));
export const assetNames = ['kibo.mjs', 'kibo.wasm', 'facade.mjs', 'LICENSE', 'NOTICE'];
export function sha256(file) { return createHash('sha256').update(fs.readFileSync(file)).digest('hex'); }

function filesIn(directory) {
  return fs.readdirSync(path.join(projectRoot, directory), { withFileTypes: true }).flatMap(entry => {
    const relative = `${directory}/${entry.name}`;
    return entry.isDirectory() ? filesIn(relative) : [relative];
  });
}

export function collectNumoptProvenance() {
  const git = (...arguments_) => execFileSync('git', arguments_, { cwd: projectRoot, encoding: 'utf8' }).trim();
  const sourceFiles = ['package.json', 'package-lock.json', 'tsconfig.json', 'tsconfig.build.json', 'rollup.config.js',
    ...filesIn('src'), ...filesIn('scripts'), ...filesIn('integration/kibo')].sort();
  return {
    commit: git('rev-parse', 'HEAD'), dirty: git('status', '--porcelain').length > 0,
    sourceHashes: Object.fromEntries(sourceFiles.map(file => [file, sha256(path.join(projectRoot, file))])),
    builtHashes: Object.fromEntries(filesIn('dist').sort().map(file => [file, sha256(path.join(projectRoot, file))]))
  };
}

export function verifyKiboAssets(directory = assetDirectory) {
  const manifest = JSON.parse(fs.readFileSync(path.join(directory, 'manifest.json'), 'utf8'));
  if (manifest.source?.commit !== toolchains.commit || manifest.source?.repository !== toolchains.repository ||
      manifest.emscriptenImage !== toolchains.emscriptenImage ||
      !manifest.compiler?.startsWith(`emcc (Emscripten gcc/clang-like replacement + linker emulating GNU ld) ${toolchains.emscriptenVersion} `))
    throw new Error('kibo source/toolchain does not match the fixed configuration');
  for (const name of assetNames) {
    if (manifest.files?.[name] !== sha256(path.join(directory, name)))
      throw new Error(`kibo asset checksum mismatch: ${name}`);
  }
  return manifest;
}
