/** Build optional kibo WASM assets from a clean, fixed external source. */
import fs from 'node:fs';
import path from 'node:path';
import { spawnSync } from 'node:child_process';
import { assetDirectory, projectRoot, toolchains, sha256, verifyKiboAssets } from './kibo-assets.mjs';

function run(command, args, cwd = projectRoot) {
  const result = spawnSync(command, args, { cwd, encoding: 'utf8', maxBuffer: 8 * 1024 * 1024 });
  if (result.error || result.status !== 0)
    throw new Error(`${command} failed: ${result.error?.message ?? result.stderr ?? result.stdout}`);
  return result.stdout.trim();
}

const argumentsList = process.argv.slice(2);
if (argumentsList.length && (argumentsList.length !== 2 || argumentsList[0] !== '--source'))
  throw new Error('Usage: node scripts/prepare-kibo-wasm.mjs [--source <clean pinned checkout>]');
const source = path.resolve(projectRoot, argumentsList[1] ?? '.cache/kibo-source');
if (!fs.existsSync(source)) {
  if (argumentsList.length) throw new Error('Explicit source checkout does not exist');
  run('git', ['clone', '--no-checkout', toolchains.repository, source]);
  run('git', ['checkout', '--detach', toolchains.commit], source);
}
const commit = run('git', ['rev-parse', 'HEAD'], source);
if (commit !== toolchains.commit || run('git', ['status', '--porcelain'], source))
  throw new Error('kibo source must be clean and at the fixed commit; existing files are never reset');
const sourceTree = run('git', ['rev-parse', 'HEAD^{tree}'], source);
fs.mkdirSync(assetDirectory, { recursive: true });
const command = ['run', '--rm', '--mount', `type=bind,source=${source},target=/work`,
  '-w', '/work', toolchains.emscriptenImage, 'sh', 'tools/wasm-build.sh'];
const compilerLog = run('docker', command);
console.log(compilerLog);
const compiler = compilerLog.split('\n')[0].trim();
fs.writeFileSync(path.join(assetDirectory, 'build.log'), `${compilerLog}\n`);
const sources = {
  'kibo.mjs': 'build/wasm/kibo.mjs', 'kibo.wasm': 'build/wasm/kibo.wasm',
  'facade.mjs': 'wasm/facade.mjs', LICENSE: 'LICENSE', NOTICE: 'NOTICE'
};
for (const [name, relative] of Object.entries(sources))
  fs.copyFileSync(path.join(source, relative), path.join(assetDirectory, name));
const manifest = {
  source: { repository: toolchains.repository, commit, tree: sourceTree },
  emscriptenImage: toolchains.emscriptenImage, compiler,
  files: Object.fromEntries(Object.keys(sources).map(name => [name, sha256(path.join(assetDirectory, name))])),
  command: ['docker', ...command], preparedAt: new Date().toISOString()
};
fs.writeFileSync(path.join(assetDirectory, 'manifest.json'), `${JSON.stringify(manifest, null, 2)}\n`);
verifyKiboAssets();
console.log(`Verified kibo ${commit}; assets: ${assetDirectory}`);
