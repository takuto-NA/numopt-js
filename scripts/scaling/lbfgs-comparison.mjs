/** Runs serial L-BFGS, BFGS and Newton comparisons under one accuracy and memory protocol. */
import { spawn } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import path from 'node:path';

const ROOT = fileURLToPath(new URL('../../', import.meta.url));
const BENCHMARK = path.join(ROOT, 'scripts', 'adjoint-scaling-benchmark.mjs');
const METHODS = 'bfgs,lbfgs,approximate,full-dense,full-hvp';
const profiles = [
  { name: 'adjoint-lbfgs-curved', families: 'curved', methods: METHODS,
    sizes: '16x128,32x128,64x128,128x128,256x128,512x128,256x256,256x512,512x512,16x1024' },
  { name: 'adjoint-lbfgs-heterogeneous', families: 'heterogeneous', methods: METHODS,
    sizes: '128x128,512x128,256x512' },
  { name: 'adjoint-lbfgs-conditioned', families: 'ill-conditioned',
    methods: `${METHODS},lbfgs-5,lbfgs-20`, sizes: '32x64,128x128' }
];

function run(profile) {
  const argumentsList = Object.entries(profile).map(([name, value]) => `--${name}=${value}`);
  return new Promise((resolve, reject) => {
    const child = spawn(process.execPath, [BENCHMARK, ...argumentsList, '--resume'], {
      cwd: ROOT, windowsHide: true, stdio: 'inherit'
    });
    child.once('error', reject);
    child.once('close', code => code === 0 ? resolve() : reject(new Error(`${profile.name} exited ${code}`)));
  });
}

for (const profile of profiles) await run(profile);
