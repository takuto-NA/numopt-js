/**
 * Serial follow-up protocol: recover the long state case, refine crossovers, and vary curvature/input structure.
 */

import { spawn } from 'node:child_process';
import { fileURLToPath } from 'node:url';

const BENCHMARK = fileURLToPath(new URL('../adjoint-scaling-benchmark.mjs', import.meta.url));
const runs = [
  ['--sizes=16x1024', '--families=curved', '--methods=approximate', '--timeout-ms=240000', '--name=adjoint-scaling-extended'],
  ['--sizes=64x128,96x128,128x128', '--families=easy', '--warmups=8', '--repeats=7', '--memory-repeats=0',
    '--timeout-ms=90000', '--name=adjoint-scaling-refine-easy'],
  ['--sizes=128x128,160x128,192x128,256x128,192x256,224x256,256x256', '--families=curved',
    '--methods=bfgs,full-dense,full-hvp', '--warmups=8', '--repeats=7', '--memory-repeats=0',
    '--timeout-ms=90000', '--name=adjoint-scaling-refine-curved'],
  ['--sizes=16x128,32x128,64x128,128x128', '--families=curved', '--methods=bfgs,approximate-fd,full-fd',
    '--timeout-ms=240000', '--name=adjoint-scaling-fd'],
  ['--sizes=128x128,512x128,256x512', '--families=heterogeneous', '--timeout-ms=240000', '--name=adjoint-scaling-heterogeneous']
];

for (const argumentsList of runs) {
  console.log(`\nFollow-up: ${argumentsList.join(' ')}`);
  const exitCode = await new Promise(resolve => {
    const child = spawn(process.execPath, [BENCHMARK, ...argumentsList, '--resume'], { windowsHide: true, stdio: 'inherit' });
    child.on('error', error => { console.error(error); resolve(1); });
    child.on('close', code => resolve(code ?? 1));
  });
  if (exitCode !== 0) { process.exitCode = exitCode; break; }
}
