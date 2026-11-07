/** Run fixed-topology length fitting and write independently checked data plus an offline replay. */

import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { fitTrajectory } from './linkage-trajectory/fit';
import { createTrajectoryProblem, TARGET_CASES } from './linkage-trajectory/model';
import { createTrajectoryReplay } from './linkage-trajectory/replay-data';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const OUTPUT_DIRECTORY = path.join(ROOT, 'benchmark-results');
const rows = TARGET_CASES.map(targetCase => {
  const problem = createTrajectoryProblem(targetCase);
  return createTrajectoryReplay(problem, fitTrajectory(problem));
});
const report = { method: 'BFGS + public IFT', trainingPhases: 32, heldOutPhases: 257,
  designVariables: ['crank length', 'coupler length', 'rocker length'],
  fixed: ['four-bar connectivity', 'ground pivots', 'marker coordinates', 'input phases', 'assembly branch', 'target trajectory'], rows };
fs.mkdirSync(OUTPUT_DIRECTORY, { recursive: true });
const jsonPath = path.join(OUTPUT_DIRECTORY, 'linkage-trajectory-example.json');
const htmlPath = path.join(OUTPUT_DIRECTORY, 'linkage-trajectory-example.html');
const template = fs.readFileSync(path.join(ROOT, 'examples/linkage-trajectory/replay.html'), 'utf8');
fs.writeFileSync(jsonPath, JSON.stringify(report, null, 2));
// Escaping '<' keeps future user-supplied labels from terminating the data script.
fs.writeFileSync(htmlPath, template.replace('__TRAJECTORY_DATA__', JSON.stringify(report).replace(/</g, '\\u003c')));
console.table(rows.map(row => ({ target: row.targetCase, passed: row.verification.passed,
  iterations: row.verification.iterations, initialRms: row.verification.initialError,
  heldOutRms: row.verification.heldOutError, maxHeldOut: row.verification.maximumHeldOutError,
  constraintNorm: row.verification.constraintNorm, gradientNorm: row.verification.gradientNorm })));
console.log(`Wrote ${jsonPath}\nOpen ${htmlPath} to replay recorded iterations and crank motion.`);
if (rows.some(row => !row.verification.passed)) process.exitCode = 1;
