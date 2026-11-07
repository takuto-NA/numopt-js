/** Saves independent multistart accuracy and expected-failure checks; no timing rankings. */

import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { execFileSync } from 'node:child_process';
import { collectPaperEnvironment } from './benchmark-harness';
import { evaluateFourBarInitializations, evaluateMechanismSafety, evaluateSliderInitializations } from '../examples/mechanical-character/robustness';

const OUTPUT_DIRECTORY = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../benchmark-results');

const environment = { ...collectPaperEnvironment(), gitDirty: execFileSync('git', ['status', '--porcelain']).length > 0 };
const rows = [...evaluateFourBarInitializations(), ...evaluateSliderInitializations()];
const checks = evaluateMechanismSafety();
const lines = ['# Mechanical-character robustness', '',
  'Synthetic accuracy and failure tests extending the regular four-bar fixture; no global convergence claim.',
  'Three starts per mechanism. Four-bars: 24 design phases, reachable/distorted targets, 3 methods.',
  'Slider-cranks: 24 design phases, reachable target, 2 methods; independent slider geometry on 257 held-out phases.',
  'Near-toggle sensitivity checks use physical rod-crank gaps 1e-2, 1e-4, 1e-6, with perturbations 1e-3 of each gap.',
  'Exact-toggle/infeasible domain checks reject the state oracle, not a new rank-detection guarantee in the public Jacobian API.', '',
  '| Mechanism | Case | Method | Accuracy pass | Solver flag | Iterations | Held-out RMS | Constraint norm |',
  '|---|---|---|---|---|---|---|---|'];
for (const row of rows) {
  lines.push(`| ${row.mechanism} | ${row.case} | ${row.method} | ${row.passed} | ${row.solverConverged} | ${row.iterations} | ${row.heldOutError.toExponential(3)} | ${row.constraintNorm.toExponential(3)} |`);
}
lines.push('', '## Safety checks', '', '| Case | Pass | Evidence |', '|---|---|---|');
for (const check of checks) lines.push(`| ${check.case} | ${check.passed} | ${check.detail.replace(/\|/g, '/')} |`);
lines.push('', '## Environment', '', '```json', JSON.stringify(environment, null, 2), '```', '');
fs.mkdirSync(OUTPUT_DIRECTORY, { recursive: true });
fs.writeFileSync(path.join(OUTPUT_DIRECTORY, 'mechanical-character-robustness.json'), JSON.stringify({ environment, rows, checks }, null, 2));
fs.writeFileSync(path.join(OUTPUT_DIRECTORY, 'mechanical-character-robustness.md'), lines.join('\n'));
console.log(`Accuracy: ${rows.filter(row => row.passed).length}/${rows.length}; safety: ${checks.filter(check => check.passed).length}/${checks.length}`);
if (rows.some(row => !row.passed) || checks.some(check => !check.passed)) process.exitCode = 1;
