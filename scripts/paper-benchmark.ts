/**
 * Paper-grade public-solver comparison.
 * Writes JSON and Markdown under benchmark-results/. Exits 1 if a hypothesis fails.
 */

import { collectPaperEnvironment } from './benchmark-harness';
import { HYPOTHESIS_FAIL_LABEL, HYPOTHESIS_PASS_LABEL } from './paper-benchmark/constants';
import { evaluateHypotheses } from './paper-benchmark/hypotheses';
import { createAllCases } from './paper-benchmark/problems';
import { writeReports } from './paper-benchmark/report';
import { summarizeCase } from './paper-benchmark/summarize';

function main(): void {
  const environment = collectPaperEnvironment();
  const rows = createAllCases().map(summarizeCase);
  const hypotheses = evaluateHypotheses(rows);
  const { jsonPath, markdownPath } = writeReports(environment, rows, hypotheses);
  console.log(`Wrote ${jsonPath}`);
  console.log(`Wrote ${markdownPath}`);
  for (const hypothesis of hypotheses) {
    const label = hypothesis.passed ? HYPOTHESIS_PASS_LABEL : HYPOTHESIS_FAIL_LABEL;
    console.log(`${label}  ${hypothesis.id}: ${hypothesis.detail}`);
  }
  if (hypotheses.some((hypothesis) => !hypothesis.passed)) {
    process.exitCode = 1;
  }
}

main();
