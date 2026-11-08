/** Paper-equation regressions: geared loops, chain-rule terms, quadrature, redundant constraints and multistart fits. */

import { describe, expect, it } from 'vitest';
import { finiteDiffConstraintPartialP, finiteDiffConstraintPartialX, finiteDiffResidualPartialP,
  finiteDiffResidualPartialX, vectorNorm } from '../src/index';
import { createPaperCases, createPaperFiveBar } from '../examples/paper-mechanism/cases';
import { GEAR_RATIOS } from '../examples/paper-mechanism/geared-five-bar';
import { assemblePaperStates, createPaperEvaluator, paperCurve } from '../examples/paper-mechanism/evaluate';
import { checkPaperEquations, checkQuadrature, checkStateAccuracy, initialPaperDesigns, runPaperFit } from '../examples/paper-mechanism/checks';
import { createPhases } from '../examples/mechanical-character/four-bar';
import { checkBoundaryStationarity } from '../examples/paper-mechanism/boundary-check';

const PARTIAL_TOLERANCE = 1e-8;
const BRANCH_TOLERANCE = 1e-9;
const TEST_PHASE = 0.7;
const NUMERICAL_TEST_TIMEOUT = 15000;
const TEST_PHASE_COUNT = 32;
const SATURATED_PARAMETER = 20;

describe('paper-guided mechanical character checks', () => {
  it.each(GEAR_RATIOS)('matches all five-bar partials and numerical assemblies to independent geometry at ratio %s', ratio => {
    const model = createPaperFiveBar(ratio);
    const parameters = model.initialParameters;
    const angles = model.oracle(parameters, [TEST_PHASE]);
    const constraint = (trial: Float64Array, fixed: Float64Array) => model.constraint(trial, fixed, TEST_PHASE);
    const marker = (trial: Float64Array, fixed: Float64Array) => model.marker(trial, fixed, TEST_PHASE);
    for (const [analytical, numerical] of [
      [model.dcdp(parameters, angles, TEST_PHASE), finiteDiffConstraintPartialP(parameters, angles, constraint)],
      [model.dcdx(parameters, angles), finiteDiffConstraintPartialX(parameters, angles, constraint)],
      [model.dmdp(parameters, angles, TEST_PHASE), finiteDiffResidualPartialP(parameters, angles, marker)],
      [model.dmdx(parameters, angles), finiteDiffResidualPartialX(parameters, angles, marker)]
    ]) expect(Math.max(...analytical.sub(numerical).to1DArray().map(Math.abs))).toBeLessThan(PARTIAL_TOLERANCE);
    const phases = createPhases(TEST_PHASE_COUNT);
    const numerical = assemblePaperStates(model, parameters, phases);
    const exact = model.oracle(parameters, phases);
    expect(vectorNorm(numerical.map((value, index) => value - exact[index]))).toBeLessThan(BRANCH_TOLERANCE);
    expect(paperCurve(model, parameters, numerical, phases).length).toBe(TEST_PHASE_COUNT);
  });

  it.each(createPaperCases())('$name satisfies Eqs. (3)-(4), rejects omitted terms and preserves redundant sensitivities', model => {
    const check = checkPaperEquations(model);
    expect(check.passed, JSON.stringify(check)).toBe(true);
    const quadrature = checkQuadrature(model);
    expect(quadrature[quadrature.length - 1].gradientError).toBeLessThan(PARTIAL_TOLERANCE);
    const accuracy = checkStateAccuracy(model);
    expect(accuracy[accuracy.length - 1].gradientError).toBeLessThan(PARTIAL_TOLERANCE);
    expect(accuracy[0].gradientError).toBeGreaterThan(accuracy[accuracy.length - 1].gradientError);
    for (const method of ['paper-ift', 'full-fd'] as const) {
      const evaluator = createPaperEvaluator(model, method);
      evaluator.gradient(model.initialParameters);
      expect(evaluator.counters.assemblies).toBe(model.phases.length * (method === 'paper-ift' ? 1 : 2 * model.initialParameters.length));
      const cached = createPaperEvaluator(model, method);
      cached.cost(model.initialParameters);
      const before = cached.counters.assemblies;
      cached.gradient(model.initialParameters);
      expect(cached.counters.assemblies - before).toBe(method === 'paper-ift' ? 0 : 2 * model.initialParameters.length * model.phases.length);
    }
  });

  it.each(createPaperCases())('$name obtains a verified fit by choosing the lowest objective among three initial designs', model => {
    for (const distorted of [false, true]) {
      const selected = distorted ? createPaperCases(TEST_PHASE_COUNT, true).find(candidate => candidate.name === model.name)! : model;
      const candidates = initialPaperDesigns(selected).map(initial => runPaperFit(selected, 'paper-ift', initial, distorted));
      const best = candidates.reduce((left, right) => left.verification.finalCost < right.verification.finalCost ? left : right);
      expect(best.verification.passed, `${model.name}: ${distorted}: ${JSON.stringify(best.verification)}`).toBe(true);
      if (!distorted) expect(candidates.every(row => row.verification.passed)).toBe(true);
    }
  }, NUMERICAL_TEST_TIMEOUT);

  it('rejects a vanished tanh gradient when the original bounded coordinate still has a feasible descent direction', () => {
    const model = { ...createPaperFiveBar(1), boundedParameterIndices: [0] };
    const parameters = new Float64Array([SATURATED_PARAMETER]);
    const falseStationary = checkBoundaryStationarity(model, parameters, trial => (Math.tanh(trial[0]) - 0.5) ** 2);
    const validBoundaryMinimum = checkBoundaryStationarity(model, parameters, trial => (Math.tanh(trial[0]) - 2) ** 2);
    expect(falseStationary.passed).toBe(false);
    expect(falseStationary.projectedGradientNorm).toBeGreaterThan(0.5);
    expect(validBoundaryMinimum.passed).toBe(true);
  });
});
