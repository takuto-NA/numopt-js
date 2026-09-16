/**
 * Reduced-space advantage: constraints eliminate the states, so adjoint
 * searches only p. A penalty solve on the concatenated (p, x) searches many
 * more decision variables for the same implicit chain.
 */

import {
  adjointGradientDescent,
  finiteDiffGradient,
  gradientDescent
} from '../src/index';
import type { ConstrainedCostFn, ConstraintFn } from '../src/core/types';
import { vectorNorm } from '../src/utils/matrix';

const STATE_COUNT = 20;
const TARGET_END = 1.0;
const EXPECTED_PARAMETER = TARGET_END / STATE_COUNT;
const PENALTY_WEIGHT = 1e4;
const SOLUTION_TOLERANCE = 5e-3;
const ADJOINT_DECISION_VARIABLE_COUNT = 1;
const PENALTY_DECISION_VARIABLE_COUNT = ADJOINT_DECISION_VARIABLE_COUNT + STATE_COUNT;
const ADJOINT_FEASIBILITY_TOLERANCE = 1e-6;

const chainCost: ConstrainedCostFn = (_parameters, states) => {
  const lastState = states[STATE_COUNT - 1];
  return (lastState - TARGET_END) * (lastState - TARGET_END);
};

const chainConstraint: ConstraintFn = (parameters, states) => {
  const constraint = new Float64Array(STATE_COUNT);
  constraint[0] = states[0] - parameters[0];
  for (let index = 1; index < STATE_COUNT; index++) {
    constraint[index] = states[index] - states[index - 1] - parameters[0];
  }
  return constraint;
};

function splitCombined(combined: Float64Array): { parameters: Float64Array; states: Float64Array } {
  return {
    parameters: combined.slice(0, 1),
    states: combined.slice(1)
  };
}

function penaltyCost(combined: Float64Array): number {
  const { parameters, states } = splitCombined(combined);
  const constraint = chainConstraint(parameters, states);
  const constraintNorm = vectorNorm(constraint);
  return chainCost(parameters, states) + PENALTY_WEIGHT * constraintNorm * constraintNorm;
}

const initialParameter = 0.2;
const initialStates = new Float64Array(STATE_COUNT);
for (let index = 0; index < STATE_COUNT; index++) {
  initialStates[index] = (index + 1) * initialParameter;
}

const initialCombined = new Float64Array(1 + STATE_COUNT);
initialCombined[0] = initialParameter;
initialCombined.set(initialStates, 1);

console.log('=== Adjoint reduced space vs penalty on (p, x) ===\n');
console.log(`Implicit chain: x[0] = p, x[i] = x[i-1] + p,  i = 1..${STATE_COUNT - 1}`);
console.log(`Objective: (x[${STATE_COUNT - 1}] - 1)^2`);
console.log(`Analytical: p = 1/${STATE_COUNT} = ${EXPECTED_PARAMETER}\n`);
console.log(`Decision variables:`);
console.log(`  Adjoint GD:  ${ADJOINT_DECISION_VARIABLE_COUNT}  (p only; x is restored from c(p, x) = 0)`);
console.log(`  Penalty GD:  ${PENALTY_DECISION_VARIABLE_COUNT}  (p and all ${STATE_COUNT} states)\n`);

const adjointStart = performance.now();
const adjointResult = adjointGradientDescent(
  new Float64Array([initialParameter]),
  new Float64Array(initialStates),
  chainCost,
  chainConstraint,
  {
    maxIterations: 200,
    tolerance: 1e-8,
    constraintTolerance: 1e-8,
    useLineSearch: true,
    logLevel: 'WARN'
  }
);
const adjointElapsedMs = performance.now() - adjointStart;

const penaltyStart = performance.now();
const penaltyResult = gradientDescent(
  initialCombined,
  penaltyCost,
  (parameters) => finiteDiffGradient(parameters, penaltyCost),
  {
    maxIterations: 2000,
    tolerance: 1e-8,
    useLineSearch: true
  }
);
const penaltyElapsedMs = performance.now() - penaltyStart;

const penaltySplit = splitCombined(penaltyResult.finalParameters);
const penaltyConstraintNorm = vectorNorm(
  chainConstraint(penaltySplit.parameters, penaltySplit.states)
);
const adjointConstraintNorm =
  adjointResult.finalConstraintNorm ??
  vectorNorm(chainConstraint(adjointResult.finalParameters, adjointResult.finalStates));

console.log('Adjoint GD');
console.log(`  decision vars: ${ADJOINT_DECISION_VARIABLE_COUNT}`);
console.log(`  p             = ${adjointResult.finalParameters[0].toFixed(8)}`);
console.log(`  f             = ${adjointResult.finalCost.toExponential(3)}`);
console.log(`  ||c||         = ${adjointConstraintNorm.toExponential(3)}`);
console.log(`  iterations    = ${adjointResult.iterations}`);
console.log(`  time          = ${adjointElapsedMs.toFixed(2)} ms`);
console.log(`  converged     = ${adjointResult.converged}\n`);

console.log('Penalty GD on (p, x)');
console.log(`  decision vars: ${PENALTY_DECISION_VARIABLE_COUNT}`);
console.log(`  p             = ${penaltySplit.parameters[0].toFixed(8)}`);
console.log(`  f             = ${chainCost(penaltySplit.parameters, penaltySplit.states).toExponential(3)}`);
console.log(`  ||c||         = ${penaltyConstraintNorm.toExponential(3)}`);
console.log(`  iterations    = ${penaltyResult.iterations}`);
console.log(`  time          = ${penaltyElapsedMs.toFixed(2)} ms`);
console.log(`  converged     = ${penaltyResult.converged}\n`);

const adjointHitsTarget = Math.abs(adjointResult.finalParameters[0] - EXPECTED_PARAMETER) < SOLUTION_TOLERANCE;
const adjointFeasible = adjointConstraintNorm < ADJOINT_FEASIBILITY_TOLERANCE;
if (!adjointResult.converged || !adjointHitsTarget || !adjointFeasible) {
  console.error('Adjoint reduced-space solve missed the feasible analytical parameter.');
  process.exit(1);
}

console.log(
  `Adjoint searched ${ADJOINT_DECISION_VARIABLE_COUNT} parameter instead of ${PENALTY_DECISION_VARIABLE_COUNT} and recovered the same implicit-chain optimum.`
);
if (adjointElapsedMs < penaltyElapsedMs) {
  console.log(
    `Wall-clock: adjoint ${adjointElapsedMs.toFixed(1)} ms vs penalty ${penaltyElapsedMs.toFixed(1)} ms.`
  );
} else {
  console.log(
    `Wall-clock is not the point of this toy: the decision-variable count is (${ADJOINT_DECISION_VARIABLE_COUNT} vs ${PENALTY_DECISION_VARIABLE_COUNT}).`
  );
}
