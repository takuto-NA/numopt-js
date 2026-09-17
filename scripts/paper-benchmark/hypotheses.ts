/**
 * Algorithm-class hypotheses. Wall-clock is never a pass/fail criterion.
 */

import {
  CHAIN_ADJOINT_DECISION_VARIABLE_COUNT,
  CHAIN_PENALTY_DECISION_VARIABLE_COUNT,
  METHOD_ADJOINT_GD,
  METHOD_BFGS,
  METHOD_CMA_ES,
  METHOD_CONSTRAINED_GN,
  METHOD_CONSTRAINED_LM,
  METHOD_GAUSS_NEWTON,
  METHOD_GRADIENT_DESCENT,
  METHOD_LBFGS,
  METHOD_LEVENBERG_MARQUARDT,
  METHOD_PENALTY_GN,
  METHOD_PENALTY_LM,
  PROBLEM_CHAIN,
  PROBLEM_CIRCLE,
  PROBLEM_EXPONENTIAL_FIT,
  PROBLEM_LINEAR_LEAST_SQUARES,
  PROBLEM_ROSENBROCK,
  PROBLEM_SPHERE
} from './constants';
import type { HypothesisResult, PaperRow } from './types';

function requireRow(rows: PaperRow[], problemName: string, methodName: string): PaperRow {
  const row = rows.find((candidate) => candidate.problemName === problemName && candidate.methodName === methodName);
  if (!row) {
    throw new Error(`Missing paper-benchmark row for ${problemName} / ${methodName}`);
  }
  return row;
}

export function evaluateHypotheses(rows: PaperRow[]): HypothesisResult[] {
  const sphereGd = requireRow(rows, PROBLEM_SPHERE, METHOD_GRADIENT_DESCENT);
  const sphereBfgs = requireRow(rows, PROBLEM_SPHERE, METHOD_BFGS);
  const sphereLbfgs = requireRow(rows, PROBLEM_SPHERE, METHOD_LBFGS);
  const sphereCma = requireRow(rows, PROBLEM_SPHERE, METHOD_CMA_ES);
  const rosenGd = requireRow(rows, PROBLEM_ROSENBROCK, METHOD_GRADIENT_DESCENT);
  const rosenBfgs = requireRow(rows, PROBLEM_ROSENBROCK, METHOD_BFGS);
  const rosenLbfgs = requireRow(rows, PROBLEM_ROSENBROCK, METHOD_LBFGS);
  const rosenCma = requireRow(rows, PROBLEM_ROSENBROCK, METHOD_CMA_ES);
  const linearGn = requireRow(rows, PROBLEM_LINEAR_LEAST_SQUARES, METHOD_GAUSS_NEWTON);
  const linearLm = requireRow(rows, PROBLEM_LINEAR_LEAST_SQUARES, METHOD_LEVENBERG_MARQUARDT);
  const expGn = requireRow(rows, PROBLEM_EXPONENTIAL_FIT, METHOD_GAUSS_NEWTON);
  const expLm = requireRow(rows, PROBLEM_EXPONENTIAL_FIT, METHOD_LEVENBERG_MARQUARDT);
  const circleAdjoint = requireRow(rows, PROBLEM_CIRCLE, METHOD_ADJOINT_GD);
  const circleGn = requireRow(rows, PROBLEM_CIRCLE, METHOD_CONSTRAINED_GN);
  const circleLm = requireRow(rows, PROBLEM_CIRCLE, METHOD_CONSTRAINED_LM);
  const chainAdjoint = requireRow(rows, PROBLEM_CHAIN, METHOD_ADJOINT_GD);
  const chainPenaltyGn = requireRow(rows, PROBLEM_CHAIN, METHOD_PENALTY_GN);
  const chainPenaltyLm = requireRow(rows, PROBLEM_CHAIN, METHOD_PENALTY_LM);

  return [
    {
      id: 'sphere-quasi-newton-fewer-iterations',
      description: 'Sphere: BFGS and L-BFGS use fewer iterations than GD; all four methods succeed',
      passed:
        sphereGd.allTrialsSucceeded &&
        sphereBfgs.allTrialsSucceeded &&
        sphereLbfgs.allTrialsSucceeded &&
        sphereCma.allTrialsSucceeded &&
        sphereBfgs.iterations.median < sphereGd.iterations.median &&
        sphereLbfgs.iterations.median < sphereGd.iterations.median,
      detail:
        `GD ${sphereGd.iterations.median} / BFGS ${sphereBfgs.iterations.median} / ` +
        `L-BFGS ${sphereLbfgs.iterations.median} / CMA-ES success ${sphereCma.successRate}`
    },
    {
      id: 'rosenbrock-quasi-newton-fewer-iterations',
      description:
        'Rosenbrock x0=(-1,1): BFGS and L-BFGS use fewer iterations than GD; CMA-ES succeeds at 1e-2',
      passed:
        rosenGd.allTrialsSucceeded &&
        rosenBfgs.allTrialsSucceeded &&
        rosenLbfgs.allTrialsSucceeded &&
        rosenCma.allTrialsSucceeded &&
        rosenBfgs.iterations.median < rosenGd.iterations.median &&
        rosenLbfgs.iterations.median < rosenGd.iterations.median,
      detail:
        `GD ${rosenGd.iterations.median} / BFGS ${rosenBfgs.iterations.median} / ` +
        `L-BFGS ${rosenLbfgs.iterations.median} / CMA-ES success ${rosenCma.successRate}`
    },
    {
      id: 'linear-least-squares-both-succeed',
      description: 'Linear least squares: Gauss-Newton and Levenberg-Marquardt both succeed',
      passed: linearGn.allTrialsSucceeded && linearLm.allTrialsSucceeded,
      detail: `GN success ${linearGn.successRate}, LM success ${linearLm.successRate}`
    },
    {
      id: 'exponential-fit-both-succeed',
      description: 'Exponential fit: Gauss-Newton and Levenberg-Marquardt both recover (a,b)~(1,1)',
      passed: expGn.allTrialsSucceeded && expLm.allTrialsSucceeded,
      detail: `GN err ${expGn.parameterError.median}, LM err ${expLm.parameterError.median}`
    },
    {
      id: 'circle-all-succeed-adjoint-more-iterations',
      description: 'Circle simple residual: all three succeed; Adjoint uses more iterations',
      passed:
        circleAdjoint.allTrialsSucceeded &&
        circleGn.allTrialsSucceeded &&
        circleLm.allTrialsSucceeded &&
        circleAdjoint.iterations.median > circleGn.iterations.median &&
        circleAdjoint.iterations.median > circleLm.iterations.median,
      detail:
        `Adjoint ${circleAdjoint.iterations.median} / C-GN ${circleGn.iterations.median} / ` +
        `C-LM ${circleLm.iterations.median}`
    },
    {
      id: 'chain-decision-variables-and-success',
      description:
        'Reduced-space chain: Adjoint has 1 decision variable, Penalty has 21; all three succeed',
      passed:
        chainAdjoint.allTrialsSucceeded &&
        chainPenaltyGn.allTrialsSucceeded &&
        chainPenaltyLm.allTrialsSucceeded &&
        chainAdjoint.decisionVariableCount === CHAIN_ADJOINT_DECISION_VARIABLE_COUNT &&
        chainPenaltyGn.decisionVariableCount === CHAIN_PENALTY_DECISION_VARIABLE_COUNT &&
        chainPenaltyLm.decisionVariableCount === CHAIN_PENALTY_DECISION_VARIABLE_COUNT,
      detail:
        `Adjoint vars ${chainAdjoint.decisionVariableCount}, Penalty GN/LM vars ` +
        `${chainPenaltyGn.decisionVariableCount}/${chainPenaltyLm.decisionVariableCount}`
    }
  ];
}
