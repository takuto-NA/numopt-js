/**
 * Planar four-bar fixture for the continuous mechanism fitting in Coros et al. (2013), section 4.2.
 * Numerical assembly uses public Gauss-Newton; circle intersections provide an independent oracle.
 */

import { Matrix } from 'ml-matrix';
import { computeEffectiveJacobian, gaussNewton, finiteDiffConstraintPartialP, vectorNorm } from '../../src/index';

export const PARAMETER_COUNT = 6;
export const COORDINATE_COUNT = 2;
export const DEFAULT_SAMPLE_COUNT = 32;
export const DIFFERENCE_STEP = 1e-5;
export const ASSEMBLY_TOLERANCE = 1e-11;
const ASSEMBLY_SOLVER_TOLERANCE = 1e-13;
const FULL_TURN = 2 * Math.PI;
const LENGTH_CENTERS = [0.45, 2.1, 1.8, 2.4];
const LENGTH_RANGES = [0.1, 0.15, 0.15, 0.15];
const MARKER_ALONG_INDEX = 4;
const MARKER_NORMAL_INDEX = 5;
const INITIAL_COUPLER_ANGLE = 1;
const INITIAL_ROCKER_ANGLE = 2;
const MAXIMUM_ASSEMBLY_ITERATIONS = 30;
const LEAST_SQUARES_COST_FACTOR = 0.5;
const CENTRAL_DIFFERENCE_DENOMINATOR = 2;
const CIRCLE_INTERSECTION_DENOMINATOR = 2;
const REDUNDANT_CONSTRAINT_MULTIPLIER = 2;
const INITIAL_PARAMETERS = [0, 0, 0, 0, 2.35, 0.25];
const TARGET_PARAMETERS = [0.3, -0.35, 0.25, 0.2, 2.6, 0.45];

export type WorkCounters = { assemblies: number; constraintEvaluations: number };
export type GradientMode = 'analytic' | 'paper' | 'finite-difference' | 'public-ift' | 'redundant-ift';
export type FourBarProblem = ReturnType<typeof createFourBarProblem>;

export function createWorkCounters(): WorkCounters {
  return { assemblies: 0, constraintEvaluations: 0 };
}

export function decodeLengths(parameters: Float64Array): number[] {
  return LENGTH_CENTERS.map((center, index) => center + LENGTH_RANGES[index] * Math.tanh(parameters[index]));
}

function lengthDerivative(parameters: Float64Array, index: number): number {
  return LENGTH_RANGES[index] * (1 - Math.tanh(parameters[index]) ** 2);
}

export function createPhases(sampleCount: number, offset: number = 0): number[] {
  if (!Number.isInteger(sampleCount) || sampleCount < COORDINATE_COUNT) {
    throw new Error('sampleCount must be an integer >= 2');
  }
  return Array.from({ length: sampleCount }, (_, index) => FULL_TURN * (index + offset) / sampleCount);
}

export function phaseConstraint(parameters: Float64Array, states: Float64Array, phase: number): Float64Array {
  const [crank, coupler, rocker, ground] = decodeLengths(parameters);
  const [couplerAngle, rockerAngle] = states;
  return new Float64Array([
    crank * Math.cos(phase) + coupler * Math.cos(couplerAngle) - ground - rocker * Math.cos(rockerAngle),
    crank * Math.sin(phase) + coupler * Math.sin(couplerAngle) - rocker * Math.sin(rockerAngle)
  ]);
}

export function phaseConstraintStateJacobian(parameters: Float64Array, states: Float64Array): Matrix {
  const [, coupler, rocker] = decodeLengths(parameters);
  const [couplerAngle, rockerAngle] = states;
  return new Matrix([
    [-coupler * Math.sin(couplerAngle), rocker * Math.sin(rockerAngle)],
    [coupler * Math.cos(couplerAngle), -rocker * Math.cos(rockerAngle)]
  ]);
}

export function phaseConstraintParameterJacobian(
  parameters: Float64Array, states: Float64Array, phase: number
): Matrix {
  const [couplerAngle, rockerAngle] = states;
  const derivatives = LENGTH_CENTERS.map((_, index) => lengthDerivative(parameters, index));
  return new Matrix([
    [derivatives[0] * Math.cos(phase), derivatives[1] * Math.cos(couplerAngle),
      -derivatives[2] * Math.cos(rockerAngle), -derivatives[3], 0, 0],
    [derivatives[0] * Math.sin(phase), derivatives[1] * Math.sin(couplerAngle),
      -derivatives[2] * Math.sin(rockerAngle), 0, 0, 0]
  ]);
}

export function markerPosition(parameters: Float64Array, states: Float64Array, phase: number): Float64Array {
  const [crank] = decodeLengths(parameters);
  const couplerAngle = states[0];
  const along = parameters[MARKER_ALONG_INDEX];
  const normal = parameters[MARKER_NORMAL_INDEX];
  return new Float64Array([
    crank * Math.cos(phase) + along * Math.cos(couplerAngle) - normal * Math.sin(couplerAngle),
    crank * Math.sin(phase) + along * Math.sin(couplerAngle) + normal * Math.cos(couplerAngle)
  ]);
}

/** Fixed states matter here: changing a marker's local coordinates also changes its path. */
export function markerParameterJacobian(parameters: Float64Array, states: Float64Array, phase: number): Matrix {
  const cosine = Math.cos(states[0]);
  const sine = Math.sin(states[0]);
  const crankDerivative = lengthDerivative(parameters, 0);
  return new Matrix([
    [crankDerivative * Math.cos(phase), 0, 0, 0, cosine, -sine],
    [crankDerivative * Math.sin(phase), 0, 0, 0, sine, cosine]
  ]);
}

export function markerStateJacobian(parameters: Float64Array, states: Float64Array): Matrix {
  const cosine = Math.cos(states[0]);
  const sine = Math.sin(states[0]);
  const along = parameters[MARKER_ALONG_INDEX];
  const normal = parameters[MARKER_NORMAL_INDEX];
  return new Matrix([[-along * sine - normal * cosine, 0], [along * cosine - normal * sine, 0]]);
}

export function addRedundantConstraints(values: Float64Array): Float64Array {
  return new Float64Array([...values, ...values.map(value => REDUNDANT_CONSTRAINT_MULTIPLIER * value)]);
}

function addRedundantRows(matrix: Matrix): Matrix {
  return new Matrix([...matrix.to2DArray(), ...matrix.clone().mul(REDUNDANT_CONSTRAINT_MULTIPLIER).to2DArray()]);
}

export function assembleStates(
  parameters: Float64Array, phases: number[], counters = createWorkCounters(), redundant = false
): Float64Array {
  const states = new Float64Array(COORDINATE_COUNT * phases.length);
  let previous: Float64Array = new Float64Array([INITIAL_COUPLER_ANGLE, INITIAL_ROCKER_ANGLE]);
  phases.forEach((phase, index) => {
    counters.assemblies++;
    const constraint = (angles: Float64Array): Float64Array => {
      counters.constraintEvaluations++;
      const values = phaseConstraint(parameters, angles, phase);
      return redundant ? addRedundantConstraints(values) : values;
    };
    const result = gaussNewton(previous, constraint, {
      jacobian: angles => {
        const matrix = phaseConstraintStateJacobian(parameters, angles);
        return redundant ? addRedundantRows(matrix) : matrix;
      },
      maxIterations: MAXIMUM_ASSEMBLY_ITERATIONS, tolerance: ASSEMBLY_SOLVER_TOLERANCE
    });
    previous = result.finalParameters;
    const constraintNorm = vectorNorm(constraint(previous));
    if (!Number.isFinite(constraintNorm) || constraintNorm > ASSEMBLY_TOLERANCE) {
      throw new Error(`Assembly failed at phase ${phase}`);
    }
    states.set(previous, COORDINATE_COUNT * index);
  });
  return states;
}

/** Circle geometry is used only for targets and verification, never inside the optimizer. */
export function oracleStates(parameters: Float64Array, phases: number[]): Float64Array {
  const [crank, coupler, rocker, ground] = decodeLengths(parameters);
  const states = new Float64Array(COORDINATE_COUNT * phases.length);
  phases.forEach((phase, index) => {
    const crankX = crank * Math.cos(phase);
    const crankY = crank * Math.sin(phase);
    const deltaX = ground - crankX;
    const deltaY = -crankY;
    const distance = Math.hypot(deltaX, deltaY);
    const along = (coupler ** 2 - rocker ** 2 + distance ** 2) / (CIRCLE_INTERSECTION_DENOMINATOR * distance);
    const height = Math.sqrt(coupler ** 2 - along ** 2);
    const jointX = crankX + (along * deltaX - height * deltaY) / distance;
    const jointY = crankY + (along * deltaY + height * deltaX) / distance;
    states.set([Math.atan2(jointY - crankY, jointX - crankX), Math.atan2(jointY, jointX - ground)],
      COORDINATE_COUNT * index);
  });
  return states;
}

export function traceCurve(parameters: Float64Array, states: Float64Array, phases: number[]): number[][] {
  return phases.map((phase, index) => Array.from(markerPosition(parameters,
    states.subarray(COORDINATE_COUNT * index, COORDINATE_COUNT * (index + 1)), phase)));
}

export function createFourBarProblem(sampleCount: number = DEFAULT_SAMPLE_COUNT, distorted: boolean = false) {
  const phases = createPhases(sampleCount);
  const initialParameters = new Float64Array(INITIAL_PARAMETERS);
  const targetParameters = new Float64Array(TARGET_PARAMETERS);
  const targetAt = (evaluationPhases: number[]): number[][] => {
    const target = traceCurve(targetParameters, oracleStates(targetParameters, evaluationPhases), evaluationPhases);
    if (distorted) evaluationPhases.forEach((phase, index) => {
      const DISTORTION_AMPLITUDE = 0.02;
      const DISTORTION_FREQUENCY = 3;
      target[index][0] += DISTORTION_AMPLITUDE * Math.cos(DISTORTION_FREQUENCY * phase);
      target[index][1] += DISTORTION_AMPLITUDE * Math.sin(DISTORTION_FREQUENCY * phase);
    });
    return target;
  };
  const target = targetAt(phases);
  // Uniform quadrature of the integral in Eq. (2), independent of sample count.
  const residualScale = 1 / Math.sqrt(sampleCount);
  const constraint = (parameters: Float64Array, states: Float64Array): Float64Array => {
    const values = new Float64Array(states.length);
    phases.forEach((phase, index) => values.set(phaseConstraint(parameters,
      states.subarray(COORDINATE_COUNT * index, COORDINATE_COUNT * (index + 1)), phase), COORDINATE_COUNT * index));
    return values;
  };
  const residual = (parameters: Float64Array, states: Float64Array): Float64Array => {
    const curve = traceCurve(parameters, states, phases);
    return Float64Array.from(curve.flatMap((point, index) => point.map((value, coordinate) =>
      residualScale * (value - target[index][coordinate]))));
  };
  const buildJacobian = (parameters: Float64Array, states: Float64Array, kind: 'cp' | 'cx' | 'rp' | 'rx'): Matrix => {
    const isParameter = kind === 'cp' || kind === 'rp';
    const matrix = Matrix.zeros(states.length, isParameter ? PARAMETER_COUNT : states.length);
    phases.forEach((phase, index) => {
      const offset = COORDINATE_COUNT * index;
      const angles = states.subarray(offset, offset + COORDINATE_COUNT);
      const block = kind === 'cp' ? phaseConstraintParameterJacobian(parameters, angles, phase)
        : kind === 'cx' ? phaseConstraintStateJacobian(parameters, angles)
        : kind === 'rp' ? markerParameterJacobian(parameters, angles, phase).mul(residualScale)
        : markerStateJacobian(parameters, angles).mul(residualScale);
      matrix.setSubMatrix(block, offset, isParameter ? 0 : offset);
    });
    return matrix;
  };
  const derivatives = {
    dcdp: (parameters: Float64Array, states: Float64Array) => buildJacobian(parameters, states, 'cp'),
    dcdx: (parameters: Float64Array, states: Float64Array) => buildJacobian(parameters, states, 'cx'),
    drdp: (parameters: Float64Array, states: Float64Array) => buildJacobian(parameters, states, 'rp'),
    drdx: (parameters: Float64Array, states: Float64Array) => buildJacobian(parameters, states, 'rx')
  };
  const cost = (parameters: Float64Array, states: Float64Array): number =>
    LEAST_SQUARES_COST_FACTOR * residual(parameters, states).reduce((sum, value) => sum + value * value, 0);
  const costPartials = {
    dfdp: (parameters: Float64Array, states: Float64Array) => Float64Array.from(derivatives.drdp(parameters, states)
      .transpose().mmul(Matrix.columnVector(residual(parameters, states))).to1DArray()),
    dfdx: (parameters: Float64Array, states: Float64Array) => Float64Array.from(derivatives.drdx(parameters, states)
      .transpose().mmul(Matrix.columnVector(residual(parameters, states))).to1DArray())
  };
  return { phases, initialParameters, targetParameters, target, targetAt, constraint, residual, cost, derivatives, costPartials };
}

/** Eq. (3-4), evaluated in small phase blocks instead of one dense cycle matrix. */
export function implicitGradient(
  problem: FourBarProblem, parameters: Float64Array, states: Float64Array,
  mode: 'analytic' | 'paper', counters = createWorkCounters()
): Float64Array {
  const gradient = new Float64Array(PARAMETER_COUNT);
  problem.phases.forEach((phase, index) => {
    const angles = states.subarray(COORDINATE_COUNT * index, COORDINATE_COUNT * (index + 1));
    const constraintPartial = mode === 'analytic' ? phaseConstraintParameterJacobian(parameters, angles, phase)
      : finiteDiffConstraintPartialP(parameters, angles, (trial, fixedStates) => {
        counters.constraintEvaluations++;
        return phaseConstraint(trial, fixedStates, phase);
      }, { stepSize: DIFFERENCE_STEP });
    const statePartial = phaseConstraintStateJacobian(parameters, angles);
    const markerStatePartial = markerStateJacobian(parameters, angles);
    const markerParameterPartial = markerParameterJacobian(parameters, angles, phase);
    // Two-by-two solve keeps the reference independent of the library's reduced-gradient helper.
    const divisor = statePartial.get(0, 0) * statePartial.get(1, 1)
      - statePartial.get(0, 1) * statePartial.get(1, 0);
    const marker = markerPosition(parameters, angles, phase);
    for (let column = 0; column < PARAMETER_COUNT; column++) {
      const sensitivity = [
        (-statePartial.get(1, 1) * constraintPartial.get(0, column)
          + statePartial.get(0, 1) * constraintPartial.get(1, column)) / divisor,
        (statePartial.get(1, 0) * constraintPartial.get(0, column)
          - statePartial.get(0, 0) * constraintPartial.get(1, column)) / divisor
      ];
      for (let coordinate = 0; coordinate < COORDINATE_COUNT; coordinate++) {
        const totalDerivative = markerParameterPartial.get(coordinate, column)
          + markerStatePartial.get(coordinate, 0) * sensitivity[0]
          + markerStatePartial.get(coordinate, 1) * sensitivity[1];
        gradient[column] += totalDerivative * (marker[coordinate] - problem.target[index][coordinate]) / problem.phases.length;
      }
    }
  });
  return gradient;
}

/** Public J_eff connects the paper's forward sensitivities to ordinary BFGS, including redundant constraints. */
export function publicImplicitGradient(
  problem: FourBarProblem, parameters: Float64Array, states: Float64Array,
  redundant: boolean, counters = createWorkCounters()
): Float64Array {
  const gradient = new Float64Array(PARAMETER_COUNT);
  const residualScale = 1 / Math.sqrt(problem.phases.length);
  problem.phases.forEach((phase, index) => {
    const angles = states.subarray(COORDINATE_COUNT * index, COORDINATE_COUNT * (index + 1));
    const residual = (trial: Float64Array, fixedStates: Float64Array): Float64Array =>
      markerPosition(trial, fixedStates, phase).map((value, coordinate) =>
        residualScale * (value - problem.target[index][coordinate]));
    const constraint = (trial: Float64Array, fixedStates: Float64Array): Float64Array => {
      counters.constraintEvaluations++;
      const values = phaseConstraint(trial, fixedStates, phase);
      return redundant ? addRedundantConstraints(values) : values;
    };
    const jacobian = computeEffectiveJacobian(parameters, angles, residual, constraint, {
      drdp: (trial, fixedStates) => markerParameterJacobian(trial, fixedStates, phase).mul(residualScale),
      drdx: (trial, fixedStates) => markerStateJacobian(trial, fixedStates).mul(residualScale),
      dcdx: (trial, fixedStates) => {
        const matrix = phaseConstraintStateJacobian(trial, fixedStates);
        return redundant ? addRedundantRows(matrix) : matrix;
      },
      stepSizeP: DIFFERENCE_STEP
    });
    const partial = jacobian.transpose().mmul(Matrix.columnVector(residual(parameters, angles))).to1DArray();
    partial.forEach((value, column) => { gradient[column] += value; });
  });
  return gradient;
}

/** One-entry cache shares an assembly between cost and gradient at exactly the same p. */
export function createReducedEvaluator(problem: FourBarProblem, mode: GradientMode, counters = createWorkCounters()) {
  let cachedParameters: Float64Array | undefined;
  let cachedStates: Float64Array;
  const statesAt = (parameters: Float64Array): Float64Array => {
    if (!cachedParameters || !parameters.every((value, index) => value === cachedParameters![index])) {
      cachedStates = assembleStates(parameters, problem.phases, counters, mode === 'redundant-ift');
      cachedParameters = parameters.slice();
    }
    return cachedStates;
  };
  const cost = (parameters: Float64Array): number => problem.cost(parameters, statesAt(parameters));
  const gradient = (parameters: Float64Array): Float64Array => {
    if (mode === 'public-ift' || mode === 'redundant-ift') {
      return publicImplicitGradient(problem, parameters, statesAt(parameters), mode === 'redundant-ift', counters);
    }
    if (mode !== 'finite-difference') return implicitGradient(problem, parameters, statesAt(parameters), mode, counters);
    return Float64Array.from(parameters, (_, index) => {
      const plus = parameters.slice();
      const minus = parameters.slice();
      plus[index] += DIFFERENCE_STEP;
      minus[index] -= DIFFERENCE_STEP;
      return (cost(plus) - cost(minus)) / (CENTRAL_DIFFERENCE_DENOMINATOR * DIFFERENCE_STEP);
    });
  };
  return { cost, gradient, statesAt, counters };
}
