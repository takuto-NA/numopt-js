/** Synthetic geared five-bar based on Fig. 4: two prescribed cranks, a closed loop, and a coupler marker. */

import { Matrix } from 'ml-matrix';

export const GEAR_RATIOS = [-2, -1, 1, 2] as const;
const CENTERS = [0.45, 2.1, 1.8, 2.4, 0.35, 0.35, 2.6, 0.45];
const RANGES = [0.08, 0.1, 0.1, 0.1, 0.06, 0.25, 0.3, 0.2];
export const FIVE_BAR_TARGET = [0.3, -0.35, 0.25, 0.2, -0.2, -0.3, 0.4, 0.3];
const HALF = 0.5;
const COORDINATE_COUNT = 2;

export function decodeFiveBar(parameters: Float64Array): number[] {
  return CENTERS.map((center, index) => center + RANGES[index] * Math.tanh(parameters[index]));
}

function derivatives(parameters: Float64Array): number[] {
  return RANGES.map((range, index) => range * (1 - Math.tanh(parameters[index]) ** 2));
}

export function fiveBarConstraint(parameters: Float64Array, states: Float64Array, phase: number, ratio: number) {
  const [crank, coupler, follower, ground, secondCrank, phaseOffset] = decodeFiveBar(parameters);
  const secondPhase = ratio * phase + phaseOffset;
  return new Float64Array([
    crank * Math.cos(phase) + coupler * Math.cos(states[0]) - ground - secondCrank * Math.cos(secondPhase) - follower * Math.cos(states[1]),
    crank * Math.sin(phase) + coupler * Math.sin(states[0]) - secondCrank * Math.sin(secondPhase) - follower * Math.sin(states[1])
  ]);
}

export function fiveBarStateJacobian(parameters: Float64Array, states: Float64Array): Matrix {
  const [, coupler, follower] = decodeFiveBar(parameters);
  return new Matrix([[-coupler * Math.sin(states[0]), follower * Math.sin(states[1])],
    [coupler * Math.cos(states[0]), -follower * Math.cos(states[1])]]);
}

export function fiveBarParameterJacobian(parameters: Float64Array, states: Float64Array, phase: number, ratio: number): Matrix {
  const [, , , , secondCrank, phaseOffset] = decodeFiveBar(parameters);
  const [crankDerivative, couplerDerivative, followerDerivative, groundDerivative, secondCrankDerivative, phaseDerivative] = derivatives(parameters);
  const secondPhase = ratio * phase + phaseOffset;
  return new Matrix([
    [crankDerivative * Math.cos(phase), couplerDerivative * Math.cos(states[0]), -followerDerivative * Math.cos(states[1]),
      -groundDerivative, -secondCrankDerivative * Math.cos(secondPhase), secondCrank * Math.sin(secondPhase) * phaseDerivative, 0, 0],
    [crankDerivative * Math.sin(phase), couplerDerivative * Math.sin(states[0]), -followerDerivative * Math.sin(states[1]),
      0, -secondCrankDerivative * Math.sin(secondPhase), -secondCrank * Math.cos(secondPhase) * phaseDerivative, 0, 0]
  ]);
}

export function fiveBarMarker(parameters: Float64Array, states: Float64Array, phase: number): Float64Array {
  const [crank, , , , , , along, normal] = decodeFiveBar(parameters);
  const angle = states[0];
  return new Float64Array([crank * Math.cos(phase) + along * Math.cos(angle) - normal * Math.sin(angle),
    crank * Math.sin(phase) + along * Math.sin(angle) + normal * Math.cos(angle)]);
}

export function fiveBarMarkerParameterJacobian(parameters: Float64Array, states: Float64Array, phase: number): Matrix {
  const [crankDerivative, , , , , , alongDerivative, normalDerivative] = derivatives(parameters);
  const angle = states[0];
  return new Matrix([
    [crankDerivative * Math.cos(phase), 0, 0, 0, 0, 0, alongDerivative * Math.cos(angle), -normalDerivative * Math.sin(angle)],
    [crankDerivative * Math.sin(phase), 0, 0, 0, 0, 0, alongDerivative * Math.sin(angle), normalDerivative * Math.cos(angle)]
  ]);
}

export function fiveBarMarkerStateJacobian(parameters: Float64Array, states: Float64Array): Matrix {
  const [, , , , , , along, normal] = decodeFiveBar(parameters);
  const angle = states[0];
  return new Matrix([[-along * Math.sin(angle) - normal * Math.cos(angle), 0],
    [along * Math.cos(angle) - normal * Math.sin(angle), 0]]);
}

/** Independent circle intersection, used only for generating targets and checking numerical assembly. */
export function fiveBarOracle(parameters: Float64Array, phases: number[], ratio: number): Float64Array {
  const [crank, coupler, follower, ground, secondCrank, phaseOffset] = decodeFiveBar(parameters);
  const states = new Float64Array(COORDINATE_COUNT * phases.length);
  phases.forEach((phase, index) => {
    const first = [crank * Math.cos(phase), crank * Math.sin(phase)];
    const secondPhase = ratio * phase + phaseOffset;
    const second = [ground + secondCrank * Math.cos(secondPhase), secondCrank * Math.sin(secondPhase)];
    const difference = second.map((value, coordinate) => value - first[coordinate]);
    const distance = Math.hypot(...difference);
    const along = HALF * (coupler ** 2 - follower ** 2 + distance ** 2) / distance;
    const heightSquared = coupler ** 2 - along ** 2;
    if (!(distance > 0 && heightSquared > 0)) throw new Error('Five-bar oracle reached a singular or infeasible triangle');
    const height = Math.sqrt(heightSquared);
    const point = [first[0] + (along * difference[0] - height * difference[1]) / distance,
      first[1] + (along * difference[1] + height * difference[0]) / distance];
    states.set([Math.atan2(point[1] - first[1], point[0] - first[0]),
      Math.atan2(point[1] - second[1], point[0] - second[0])], COORDINATE_COUNT * index);
  });
  return states;
}
