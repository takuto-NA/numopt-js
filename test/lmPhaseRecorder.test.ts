/** Diagnostic timers must partition nested time and preserve normal returns/errors. */
import { PhaseRecorder } from '../scripts/lm-phase-recorder.mjs';
import { instrumentSource } from '../scripts/instrument-lm-profile.mjs';

it('accounts child time once while retaining inclusive totals', () => {
  let now = 0;
  const recorder = new PhaseRecorder(() => now);
  recorder.enter('lm');
  now = 2;
  recorder.enter('jacobian');
  now = 5;
  recorder.exit('jacobian');
  now = 10;
  recorder.exit('lm');
  expect(recorder.snapshot()).toEqual({
    jacobian: { calls: 1, inclusiveMilliseconds: 3, exclusiveMilliseconds: 3 },
    lm: { calls: 1, inclusiveMilliseconds: 10, exclusiveMilliseconds: 7 }
  });
});

it('keeps phase accounting balanced after an exception', () => {
  const recorder = new PhaseRecorder();
  const failure = new Error('expected failure');
  expect(() => recorder.measure('lm', () => { throw failure; })).toThrow(failure);
  expect(recorder.snapshot().lm.calls).toBe(1);
  recorder.reset();
  expect(recorder.measure('lm', () => 42)).toBe(42);
});

it('refuses incomplete phases and source drift instead of publishing misleading totals', () => {
  const recorder = new PhaseRecorder();
  recorder.enter('lm');
  expect(() => recorder.snapshot()).toThrow(/active phase/);
  expect(() => instrumentSource('export function renamed() {}', {
    file: 'fixture.js', functions: { expected: 'lm' }
  })).toThrow(/anchor/);
});
