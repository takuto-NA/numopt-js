/** Account nested diagnostic phases without double-counting child durations. */
export class PhaseRecorder {
  constructor(clock = () => performance.now()) {
    this.clock = clock;
    this.reset();
  }

  reset() {
    if (this.stack?.length) throw new Error('Cannot reset an active phase');
    this.stack = [];
    this.phases = new Map();
  }

  contains(name) { return this.stack.some(frame => frame.name === name); }

  enter(name) { this.stack.push({ name, started: this.clock(), children: 0 }); }

  exit(name) {
    const stopped = this.clock();
    const frame = this.stack.pop();
    if (!frame || frame.name !== name) throw new Error(`Unbalanced phase: ${name}`);
    const inclusiveMilliseconds = stopped - frame.started;
    const phase = this.phases.get(name) ?? { calls: 0, inclusiveMilliseconds: 0, exclusiveMilliseconds: 0 };
    phase.calls++;
    phase.inclusiveMilliseconds += inclusiveMilliseconds;
    phase.exclusiveMilliseconds += inclusiveMilliseconds - frame.children;
    this.phases.set(name, phase);
    if (this.stack.length) this.stack[this.stack.length - 1].children += inclusiveMilliseconds;
  }

  measure(name, operation) {
    this.enter(name);
    try { return operation(); }
    finally { this.exit(name); }
  }

  snapshot() {
    if (this.stack.length) throw new Error('Cannot snapshot an active phase');
    return Object.fromEntries([...this.phases].map(([name, phase]) => [name, { ...phase }]));
  }
}
