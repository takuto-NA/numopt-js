# Coding Rules

This document defines the coding standards for numopt-js.

## 1. No Magic Numbers

All numeric constants must be named constants with clear meaning.

**Bad:**
```typescript
if (lambda > 1e-6) {
  // ...
}
```

**Good:**
```typescript
const MINIMUM_LAMBDA_THRESHOLD = 1e-6;
if (lambda > MINIMUM_LAMBDA_THRESHOLD) {
  // ...
}
```

## 2. File Header Comments

Every file must start with a short comment stating its responsibility (what it owns). Optional bullets for main entry points or constraints are fine when they add facts a reader cannot get from the signature alone.

Do not pad headers with roadmap language (Phase/MVP), marketing adjectives, or throat-clearing “for first-time readers” sections that only restate the file name.

**Example:**
```typescript
/**
 * Levenberg-Marquardt nonlinear least squares solver.
 * Uses numerical differentiation when a Jacobian is omitted.
 * Entry point: `levenbergMarquardt`.
 */
```

## 3. No Abbreviations

Function and variable names must be self-documenting. Obvious conventions like `i` for loop counters are acceptable.

**Bad:**
```typescript
function calcJ(p: Float64Array): Matrix {
  // ...
}
```

**Good:**
```typescript
function computeJacobianMatrix(parameters: Float64Array): Matrix {
  // ...
}
```

## 4. Comment Philosophy: WHY, not WHAT

Comments should explain **WHY**, not **WHAT**. Code should be self-explanatory.

**Bad:**
```typescript
// Calculate the gradient
const gradient = computeGradient(parameters);
```

**Good:**
```typescript
// Using central difference instead of forward difference for better accuracy
// at the cost of one additional function evaluation per parameter
const gradient = computeGradient(parameters);
```

## 5. Variable Naming Awareness

Distinguish between:
- **Constants** (never change): `UPPER_SNAKE_CASE`
- **Configurable values** (may change): `camelCase` with clear names
- **User-provided values** (user can change): `camelCase` with descriptive names

**Example:**
```typescript
const MAXIMUM_ITERATIONS = 1000; // Constant
const stepSize = options.stepSize ?? DEFAULT_STEP_SIZE; // Configurable
const userProvidedParameters = initialParams; // User-provided
```

## 6. Function and File Size

Prefer small, focused functions. When a function grows past ~50 lines, extract named helpers with clear responsibilities.

Treat files approaching or exceeding ~1000 lines as a decomposition smell by default (extract helpers/modules) unless there is a compelling structural reason to keep them together.

## 7. No Deep Nesting

Use early returns and guard clauses to reduce nesting depth.

**Bad:**
```typescript
function processData(data: Data) {
  if (data !== null) {
    if (data.isValid) {
      if (data.values.length > 0) {
        // process...
      }
    }
  }
}
```

**Good:**
```typescript
function processData(data: Data) {
  if (data === null) return;
  if (!data.isValid) return;
  if (data.values.length === 0) return;
  
  // process...
}
```

## 8. DRY Principle

Strictly avoid code duplication. Extract common logic into reusable functions.

**Bad:**
```typescript
function computeA() {
  const result = expensiveComputation();
  return result * 2;
}

function computeB() {
  const result = expensiveComputation();
  return result * 3;
}
```

**Good:**
```typescript
function expensiveComputation(): number {
  // ...
}

function computeA(): number {
  return expensiveComputation() * 2;
}

function computeB(): number {
  return expensiveComputation() * 3;
}
```

