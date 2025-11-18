/**
 * This file provides utility functions for converting between Float64Array
 * and ml-matrix Matrix types, and for vector operations.
 * 
 * Role in system:
 * - Bridges the gap between native JavaScript arrays and ml-matrix library
 * - Provides efficient conversion utilities used throughout the codebase
 * - Implements common vector operations needed by optimization algorithms
 * 
 * For first-time readers:
 * - These are helper functions used by core algorithms
 * - Focus on understanding the conversion functions first
 * - Vector norm computation is used for convergence checks
 */

import { Matrix } from 'ml-matrix';

/**
 * Converts a Float64Array to an ml-matrix Matrix (column vector).
 * This is efficient for single-column matrices used in optimization.
 */
export function float64ArrayToMatrix(vector: Float64Array): Matrix {
  const rows = vector.length;
  const matrixData: number[][] = [];
  
  for (let i = 0; i < rows; i++) {
    matrixData.push([vector[i]]);
  }
  
  return new Matrix(matrixData);
}

/**
 * Converts an ml-matrix Matrix to a Float64Array.
 * Assumes the matrix is a column vector (single column).
 */
export function matrixToFloat64Array(matrix: Matrix): Float64Array {
  const rows = matrix.rows;
  const result = new Float64Array(rows);
  
  for (let i = 0; i < rows; i++) {
    result[i] = matrix.get(i, 0);
  }
  
  return result;
}

/**
 * Converts a 2D ml-matrix Matrix to a Float64Array (row-major order).
 * Used for converting Jacobian matrices to flat arrays when needed.
 */
export function matrixToFloat64Array2D(matrix: Matrix): Float64Array {
  const rows = matrix.rows;
  const cols = matrix.columns;
  const result = new Float64Array(rows * cols);
  
  for (let i = 0; i < rows; i++) {
    for (let j = 0; j < cols; j++) {
      result[i * cols + j] = matrix.get(i, j);
    }
  }
  
  return result;
}

/**
 * Computes the L2 (Euclidean) norm of a Float64Array vector.
 * Used for convergence checks and gradient norm calculations.
 */
export function vectorNorm(vector: Float64Array): number {
  let sumOfSquares = 0.0;
  
  for (let i = 0; i < vector.length; i++) {
    const value = vector[i];
    sumOfSquares += value * value;
  }
  
  return Math.sqrt(sumOfSquares);
}

/**
 * Computes the dot product of two Float64Array vectors.
 * Both vectors must have the same length.
 */
export function dotProduct(vectorA: Float64Array, vectorB: Float64Array): number {
  if (vectorA.length !== vectorB.length) {
    throw new Error('Vectors must have the same length for dot product');
  }
  
  let sum = 0.0;
  
  for (let i = 0; i < vectorA.length; i++) {
    sum += vectorA[i] * vectorB[i];
  }
  
  return sum;
}

/**
 * Adds two Float64Array vectors element-wise.
 * Returns a new Float64Array with the result.
 */
export function addVectors(vectorA: Float64Array, vectorB: Float64Array): Float64Array {
  if (vectorA.length !== vectorB.length) {
    throw new Error('Vectors must have the same length for addition');
  }
  
  const result = new Float64Array(vectorA.length);
  
  for (let i = 0; i < vectorA.length; i++) {
    result[i] = vectorA[i] + vectorB[i];
  }
  
  return result;
}

/**
 * Subtracts vectorB from vectorA element-wise.
 * Returns a new Float64Array with the result.
 */
export function subtractVectors(vectorA: Float64Array, vectorB: Float64Array): Float64Array {
  if (vectorA.length !== vectorB.length) {
    throw new Error('Vectors must have the same length for subtraction');
  }
  
  const result = new Float64Array(vectorA.length);
  
  for (let i = 0; i < vectorA.length; i++) {
    result[i] = vectorA[i] - vectorB[i];
  }
  
  return result;
}

/**
 * Multiplies a Float64Array vector by a scalar.
 * Returns a new Float64Array with the result.
 */
export function scaleVector(vector: Float64Array, scalar: number): Float64Array {
  const result = new Float64Array(vector.length);
  
  for (let i = 0; i < vector.length; i++) {
    result[i] = vector[i] * scalar;
  }
  
  return result;
}

