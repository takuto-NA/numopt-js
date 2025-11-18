/**
 * Type definitions for ml-matrix library.
 * This file provides TypeScript types for ml-matrix when official types are not available.
 * 
 * Role in system:
 * - Provides type safety for ml-matrix Matrix class
 * - Ensures compatibility with our Float64Array-based code
 * 
 * For first-time readers:
 * - These are type definitions only, no implementation
 * - Used throughout the codebase for matrix operations
 */

declare module 'ml-matrix' {
  export class Matrix {
    constructor(data: number[][] | number[]);
    
    rows: number;
    columns: number;
    
    get(row: number, column: number): number;
    set(row: number, column: number, value: number): void;
    
    transpose(): Matrix;
    mmul(other: Matrix): Matrix;
    mul(scalar: number): Matrix;
    add(other: Matrix): Matrix;
    
    static eye(rows: number, columns: number): Matrix;
    static zeros(rows: number, columns: number): Matrix;
    
    to1DArray(): number[];
  }

  export function solve(A: Matrix, b: Matrix): Matrix;

  export class CholeskyDecomposition {
    constructor(value: Matrix | number[][] | number[]);
    isPositiveDefinite(): boolean;
    solve(value: Matrix): Matrix;
    readonly lowerTriangularMatrix: Matrix;
  }
}

declare module 'ml-matrix/src/dc/cholesky' {
  import { CholeskyDecomposition } from 'ml-matrix';
  export default CholeskyDecomposition;
}
