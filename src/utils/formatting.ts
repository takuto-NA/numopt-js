/**
 * This file provides number formatting utilities for consistent display
 * of optimization results across the library.
 * 
 * Role in system:
 * - Shared formatting logic used by Logger and ResultFormatter
 * - Ensures consistent number representation in console output
 * - Handles edge cases (NaN, Infinity, very small/large numbers)
 * 
 * For first-time readers:
 * - formatNumber() is the main function for formatting numbers
 * - Uses scientific notation for very small/large numbers
 * - Uses fixed notation for standard range numbers
 */

const SCIENTIFIC_NOTATION_LOWER_THRESHOLD = 0.01;
const SCIENTIFIC_NOTATION_UPPER_THRESHOLD = 1000;
const SCIENTIFIC_NOTATION_FRACTION_DIGITS = 3;
const FIXED_NOTATION_FRACTION_DIGITS = 6;

/**
 * Formats a number using scientific notation for readability.
 * Small numbers (< 0.01) and large numbers (> 1000) use scientific notation.
 * Otherwise, uses standard decimal notation.
 * Handles non-numeric values gracefully.
 */
export function formatNumber(value: number | string): string {
  // Handle non-numeric values
  if (typeof value !== 'number' || !isFinite(value)) {
    return String(value);
  }
  
  const absoluteValue = Math.abs(value);
  if (absoluteValue === 0) {
    return '0';
  }
  // Use scientific notation for very small or very large numbers to avoid long strings of zeros
  if (absoluteValue < SCIENTIFIC_NOTATION_LOWER_THRESHOLD || absoluteValue >= SCIENTIFIC_NOTATION_UPPER_THRESHOLD) {
    return value.toExponential(SCIENTIFIC_NOTATION_FRACTION_DIGITS);
  }
  // Use fixed notation for standard range numbers for easier reading
  return value.toFixed(FIXED_NOTATION_FRACTION_DIGITS).replace(/\.?0+$/, '');
}

/**
 * Formats a number with a specific number of decimal places.
 * Used when precise control over formatting is needed.
 */
export function formatNumberWithPrecision(value: number, precision: number): string {
  if (typeof value !== 'number' || !isFinite(value)) {
    return String(value);
  }
  
  if (value === 0) {
    return '0.' + '0'.repeat(precision);
  }
  
  return value.toFixed(precision);
}

