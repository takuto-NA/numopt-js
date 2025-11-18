/**
 * This file implements a user-friendly and trackable logging system for optimization algorithms.
 * 
 * Role in system:
 * - Provides structured logging with log levels (DEBUG, INFO, WARN, ERROR)
 * - Formats logs with timestamps, emojis, and aligned indentation
 * - Optimized for performance when logging is disabled
 * 
 * For first-time readers:
 * - Start with LogLevel enum and Logger class
 * - Understand how logLevel and verbose options work together
 * - Check formatLogMessage for formatting details
 */

export type LogLevel = 'DEBUG' | 'INFO' | 'WARN' | 'ERROR';

/**
 * Log level priority order (higher number = higher priority).
 * Used to determine if a log message should be displayed.
 */
const LOG_LEVEL_PRIORITY: Record<LogLevel, number> = {
  DEBUG: 0,
  INFO: 1,
  WARN: 2,
  ERROR: 3
};

/**
 * Emoji symbols for each log level.
 * Used to provide visual distinction while maintaining academic tone.
 */
const LOG_LEVEL_EMOJI: Record<LogLevel, string> = {
  DEBUG: '🔍',
  INFO: '✅',
  WARN: '⚠️',
  ERROR: '❌'
};

/**
 * Determines the effective log level from options.
 * Handles backward compatibility with verbose option.
 * 
 * Priority: logLevel > verbose
 * - If logLevel is specified, use it
 * - If verbose is true, treat as INFO level
 * - Otherwise, logging is disabled (returns undefined)
 */
export function getEffectiveLogLevel(
  logLevel: LogLevel | undefined,
  verbose: boolean | undefined
): LogLevel | undefined {
  if (logLevel !== undefined) {
    return logLevel;
  }
  if (verbose === true) {
    return 'INFO';
  }
  return undefined;
}

/**
 * Checks if a log message should be displayed based on the effective log level.
 * 
 * A message is displayed if its level priority is >= the effective log level priority.
 * For example, if effective level is INFO, then INFO, WARN, and ERROR are shown, but DEBUG is not.
 */
export function shouldLog(
  messageLevel: LogLevel,
  effectiveLogLevel: LogLevel | undefined
): boolean {
  if (effectiveLogLevel === undefined) {
    return false;
  }
  return LOG_LEVEL_PRIORITY[messageLevel] >= LOG_LEVEL_PRIORITY[effectiveLogLevel];
}

/**
 * Formats a number using scientific notation for readability.
 * Small numbers (< 0.01) and large numbers (> 1000) use scientific notation.
 * Otherwise, uses standard decimal notation.
 */
function formatNumber(value: number): string {
  const absValue = Math.abs(value);
  if (absValue === 0) {
    return '0';
  }
  if (absValue < 0.01 || absValue >= 1000) {
    return value.toExponential(3);
  }
  return value.toFixed(6).replace(/\.?0+$/, '');
}

/**
 * Formats key-value pairs with aligned indentation.
 * All labels are padded to the same width so values align vertically.
 * 
 * Example:
 *   Cost:           1.234e-6
 *   Gradient norm:  9.876e-7
 *   Step size:      1.234e-8
 */
function formatKeyValuePairs(pairs: Array<{ key: string; value: number }>): string {
  if (pairs.length === 0) {
    return '';
  }

  // Find maximum label length
  const maxLabelLength = Math.max(...pairs.map(p => p.key.length));

  // Format each pair with aligned labels
  return pairs
    .map(({ key, value }) => {
      const paddedKey = key.padEnd(maxLabelLength);
      const formattedValue = formatNumber(value);
      return `  ${paddedKey}  ${formattedValue}`;
    })
    .join('\n');
}

/**
 * Formats a log message with timestamp, level, algorithm, iteration, and details.
 * 
 * Format: emoji [timestamp] [LEVEL] [algorithm] [iteration] message
 * 
 * Emoji is placed at the beginning for better visual scanning.
 * If details are provided as key-value pairs, they are formatted with aligned indentation.
 */
function formatLogMessage(
  level: LogLevel,
  algorithm: string,
  iteration: number | undefined,
  message: string,
  details?: Array<{ key: string; value: number }>
): string {
  const timestamp = new Date().toISOString();
  const emoji = LOG_LEVEL_EMOJI[level];
  const levelLabel = `[${level}]`;
  
  const iterationPart = iteration !== undefined ? `Iteration ${iteration}` : '';
  const header = `${emoji} [${timestamp}] ${levelLabel} [${algorithm}]${iterationPart ? ` ${iterationPart}` : ''}: ${message}`;
  
  if (details && details.length > 0) {
    const detailsFormatted = formatKeyValuePairs(details);
    return `${header}\n${detailsFormatted}`;
  }
  
  return header;
}

/**
 * Logger class for optimization algorithms.
 * Provides structured logging with log levels and formatted output.
 */
export class Logger {
  private effectiveLogLevel: LogLevel | undefined;

  constructor(logLevel: LogLevel | undefined, verbose: boolean | undefined) {
    this.effectiveLogLevel = getEffectiveLogLevel(logLevel, verbose);
  }

  /**
   * Logs a DEBUG level message.
   * Used for detailed progress information (cost, gradient norm, step size, etc.).
   */
  debug(
    algorithm: string,
    iteration: number | undefined,
    message: string,
    details?: Array<{ key: string; value: number }>
  ): void {
    if (!shouldLog('DEBUG', this.effectiveLogLevel)) {
      return;
    }
    const formatted = formatLogMessage('DEBUG', algorithm, iteration, message, details);
    console.log(formatted);
  }

  /**
   * Logs an INFO level message.
   * Used for convergence messages and important state changes.
   */
  info(
    algorithm: string,
    iteration: number | undefined,
    message: string,
    details?: Array<{ key: string; value: number }>
  ): void {
    if (!shouldLog('INFO', this.effectiveLogLevel)) {
      return;
    }
    const formatted = formatLogMessage('INFO', algorithm, iteration, message, details);
    console.log(formatted);
  }

  /**
   * Logs a WARN level message.
   * Used for warnings (singular matrix, max iterations reached, line search failure, etc.).
   */
  warn(
    algorithm: string,
    iteration: number | undefined,
    message: string,
    details?: Array<{ key: string; value: number }>
  ): void {
    if (!shouldLog('WARN', this.effectiveLogLevel)) {
      return;
    }
    const formatted = formatLogMessage('WARN', algorithm, iteration, message, details);
    console.log(formatted);
  }

  /**
   * Logs an ERROR level message.
   * Used for fatal errors (currently not used, reserved for future extensions).
   */
  error(
    algorithm: string,
    iteration: number | undefined,
    message: string,
    details?: Array<{ key: string; value: number }>
  ): void {
    if (!shouldLog('ERROR', this.effectiveLogLevel)) {
      return;
    }
    const formatted = formatLogMessage('ERROR', algorithm, iteration, message, details);
    console.error(formatted);
  }
}

