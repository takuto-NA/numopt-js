import { Logger, getEffectiveLogLevel, shouldLog, LogLevel } from '../src/core/logger';

describe('Logger', () => {
    describe('getEffectiveLogLevel', () => {
        it('should return logLevel if provided', () => {
            expect(getEffectiveLogLevel('DEBUG', undefined)).toBe('DEBUG');
            expect(getEffectiveLogLevel('INFO', true)).toBe('INFO');
        });

        it('should return INFO if verbose is true and logLevel is undefined', () => {
            expect(getEffectiveLogLevel(undefined, true)).toBe('INFO');
        });

        it('should return undefined if neither is provided', () => {
            expect(getEffectiveLogLevel(undefined, false)).toBe(undefined);
            expect(getEffectiveLogLevel(undefined, undefined)).toBe(undefined);
        });
    });

    describe('shouldLog', () => {
        it('should return false if effectiveLogLevel is undefined', () => {
            expect(shouldLog('ERROR', undefined)).toBe(false);
        });

        it('should return true if message level is higher or equal to effective level', () => {
            expect(shouldLog('INFO', 'DEBUG')).toBe(true);
            expect(shouldLog('INFO', 'INFO')).toBe(true);
            expect(shouldLog('WARN', 'INFO')).toBe(true);
            expect(shouldLog('ERROR', 'INFO')).toBe(true);
        });

        it('should return false if message level is lower than effective level', () => {
            expect(shouldLog('DEBUG', 'INFO')).toBe(false);
            expect(shouldLog('INFO', 'WARN')).toBe(false);
        });
    });

    describe('Logger class', () => {
        let consoleLogSpy: any;
        let consoleErrorSpy: any;

        beforeEach(() => {
            consoleLogSpy = vi.spyOn(console, 'log').mockImplementation(() => { });
            consoleErrorSpy = vi.spyOn(console, 'error').mockImplementation(() => { });
        });

        afterEach(() => {
            vi.restoreAllMocks();
        });

        it('should log debug messages when level is DEBUG', () => {
            const logger = new Logger('DEBUG', false);
            logger.debug('TestAlgo', 1, 'Debug message');
            expect(consoleLogSpy).toHaveBeenCalled();
            const logCall = consoleLogSpy.mock.calls[0][0];
            expect(logCall).toContain('[DEBUG]');
            expect(logCall).toContain('Debug message');
        });

        it('should not log debug messages when level is INFO', () => {
            const logger = new Logger('INFO', false);
            logger.debug('TestAlgo', 1, 'Debug message');
            expect(consoleLogSpy).not.toHaveBeenCalled();
        });

        it('should log info messages when level is INFO', () => {
            const logger = new Logger('INFO', false);
            logger.info('TestAlgo', 1, 'Info message');
            expect(consoleLogSpy).toHaveBeenCalled();
            const logCall = consoleLogSpy.mock.calls[0][0];
            expect(logCall).toContain('[INFO]');
            expect(logCall).toContain('Info message');
        });

        it('should log warn messages when level is WARN', () => {
            const logger = new Logger('WARN', false);
            logger.warn('TestAlgo', 1, 'Warn message');
            expect(consoleLogSpy).toHaveBeenCalled();
            const logCall = consoleLogSpy.mock.calls[0][0];
            expect(logCall).toContain('[WARN]');
            expect(logCall).toContain('Warn message');
        });

        it('should log error messages to console.error', () => {
            const logger = new Logger('ERROR', false);
            logger.error('TestAlgo', 1, 'Error message');
            expect(consoleErrorSpy).toHaveBeenCalled();
            const logCall = consoleErrorSpy.mock.calls[0][0];
            expect(logCall).toContain('[ERROR]');
            expect(logCall).toContain('Error message');
        });

        it('should format details correctly', () => {
            const logger = new Logger('INFO', false);
            logger.info('TestAlgo', 1, 'Message with details', [
                { key: 'Cost', value: 100 },
                { key: 'Small', value: 1e-5 }
            ]);
            expect(consoleLogSpy).toHaveBeenCalled();
            const logCall = consoleLogSpy.mock.calls[0][0];
            expect(logCall).toContain('Cost   100');
            expect(logCall).toContain('Small  1.000e-5'); // 1e-5 is < 0.01, so scientific notation
        });
    });
});
