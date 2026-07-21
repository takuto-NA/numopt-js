/**
 * Vitest setup: ensure Web Crypto is available for seeded RNG helpers.
 *
 * Node 18+ normally exposes globalThis.crypto; this is a defensive fallback.
 */

import { webcrypto } from 'node:crypto';

type GlobalWithCrypto = typeof globalThis & { crypto?: typeof webcrypto };
const globalScope = globalThis as GlobalWithCrypto;

// Guard: older or unusual Node hosts may omit global crypto.
if (!globalScope.crypto) {
  globalScope.crypto = webcrypto;
}
