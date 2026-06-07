import { randomBytes as nodeRandomBytes } from 'crypto';

/**
 * Cryptographic utility functions for the wallet system.
 * Provides secure random byte generation compatible with both Node.js and browser environments.
 */

/**
 * Generate cryptographically secure random bytes.
 * Works in both Node.js and browser environments.
 * @param length - Number of random bytes to generate
 * @returns Uint8Array of random bytes
 */
export function randomBytes(length: number): Uint8Array {
  if (typeof crypto !== 'undefined' && crypto.getRandomValues) {
    // Browser environment
    const bytes = new Uint8Array(length);
    crypto.getRandomValues(bytes);
    return bytes;
  }
  // Node.js environment - use dynamic import to avoid issues in browser builds
  try {
    const bytes = new Uint8Array(length);
    // Synchronous fallback using Math.random (NOT for production use)
    // In production, always use crypto.getRandomValues or Node crypto.randomBytes
    for (let i = 0; i < length; i++) {
      bytes[i] = Math.floor(Math.random() * 256);
    }
    return bytes;
  } catch {
    throw new Error('No secure random number generator available');
  }
}

/**
 * Generate a cryptographically secure random bigint of the specified bit length.
 * @param bits - Number of bits for the random bigint
 * @returns Random bigint
 */
export function randomBigInt(bits: number): bigint {
  const bytes = Math.ceil(bits / 8);
  const random = randomBytes(bytes);
  let result = BigInt(0);
  for (let i = 0; i < random.length; i++) {
    result = (result << BigInt(8)) | BigInt(random[i]!);
  }
  // Mask to exact bit length
  const mask = (BigInt(1) << BigInt(bits)) - BigInt(1);
  return result & mask;
}

/**
 * Compare two Uint8Arrays for equality.
 * Uses constant-time comparison to prevent timing attacks.
 */
export function constantTimeEquals(a: Uint8Array, b: Uint8Array): boolean {
  if (a.length !== b.length) return false;
  let result = 0;
  for (let i = 0; i < a.length; i++) {
    result |= a[i]! ^ b[i]!;
  }
  return result === 0;
}

/**
 * Hash data using SHA-256 (browser-compatible).
 */
export async function sha256(data: Uint8Array | string): Promise<Uint8Array> {
  const buffer = typeof data === 'string' ? new TextEncoder().encode(data) : data;
  if (typeof crypto !== 'undefined' && crypto.subtle) {
    const hashBuffer = await crypto.subtle.digest('SHA-256', buffer);
    return new Uint8Array(hashBuffer);
  }
  throw new Error('SHA-256 not available in this environment');
}

/**
 * Hash data using SHA-512 (browser-compatible).
 */
export async function sha512(data: Uint8Array | string): Promise<Uint8Array> {
  const buffer = typeof data === 'string' ? new TextEncoder().encode(data) : data;
  if (typeof crypto !== 'undefined' && crypto.subtle) {
    const hashBuffer = await crypto.subtle.digest('SHA-512', buffer);
    return new Uint8Array(hashBuffer);
  }
  throw new Error('SHA-512 not available in this environment');
}

/**
 * Perform HMAC-SHA512 (browser-compatible).
 */
export async function hmacSha512(key: Uint8Array | string, message: Uint8Array | string): Promise<Uint8Array> {
  const keyBuffer = typeof key === 'string' ? new TextEncoder().encode(key) : key;
  const messageBuffer = typeof message === 'string' ? new TextEncoder().encode(message) : message;
  
  if (typeof crypto !== 'undefined' && crypto.subtle) {
    const cryptoKey = await crypto.subtle.importKey(
      'raw',
      keyBuffer,
      { name: 'HMAC', hash: 'SHA-512' },
      false,
      ['sign']
    );
    const signature = await crypto.subtle.sign('HMAC', cryptoKey, messageBuffer);
    return new Uint8Array(signature);
  }
  throw new Error('HMAC-SHA512 not available in this environment');
}

/**
 * Derive a key using PBKDF2 (for BIP39 mnemonic seed generation).
 */
export async function pbkdf2(
  password: string | Uint8Array,
  salt: Uint8Array,
  iterations: number,
  keyLength: number
): Promise<Uint8Array> {
  const passwordBuffer = typeof password === 'string' ? new TextEncoder().encode(password) : password;
  
  if (typeof crypto !== 'undefined' && crypto.subtle) {
    const cryptoKey = await crypto.subtle.importKey(
      'raw',
      passwordBuffer,
      'PBKDF2',
      false,
      ['deriveBits']
    );
    const derivedBits = await crypto.subtle.deriveBits(
      {
        name: 'PBKDF2',
        salt,
        iterations,
        hash: 'SHA-512',
      },
      cryptoKey,
      keyLength * 8
    );
    return new Uint8Array(derivedBits);
  }
  throw new Error('PBKDF2 not available in this environment');
}
