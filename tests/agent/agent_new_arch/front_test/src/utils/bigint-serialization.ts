import type { HexString, Wei, BlockNumber, ChainId, Nonce, Timestamp } from '../types/index.js';

/**
 * BigInt Serialization Utilities
 * Solves the JSON.stringify losing bigint issue in frontend contexts.
 * Provides safe serialization/deserialization for all numeric domain types.
 */

interface SerializedBigInt {
  __type: 'bigint';
  value: string;
}

function createSerializedBigInt(value: bigint): SerializedBigInt {
  return { __type: 'bigint', value: value.toString() };
}

function isSerializedBigInt(obj: unknown): obj is SerializedBigInt {
  return (
    typeof obj === 'object' &&
    obj !== null &&
    '__type' in obj &&
    (obj as SerializedBigInt).__type === 'bigint' &&
    typeof (obj as SerializedBigInt).value === 'string'
  );
}

/**
 * Reviver function for JSON.parse to restore bigint values.
 */
export function bigintReviver(_key: string, value: unknown): unknown {
  if (isSerializedBigInt(value)) {
    return BigInt(value.value);
  }
  return value;
}

/**
 * Replacer function for JSON.stringify to preserve bigint values.
 */
export function bigintReplacer(_key: string, value: unknown): unknown {
  if (typeof value === 'bigint') {
    return createSerializedBigInt(value);
  }
  return value;
}

/**
 * Safely serialize a value that contains bigint fields.
 */
export function safeSerialize(value: unknown): string {
  return JSON.stringify(value, bigintReplacer);
}

/**
 * Safely deserialize a value that contains bigint fields.
 */
export function safeDeserialize<T>(json: string): T {
  return JSON.parse(json, bigintReviver) as T;
}

/**
 * Convert Wei to Ether string representation for display.
 */
export function weiToEther(wei: Wei): string {
  const ether = Number(wei) / 1e18;
  return ether.toFixed(6).replace(/\.?0+$/, '');
}

/**
 * Convert Ether string to Wei.
 */
export function etherToWei(ether: string): Wei {
  const parts = ether.split('.');
  const whole = parts[0] ?? '0';
  let fractional = parts[1] ?? '';
  fractional = fractional.padEnd(18, '0').slice(0, 18);
  return BigInt(whole + fractional);
}

/**
 * Format a bigint as a hexadecimal string.
 */
export function bigintToHex(value: bigint): HexString {
  return `0x${value.toString(16)}` as HexString;
}

/**
 * Parse a hexadecimal string to bigint.
 */
export function hexToBigInt(hex: HexString): bigint {
  return BigInt(hex);
}

/**
 * Format a bigint as a locale-aware display string with optional decimals.
 */
export function formatBigInt(value: bigint, decimals: number = 18, displayDecimals: number = 6): string {
  const divisor = BigInt(10 ** decimals);
  const wholePart = value / divisor;
  const fractionalPart = value % divisor;
  const fractionalStr = fractionalPart.toString().padStart(decimals, '0').slice(0, displayDecimals);
  return `${wholePart.toLocaleString()}.${fractionalStr}`;
}
