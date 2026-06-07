export type Result<T, E = string> =
  | { success: true; data: T }
  | { success: false; error: E };

export function success<T>(data: T): Result<T> {
  return { success: true, data };
}

export function failure<E = string>(error: E): Result<never, E> {
  return { success: false, error };
}

export function isSuccess<T, E>(result: Result<T, E>): result is { success: true; data: T } {
  return result.success === true;
}

export function isFailure<T, E>(result: Result<T, E>): result is { success: false; error: E } {
  return result.success === false;
}

export async function mapResult<T, U, E>(
  result: Result<T, E>,
  fn: (data: T) => U | Promise<U>
): Promise<Result<U, E>> {
  if (result.success) {
    return success(await fn(result.data));
  }
  return result;
}

export async function flatMapResult<T, U, E>(
  result: Result<T, E>,
  fn: (data: T) => Result<U, E> | Promise<Result<U, E>>
): Promise<Result<U, E>> {
  if (result.success) {
    return await fn(result.data);
  }
  return result;
}

export interface AsyncResult<T, E = string> {
  then<U>(
    onFulfilled: (result: Result<T, E>) => U | Promise<U>
  ): Promise<U>;
  catch<U>(
    onRejected: (error: E) => U | Promise<U>
  ): Promise<U>;
}

export type Mnemonic = string;
export type DerivationPath = string;

export interface MnemonicStrength {
  readonly words: number;
  readonly bits: number;
}

export const MNEMONIC_STRENGTHS: Record<string, MnemonicStrength> = {
  WORDS_12: { words: 12, bits: 128 },
  WORDS_15: { words: 15, bits: 160 },
  WORDS_18: { words: 18, bits: 192 },
  WORDS_21: { words: 21, bits: 224 },
  WORDS_24: { words: 24, bits: 256 },
};
