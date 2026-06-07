export interface IEventEmitter {
  on(event: string, listener: (...args: unknown[]) => void): void;
  off(event: string, listener: (...args: unknown[]) => void): void;
  emit(event: string, ...args: unknown[]): void;
  once(event: string, listener: (...args: unknown[]) => void): void;
  removeAllListeners(event?: string): void;
  listenerCount(event: string): number;
}

export interface IWalletEventEmitter extends IEventEmitter {
  on(event: 'accountChanged', listener: (address: string | null) => void): void;
  on(event: 'chainChanged', listener: (chainId: string) => void): void;
  on(event: 'lockStateChanged', listener: (isLocked: boolean) => void): void;
  on(event: 'transactionSubmitted', listener: (hash: string) => void): void;
  on(event: 'transactionConfirmed', listener: (hash: string, confirmations: number) => void): void;
  on(event: 'transactionFailed', listener: (hash: string, error: string) => void): void;
  on(event: 'balanceUpdated', listener: (address: string, balance: string) => void): void;
}
