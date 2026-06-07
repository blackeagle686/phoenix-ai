import type { EncryptedVault, PublicKey } from '../types/index.js';

export interface IVault {
  isInitialized(): boolean;
  isLocked(): boolean;
  initialize(password: string): Promise<void>;
  lock(): void;
  unlock(password: string): Promise<boolean>;
  storeKey(publicKey: PublicKey, encryptedPrivateKey: string): Promise<void>;
  retrieveKey(publicKey: PublicKey): Promise<string | null>;
  deleteKey(publicKey: PublicKey): Promise<void>;
  listKeys(): Promise<PublicKey[]>;
  changePassword(oldPassword: string, newPassword: string): Promise<void>;
  exportVault(): Promise<EncryptedVault>;
  importVault(vault: EncryptedVault, password: string): Promise<void>;
}

export interface VaultConfig {
  kdfIterations?: number;
  kdfAlgorithm?: string;
  encryptionAlgorithm?: string;
  derivationPath?: string;
}

export interface ISecureStorage {
  setItem(key: string, value: string): Promise<void>;
  getItem(key: string): Promise<string | null>;
  removeItem(key: string): Promise<void>;
  clear(): Promise<void>;
  hasItem(key: string): Promise<boolean>;
}
