import type { Address, PublicKey, PrivateKey, KeyPair, Signature, HexString, EncryptedVault } from '../types/index.js';

export interface IKeyManager {
  generateKeyPair(): Promise<KeyPair>;
  deriveKeyPairFromMnemonic(mnemonic: string, derivationPath: string): Promise<KeyPair>;
  sign(data: HexString, privateKey: PrivateKey): Promise<Signature>;
  verify(data: HexString, signature: Signature, publicKey: PublicKey): Promise<boolean>;
  getAddress(publicKey: PublicKey): Promise<Address>;
  exportPrivateKey(publicKey: PublicKey): Promise<PrivateKey>;
  hasKey(publicKey: PublicKey): boolean;
}

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

export interface KeyManagerConfig {
  vault: IVault;
  defaultDerivationPath?: string;
}

export interface ISecureStorage {
  setItem(key: string, value: string): Promise<void>;
  getItem(key: string): Promise<string | null>;
  removeItem(key: string): Promise<void>;
  clear(): Promise<void>;
  hasItem(key: string): Promise<boolean>;
}
