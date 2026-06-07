import type {
  Address,
  PublicKey,
  WalletAccount,
  WalletState,
  ChainId,
  KeyPair,
  EncryptedVault,
  HexString,
} from '../types/index.js';
import type { IKeyManager } from './key-manager.js';
import type { IVault } from './vault.js';
import type { IEventEmitter } from './event-emitter.js';

export interface IWallet {
  readonly state: WalletState;
  readonly eventEmitter: IEventEmitter;
  createFromMnemonic(mnemonic: string, password: string, derivationPath?: string): Promise<WalletAccount>;
  createFromPrivateKey(privateKey: HexString, password: string): Promise<WalletAccount>;
  importFromVault(vault: EncryptedVault, password: string): Promise<void>;
  exportVault(password: string): Promise<EncryptedVault>;
  recoverFromMnemonic(mnemonic: string, password: string, derivationPath?: string): Promise<WalletAccount>;
  addAccount(derivationPath?: string): Promise<WalletAccount>;
  removeAccount(address: Address): Promise<void>;
  setActiveAccount(address: Address): Promise<void>;
  getAccount(address: Address): WalletAccount | null;
  listAccounts(): WalletAccount[];
  lock(): void;
  unlock(password: string): Promise<boolean>;
  isLocked(): boolean;
  changePassword(oldPassword: string, newPassword: string): Promise<void>;
  setChain(chainId: ChainId): Promise<void>;
  signMessage(address: Address, message: HexString): Promise<HexString>;
  verifyMessage(message: HexString, signature: HexString): Promise<Address | null>;
  exportPrivateKey(address: Address, password: string): Promise<HexString>;
}

export interface WalletConfig {
  defaultDerivationPath: string;
  defaultChainId: ChainId;
  vaultKdfIterations: number;
  maxAccounts: number;
  autoLockTimeoutMs: number;
}

export interface WalletCreationResult {
  success: boolean;
  account?: WalletAccount;
  error?: string;
}

export interface WalletRecoveryResult {
  success: boolean;
  account?: WalletAccount;
  accountsRestored?: number;
  error?: string;
}

export interface IWalletFactory {
  create(config: WalletConfig): Promise<IWallet>;
  createWithMnemonic(mnemonic: string, password: string, config: WalletConfig): Promise<IWallet>;
  createWithPrivateKey(privateKey: HexString, password: string, config: WalletConfig): Promise<IWallet>;
  restoreFromVault(vault: EncryptedVault, password: string, config: WalletConfig): Promise<IWallet>;
}

export interface IWalletState {
  getState(): WalletState;
  setState(state: Partial<WalletState>): void;
  subscribe(listener: (state: WalletState) => void): () => void;
}
