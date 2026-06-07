import type {
  UnsignedTransaction,
  SignedTransaction,
  TransactionRecord,
  TransactionStatus,
  TransactionHash,
  Address,
  Wei,
  HexString,
  BlockNumber,
} from '../types/index.js';

export interface ITransactionManager {
  createTransaction(params: CreateTransactionParams): Promise<UnsignedTransaction>;
  signTransaction(
    unsigned: UnsignedTransaction,
    privateKey: HexString
  ): Promise<SignedTransaction>;
  serializeTransaction(signed: SignedTransaction): HexString;
  estimateGas(transaction: UnsignedTransaction): Promise<Wei>;
  getNonce(address: Address): Promise<bigint>;
  getRecommendedFees(): Promise<{ maxFeePerGas: Wei; maxPriorityFeePerGas: Wei }>;
}

export interface ITransactionBroadcaster {
  broadcast(signedTx: SignedTransaction): Promise<TransactionHash>;
  getTransactionReceipt(hash: TransactionHash): Promise<TransactionReceipt | null>;
  getTransactionByHash(hash: TransactionHash): Promise<TransactionOnChain | null>;
  waitForConfirmation(
    hash: TransactionHash,
    confirmations?: number,
    timeoutMs?: number
  ): Promise<TransactionReceipt>;
}

export interface ITransactionHistory {
  add(record: TransactionRecord): Promise<void>;
  get(id: string): Promise<TransactionRecord | null>;
  getByAddress(address: Address): Promise<TransactionRecord[]>;
  getByStatus(status: TransactionStatus): Promise<TransactionRecord[]>;
  update(id: string, updates: Partial<TransactionRecord>): Promise<void>;
  getAll(): Promise<TransactionRecord[]>;
}

export interface CreateTransactionParams {
  from: Address;
  to: Address;
  value: Wei;
  data?: HexString;
  gasLimit?: Wei;
  maxFeePerGas?: Wei;
  maxPriorityFeePerGas?: Wei;
  nonce?: bigint;
}

export interface TransactionReceipt {
  hash: TransactionHash;
  from: Address;
  to: Address | null;
  blockNumber: BlockNumber;
  gasUsed: Wei;
  status: boolean;
  logs: TransactionLog[];
}

export interface TransactionLog {
  address: Address;
  topics: HexString[];
  data: HexString;
  logIndex: number;
}

export interface TransactionOnChain {
  hash: TransactionHash;
  from: Address;
  to: Address | null;
  value: Wei;
  blockNumber: BlockNumber | null;
  confirmations: number;
}

export interface TransactionManagerConfig {
  blockchainProvider: IBlockchainProvider;
  keyManager: IKeyManager;
  history: ITransactionHistory;
  requiredConfirmations?: number;
}

// Forward declarations to avoid circular imports
import type { IBlockchainProvider } from './blockchain-provider.js';
import type { IKeyManager } from './key-manager.js';
