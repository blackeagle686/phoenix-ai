export type HexString = `0x${string}`;
export type Address = HexString;
export type PrivateKey = HexString;
export type PublicKey = HexString;
export type Signature = HexString;
export type TransactionHash = HexString;
export type BlockNumber = bigint;
export type Nonce = bigint;
export type Wei = bigint;
export type Timestamp = bigint;
export type ChainId = bigint;
export interface ChainConfig {
  chainId: ChainId;
  name: string;
  rpcUrl: string;
  currencySymbol: string;
  blockExplorerUrl?: string;
}
export interface KeyPair {
  publicKey: PublicKey;
  privateKey: PrivateKey;
}
export enum TransactionStatus {
  DRAFT = 'DRAFT',
  CONSTRUCTED = 'CONSTRUCTED',
  SIGNED = 'SIGNED',
  BROADCASTED = 'BROADCASTED',
  CONFIRMED = 'CONFIRMED',
  FAILED = 'FAILED,
  DROPPED = 'DROPPED'
}
export interface UnsignedTransaction {
  to: Address;
  value: Wei;
  data: HexString;
  gasLimit: Wei;
  maxFeePerGas: Wei;
  maxPriorityFeePerGas: Wei;
  nonce: Nonce;
  chainId: ChainId;
}
export interface SignedTransaction extends UnsignedTransaction {
  signature: Signature;
  serialized: HexString;
}
export interface TransactionRecord {
  id: string;
  unsigned: UnsignedTransaction;
  signed?: SignedTransaction;
  hash?: TransactionHash;
  status: TransactionStatus;
  createdAt: Timestamp;
  updatedAt: Timestamp;
  error?: string;
  blockNumber?: BlockNumber;
  confirmations?: number;
}
export interface BalanceInfo {
  address: Address;
  balance: Wei;
  blockNumber: BlockNumber;
}
export interface Block {
  number: BlockNumber;
  hash: HexString;
  parentHash: HexString;
  timestamp: Timestamp;
  transactions: TransactionHash[];
}
export interface EventLog {
  address: Address;
  topics: HexString[];
  data: HexString;
  blockNumber: BlockNumber;
  transactionHash: TransactionHash;
  logIndex: number;
}
export interface Subscription {
  id: string;
  unsubscribe: () => void;
}
export interface WalletAccount {
  address: Address;
  publicKey: PublicKey;
  createdAt: Timestamp;
}
export interface WalletState {
  accounts: WalletAccount[];
  activeAccount: Address | null;
  chainId: ChainId | null;
  isLocked: boolean;
}
export interface EncryptedVault {
  ciphertext: string;
  iv: string;
  salt: string;
  version: number;
}
