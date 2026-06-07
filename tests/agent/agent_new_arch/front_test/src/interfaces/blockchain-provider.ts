import type {
  Block,
  BlockNumber,
  BalanceInfo,
  ChainConfig,
  ChainId,
  Address,
  TransactionHash,
  EventLog,
  Subscription,
  HexString,
  TransactionReceipt,
  TransactionOnChain,
  SignedTransaction,
  Wei,
} from '../types/index.js';

export interface IBlockchainProvider {
  readonly chainId: ChainId;
  getBlockNumber(): Promise<BlockNumber>;
  getBlock(blockNumber: BlockNumber): Promise<Block | null>;
  getBlockByHash(blockHash: HexString): Promise<Block | null>;
  getBalance(address: Address): Promise<Wei>;
  getBalanceInfo(address: Address): Promise<BalanceInfo>;
  getTransactionCount(address: Address): Promise<bigint>;
  getCode(address: Address): Promise<HexString>;
  call(transaction: Record<string, unknown>): Promise<HexString>;
  estimateGas(transaction: Record<string, unknown>): Promise<Wei>;
  getGasPrice(): Promise<Wei>;
  getMaxPriorityFeePerGas(): Promise<Wei>;
  sendRawTransaction(signedTx: SignedTransaction): Promise<TransactionHash>;
  getTransactionReceipt(hash: TransactionHash): Promise<TransactionReceipt | null>;
  getTransaction(hash: TransactionHash): Promise<TransactionOnChain | null>;
  waitForTransaction(
    hash: TransactionHash,
    confirmations?: number,
    timeoutMs?: number
  ): Promise<TransactionReceipt>;
  getLogs(filter: LogFilter): Promise<EventLog[]>;
  subscribe(event: string, callback: (log: EventLog) => void): Promise<Subscription>;
  subscribeNewHeads(callback: (block: Block) => void): Promise<Subscription>;
  subscribePendingTransactions(callback: (hash: TransactionHash) => void): Promise<Subscription>;
  isConnected(): boolean;
  getChainConfig(): ChainConfig;
}

export interface LogFilter {
  fromBlock?: BlockNumber;
  toBlock?: BlockNumber;
  address?: Address | Address[];
  topics?: (HexString | HexString[] | null)[];
}

export interface BlockchainProviderConfig {
  chainConfig: ChainConfig;
  timeoutMs?: number;
  retryCount?: number;
  retryDelayMs?: number;
}

export interface IBlockchainProviderFactory {
  create(config: BlockchainProviderConfig): IBlockchainProvider;
  createFromChainId(chainId: ChainId): IBlockchainProvider;
}
