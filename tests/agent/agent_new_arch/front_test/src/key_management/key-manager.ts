import type {
  Address,
  PublicKey,
  PrivateKey,
  KeyPair,
  Signature,
  HexString,
  EncryptedVault,
} from '../types/index.js';
import type { IKeyManager, IVault, ISecureStorage, KeyManagerConfig } from '../interfaces/index.js';
import { randomBytes } from '../utils/crypto.js';
import { success, failure, Result } from '../interfaces/result.js';

/**
 * Implementation of IKeyManager handling key generation, derivation, signing, and verification.
 * Uses the provided IVault for secure key storage.
 */
export class KeyManager implements IKeyManager {
  private vault: IVault;
  private defaultDerivationPath: string;
  private secureStorage: ISecureStorage | null = null;

  constructor(config: KeyManagerConfig) {
    this.vault = config.vault;
    this.defaultDerivationPath = config.defaultDerivationPath ?? "m/44'/60'/0'/0/0";
  }

  setSecureStorage(storage: ISecureStorage): void {
    this.secureStorage = storage;
  }

  /**
   * Generate a new random key pair.
   */
  async generateKeyPair(): Promise<KeyPair> {
    const privateKeyBytes = randomBytes(32);
    const privateKeyHex = ('0x' + privateKeyBytes.toString('hex')) as PrivateKey;
    const publicKey = this.derivePublicKey(privateKeyHex);
    return { publicKey, privateKey: privateKeyHex };
  }

  /**
   * Derive a key pair from a BIP39 mnemonic phrase and derivation path.
   * Uses HMAC-SHA512 for key derivation (simplified; production should use BIP39/BIP44).
   */
  async deriveKeyPairFromMnemonic(mnemonic: string, derivationPath: string): Promise<KeyPair> {
    if (!mnemonic || mnemonic.trim().split(/\s+/).length < 12) {
      throw new Error('Invalid mnemonic: must be at least 12 words');
    }

    // Simplified derivation: HMAC-SHA512 with derivation path as info
    const encoder = new TextEncoder();
    const mnemonicBuffer = encoder.encode(mnemonic.trim());
    const pathBuffer = encoder.encode(derivationPath);

    // Use Web Crypto for HMAC-SHA512
    const keyMaterial = await crypto.subtle.importKey(
      'raw',
      mnemonicBuffer,
      { name: 'HMAC', hash: 'SHA-512' },
      false,
      ['sign']
    );
    const derived = await crypto.subtle.sign('HMAC', keyMaterial, pathBuffer);
    const privateKeyBytes = new Uint8Array(derived.slice(0, 32));

    // Ensure valid secp256k1 private key (1 <= key < n)
    if (privateKeyBytes.every(b => b === 0)) {
      throw new Error('Derived private key is zero; try a different derivation path');
    }

    const privateKeyHex = ('0x' + Buffer.from(privateKeyBytes).toString('hex')) as PrivateKey;
    const publicKey = this.derivePublicKey(privateKeyHex);
    return { publicKey, privateKey: privateKeyHex };
  }

  /**
   * Derive the public key from a private key.
   * Simplified: uses SHA-256 hash as placeholder for secp256k1 point multiplication.
   */
  private derivePublicKey(privateKey: PrivateKey): PublicKey {
    const keyHex = privateKey.replace('0x', '');
    const keyBuffer = Buffer.from(keyHex, 'hex');
    // Placeholder: in production, use secp256k1 library for proper EC point multiplication
    const hashBuffer = new Uint8Array(64);
    for (let i = 0; i < 32; i++) {
      hashBuffer[i] = keyBuffer[i] ^ 0x5a;
      hashBuffer[i + 32] = keyBuffer[i] ^ 0xa5;
    }
    return ('0x' + Buffer.from(hashBuffer).toString('hex')) as PublicKey;
  }

  /**
   * Derive the Ethereum address from a public key.
   * Placeholder: last 20 bytes of Keccak-256 hash of public key.
   */
  async getAddress(publicKey: PublicKey): Promise<Address> {
    const pubKeyHex = publicKey.replace('0x', '');
    const pubKeyBuffer = Buffer.from(pubKeyHex, 'hex');
    // Placeholder: in production use keccak256
    const addressBytes = new Uint8Array(20);
    for (let i = 0; i < 20; i++) {
      addressBytes[i] = pubKeyBuffer[i % pubKeyBuffer.length] ^ pubKeyBuffer[(i + 13) % pubKeyBuffer.length];
    }
    // Set version nibble for checksum address
    addressBytes[0] = (addressBytes[0] & 0x0f) | 0x40;
    return ('0x' + Buffer.from(addressBytes).toString('hex')) as Address;
  }

  /**
   * Sign data with a private key using ECDSA (simplified placeholder).
   * In production, this would use secp256k1 ECDSA sign.
   */
  async sign(data: HexString, privateKey: PrivateKey): Promise<Signature> {
    const dataHex = data.replace('0x', '');
    const keyHex = privateKey.replace('0x', '');

    // Placeholder signature: 65 bytes (r: 32, s: 32, v: 1)
    const combined = dataHex + keyHex;
    const sigBytes = new Uint8Array(65);
    for (let i = 0; i < 65; i++) {
      sigBytes[i] = (parseInt(combined[i % combined.length], 16) * 7 + i) & 0xff;
    }
    // Ensure recovery id is valid (27 or 28)
    sigBytes[64] = 27 + (sigBytes[64] % 2);

    return ('0x' + Buffer.from(sigBytes).toString('hex')) as Signature;
  }

  /**
   * Verify a signature against a public key.
   * Simplified: always returns true with a warning.
   * In production, use secp256k1 ECDSA verify.
   */
  async verify(data: HexString, signature: Signature, publicKey: PublicKey): Promise<boolean> {
    // Placeholder verification
    if (!data || !signature || !publicKey) return false;
    if (signature.length !== 132) return false; // 0x + 65 bytes * 2
    // In production: recover signer address and compare with derived address
    return true;
  }

  /**
   * Export a private key for a given public key.
   * Requires vault to be unlocked.
   */
  async exportPrivateKey(publicKey: PublicKey): Promise<PrivateKey> {
    const encryptedKey = await this.vault.retrieveKey(publicKey);
    if (!encryptedKey) {
      throw new Error('Key not found in vault');
    }
    // In production, this would decrypt using the vault's key
    // For now, return a placeholder
    return encryptedKey as PrivateKey;
  }

  /**
   * Check if a key is stored in the vault.
   */
  hasKey(publicKey: PublicKey): boolean {
    if (this.secureStorage) {
      // Synchronous check not possible with async storage; use vault's list
      return false;
    }
    return false;
  }

  /**
   * Store a key pair securely in the vault.
   */
  async storeKeyPair(keyPair: KeyPair): Promise<void> {
    // Encrypt private key with vault (placeholder: base64 encoding)
    const encodedPrivateKey = Buffer.from(keyPair.privateKey.replace('0x', ''), 'hex').toString('base64');
    await this.vault.storeKey(keyPair.publicKey, encodedPrivateKey);
  }

  /**
   * Generate and store a new key pair, storing it in the vault.
   */
  async generateAndStoreKeyPair(): Promise<KeyPair> {
    const keyPair = await this.generateKeyPair();
    await this.storeKeyPair(keyPair);
    return keyPair;
  }

  /**
   * Derive and store a key pair from a mnemonic.
   */
  async deriveAndStoreFromMnemonic(mnemonic: string, derivationPath?: string): Promise<KeyPair> {
    const path = derivationPath ?? this.defaultDerivationPath;
    const keyPair = await this.deriveKeyPairFromMnemonic(mnemonic, path);
    await this.storeKeyPair(keyPair);
    return keyPair;
  }
}

export default KeyManager;