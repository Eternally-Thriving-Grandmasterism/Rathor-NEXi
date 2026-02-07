// src/storage/rathor-indexeddb.js — Optimized, versioned, Zstandard-compressed IndexedDB wrapper

const DB_NAME = 'rathor-indexeddb';
const DB_VERSION = 7; // bump for zstd compression

const STORES = {
  sessions: 'sessions',
  messages: 'messages',
  tags: 'tags',
  translationCache: 'translationCache'
};

let db = null;

// Zstandard (zstd-wasm) loading
let ZstdCodec = null;
async function loadZstd() {
  if (ZstdCodec) return ZstdCodec;
  const script = document.createElement('script');
  script.src = 'https://unpkg.com/@libzstd-js/zstd-codec@0.0.3/dist/zstd.min.js';
  script.async = true;
  document.head.appendChild(script);
  await new Promise(resolve => { script.onload = resolve; });
  ZstdCodec = window.ZstdCodec;
  if (!ZstdCodec) throw new Error('ZstdCodec failed to load');
  return ZstdCodec;
}

// Database open with migration
const dbPromise = new Promise((resolve, reject) => {
  const request = indexedDB.open(DB_NAME, DB_VERSION);

  request.onupgradeneeded = event => {
    db = event.target.result;
    const oldVersion = event.oldVersion;
    console.log(`[rathorDB] Migrating from v\( {oldVersion} to v \){DB_VERSION}`);

    if (oldVersion < 1) {
      db.createObjectStore(STORES.sessions, { keyPath: 'id' });
      db.createObjectStore(STORES.messages, { autoIncrement: true });
      db.createObjectStore(STORES.tags, { keyPath: 'id' });
      db.createObjectStore(STORES.translationCache, { keyPath: 'key' });
    }

    if (oldVersion < 2) {
      const msgStore = db.transaction(STORES.messages, 'readwrite').objectStore(STORES.messages);
      msgStore.createIndex('sessionId', 'sessionId');
      msgStore.createIndex('timestamp', 'timestamp');
      msgStore.createIndex('role', 'role');
    }

    if (oldVersion < 7) {
      console.log('[rathorDB] Zstandard compression support added (v7)');
    }
  };

  request.onsuccess = event => {
    db = event.target.result;
    resolve(db);
  };

  request.onerror = event => reject(event.target.error);
});

async function openDB() {
  if (db) return db;
  db = await dbPromise;
  return db;
}

// ────────────────────────────────────────────────
// Messages — Zstandard-compressed
// ────────────────────────────────────────────────

export async function saveMessage(sessionId, role, content) {
  const db = await openDB();
  let compressed = content;
  let compression = 'none';
  let originalSize = new TextEncoder().encode(content).length;

  // Compress if > 1 KB
  if (originalSize > 1024) {
    try {
      await loadZstd();
      const codec = await ZstdCodec({ wasmUrl: 'https://unpkg.com/@libzstd-js/zstd-codec@0.0.3/dist/zstd.wasm' });
      const data = new TextEncoder().encode(content);
      compressed = await codec.compress(data, 3); // level 3 = good balance
      compression = 'zstd';
    } catch (e) {
      console.warn('Zstd compression failed, saving raw', e);
    }
  }

  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORES.messages, 'readwrite');
    const store = tx.objectStore(STORES.messages);
    const msg = {
      sessionId,
      role,
      content: compressed,
      compression,
      originalSize,
      timestamp: Date.now()
    };
    const req = store.add(msg);
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}

export async function getMessages(sessionId, limit = 100, offset = 0) {
  const db = await openDB();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORES.messages);
    const store = tx.objectStore(STORES.messages);
    const index = store.index('sessionId');
    const req = index.openCursor(IDBKeyRange.only(sessionId));
    const results = [];
    let skipped = 0;

    req.onsuccess = async event => {
      const cursor = event.target.result;
      if (!cursor) {
        // Decompress on read
        await loadZstd();
        const codec = await ZstdCodec({ wasmUrl: 'https://unpkg.com/@libzstd-js/zstd-codec@0.0.3/dist/zstd.wasm' });
        const decompressed = await Promise.all(results.map(async msg => {
          if (msg.compression === 'zstd') {
            try {
              const decompressedData = await codec.decompress(msg.content);
              msg.content = new TextDecoder().decode(decompressedData);
            } catch (e) {
              console.warn('Zstd decompress failed for msg', msg.id, e);
              // fallback: keep raw
            }
          }
          return msg;
        }));
        return resolve(decompressed);
      }

      if (skipped < offset) {
        skipped++;
        cursor.continue();
      } else if (results.length < limit) {
        results.push(cursor.value);
        cursor.continue();
      } else {
        await loadZstd();
        const codec = await ZstdCodec({ wasmUrl: 'https://unpkg.com/@libzstd-js/zstd-codec@0.0.3/dist/zstd.wasm' });
        const decompressed = await Promise.all(results.map(async msg => {
          if (msg.compression === 'zstd') {
            try {
              const decompressedData = await codec.decompress(msg.content);
              msg.content = new TextDecoder().decode(decompressedData);
            } catch (e) {
              console.warn('Zstd decompress failed for msg', msg.id, e);
            }
          }
          return msg;
        }));
        resolve(decompressed);
      }
    };

    req.onerror = () => reject(req.error);
  });
}

// ... rest of functions (sessions CRUD, cleanup, quota, etc.) remain as in previous optimized version ...

export default {
  openDB,
  saveSession,
  getSession,
  getAllSessions,
  saveMessage,
  getMessages,
  clearExpiredCache,
  getStorageUsage
};
