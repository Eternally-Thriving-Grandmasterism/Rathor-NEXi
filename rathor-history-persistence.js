// rathor-history-persistence.js – v3 optimized: bulk ops, cursor streaming, error retry
// MIT License – Autonomicity Games Inc. 2026

const DB_NAME = 'RathorEternalLattice';
const DB_VERSION = 3;
const STORE_NAME = 'chatHistory';
let db = null;

async function openDB() {
  if (db) return db;
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(DB_NAME, DB_VERSION);
    req.onerror = e => reject(e.target.error);
    req.onsuccess = () => { db = req.result; resolve(db); };
    req.onupgradeneeded = e => {
      const upgradeDb = e.target.result;
      let store = upgradeDb.objectStoreNames.contains(STORE_NAME) 
        ? e.target.transaction.objectStore(STORE_NAME) 
        : upgradeDb.createObjectStore(STORE_NAME, { keyPath: 'id', autoIncrement: true });
      ['timestamp', 'sessionId', 'conversationId', 'parentId'].forEach(idx => {
        if (!store.indexNames.contains(idx)) store.createIndex(idx, idx, { unique: false });
      });
    };
  });
}

async function saveBulkMessages(entries) {
  const dbInstance = await openDB();
  return new Promise((resolve, reject) => {
    const tx = dbInstance.transaction(STORE_NAME, 'readwrite');
    const store = tx.objectStore(STORE_NAME);
    entries.forEach(entry => store.add(entry));
    tx.oncomplete = () => resolve();
    tx.onerror = e => reject(e.target.error);
  }).catch(err => {
    console.warn("[History] Bulk save error – retrying individually");
    return Promise.all(entries.map(entry => saveMessage(entry))); // fallback
  });
}

// Cursor streaming for large reads (yield instead of array)
async function* streamThread(conversationId = SESSION_ID) {
  const dbInstance = await openDB();
  const tx = dbInstance.transaction(STORE_NAME, 'readonly');
  const index = tx.objectStore(STORE_NAME).index('conversationId');
  const req = index.openCursor(IDBKeyRange.only(conversationId));
  let cursor = await new Promise(r => req.onsuccess = e => r(e.target.result));
  while (cursor) {
    yield cursor.value;
    cursor = await new Promise(r => cursor.continue() || r(null));
  }
}

export { saveBulkMessages, streamThread };
