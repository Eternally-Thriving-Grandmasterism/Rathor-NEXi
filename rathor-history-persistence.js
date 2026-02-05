// rathor-history-persistence.js – sovereign IndexedDB chat/valence history v1
// Persistent across sessions, mercy-gated load/save, conversation threading stub
// MIT License – Autonomicity Games Inc. 2026

const DB_NAME = 'RathorEternalLattice';
const DB_VERSION = 1;
const STORE_NAME = 'chatHistory';
let db = null;

async function openDB() {
  if (db) return db;
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(DB_NAME, DB_VERSION);
    request.onerror = () => reject(request.error);
    request.onsuccess = () => {
      db = request.result;
      resolve(db);
    };
    request.onupgradeneeded = (event) => {
      const upgradeDb = event.target.result;
      if (!upgradeDb.objectStoreNames.contains(STORE_NAME)) {
        const store = upgradeDb.createObjectStore(STORE_NAME, { keyPath: 'id', autoIncrement: true });
        store.createIndex('timestamp', 'timestamp', { unique: false });
        store.createIndex('sessionId', 'sessionId', { unique: false });
      }
    };
  });
}

async function saveMessage(sessionId = 'default', role, content, valence = 0.9999999) {
  const dbInstance = await openDB();
  return new Promise((resolve, reject) => {
    const tx = dbInstance.transaction(STORE_NAME, 'readwrite');
    const store = tx.objectStore(STORE_NAME);
    const entry = {
      sessionId,
      role,
      content,
      valence,
      timestamp: Date.now()
    };
    const req = store.add(entry);
    req.onsuccess = () => resolve(entry);
    req.onerror = () => reject(req.error);
    tx.oncomplete = () => console.log("[History] Message persisted eternally");
  });
}

async function loadHistory(sessionId = 'default', limit = 50) {
  const dbInstance = await openDB();
  return new Promise((resolve, reject) => {
    const tx = dbInstance.transaction(STORE_NAME, 'readonly');
    const store = tx.objectStore(STORE_NAME);
    const index = store.index('timestamp');
    const req = index.openCursor(IDBKeyRange.lowerBound(0), 'prev'); // newest first
    const results = [];
    let count = 0;
    req.onsuccess = (event) => {
      const cursor = event.target.result;
      if (cursor && count < limit) {
        if (cursor.value.sessionId === sessionId) {
          results.push(cursor.value);
          count++;
        }
        cursor.continue();
      } else {
        resolve(results.reverse()); // oldest to newest
      }
    };
    req.onerror = () => reject(req.error);
  });
}

async function clearHistory(sessionId = 'default') {
  const dbInstance = await openDB();
  return new Promise((resolve, reject) => {
    const tx = dbInstance.transaction(STORE_NAME, 'readwrite');
    const store = tx.objectStore(STORE_NAME);
    const req = store.delete(IDBKeyRange.only(sessionId)); // simplistic; expand for range
    req.onsuccess = () => resolve();
    req.onerror = () => reject(req.error);
  });
}

export { saveMessage, loadHistory, clearHistory };
