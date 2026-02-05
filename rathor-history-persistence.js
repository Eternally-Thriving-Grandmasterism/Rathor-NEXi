// rathor-history-persistence.js – v2 with threading, error handling, retry
// MIT License – Autonomicity Games Inc. 2026

const DB_NAME = 'RathorEternalLattice';
const DB_VERSION = 2; // bumped for schema
const STORE_NAME = 'chatHistory';
let db = null;

async function openDB() {
  if (db) return db;
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(DB_NAME, DB_VERSION);
    request.onerror = (e) => {
      console.error("[History] DB open error:", e);
      reject(e.target.error);
    };
    request.onsuccess = () => {
      db = request.result;
      resolve(db);
    };
    request.onupgradeneeded = (event) => {
      const upgradeDb = event.target.result;
      let store;
      if (!upgradeDb.objectStoreNames.contains(STORE_NAME)) {
        store = upgradeDb.createObjectStore(STORE_NAME, { keyPath: 'id', autoIncrement: true });
      } else {
        store = event.target.transaction.objectStore(STORE_NAME);
      }
      if (!store.indexNames.contains('timestamp')) store.createIndex('timestamp', 'timestamp', { unique: false });
      if (!store.indexNames.contains('sessionId')) store.createIndex('sessionId', 'sessionId', { unique: false });
      if (!store.indexNames.contains('conversationId')) store.createIndex('conversationId', 'conversationId', { unique: false });
      if (!store.indexNames.contains('parentId')) store.createIndex('parentId', 'parentId', { unique: false });
    };
  });
}

async function saveMessage({ sessionId = 'default', conversationId = SESSION_ID, parentId = null, role, content, valence = 0.9999999 }) {
  const dbInstance = await openDB().catch(() => null);
  if (!dbInstance) return console.warn("[History] Save skipped – DB unavailable");
  return new Promise((resolve, reject) => {
    const tx = dbInstance.transaction(STORE_NAME, 'readwrite');
    const store = tx.objectStore(STORE_NAME);
    const entry = {
      sessionId,
      conversationId,
      parentId,
      role,
      content,
      valence,
      timestamp: Date.now()
    };
    const req = store.add(entry);
    req.onsuccess = () => resolve(entry);
    req.onerror = (e) => {
      console.error("[History] Save error:", e);
      reject(e.target.error);
    };
    tx.onerror = (e) => console.error("[History] TX error:", e);
  }).catch((err) => {
    console.warn("[History] Retry save after error...");
    // Simple retry once
    return saveMessage({ sessionId, conversationId, parentId, role, content, valence });
  });
}

async function loadThread(conversationId = SESSION_ID, limit = 100) {
  const dbInstance = await openDB().catch(() => null);
  if (!dbInstance) return [];
  return new Promise((resolve, reject) => {
    const tx = dbInstance.transaction(STORE_NAME, 'readonly');
    const store = tx.objectStore(STORE_NAME);
    const index = store.index('conversationId');
    const req = index.openCursor(IDBKeyRange.only(conversationId));
    const results = [];
    req.onsuccess = (event) => {
      const cursor = event.target.result;
      if (cursor) {
        results.push(cursor.value);
        cursor.continue();
      } else {
        // Sort by timestamp, build tree if needed (client-side for now)
        results.sort((a, b) => a.timestamp - b.timestamp);
        resolve(results.slice(-limit));
      }
    };
    req.onerror = (e) => reject(e.target.error);
  }).catch((err) => {
    console.error("[History] Load thread error:", err);
    return [];
  });
}

async function loadRepliesTo(parentId) {
  // ... similar cursor on parentId index
  // stub for full threading UI
}

export { saveMessage, loadThread, loadRepliesTo };
