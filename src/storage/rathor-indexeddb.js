// src/storage/rathor-indexeddb.js — Optimized, versioned, migratable IndexedDB wrapper

const DB_NAME = 'rathor-indexeddb';
const DB_VERSION = 4; // current version — bump when adding breaking changes

const STORES = {
  sessions: 'sessions',
  messages: 'messages',
  tags: 'tags',
  translationCache: 'translationCache'
};

let db = null;

const dbPromise = new Promise((resolve, reject) => {
  const request = indexedDB.open(DB_NAME, DB_VERSION);

  request.onupgradeneeded = event => {
    db = event.target.result;
    const oldVersion = event.oldVersion;
    console.log(`[rathorDB] Migrating from v\( {oldVersion} to v \){DB_VERSION}`);

    // v1: initial schema
    if (oldVersion < 1) {
      db.createObjectStore(STORES.sessions, { keyPath: 'id' });
      db.createObjectStore(STORES.messages, { autoIncrement: true });
      db.createObjectStore(STORES.tags, { keyPath: 'id' });
      db.createObjectStore(STORES.translationCache, { keyPath: 'key' });
      console.log('[rathorDB] Created initial stores (v1)');
    }

    // v2: add indexes for performance
    if (oldVersion < 2) {
      const msgStore = db.transaction(STORES.messages, 'readwrite').objectStore(STORES.messages);
      if (!msgStore.indexNames.contains('sessionId')) msgStore.createIndex('sessionId', 'sessionId');
      if (!msgStore.indexNames.contains('timestamp')) msgStore.createIndex('timestamp', 'timestamp');
      if (!msgStore.indexNames.contains('role')) msgStore.createIndex('role', 'role');
      console.log('[rathorDB] Added indexes on messages (v2)');
    }

    // v3: add compression flag & quota tracking stub
    if (oldVersion < 3) {
      // Future: add 'compressed' field to messages if needed
      console.log('[rathorDB] Schema v3 ready (compression stub)');
    }

    // v4: add cleanup index or quota metadata
    if (oldVersion < 4) {
      // Future: add 'ttl' or 'lastAccessed' field
      console.log('[rathorDB] Schema v4 ready (cleanup & quota)');
    }
  };

  request.onsuccess = event => {
    db = event.target.result;
    db.onerror = err => console.error('[rathorDB] IndexedDB error:', err);
    console.log(`[rathorDB] Opened v${DB_VERSION}`);
    resolve(db);
  };

  request.onerror = event => {
    console.error('[rathorDB] Open failed:', event.target.error);
    reject(event.target.error);
  };
});

async function openDB() {
  if (db) return db;
  db = await dbPromise;
  return db;
}

// ────────────────────────────────────────────────
// Sessions CRUD
// ────────────────────────────────────────────────

export async function saveSession(session) {
  const db = await openDB();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORES.sessions, 'readwrite');
    const store = tx.objectStore(STORES.sessions);
    const req = store.put(session);
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}

export async function getSession(id) {
  const db = await openDB();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORES.sessions);
    const store = tx.objectStore(STORES.sessions);
    const req = store.get(id);
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}

export async function getAllSessions() {
  const db = await openDB();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORES.sessions);
    const store = tx.objectStore(STORES.sessions);
    const req = store.getAll();
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}

// ────────────────────────────────────────────────
// Messages CRUD — batched, paginated, indexed
// ────────────────────────────────────────────────

export async function saveMessages(sessionId, messages) {
  const db = await openDB();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORES.messages, 'readwrite');
    const store = tx.objectStore(STORES.messages);
    let count = 0;
    messages.forEach(msg => {
      msg.sessionId = sessionId;
      msg.timestamp = Date.now();
      const req = store.add(msg);
      req.onsuccess = () => { count++; if (count === messages.length) resolve(); };
      req.onerror = () => reject(req.error);
    });
    tx.oncomplete = resolve;
    tx.onerror = reject;
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

    req.onsuccess = event => {
      const cursor = event.target.result;
      if (!cursor) return resolve(results);

      if (skipped < offset) {
        skipped++;
        cursor.continue();
      } else if (results.length < limit) {
        results.push(cursor.value);
        cursor.continue();
      } else {
        resolve(results);
      }
    };

    req.onerror = () => reject(req.error);
  });
}

// ────────────────────────────────────────────────
// Cleanup & Quota Management
// ────────────────────────────────────────────────

export async function clearExpiredCache(days = 30) {
  const db = await openDB();
  const cutoff = Date.now() - days * 24 * 60 * 60 * 1000;
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORES.messages, 'readwrite');
    const store = tx.objectStore(STORES.messages);
    const index = store.index('timestamp');
    const req = index.openCursor(IDBKeyRange.upperBound(cutoff));
    req.onsuccess = e => {
      const cursor = e.target.result;
      if (cursor) {
        cursor.delete();
        cursor.continue();
      } else {
        resolve();
      }
    };
    req.onerror = () => reject(req.error);
  });
}

export async function getStorageUsage() {
  if (!navigator.storage?.estimate) return { usage: 0, quota: 0 };
  const estimate = await navigator.storage.estimate();
  return {
    usage: estimate.usage,
    quota: estimate.quota,
    percentUsed: (estimate.usage / estimate.quota) * 100
  };
}

// Export
export default {
  openDB,
  saveSession,
  getSession,
  getAllSessions,
  saveMessages,
  getMessages,
  clearExpiredCache,
  getStorageUsage
};  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORES.sessions);
    const store = tx.objectStore(STORES.sessions);
    const req = store.getAll();
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}

// ────────────────────────────────────────────────
// Messages CRUD (batched + paginated)
// ────────────────────────────────────────────────

export async function saveMessages(sessionId, messages) {
  const db = await openDB();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORES.messages, 'readwrite');
    const store = tx.objectStore(STORES.messages);
    let count = 0;
    messages.forEach(msg => {
      msg.sessionId = sessionId;
      msg.timestamp = Date.now();
      const req = store.add(msg);
      req.onsuccess = () => { count++; if (count === messages.length) resolve(); };
      req.onerror = () => reject(req.error);
    });
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

    req.onsuccess = event => {
      const cursor = event.target.result;
      if (!cursor) return resolve(results);

      if (skipped < offset) {
        skipped++;
        cursor.continue();
      } else if (results.length < limit) {
        results.push(cursor.value);
        cursor.continue();
      } else {
        resolve(results);
      }
    };

    req.onerror = () => reject(req.error);
  });
}

// ────────────────────────────────────────────────
// Utility: Clear old data (quota management)
// ────────────────────────────────────────────────

export async function clearExpiredCache(days = 30) {
  const db = await openDB();
  const cutoff = Date.now() - days * 24 * 60 * 60 * 1000;
  return new Promise((resolve, reject) => {
    const tx = db.transaction([STORES.messages, STORES.sessions], 'readwrite');
    const msgStore = tx.objectStore(STORES.messages);
    const msgIndex = msgStore.index('timestamp');
    const req = msgIndex.openCursor(IDBKeyRange.upperBound(cutoff));
    req.onsuccess = e => {
      const cursor = e.target.result;
      if (cursor) {
        cursor.delete();
        cursor.continue();
      }
    };
    tx.oncomplete = resolve;
    tx.onerror = reject;
  });
}
