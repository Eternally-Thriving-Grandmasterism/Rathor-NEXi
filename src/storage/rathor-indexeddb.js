const DB_NAME = 'RathorNEXiDB';
const DB_VERSION = 1; // Bump for schema changes — onupgradeneeded triggers
const STORES = {
  CHAT_HISTORY: 'chat-history',
  MERCY_LOGS: 'mercy-logs',
  EVOLUTION_STATES: 'evolution-states',
  USER_PREFERENCES: 'user-preferences'
};

class RathorIndexedDB {
  constructor() {
    this.db = null;
  }

  async open() {
    if (this.db) return this.db;

    return new Promise((resolve, reject) => {
      const request = indexedDB.open(DB_NAME, DB_VERSION);

      request.onerror = (event) => {
        console.error('[Rathor IndexedDB] Open failed:', event.target.error);
        reject(event.target.error);
      };

      request.onsuccess = (event) => {
        this.db = event.target.result;
        console.log('[Rathor IndexedDB] Opened successfully');
        resolve(this.db);
      };

      request.onupgradeneeded = (event) => {
        const db = event.target.result;
        console.log('[Rathor IndexedDB] Upgrading schema to v', DB_VERSION);

        // Create stores if not exist
        if (!db.objectStoreNames.contains(STORES.CHAT_HISTORY)) {
          const store = db.createObjectStore(STORES.CHAT_HISTORY, { keyPath: 'id', autoIncrement: true });
          store.createIndex('timestamp', 'timestamp', { unique: false });
          store.createIndex('role', 'role', { unique: false });
        }

        if (!db.objectStoreNames.contains(STORES.MERCY_LOGS)) {
          const store = db.createObjectStore(STORES.MERCY_LOGS, { keyPath: 'id', autoIncrement: true });
          store.createIndex('timestamp', 'timestamp');
          store.createIndex('valence', 'valence');
        }

        if (!db.objectStoreNames.contains(STORES.EVOLUTION_STATES)) {
          db.createObjectStore(STORES.EVOLUTION_STATES, { keyPath: 'bloomId' });
        }

        if (!db.objectStoreNames.contains(STORES.USER_PREFERENCES)) {
          db.createObjectStore(STORES.USER_PREFERENCES, { keyPath: 'key' });
        }
      };
    });
  }

  async add(storeName, data) {
    const db = await this.open();
    return new Promise((resolve, reject) => {
      const tx = db.transaction(storeName, 'readwrite');
      const store = tx.objectStore(storeName);
      const request = store.add({ ...data, timestamp: Date.now() });

      request.onsuccess = () => resolve(request.result);
      request.onerror = () => reject(request.error);
      tx.oncomplete = () => console.log(`[Rathor DB] Added to ${storeName}`);
      tx.onerror = (e) => reject(e.target.error);
    });
  }

  async getAll(storeName, query = null, direction = 'prev') {
    const db = await this.open();
    return new Promise((resolve, reject) => {
      const tx = db.transaction(storeName, 'readonly');
      const store = tx.objectStore(storeName);
      let request;

      if (query) {
        const index = store.index(query.index || 'timestamp');
        request = index.openCursor(query.range || IDBKeyRange.lowerBound(0), direction);
      } else {
        request = store.openCursor(null, direction);
      }

      const results = [];
      request.onsuccess = (event) => {
        const cursor = event.target.result;
        if (cursor) {
          results.push(cursor.value);
          cursor.continue();
        } else {
          resolve(results);
        }
      };
      request.onerror = () => reject(request.error);
    });
  }

  async getLatestChat(limit = 50) {
    const history = await this.getAll(STORES.CHAT_HISTORY, { index: 'timestamp' }, 'prev');
    return history.slice(0, limit).reverse(); // Latest first
  }

  async clearStore(storeName) {
    const db = await this.open();
    return new Promise((resolve, reject) => {
      const tx = db.transaction(storeName, 'readwrite');
      const store = tx.objectStore(storeName);
      const request = store.clear();
      request.onsuccess = resolve;
      request.onerror = reject;
    });
  }

  // Mercy valence gate example — reject low-valence writes
  async addWithValenceCheck(storeName, data) {
    if (storeName === STORES.MERCY_LOGS && (data.valence ?? 0) < 0.999) {
      console.warn('[Rathor DB] Valence too low — write blocked');
      throw new Error('Mercy valence threshold not met');
    }
    return this.add(storeName, data);
  }
}

export const rathorDB = new RathorIndexedDB();

// Usage example (in chat init):
// await rathorDB.open();
// const history = await rathorDB.getLatestChat();
// await rathorDB.add(STORES.CHAT_HISTORY, { role: 'user', content: 'Hello thunder' });
