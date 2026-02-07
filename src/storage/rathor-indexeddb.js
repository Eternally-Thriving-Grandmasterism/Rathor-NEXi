const DB_NAME = 'RathorNEXiDB';
const DB_VERSION = 3; // Bump to 3 for multi-session index + backfill support

const STORES = {
  CHAT_HISTORY: 'chat-history',
  MERCY_LOGS: 'mercy-logs',
  EVOLUTION_STATES: 'evolution-states',
  USER_PREFERENCES: 'user-preferences'
};

class RathorIndexedDB {
  constructor() {
    this.db = null;
    this.activeSessionId = localStorage.getItem('rathor_active_session') || 'default';
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
        console.log('[Rathor IndexedDB] Opened v' + this.db.version);
        resolve(this.db);
      };

      request.onupgradeneeded = (event) => {
        const db = event.target.result;
        const tx = event.target.transaction;
        const oldVersion = event.oldVersion || 0;
        console.log(`[Rathor IndexedDB] Upgrading from v\( {oldVersion} to v \){DB_VERSION}`);

        const createOrUpdateStore = (name, options = {}, indexes = []) => {
          let store;
          if (db.objectStoreNames.contains(name)) {
            store = tx.objectStore(name);
          } else {
            store = db.createObjectStore(name, options);
          }
          indexes.forEach(([keyPath, unique = false]) => {
            if (!store.indexNames.contains(keyPath)) {
              store.createIndex(keyPath, keyPath, { unique });
            }
          });
          return store;
        };

        if (oldVersion < 1) {
          createOrUpdateStore(STORES.CHAT_HISTORY, { keyPath: 'id', autoIncrement: true }, [
            ['timestamp', false],
            ['role', false]
          ]);
          // ... other initial stores
        }

        if (oldVersion < 2) {
          const chatStore = tx.objectStore(STORES.CHAT_HISTORY);
          if (!chatStore.indexNames.contains('sessionId')) {
            chatStore.createIndex('sessionId', 'sessionId', { unique: false });
          }
        }

        if (oldVersion < 3) {
          // v3: Ensure composite-like query support + backfill legacy messages to 'default'
          const chatStore = tx.objectStore(STORES.CHAT_HISTORY);
          const cursorReq = chatStore.openCursor();

          cursorReq.onsuccess = (e) => {
            const cursor = e.target.result;
            if (cursor) {
              const value = cursor.value;
              if (!value.sessionId) {
                value.sessionId = 'default';
                cursor.update(value);
              }
              cursor.continue();
            }
          };
        }
      };

      request.onblocked = () => {
        console.warn('[Rathor IndexedDB] Upgrade blocked — close other tabs');
      };
    });
  }

  // ────────────────────────────────────────────────
  // Session Management
  // ────────────────────────────────────────────────

  getActiveSessionId() {
    return this.activeSessionId;
  }

  async switchSession(sessionId) {
    if (!sessionId || typeof sessionId !== 'string') throw new Error('Invalid sessionId');
    this.activeSessionId = sessionId;
    localStorage.setItem('rathor_active_session', sessionId);
    console.log('[Rathor Session] Switched to:', sessionId);
  }

  async listSessions() {
    const allMessages = await this.getAll(STORES.CHAT_HISTORY, 'sessionId');
    const sessions = new Set(allMessages.map(m => m.sessionId || 'default'));
    return Array.from(sessions);
  }

  async createSession(sessionId) {
    // Just switch — no need to pre-create (lazy creation)
    await this.switchSession(sessionId);
  }

  // ────────────────────────────────────────────────
  // Chat Persistence with Session Isolation
  // ────────────────────────────────────────────────

  async saveMessage(message) {
    const enhanced = {
      ...message,
      sessionId: this.activeSessionId,
      timestamp: message.timestamp || Date.now()
    };

    return this._transaction(STORES.CHAT_HISTORY, 'readwrite', (tx) => {
      const store = tx.objectStore(STORES.CHAT_HISTORY);
      const req = store.add(enhanced);
      return new Promise((res, rej) => {
        req.onsuccess = () => res(req.result);
        req.onerror = () => rej(req.error);
      });
    });
  }

  async loadHistory(limit = 100) {
    const sessionId = this.activeSessionId;
    return this._transaction(STORES.CHAT_HISTORY, 'readonly', (tx) => {
      const store = tx.objectStore(STORES.CHAT_HISTORY);
      const index = store.index('timestamp');
      const req = index.openCursor(null, 'prev'); // newest first

      const messages = [];
      return new Promise((resolve, reject) => {
        let count = 0;
        req.onsuccess = (event) => {
          const cursor = event.target.result;
          if (cursor && count < limit) {
            if (cursor.value.sessionId === sessionId) {
              messages.push(cursor.value);
              count++;
            }
            cursor.continue();
          } else {
            resolve(messages.reverse()); // oldest first for UI
          }
        };
        req.onerror = () => reject(req.error);
      });
    });
  }

  async clearSessionHistory(sessionId = null) {
    const target = sessionId || this.activeSessionId;
    return this._transaction(STORES.CHAT_HISTORY, 'readwrite', (tx) => {
      const store = tx.objectStore(STORES.CHAT_HISTORY);
      const index = store.index('sessionId');
      const req = index.openCursor(IDBKeyRange.only(target));
      req.onsuccess = (event) => {
        const cursor = event.target.result;
        if (cursor) {
          cursor.delete();
          cursor.continue();
        }
      };
    });
  }

  // ... keep prior _transaction, add/put/get/getAll/clear/addWithValence methods unchanged ...
}

export const rathorDB = new RathorIndexedDB();
