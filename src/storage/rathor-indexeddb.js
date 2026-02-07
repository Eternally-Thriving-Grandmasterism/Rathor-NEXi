const DB_NAME = 'RathorNEXiDB';
const DB_VERSION = 4; // Bump to 4 for session-metadata store

const STORES = {
  CHAT_HISTORY: 'chat-history',
  SESSION_METADATA: 'session-metadata',
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
            ['role', false],
            ['sessionId', false]
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

        if (oldVersion < 4) {
          // v4: Add session-metadata store
          createOrUpdateStore(STORES.SESSION_METADATA, { keyPath: 'sessionId' }, [
            ['lastActive', false]
          ]);

          // Backfill metadata for existing sessions
          const sessions = new Set();
          const chatStore = tx.objectStore(STORES.CHAT_HISTORY);
          const chatCursor = chatStore.openCursor();
          chatCursor.onsuccess = (e) => {
            const cursor = e.target.result;
            if (cursor) {
              const value = cursor.value;
              if (value.sessionId) sessions.add(value.sessionId);
              cursor.continue();
            }
          };

          chatCursor.addEventListener('success', () => {
            const metaStore = tx.objectStore(STORES.SESSION_METADATA);
            sessions.forEach(sid => {
              metaStore.put({
                sessionId: sid,
                name: sid === 'default' ? 'Default Session' : `Session ${sid.slice(0,8)}`,
                createdAt: Date.now(),
                lastActive: Date.now(),
                description: '',
                colorTag: '#ffaa00'
              });
            });
          });
        }
      };

      request.onblocked = () => {
        console.warn('[Rathor IndexedDB] Upgrade blocked — close other tabs');
      };
    });
  }

  // ────────────────────────────────────────────────
  // Session Management with Metadata
  // ────────────────────────────────────────────────

  getActiveSessionId() {
    return this.activeSessionId;
  }

  async switchSession(sessionId) {
    if (!sessionId || typeof sessionId !== 'string') throw new Error('Invalid sessionId');
    this.activeSessionId = sessionId;
    localStorage.setItem('rathor_active_session', sessionId);
    // Update lastActive
    await this.updateSessionMetadata({ lastActive: Date.now() });
    console.log('[Rathor Session] Switched to:', sessionId);
  }

  async listSessions() {
    return this._transaction(STORES.SESSION_METADATA, 'readonly', (tx) => {
      const store = tx.objectStore(STORES.SESSION_METADATA);
      const req = store.openCursor();
      const sessions = [];
      return new Promise((resolve, reject) => {
        req.onsuccess = (event) => {
          const cursor = event.target.result;
          if (cursor) {
            sessions.push(cursor.value);
            cursor.continue();
          } else {
            resolve(sessions);
          }
        };
        req.onerror = () => reject(req.error);
      });
    });
  }

  async createSession(name = null) {
    const sessionId = name?.trim() || `Session-${Date.now()}`;
    const metadata = {
      sessionId,
      name: name?.trim() || `Session ${sessionId.slice(0,8)}`,
      createdAt: Date.now(),
      lastActive: Date.now(),
      description: '',
      colorTag: '#ffaa00' // default thunder gold
    };
    await this._transaction(STORES.SESSION_METADATA, 'readwrite', (tx) => {
      tx.objectStore(STORES.SESSION_METADATA).put(metadata);
    });
    await this.switchSession(sessionId);
    return sessionId;
  }

  async updateSessionMetadata(updates) {
    const current = await this.getSessionMetadata(this.activeSessionId);
    if (!current) return;
    const updated = { ...current, ...updates, lastActive: Date.now() };
    await this._transaction(STORES.SESSION_METADATA, 'readwrite', (tx) => {
      tx.objectStore(STORES.SESSION_METADATA).put(updated);
    });
  }

  async getSessionMetadata(sessionId) {
    return this._transaction(STORES.SESSION_METADATA, 'readonly', (tx) => {
      const store = tx.objectStore(STORES.SESSION_METADATA);
      const req = store.get(sessionId);
      return new Promise((res, rej) => {
        req.onsuccess = () => res(req.result);
        req.onerror = () => rej(req.error);
      });
    });
  }

  // ────────────────────────────────────────────────
  // Enhanced saveMessage with metadata touch
  // ────────────────────────────────────────────────

  async saveMessage(message) {
    const enhanced = {
      ...message,
      sessionId: this.activeSessionId,
      timestamp: message.timestamp || Date.now()
    };

    await this._transaction(STORES.CHAT_HISTORY, 'readwrite', (tx) => {
      tx.objectStore(STORES.CHAT_HISTORY).add(enhanced);
    });

    // Touch metadata lastActive
    await this.updateSessionMetadata({});
  }

  // ... keep prior loadHistory, clearSessionHistory, _transaction, etc. unchanged ...
}

export const rathorDB = new RathorIndexedDB();            }
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
