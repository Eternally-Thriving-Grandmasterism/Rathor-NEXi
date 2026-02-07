const DB_NAME = 'RathorNEXiDB';
const DB_VERSION = 2; // Increment this for schema changes — triggers onupgradeneeded

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
        console.error('[Rathor IndexedDB] Open failed:', event.target.error?.name || event.target.error);
        reject(event.target.error);
      };

      request.onsuccess = (event) => {
        this.db = event.target.result;
        this.db.onerror = (e) => console.error('[Rathor DB] Global error:', e.target.error);
        console.log('[Rathor IndexedDB] Opened v' + this.db.version);
        resolve(this.db);
      };

      request.onupgradeneeded = (event) => {
        const db = event.target.result;
        const oldVersion = event.oldVersion || 0;
        const transaction = event.target.transaction; // Version change tx — auto readwrite
        console.log(`[Rathor IndexedDB] Upgrading from v\( {oldVersion} to v \){DB_VERSION}`);

        // Conditional schema creation/updates — safe from any oldVersion
        const createOrUpdateStore = (name, options = {}, indexes = []) => {
          let store;
          if (db.objectStoreNames.contains(name)) {
            store = transaction.objectStore(name);
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
          // Initial schema (v1)
          createOrUpdateStore(STORES.CHAT_HISTORY, { keyPath: 'id', autoIncrement: true }, [
            ['timestamp', false],
            ['role', false]
          ]);
          createOrUpdateStore(STORES.MERCY_LOGS, { keyPath: 'id', autoIncrement: true }, [
            ['timestamp', false],
            ['valence', false]
          ]);
          createOrUpdateStore(STORES.EVOLUTION_STATES, { keyPath: 'bloomId' });
          createOrUpdateStore(STORES.USER_PREFERENCES, { keyPath: 'key' });
        }

        if (oldVersion < 2 && DB_VERSION >= 2) {
          // Example migration v2: Add new index or migrate data
          // e.g., if we add 'contentHash' index to chat-history
          const chatStore = transaction.objectStore(STORES.CHAT_HISTORY);
          if (!chatStore.indexNames.contains('contentHash')) {
            chatStore.createIndex('contentHash', 'contentHash', { unique: false });
          }

          // Example data migration: If old records had 'name' → split to 'firstName'/'lastName' (hypothetical)
          // Use cursor for transformation
          // const cursorReq = chatStore.openCursor();
          // cursorReq.onsuccess = (e) => {
          //   const cursor = e.target.result;
          //   if (cursor) {
          //     const value = cursor.value;
          //     if (value.name && !value.firstName) {
          //       const names = value.name.split(' ');
          //       value.firstName = names.shift() || '';
          //       value.lastName = names.join(' ');
          //       delete value.name;
          //       cursor.update(value);
          //     }
          //     cursor.continue();
          //   }
          // };
        }

        // Future versions: Add more if (oldVersion < X) blocks
      };

      request.onblocked = () => {
        console.warn('[Rathor IndexedDB] Upgrade blocked — close other tabs');
        alert('Please close other tabs/windows using Rathor to complete lattice upgrade.');
      };
    });
  }

  // ... rest of class (add, put, get, getAll, batchAdd, clear, addWithValence) as in previous full version ...
  // (Omit repeating them here for brevity — keep them identical in your overwrite)

  async migrateExampleData() {
    // Optional: Manual migration trigger if needed outside upgrade
    // But prefer doing in onupgradeneeded
  }
}

export const rathorDB = new RathorIndexedDB();
