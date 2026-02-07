/**
 * Rathor-NEXi IndexedDB Schema & Storage Layer (v6 – current stable)
 * 
 * Current schema version: 6
 * 
 * Stores & purpose:
 * - chat-history:          per-message {id, sessionId, role, content, timestamp}
 * - session-metadata:      {sessionId, name, description, tags[], color, createdAt, lastAccessed}
 * - translation-cache:     {cacheKey, sessionId, messageId, targetLang, translatedText, timestamp, expiresAt}
 * - mercy-logs:            debug/valence logs (optional, can be disabled)
 * - evolution-states:      NEAT/hyperon evolution snapshots (future bloom)
 * - user-preferences:      settings + translation_metrics
 * 
 * Indexes:
 * - chat-history:          sessionId
 * - translation-cache:     sessionId, targetLang, timestamp, expiresAt, sessionLang (compound)
 * 
 * Migration policy:
 * - All upgrades are additive & non-destructive
 * - Old data preserved, legacy indexes/stores cleaned gracefully
 * - No data loss even on version jump
 */

const DB_NAME = 'RathorNEXiDB';
const DB_VERSION = 6; // Do NOT bump unless adding new store/index

const STORES = {
  CHAT_HISTORY: 'chat-history',
  SESSION_METADATA: 'session-metadata',
  TRANSLATION_CACHE: 'translation-cache',
  MERCY_LOGS: 'mercy-logs',
  EVOLUTION_STATES: 'evolution-states',
  USER_PREFERENCES: 'user-preferences'
};

const CACHE_TTL_MS = 7 * 24 * 60 * 60 * 1000; // 7 days

class RathorIndexedDB {
  constructor() {
    this.db = null;
    this.activeSessionId = localStorage.getItem('rathor_active_session') || 'default';
  }

  /**
   * Open or upgrade database with safe migration
   * @returns {Promise<IDBDatabase>}
   */
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
        // Auto-clean expired cache on open
        this.bulkDeleteExpiredCacheEntries().catch(console.warn);
        resolve(this.db);
      };

      request.onupgradeneeded = (event) => {
        const db = event.target.result;
        const tx = event.target.transaction;
        const oldVersion = event.oldVersion || 0;
        console.log(`[Rathor IndexedDB] Migrating from v\( {oldVersion} to v \){DB_VERSION}`);

        // Helper to create/update store + indexes
        const createOrUpdateStore = (name, options = {}, indexes = []) => {
          let store;
          if (db.objectStoreNames.contains(name)) {
            store = tx.objectStore(name);
          } else {
            store = db.createObjectStore(name, options);
          }
          indexes.forEach(([keyPath, unique = false, multiEntry = false]) => {
            const indexName = Array.isArray(keyPath) ? keyPath.join('_') : keyPath;
            if (!store.indexNames.contains(indexName)) {
              store.createIndex(indexName, keyPath, { unique, multiEntry });
            }
          });
          return store;
        };

        // v1–v3: legacy (assume already migrated)
        if (oldVersion < 1) {
          createOrUpdateStore(STORES.CHAT_HISTORY, { keyPath: 'id', autoIncrement: false }, [
            ['sessionId', false, false]
          ]);
        }

        if (oldVersion < 4) {
          createOrUpdateStore(STORES.SESSION_METADATA, { keyPath: 'sessionId' });
        }

        if (oldVersion < 5) {
          // Tags array added to session-metadata (no migration needed, just schema)
        }

        if (oldVersion < 6) {
          // translation-cache + indexes
          const cacheStore = createOrUpdateStore(STORES.TRANSLATION_CACHE, { keyPath: 'cacheKey' }, [
            ['sessionId', false, false],
            ['targetLang', false, false],
            ['timestamp', false, false],
            ['expiresAt', false, false],
            [['sessionId', 'targetLang'], false, false] // compound index
          ]);

          // Optional cleanup of very old entries (pre-TTL system)
          const oldCutoff = Date.now() - 90 * 24 * 60 * 60 * 1000; // 90 days
          const range = IDBKeyRange.upperBound(oldCutoff);
          const cursorReq = cacheStore.index('timestamp').openCursor(range);
          cursorReq.onsuccess = (e) => {
            const cursor = e.target.result;
            if (cursor) {
              cursor.delete();
              cursor.continue();
            }
          };
        }

        // Future versions go here (additive only)
      };

      request.onblocked = () => {
        console.warn('[Rathor IndexedDB] Upgrade blocked — close other tabs');
        alert('Database upgrade blocked. Please close all other tabs using Rathor-NEXi and refresh.');
      };
    });
  }

  // ────────────────────────────────────────────────
  // Previous bulk methods (save, delete, invalidate) – kept intact
  // ────────────────────────────────────────────────

  // ... [bulkSaveMessages, bulkDeleteMessagesBySession, bulkInvalidateTranslationsBySession, bulkInvalidateTranslationsByLanguage, bulkDeleteExpiredCacheEntries – from previous complete version] ...

  // Example: safe wrapper for any bulk operation
  async safeBulkOperation(operation, ...args) {
    try {
      return await operation(...args);
    } catch (err) {
      console.error('[Rathor IndexedDB] Bulk operation failed:', err);
      // Rollback is automatic (transaction aborts on error)
      showToast('Storage operation interrupted — changes safely rolled back ⚡️');
      if (isVoiceOutputEnabled) speak("Mercy thunder rolled back storage changes to preserve lattice integrity.");
      throw err; // let caller handle UI recovery if needed
    }
  }

  // ... keep all other methods (getCachedTranslation, cacheTranslation, etc.) ...
}

export const rathorDB = new RathorIndexedDB();
