const DB_NAME = 'RathorNEXiDB';
const DB_VERSION = 6; // Already v6 – no bump needed

const STORES = {
  CHAT_HISTORY: 'chat-history',
  SESSION_METADATA: 'session-metadata',
  TRANSLATION_CACHE: 'translation-cache',
  MERCY_LOGS: 'mercy-logs',
  EVOLUTION_STATES: 'evolution-states',
  USER_PREFERENCES: 'user-preferences'
};

// Cache expiration: 7 days default
const CACHE_TTL_MS = 7 * 24 * 60 * 60 * 1000;

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
        // ... previous migrations kept unchanged ...
        // v6 already has translation_cache – no new schema change needed
      };

      request.onblocked = () => {
        console.warn('[Rathor IndexedDB] Upgrade blocked — close other tabs');
      };
    });
  }

  // ────────────────────────────────────────────────
  // Translation Cache with Invalidation & Expiration
  // ────────────────────────────────────────────────

  async getCachedTranslation(sessionId, messageId, targetLang) {
    const cacheKey = `\( {sessionId}_ \){messageId}_${targetLang}`;
    const entry = await this._transaction(STORES.TRANSLATION_CACHE, 'readonly', (tx) => {
      const store = tx.objectStore(STORES.TRANSLATION_CACHE);
      return store.get(cacheKey);
    });

    if (entry) {
      // Check expiration
      if (entry.expiresAt && entry.expiresAt < Date.now()) {
        await this.invalidateTranslationCache(cacheKey);
        return null;
      }
      return entry.translatedText;
    }
    return null;
  }

  async cacheTranslation(sessionId, messageId, targetLang, translatedText) {
    const cacheKey = `\( {sessionId}_ \){messageId}_${targetLang}`;
    const entry = {
      cacheKey,
      sessionId,
      messageId,
      targetLang,
      translatedText,
      timestamp: Date.now(),
      expiresAt: Date.now() + CACHE_TTL_MS // 7 days expiration
    };

    await this._transaction(STORES.TRANSLATION_CACHE, 'readwrite', (tx) => {
      tx.objectStore(STORES.TRANSLATION_CACHE).put(entry);
    });
  }

  async invalidateTranslationCache(cacheKeyOrSessionId, targetLang = null) {
    await this._transaction(STORES.TRANSLATION_CACHE, 'readwrite', (tx) => {
      const store = tx.objectStore(STORES.TRANSLATION_CACHE);

      if (targetLang) {
        // Specific lang invalidation
        const key = `\( {cacheKeyOrSessionId}_ \){targetLang}`;
        store.delete(key);
      } else if (cacheKeyOrSessionId.includes('_')) {
        // Single cache key
        store.delete(cacheKeyOrSessionId);
      } else {
        // All for sessionId
        const index = store.index('sessionId');
        const req = index.openCursor(IDBKeyRange.only(cacheKeyOrSessionId));
        req.onsuccess = (event) => {
          const cursor = event.target.result;
          if (cursor) {
            cursor.delete();
            cursor.continue();
          }
        };
      }
    });
  }

  async invalidateAllForLanguage(targetLang) {
    await this._transaction(STORES.TRANSLATION_CACHE, 'readwrite', (tx) => {
      const store = tx.objectStore(STORES.TRANSLATION_CACHE);
      const index = store.index('targetLang');
      const req = index.openCursor(IDBKeyRange.only(targetLang));
      req.onsuccess = (event) => {
        const cursor = event.target.result;
        if (cursor) {
          cursor.delete();
          cursor.continue();
        }
      };
    });
  }

  async clearExpiredCache() {
    const now = Date.now();
    await this._transaction(STORES.TRANSLATION_CACHE, 'readwrite', (tx) => {
      const store = tx.objectStore(STORES.TRANSLATION_CACHE);
      const index = store.index('expiresAt');
      const req = index.openCursor(IDBKeyRange.upperBound(now));
      req.onsuccess = (event) => {
        const cursor = event.target.result;
        if (cursor) {
          cursor.delete();
          cursor.continue();
        }
      };
    });
  }

  // ... keep all previous methods (createSession, updateSessionMetadata, saveMessage, loadHistory, etc.) ...
}

export const rathorDB = new RathorIndexedDB();
