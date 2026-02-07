// js/common.js — Shared utilities & i18n across all Rathor-NEXi pages

const registryUrl = '/locales/languages.json';

// ────────────────────────────────────────────────
// i18n Initialization & Language Management
// ────────────────────────────────────────────────

async function initI18n() {
  await i18next.init({
    lng: localStorage.getItem('rathor_lang') || getBestLanguage(),
    fallbackLng: 'en',
    debug: false,
    ns: 'translation',
    defaultNS: 'translation',
    interpolation: { escapeValue: false }
  });

  // Load language registry once
  const registryResp = await fetch(registryUrl);
  const registry = await registryResp.json();

  // Add empty bundles for all languages (actual content loaded on demand)
  registry.languages.forEach(lang => {
    i18next.addResourceBundle(lang.code, 'translation', {}, true, true);
  });

  // Load initial language content
  await loadLanguage(i18next.language);
  updateContent();
  applyRTL(i18next.language);
}

// Load specific language JSON dynamically + cache it
async function loadLanguage(lng) {
  if (i18next.services.resourceStore.hasResourceBundle(lng, 'translation') && 
      Object.keys(i18next.services.resourceStore.data[lng]?.translation || {}).length > 0) {
    return;
  }

  const cache = await caches.open('rathor-locales');
  const cached = await cache.match(`/locales/${lng}.json`);

  if (cached) {
    const json = await cached.json();
    i18next.addResourceBundle(lng, 'translation', json, true, true);
    return;
  }

  try {
    const response = await fetch(`/locales/${lng}.json`);
    if (!response.ok) throw new Error(`Locale not found: ${lng}`);
    const json = await response.json();

    await cache.put(`/locales/${lng}.json`, new Response(JSON.stringify(json), {
      headers: { 'Content-Type': 'application/json' }
    }));

    i18next.addResourceBundle(lng, 'translation', json, true, true);
  } catch (err) {
    console.warn(`Failed to load ${lng}, falling back to en`, err);
    if (lng !== 'en') await loadLanguage('en');
  }
}

// Change language + update UI + RTL + toast
async function changeLanguage(lng) {
  await loadLanguage(lng);
  i18next.changeLanguage(lng, () => {
    updateContent();
    applyRTL(lng);
    localStorage.setItem('rathor_lang', lng);
    showToast(`Language switched to \( {i18next.t(`language. \){lng}`)} ⚡️`);
  });
}

// Apply RTL layout for right-to-left languages
function applyRTL(lng) {
  const rtlLanguages = ['ar', 'he', 'fa', 'ur'];
  const isRTL = rtlLanguages.includes(lng);
  document.body.classList.toggle('rtl', isRTL);
  document.documentElement.setAttribute('dir', isRTL ? 'rtl' : 'ltr');
}

// Update all translatable elements on page
function updateContent() {
  document.querySelectorAll('[data-i18n]').forEach(el => {
    const key = el.getAttribute('data-i18n');
    el.innerHTML = i18next.t(key);
  });

  // Dynamic placeholders
  const chatInput = document.getElementById('chat-input');
  if (chatInput) chatInput.placeholder = i18next.t('placeholders.chatInput');

  const sessionSearch = document.getElementById('session-search');
  if (sessionSearch) sessionSearch.placeholder = i18next.t('placeholders.sessionSearch');

  const langSearch = document.getElementById('lang-search-input');
  if (langSearch) langSearch.placeholder = i18next.t('placeholders.languageSearch');

  // ... add more dynamic updates as needed ...
}

// Best initial language from browser preferences
function getBestLanguage() {
  const preferred = navigator.languages || [navigator.language];
  const supported = ['en','ar','es','fr','de','nl','it','pt','ru','ja','zh','hi','sw','id','tr','ko' /* extend later */];
  for (const lang of preferred) {
    const short = lang.split('-')[0].toLowerCase();
    if (supported.includes(short)) return short;
  }
  return 'en';
}

// Shared toast utility
function showToast(message) {
  const toast = document.createElement('div');
  toast.textContent = message;
  toast.style.cssText = `
    position: fixed; bottom: 80px; left: 50%; transform: translateX(-50%);
    background: var(--thunder-gold); color: #000; padding: 12px 24px;
    border-radius: 8px; box-shadow: 0 4px 20px rgba(255,170,0,0.4);
    z-index: 4000; font-weight: 600; white-space: pre-wrap; max-width: 90%;
  `;
  document.body.appendChild(toast);
  setTimeout(() => toast.remove(), 4000);
}

// Export shared utilities
window.rathorCommon = {
  initI18n,
  loadLanguage,
  changeLanguage,
  applyRTL,
  updateContent,
  getBestLanguage,
  showToast
};
