// js/common.js — Shared across all pages

const registryUrl = '/locales/languages.json';

async function loadLanguageRegistry() {
  const res = await fetch(registryUrl);
  return await res.json();
}

async function initI18n() {
  await i18next.init({
    lng: localStorage.getItem('rathor_lang') || getBestLanguage(),
    fallbackLng: 'en',
    debug: false
  });

  const registry = await loadLanguageRegistry();
  registry.languages.forEach(lang => {
    i18next.addResourceBundle(lang.code, 'translation', {}, true, true);
  });

  await loadLanguage(i18next.language);
  updateContent();
}

async function loadLanguage(lng) {
  if (i18next.services.resourceStore.hasResourceBundle(lng, 'translation') && Object.keys(i18next.services.resourceStore.data[lng].translation || {}).length > 0) {
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
    if (!response.ok) throw new Error('Locale not found');
    const json = await response.json();

    cache.put(`/locales/${lng}.json`, new Response(JSON.stringify(json), {
      headers: { 'Content-Type': 'application/json' }
    }));

    i18next.addResourceBundle(lng, 'translation', json, true, true);
  } catch (err) {
    console.warn(`Failed to load ${lng}, falling back to en`, err);
    if (lng !== 'en') await loadLanguage('en');
  }
}

async function changeLanguage(lng) {
  await loadLanguage(lng);
  i18next.changeLanguage(lng, () => {
    updateContent();
    applyRTL(lng);
    localStorage.setItem('rathor_lang', lng);
    showToast(`Language switched to \( {i18next.t(`language. \){lng}`)} ⚡️`);
  });
}

function applyRTL(lng) {
  const rtlLanguages = ['ar', 'he', 'fa', 'ur'];
  const isRTL = rtlLanguages.includes(lng);
  document.body.classList.toggle('rtl', isRTL);
  document.documentElement.setAttribute('dir', isRTL ? 'rtl' : 'ltr');
}

function updateContent() {
  document.querySelectorAll('[data-i18n]').forEach(el => {
    const key = el.getAttribute('data-i18n');
    el.innerHTML = i18next.t(key);
  });

  document.getElementById('chat-input').placeholder = i18next.t('placeholders.chatInput');
  document.getElementById('session-search').placeholder = i18next.t('placeholders.sessionSearch');
  document.getElementById('voice-output-btn').title = i18next.t('actions.voiceOutputToggle');
  // ... update more dynamic content ...
}

function getBestLanguage() {
  const preferred = navigator.languages || [navigator.language];
  const supported = ['en','ar','es','fr','de','nl','it','pt','ru','ja','zh','hi','sw','id','tr','ko'];
  for (const lang of preferred) {
    const short = lang.split('-')[0].toLowerCase();
    if (supported.includes(short)) return short;
  }
  return 'en';
}

function showToast(message) {
  const toast = document.createElement('div');
  toast.style.position = 'fixed';
  toast.style.bottom = '80px';
  toast.style.left = '50%';
  toast.style.transform = 'translateX(-50%)';
  toast.style.background = 'var(--thunder-gold)';
  toast.style.color = '#000';
  toast.style.padding = '12px 24px';
  toast.style.borderRadius = '8px';
  toast.style.boxShadow = '0 4px 20px rgba(255,170,0,0.4)';
  toast.style.zIndex = '4000';
  toast.style.fontWeight = '600';
  toast.textContent = message;
  document.body.appendChild(toast);
  setTimeout(() => toast.remove(), 3000);
}
