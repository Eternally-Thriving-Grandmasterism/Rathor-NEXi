// js/common.js — Shared utilities, i18n, language registry

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
}

// ... rest of shared functions (changeLanguage, applyRTL, updateContent, initLanguageSearch, getBestLanguage, etc.) ...
