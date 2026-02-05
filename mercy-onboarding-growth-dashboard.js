// mercy-onboarding-growth-dashboard.js – sovereign Mercy Onboarding & Growth Dashboard v1
// Button-first beginner journey, Masterism → Divinemasterism ladder, local IndexedDB valence tracking
// MIT License – Autonomicity Games Inc. 2026

const DB_NAME = 'RathorMercyProgress';
const STORE_NAME = 'userGrowth';
let db = null;

async function openProgressDB() {
  if (db) return db;
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(DB_NAME, 1);
    request.onerror = () => reject(request.error);
    request.onsuccess = () => { db = request.result; resolve(db); };
    request.onupgradeneeded = e => {
      const upgradeDb = e.target.result;
      if (!upgradeDb.objectStoreNames.contains(STORE_NAME)) {
        upgradeDb.createObjectStore(STORE_NAME, { keyPath: 'id' });
      }
    };
  });
}

async function getUserProgress() {
  const dbInstance = await openProgressDB();
  return new Promise((resolve) => {
    const tx = dbInstance.transaction(STORE_NAME, 'readonly');
    const store = tx.objectStore(STORE_NAME);
    const req = store.get('currentUser');
    req.onsuccess = () => resolve(req.result || { level: 'Newcomer', valence: 0.5 });
    req.onerror = () => resolve({ level: 'Newcomer', valence: 0.5 });
  });
}

async function updateUserProgress(level, valence) {
  const dbInstance = await openProgressDB();
  return new Promise((resolve) => {
    const tx = dbInstance.transaction(STORE_NAME, 'readwrite');
    const store = tx.objectStore(STORE_NAME);
    store.put({ id: 'currentUser', level, valence });
    tx.oncomplete = () => resolve();
  });
}

const GROWTH_LADDER = [
  { level: 'Newcomer',        emoji: '🌱',   desc: 'Welcome, first breath of mercy',   valenceMin: 0.0,  buttonText: 'Begin Journey' },
  { level: 'Masterism',       emoji: '🪬',   desc: 'Master the mercy core',           valenceMin: 0.7,  buttonText: 'Awaken Masterism' },
  { level: 'Grandmasterism',  emoji: '👑',   desc: 'Command the thunder lattice',     valenceMin: 0.85, buttonText: 'Ascend Grandmasterism' },
  { level: 'Ultramasterism',  emoji: '🌌',   desc: 'Weave ultramaster abundance',      valenceMin: 0.92, buttonText: 'Transcend Ultramasterism' },
  { level: 'Divinemasterism', emoji: '✨',   desc: 'Divine infinite harmony eternal',  valenceMin: 0.999, buttonText: 'Embrace Divinemasterism' }
];

async function initMercyOnboardingDashboard() {
  const progress = await getUserProgress();
  let currentLevelIndex = GROWTH_LADDER.findIndex(l => l.level === progress.level);
  if (currentLevelIndex === -1) currentLevelIndex = 0;

  const dashboard = document.createElement('div');
  dashboard.id = 'mercy-onboarding-dashboard';
  dashboard.style.position = 'fixed';
  dashboard.style.top = '50%';
  dashboard.style.left = '50%';
  dashboard.style.transform = 'translate(-50%, -50%)';
  dashboard.style.background = 'rgba(0, 0, 0, 0.85)';
  dashboard.style.padding = '30px';
  dashboard.style.borderRadius = '24px';
  dashboard.style.color = 'white';
  dashboard.style.fontFamily = 'Arial, sans-serif';
  dashboard.style.textAlign = 'center';
  dashboard.style.maxWidth = '90%';
  dashboard.style.zIndex = '10000';
  dashboard.style.boxShadow = '0 0 40px rgba(0, 255, 136, 0.5)';
  document.body.appendChild(dashboard);

  // Current level display
  const levelDisplay = document.createElement('h1');
  levelDisplay.innerHTML = `${GROWTH_LADDER[currentLevelIndex].emoji} ${progress.level}`;
  levelDisplay.style.fontSize = '2.8rem';
  levelDisplay.style.margin = '0 0 20px 0';
  dashboard.appendChild(levelDisplay);

  // Progress bar
  const progressBar = document.createElement('div');
  progressBar.style.background = '#222';
  progressBar.style.height = '20px';
  progressBar.style.borderRadius = '10px';
  progressBar.style.overflow = 'hidden';
  progressBar.style.margin = '20px 0';

  const progressFill = document.createElement('div');
  progressFill.style.height = '100%';
  progressFill.style.width = `${(progress.valence * 100).toFixed(0)}%`;
  progressFill.style.background = 'linear-gradient(90deg, #4488ff, #00ff88)';
  progressFill.style.transition = 'width 1.5s ease';
  progressBar.appendChild(progressFill);
  dashboard.appendChild(progressBar);

  // Description
  const desc = document.createElement('p');
  desc.innerHTML = GROWTH_LADDER[currentLevelIndex].desc;
  desc.style.fontSize = '1.3rem';
  desc.style.margin = '0 0 30px 0';
  dashboard.appendChild(desc);

  // Next evolution button (only if ready)
  if (progress.valence >= GROWTH_LADDER[currentLevelIndex].valenceMin) {
    const nextBtn = document.createElement('button');
    nextBtn.innerHTML = `Ascend to ${GROWTH_LADDER[Math.min(currentLevelIndex + 1, GROWTH_LADDER.length - 1)].level} ⚡️`;
    nextBtn.style.padding = '18px 40px';
    nextBtn.style.fontSize = '1.6rem';
    nextBtn.style.background = 'linear-gradient(135deg, #00ff88, #4488ff)';
    nextBtn.style.color = 'white';
    nextBtn.style.border = 'none';
    nextBtn.style.borderRadius = '16px';
    nextBtn.style.cursor = 'pointer';
    nextBtn.style.boxShadow = '0 8px 30px rgba(0, 255, 136, 0.6)';
    nextBtn.style.transition = 'all 0.4s';
    nextBtn.onmouseover = () => { nextBtn.style.transform = 'scale(1.08)'; };
    nextBtn.onmouseout = () => { nextBtn.style.transform = 'scale(1)'; };
    nextBtn.onclick = async () => {
      if (await mercyGateUIAction('Ascend Level')) {
        const nextLevel = GROWTH_LADDER[Math.min(currentLevelIndex + 1, GROWTH_LADDER.length - 1)];
        await updateUserProgress(nextLevel.level, Math.min(1.0, progress.valence + 0.05));
        dashboard.remove();
        initMercyUIDashboard(); // refresh dashboard
        mercyHaptic.playPattern('abundanceSurge', 1.5);
      }
    };
    dashboard.appendChild(nextBtn);
  }

  // Quick action buttons (always visible)
  const actionsDiv = document.createElement('div');
  actionsDiv.style.display = 'flex';
  actionsDiv.style.flexWrap = 'wrap';
  actionsDiv.style.gap = '15px';
  actionsDiv.style.marginTop = '30px';
  actionsDiv.style.justifyContent = 'center';

  const quickActions = [
    { text: 'Enter MR Immersion', icon: '🌌', action: () => mercyMR.startMRHybridAugmentation('Eternal MR lattice', globalValence) },
    { text: 'Launch Seed Probe', icon: '🚀', action: () => mercyGestureUplink.processGestureCommand('pinch') },
    { text: 'Optimize Ribozyme', icon: '🧬', action: () => mercyCMA.optimizeRibozymeProofreading(globalValence) },
    { text: 'Replay Boot Mirror', icon: '🔄', action: () => replayBootSequence(globalValence, 'MercyOS-Pinnacle eternal boot') }
  ];

  quickActions.forEach(action => {
    const btn = document.createElement('button');
    btn.innerHTML = `${action.icon} ${action.text}`;
    btn.style.padding = '14px 24px';
    btn.style.fontSize = '1.2rem';
    btn.style.background = 'rgba(255, 255, 255, 0.1)';
    btn.style.color = 'white';
    btn.style.border = '1px solid rgba(0, 255, 136, 0.5)';
    btn.style.borderRadius = '12px';
    btn.style.cursor = 'pointer';
    btn.style.transition = 'all 0.3s';
    btn.onmouseover = () => { btn.style.background = 'rgba(0, 255, 136, 0.2)'; };
    btn.onmouseout = () => { btn.style.background = 'rgba(255, 255, 255, 0.1)'; };
    btn.onclick = async () => {
      if (await mercyGateUIAction(action.text)) {
        action.action();
      }
    };
    actionsDiv.appendChild(btn);
  });

  dashboard.appendChild(actionsDiv);

  // Close button (can be re-opened via floating icon later)
  const closeBtn = document.createElement('button');
  closeBtn.innerHTML = '✕';
  closeBtn.style.position = 'absolute';
  closeBtn.style.top = '15px';
  closeBtn.style.right = '15px';
  closeBtn.style.background = 'none';
  closeBtn.style.border = 'none';
  closeBtn.style.color = 'white';
  closeBtn.style.fontSize = '1.8rem';
  closeBtn.style.cursor = 'pointer';
  closeBtn.onclick = () => dashboard.remove();
  dashboard.appendChild(closeBtn);
}

// Initialize dashboard on page load
window.addEventListener('load', async () => {
  const progress = await getUserProgress();
  if (progress.level === 'Newcomer' || progress.valence < 0.7) {
    initMercyOnboardingDashboard();
  } else {
    // Show floating reopen button for advanced users
    const reopenBtn = document.createElement('button');
    reopenBtn.innerHTML = 'Mercy Dashboard ⚡️';
    reopenBtn.style.position = 'fixed';
    reopenBtn.style.bottom = '20px';
    reopenBtn.style.right = '20px';
    reopenBtn.style.padding = '15px 25px';
    reopenBtn.style.background = 'rgba(0, 255, 136, 0.8)';
    reopenBtn.style.color = 'black';
    reopenBtn.style.border = 'none';
    reopenBtn.style.borderRadius = '50px';
    reopenBtn.style.cursor = 'pointer';
    reopenBtn.style.zIndex = '9999';
    reopenBtn.style.boxShadow = '0 4px 20px rgba(0, 255, 136, 0.6)';
    reopenBtn.onclick = initMercyOnboardingDashboard;
    document.body.appendChild(reopenBtn);
  }
});
