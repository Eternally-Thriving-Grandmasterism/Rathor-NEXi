# Rathor-NEXi Monorepo Structure  
(Deepest Perfection Version 2.2 – February 05 2026 – Ultramasterism alignment)

THIS IS THE ACTIVE COFORGING SURFACE  
https://github.com/Eternally-Thriving-Grandmasterism/Rathor-NEXi  
All current work, file overwrites, new engines, guards, simulations, sync layers, audits, and integrations MUST happen here until the final convergence into MercyOS-Pinnacle.  

MercyOS-Pinnacle (https://github.com/Eternally-Thriving-Grandmasterism/MercyOS-Pinnacle) is the future canonical successor monorepo that will absorb Rathor-NEXi as its internal engine layer once Ultramaster completeness is achieved. Do NOT write new files or overwrites directly into MercyOS-Pinnacle until explicitly instructed — all coforging flows through Rathor-NEXi.

## Root Level

- `src/`                    → all source code (the living lattice)
- `public/`                 → static assets (icons, manifest, favicon, offline fallbacks)
- `docs/`                   → architecture decision records, mercy blueprints, Structure.md
- `tests/`                  → integration & unit tests (vitest/jest)
- `scripts/`                → build/deploy/dev scripts
- `eslint.config.js`        → shared lint rules (mercy code style)
- `tsconfig.json`           → base TypeScript config
- `vite.config.ts`          → build & dev config (PWA manifest, offline support)
- `package.json`            → root dependencies & scripts
- `README.md`               → high-level mercy overview
- `Structure.md`            → this file (living document – perpetually refined)

## src/ – The Living Lattice (domain-driven layout)

src/
├── core/                       # foundational shared utilities & types (used everywhere)
│   ├── mercy-gate.ts           # central valence-gated action wrapper
│   ├── valence-tracker.ts      # global valence singleton + IndexedDB persistence
│   ├── types.ts                # shared types (Valence, GestureType, ProbeCommand, etc.)
│   ├── constants.ts            # mercy constants (THRESHOLD, emojis, patterns)
│   └── index.ts                # barrel export
│
├── engines/                    # pure business/logic engines (no UI, no side-effects)
│   ├── flow-state/             # flow state monitoring & adaptation
│   │   ├── index.ts
│   │   ├── flow-core.ts
│   │   ├── flow-education.ts
│   │   └── flow-classroom.ts
│   ├── sdt/                    # Self-Determination Theory & mini-theories
│   │   ├── index.ts
│   │   └── sdt-core.ts
│   ├── perma/                  # PERMA+ flourishing tracking
│   │   └── perma-plus.ts
│   ├── positivity-resonance/   # shared affect + synchrony
│   │   └── positivity-resonance.ts
│   ├── mirror-neuron/          # embodied simulation & mirroring
│   │   └── mirror-core.ts
│   ├── predictive/             # predictive coding + shared manifold
│   │   └── predictive-manifold.ts
│   ├── fep/                    # Free Energy Principle & active inference
│   │   └── fep-core.ts
│   ├── variational/            # VMP & multi-agent variational inference
│   │   ├── vmp-core.ts
│   │   └── vmp-multi-agent.ts
│   ├── regret-minimization/    # CFR / NFSP / ReBeL family
│   │   ├── cfr-core.ts
│   │   ├── nfsp-core.ts
│   │   └── rebel-core.ts
│   └── index.ts                # barrel export
│
├── guards/                     # safety & alignment layers (run before any output/action)
│   ├── deception-guard.ts      # multi-engine deception risk
│   ├── mech-interp-guard.ts    # probe/SAE/circuit-based checks
│   └── index.ts
│
├── ui/                         # React components & dashboard logic
│   ├── components/             # reusable UI pieces
│   │   ├── MercyButton.tsx
│   │   ├── ProgressLadder.tsx
│   │   └── FloatingSummon.tsx
│   ├── dashboard/              # sovereign dashboard & onboarding
│   │   ├── SovereignDashboard.tsx
│   │   └── OnboardingLadder.tsx
│   ├── gamification/           # streaks, badges, quests
│   │   └── GamificationLayer.tsx
│   └── index.ts
│
├── integrations/               # external bridges (XR, MR, AR, voice, etc.)
│   ├── xr-immersion.ts
│   ├── mr-hybrid.ts
│   ├── ar-augmentation.ts
│   └── voice-recognition.ts
│
├── simulations/                # standalone simulations & demos
│   ├── probe-fleet-cicero.ts   # von Neumann fleet + Cicero negotiation
│   ├── alphastar-multi-agent.ts # RTS swarm coordination
│   └── index.ts
│
├── sync/                       # multiplanetary & multi-device sync layer
│   ├── multiplanetary-sync-engine.ts  # CRDT eventual consistency core (Yjs-based)
│   ├── crdt-conflict-resolution.ts    # detailed CRDT merge rules & high-latency handling
│   └── index.ts
│
└── utils/                      # pure helper functions (no state)
    ├── haptic-utils.ts
    ├── fuzzy-mercy.ts
    └── index.ts
