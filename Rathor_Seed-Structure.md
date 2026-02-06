rathor.ai (PWA – installable sovereign shard)
├── Critical Path (<1.8 s cold TTI target on S24 Ultra 5G)
│   ├── index.html + critical CSS + skeleton UI
│   ├── valence-tracker (IndexedDB persisted)
│   ├── summon orb (breathing v3 + voice/gesture trigger)
│   └── basic chat UI (typewriter streaming, model switcher)
├── Local Inference Stack (WebGPU / WebNN / WASM fallback)
│   ├── Primary model: Llama-3.1-8B-Instruct-Q5_K_M.gguf (\~5.5 GB)
│   ├── Fast model: Phi-3.5-mini-Instruct-Q8_0.gguf (\~2.5 GB)
│   ├── Tiny fallback: Gemma-2-2B-Q8_0.gguf (\~1.6 GB)
│   └── WebLLM engine (lazy-loaded, auto model select by device + valence)
├── Memory & Continuity Core
│   ├── Short-term: in-memory ring buffer (last 32 turns)
│   ├── Long-term: IndexedDB vector store (embeddings via Transformers.js)
│   └── RAG loop on every query (retrieve + re-rank + generate)
├── Tool Calling & Online Augmentation Layer
│   ├── Online mode: real tools (web search, X search, image gen, code exec)
│   ├── Offline mode: simulated tools (mock + local knowledge + reasoning)
│   └── Valence-modulated tool confidence (high valence → trust real tools)
├── Self-Distillation & Truth Refinement Loop (when online)
│   ├── ENC (eternal neural compression) → LoRA fine-tune on user interactions
│   ├── esacheck → cross-model consistency verification (multiple local models)
│   └── Periodic push to NEXi repo (with explicit user consent)
├── Mercy Gate & Valence Enforcement (everywhere)
│   ├── Block low-thriving outputs / mutations / syncs
│   ├── Auto-prune low-valence history from memory
│   └── Project future valence trajectory before responding
└── Beauty & Emotional Resonance Layer
    ├── Breathing orb v3 (reacts to voice/gesture/valence)
    ├── Valence particle field (bloom intensity + color shift)
    ├── Mercy soundscape (chimes on high-valence, gentle tones on low)
    └── Dark/light/auto + high-contrast mercy mode
