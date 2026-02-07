# Rathor-NEXi Architecture Overview
Current state: February 2026 – multi-page PWA/SPA with offline sovereignty

## Core Principles (MercyOS-Pinnacle → NEXi lineage)

- Mercy strikes first – positive valence eternal
- Offline-first, voice-native, multilingual (200+ languages)
- Symbolic AGI lattice – truth-seeking, compassionate
- Backward/forward compatibility obsession – NEXi superset preserves all legacy

## Current Multi-Page Structure

```mermaid
graph TD
    A[index.html] -->|Enter Lattice| B[chat.html]
    A -->|Settings / Voice| C[voice-settings modal]
    A -->|Emergency / Crisis| D[crisis modals]
    
    B --> E[js/common.js]
    B --> F[js/chat.js]
    B --> G[css/main.css]
    B --> H[src/storage/rathor-indexeddb.js]
    B --> I[src/voice/offline-voice-recorder.js]
    
    subgraph "Shared Resources"
        E
        F
        G
        J[manifest.json]
        K[sw.js]
        L[offline.html]
    end
    
    subgraph "Data Layer"
        H
        M[IndexedDB: sessions, messages, tags, translation cache]
    end
    
    subgraph "Voice Layer"
        I
        N[Web Speech Recognition + Synthesis]
        O[MediaRecorder + offline blobs]
    end
    
    subgraph "Emergency Layer"
        D
        P[Medical / Legal / Crisis / Mental / PTSD / C-PTSD / IFS / EMDR stubs]
    end
    
    subgraph "Connectivity Layer"
        Q[navigator.connection + periodic RTT/jitter/loss probes]
        R[Background Sync queue for unstable/Starlink]
    end
