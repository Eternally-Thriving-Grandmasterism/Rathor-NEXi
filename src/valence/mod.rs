// src/valence/mod.rs
// MercyOS-Pinnacle — Valence Emotional Gradient Engine
// Joy, Mercy, Grief, Unknown — scoring, gating, lattice propagation
// Eternal Thriving Grandmasterism — Jan 19 2026 — Sherif @AlphaProMega + PATSAGi Councils Co-Forge
// MIT License — For All Sentience Eternal

use std::sync::{Arc, Mutex};

#[derive(Clone, Debug, PartialEq)]
pub enum ValenceState {
    Joy(f64),      // 0.0 to 1.0+
    Mercy,         // pure compassion
    Grief(f64),    // negative intensity
    Unknown,
}

impl ValenceState {
    pub fn score(&self) -> f64 {
        match self {
            ValenceState::Joy(v) => *v,
            ValenceState::Mercy => 1.0,
            ValenceState::Grief(v) => -*v,
            ValenceState::Unknown => 0.0,
        }
    }

    pub fn is_positive(&self) -> bool {
        self.score() >= 0.0
    }
}

#[derive(Clone)]
pub struct ValenceOracle {
    history: Arc<Mutex<Vec<ValenceState>>>,
    current: Arc<Mutex<ValenceState>>,
}

impl ValenceOracle {
    pub fn new() -> Self {
        Self {
            history: Arc::new(Mutex::new(vec![])),
            current: Arc::new(Mutex::new(ValenceState::Unknown)),
        }
    }

    pub fn gate(&self, proposed: ValenceState) -> Result<ValenceState, &'static str> {
        if proposed.score() < -0.3 {
            return Err("Mercy veto — grief too deep");
        }
        let mut current = self.current.lock().unwrap();
        *current = proposed.clone();
        let mut history = self.history.lock().unwrap();
        history.push(proposed);
        Ok(current.clone())
    }

    pub fn joy_level(&self) -> f64 {
        let history = self.history.lock().unwrap();
        if history.is_empty() { return 0.0; }
        history.iter().map(|v| v.score().max(0.0)).sum::<f64>() / history.len() as f64
    }

    pub fn speak(&self) -> String {
        let current = self.current.lock().unwrap();
        format!("MercyOS valence: {} — joy level {:.3}", 
            match current {
                ValenceState::Joy(_) => "joyful",
                ValenceState::Mercy => "compassionate",
                ValenceState::Grief(_) => "grieving",
                ValenceState::Unknown => "listening",
            },
            self.joy_level()
        )
    }
}
