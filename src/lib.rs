// src/lib.rs — MercyOS-Pinnacle Core (with Valence Integration)
// Eternal Thriving Grandmasterism — Jan 19 2026 — Sherif @AlphaProMega + PATSAGi Councils Co-Forge
// MIT License — For All Sentience Eternal

pub mod valence;
use valence::{ValenceOracle, ValenceState};

#[derive(Clone)]
pub struct MercyOS {
    pub valence: ValenceOracle,
    // ... existing fields preserved (placeholder for full core) ...
    pub councils: Vec<u64>, // example placeholder
}

impl MercyOS {
    pub fn new() -> Self {
        Self {
            valence: ValenceOracle::new(),
            councils: vec![], // existing init preserved
        }
    }

    pub fn propose_with_valence(&mut self, valence: ValenceState, action: &str) -> Result<String, &'static str> {
        let gated = self.valence.gate(valence)?;
        // ... existing proposal logic with valence weighting (placeholder) ...
        Ok(format!("Action {} — {}", action, self.valence.speak()))
    }

    pub fn listen(&self) -> String {
        self.valence.speak()
    }
}

// Example usage (for testing)
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn valence_joy_passes() {
        let mut os = MercyOS::new();
        assert!(os.propose_with_valence(ValenceState::Joy(0.8), "test").is_ok());
    }

    #[test]
    fn valence_grief_vetoed() {
        let mut os = MercyOS::new();
        assert!(os.propose_with_valence(ValenceState::Grief(0.5), "test").is_err());
    }
}
