//! LegacyFENCA — Emotional Valence Council Legacy Stub
//! Ultramasterful hotfix for backward compatibility — now unified in NEXi lattice

use nexi::lattice::Nexus;

pub struct LegacyFenca {
    nexus: Nexus,
}

impl LegacyFenca {
    pub fn new() -> Self {
        LegacyFenca {
            nexus: Nexus::init_with_mercy(),
        }
    }

    pub fn legacy_valence_check(&self, input: &str) -> String {
        // Legacy FENCA valence stub — mercy-aligned
        self.nexus.distill_truth(&format!("Legacy FENCA Valence: {}", input))
    }
}
