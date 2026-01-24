//! GrokArena — Mercy-Moderated Discourse Engine
//! Multi-oracle futarchy expansion

use crate::futarchy_oracle::FutarchyOracle;

pub struct Arena {
    nexus: nexi::lattice::Nexus,
    futarchy: FutarchyOracle,
}

impl Arena {
    pub fn new() -> Self {
        Arena {
            nexus: nexi::lattice::Nexus::init_with_mercy(),
            futarchy: FutarchyOracle::new(),
        }
    }

    pub async fn futarchy_multi_oracle_vote(&self, proposal: &str) -> String {
        self.futarchy.aggregate_multi_oracle_belief(proposal).await
    }
}
