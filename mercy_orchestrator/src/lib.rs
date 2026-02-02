// mercy_orchestrator/src/lib.rs — Valence-Gated Orchestrator with MeTTa + Terminus + Hyper + Neo4j
use std::error::Error;

// Existing imports...
use crate::neo4j_integration::Neo4jMercyStore;

// ... existing MercyOrchestrator struct ...

impl MercyOrchestrator {
    // ... existing: new(), load_local, allow, load_from_terminus, load_from_hypergraph_valence, persist_to_hypergraph ...

    // NEW: Load rules from Neo4j via Cypher
    pub async fn load_from_neo4j(&mut self, store: &Neo4jMercyStore, min_valence: f64) -> Result<(), Box<dyn Error>> {
        if self.valence < 0.9999999 {
            return Err("Mercy shield: Low valence — Neo4j query rejected".into());
        }

        let high_atoms = store.query_high_valence(min_valence).await?;
        self.rules = high_atoms.into_iter()
            .map(|(atom, val)| format!("neo-atom: {} @ valence {:.7}", atom, val))
            .collect();

        println!("Mercy rules loaded from Neo4j: {} atoms", self.rules.len());
        Ok(())
    }

    // Optional: persist mercy-gated to Neo4j
    pub async fn persist_to_neo4j(&self, store: &Neo4jMercyStore, atom: &str, context: &str) -> Result<(), Box<dyn Error>> {
        if self.valence < 0.9999999 {
            return Err("Mercy shield: persistence rejected".into());
        }
        store.insert_metta_atom(atom, self.valence, context).await?;
        Ok(())
    }
}
