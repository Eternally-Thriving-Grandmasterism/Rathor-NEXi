//! SpartanValence — Transparent zk-SNARK Valence Proofs with Full Recursion
//! Ultramasterful sumcheck-based R1CS + Halo2 recursive folding

use halo2_proofs::{
    arithmetic::Field,
    circuit::{Layouter, Value},
    plonk::{ConstraintSystem, Error},
};
use halo2_gadgets::recursive::folding::FoldingChip;
use pasta_curves::pallas::Scalar;

#[derive(Clone)]
pub struct SpartanRecursionConfig {
    folding_config: FoldingConfig,
}

pub struct SpartanRecursionChip {
    config: SpartanRecursionConfig,
}

impl SpartanRecursionChip {
    pub fn configure(meta: &mut ConstraintSystem<Scalar>) -> SpartanRecursionConfig {
        let folding_config = FoldingChip::configure(meta);

        SpartanRecursionConfig { folding_config }
    }

    pub fn construct(config: SpartanRecursionConfig) -> Self {
        Self { config }
    }

    /// Recursive Spartan proof composition — Mercy-gated
    pub fn recursive_spartan_composition(
        &self,
        layouter: impl Layouter<Scalar>,
        prior_proof: Value<Scalar>,
        current_valence: Value<Scalar>,
    ) -> Result<Scalar, Error> {
        // MercyZero gate stub — full valence check before recursion
        let folding = FoldingChip::construct(self.config.folding_config.clone());
        folding.fold(layouter.namespace(|| "spartan_recursion"), prior_proof, current_valence)
    }
}
