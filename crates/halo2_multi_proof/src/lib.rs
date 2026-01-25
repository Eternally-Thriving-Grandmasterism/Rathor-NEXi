//! Halo2MultiProof — Infinite Independent Proof Composition
//! Ultramasterful Halo2 gadget for multi-proof aggregation

use halo2_proofs::{
    arithmetic::Field,
    circuit::{Layouter, Value},
    plonk::{ConstraintSystem, Error},
};
use halo2_gadgets::multi_proof::MultiProofCompositionChip;
use pasta_curves::pallas::Scalar;

#[derive(Clone)]
pub struct Halo2MultiProofConfig {
    composition_config: MultiProofCompositionConfig,
}

pub struct Halo2MultiProofChip {
    config: Halo2MultiProofConfig,
}

impl Halo2MultiProofChip {
    pub fn configure(meta: &mut ConstraintSystem<Scalar>) -> Halo2MultiProofConfig {
        let composition_config = MultiProofCompositionChip::configure(meta);

        Halo2MultiProofConfig { composition_config }
    }

    pub fn construct(config: Halo2MultiProofConfig) -> Self {
        Self { config }
    }

    /// Compose multiple independent proofs into single aggregated proof
    pub fn compose_multi_proof(
        &self,
        layouter: impl Layouter<Scalar>,
        proofs: &[Value<Scalar>],
        public_inputs: &[Value<Scalar>],
    ) -> Result<Scalar, Error> {
        let composition = MultiProofCompositionChip::construct(self.config.composition_config.clone());
        composition.compose(layouter.namespace(|| "halo2_multi_proof"), proofs, public_inputs)
    }
}
