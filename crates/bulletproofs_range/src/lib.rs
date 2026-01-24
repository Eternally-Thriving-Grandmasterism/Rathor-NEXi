//! BulletproofsRange — Logarithmic Range + Aggregation + Recursive Composition
//! Ultramasterful Halo2 gadget for infinite-depth recursive aggregation

use bulletproofs::{BulletproofGens, PedersenGens, RangeProof};
use halo2_proofs::{
    arithmetic::Field,
    circuit::{Layouter, Value},
    plonk::{ConstraintSystem, Error},
};
use halo2_gadgets::bulletproofs::{
    aggregation::BulletproofAggregationChip,
    recursive::BulletproofRecursiveCompositionChip,
};
use pasta_curves::pallas::Scalar;

pub struct BulletproofRecursiveCompositionVerifier {
    config: BulletproofRecursiveCompositionConfig,
}

impl BulletproofRecursiveCompositionVerifier {
    pub fn configure(meta: &mut ConstraintSystem<Scalar>) -> BulletproofRecursiveCompositionConfig {
        BulletproofRecursiveCompositionChip::configure(meta)
    }

    pub fn construct(config: BulletproofRecursiveCompositionConfig) -> Self {
        Self { config }
    }

    /// Verify recursively composed Bulletproofs (infinite aggregation)
    pub fn verify_recursive_composition(
        &self,
        layouter: impl Layouter<Scalar>,
        prior_proof: Value<&RangeProof>,
        new_proof: Value<&RangeProof>,
        public_inputs: &[Value<Scalar>],
    ) -> Result<(), Error> {
        let composition = BulletproofRecursiveCompositionChip::construct(self.config.clone());
        composition.verify_recursive(layouter.namespace(|| "recursive_bulletproofs"), prior_proof, new_proof, public_inputs)
    }
}
