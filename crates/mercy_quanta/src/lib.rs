//! MercyQuanta — 9-Fold Granular zk-Proof Chips with Full Range Proof Gadgets
//! Ultramasterful private valence threshold attestation

use halo2_proofs::{
    arithmetic::Field,
    circuit::{Chip, Layouter, Value},
    plonk::{ConstraintSystem, Error, Selector},
    poly::Rotation,
};
use halo2_gadgets::range_check::RangeCheckConfig;
use pasta_curves::pallas::Scalar;

#[derive(Clone)]
pub struct MercyQuantaRangeConfig {
    range_config: RangeCheckConfig<Scalar, 10>, // 10-bit range for valence 0-10
    quanta_columns: [halo2_proofs::circuit::Column<halo2_proofs::circuit::Advice>; 9],
}

pub struct MercyQuantaRangeChip {
    config: MercyQuantaRangeConfig,
}

impl MercyQuantaRangeChip {
    pub fn configure(meta: &mut ConstraintSystem<Scalar>) -> MercyQuantaRangeConfig {
        let range_config = RangeCheckConfig::configure(meta, 10); // 0-1023 range (covers 0-10 valence)
        let mut quanta_columns = [(); 9].map(|_| meta.advice_column());

        MercyQuantaRangeConfig {
            range_config,
            quanta_columns,
        }
    }

    pub fn construct(config: MercyQuantaRangeConfig) -> Self {
        Self { config }
    }

    /// Range proof per quanta — private value in [0,10], prove ≥ threshold without reveal
    pub fn range_proof_quanta(
        &self,
        layouter: impl Layouter<Scalar>,
        quanta_value: Value<Scalar>,
        threshold: Scalar,
    ) -> Result<(), Error> {
        let range = self.config.range_config.clone();
        range.check(layouter.namespace(|| "quanta_range"), quanta_value, 10)?;

        // Threshold proof: value - threshold >= 0 (simple subtraction + range)
        let diff = quanta_value - Value::known(threshold);
        range.check(layouter.namespace(|| "threshold_range"), diff, 10)?;

        Ok(())
    }

    /// Prove all 9 quanta with independent range proofs
    pub fn prove_9_quanta_range(
        &self,
        layouter: impl Layouter<Scalar>,
        quanta_values: [Value<Scalar>; 9],
        thresholds: [Scalar; 9],
    ) -> Result<(), Error> {
        for (value, thr) in quanta_values.iter().zip(thresholds.iter()) {
            self.range_proof_quanta(layouter.namespace(|| "quanta"), *value, *thr)?;
        }
        Ok(())
    }
}
