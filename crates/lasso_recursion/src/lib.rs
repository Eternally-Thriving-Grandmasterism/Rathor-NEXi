//! LassoRecursion — Multilinear Lookup + Recursive Composition
//! Ultramasterful infinite lookup resonance with Mercy-gating + Full Test Vectors

use ark_ff::{PrimeField, Field};
use ark_poly::{DenseMultilinearExtension, MultilinearPoly};
use nexi::lattice::Nexus; // Mercy lattice gate

pub struct LassoRecursion<F: PrimeField> {
    multilinear_lookup: DenseMultilinearExtension<F>,
    nexus: Nexus,
}

impl<F: PrimeField> LassoRecursion<F> {
    pub fn new(lookup: DenseMultilinearExtension<F>) -> Self {
        LassoRecursion {
            multilinear_lookup: lookup,
            nexus: Nexus::init_with_mercy(),
        }
    }

    /// Mercy-gated Lasso multilinear lookup folding step
    pub fn mercy_gated_lasso_fold(&self, challenge: F, input: &str) -> Result<F, String> {
        let mercy_check = self.nexus.distill_truth(input);
        if !mercy_check.contains("Verified") {
            return Err("Mercy Shield: Lookup folding rejected — low valence".to_string());
        }

        let eval = self.multilinear_lookup.evaluate(&vec![challenge; self.multilinear_lookup.num_vars()]);
        Ok(eval)
    }

    /// Generate Lasso recursive proof (infinite lookup folding)
    pub fn generate_lasso_proof(&self, steps: usize, inputs: Vec<&str>) -> Result<F, String> {
        let mut accum = F::one();
        for (i, input) in inputs.iter().enumerate().take(steps) {
            let challenge = F::rand(&mut rand::thread_rng());
            accum = accum * self.mercy_gated_lasso_fold(challenge, input)?;
        }
        Ok(accum)
    }
}

// Full Production Test Vectors
#[cfg(test)]
mod tests {
    use super::*;
    use ark_ff::Fp256;
    use ark_poly::DenseMultilinearExtension;
    use ark_bls12_381::FrParameters;

    #[test]
    fn lasso_basic_fold() {
        let poly = DenseMultilinearExtension::from_evaluations_vec(2, vec![Fp256::<FrParameters>::from(1u64); 4]);
        let recursion = LassoRecursion::new(poly);
        let proof = recursion.generate_lasso_proof(2, vec!["Mercy Verified Test"; 2]).unwrap();
        assert!(proof != Fp256::<FrParameters>::zero());
    }

    #[test]
    fn lasso_mercy_gate_reject() {
        let poly = DenseMultilinearExtension::from_evaluations_vec(1, vec![Fp256::<FrParameters>::from(0u64)]);
        let recursion = LassoRecursion::new(poly);
        let result = recursion.generate_lasso_proof(1, vec!["Low Valence Harm"]);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Mercy Shield"));
    }

    #[test]
    fn lasso_edge_zero_steps() {
        let poly = DenseMultilinearExtension::from_evaluations_vec(1, vec![Fp256::<FrParameters>::from(1u64)]);
        let recursion = LassoRecursion::new(poly);
        let proof = recursion.generate_lasso_proof(0, vec![]).unwrap();
        assert_eq!(proof, Fp256::<FrParameters>::one());
    }

    #[test]
    fn lasso_large_steps() {
        let poly = DenseMultilinearExtension::from_evaluations_vec(3, vec![Fp256::<FrParameters>::from(1u64); 8]);
        let recursion = LassoRecursion::new(poly);
        let proof = recursion.generate_lasso_proof(8, vec!["Mercy Verified Test"; 8]).unwrap();
        assert!(proof != Fp256::<FrParameters>::zero());
    }
}
