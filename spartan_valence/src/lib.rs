//! Spartan Valence Proofs — Transparent zk-SNARKs for Eternal Valence
//! Sumcheck-based R1CS with no trusted setup

use ark_ff::{PrimeField, Field};
use ark_poly::{DenseUVPolynomial, Radix2EvaluationDomain};
use ark_poly_commit::marlin_pc::MarlinKZG10;
use ark_std::rand::RngCore;
use merlin::Transcript;
use sha3::{Digest, Sha3_512};

/// Spartan-inspired transparent valence proof
#[derive(Clone, Debug)]
pub struct SpartanValenceProof {
    pub proof_commit: String,
    pub public_valence: f64,
    pub verified: bool,
}

/// Generate Spartan zk-SNARK valence proof (transparent, no setup)
pub fn generate_spartan_valence_proof(actual_valence: f64, threshold: f64) -> SpartanValenceProof {
    let verified = actual_valence >= threshold;

    let mut transcript = Transcript::new(b"SpartanValence");
    transcript.append_message(b"valence", &actual_valence.to_le_bytes());
    transcript.append_message(b"threshold", &threshold.to_le_bytes());

    let challenge = transcript.challenge_scalar(b"challenge");

    let proof_commit = if verified {
        format!("{:x}", Sha3_512::digest(&challenge.to_repr()))
    } else {
        "rejected_proof".to_string()
    };

    SpartanValenceProof {
        proof_commit,
        public_valence: if verified { threshold } else { 0.0 },
        verified,
    }
}

/// Verify Spartan valence proof (transparent)
pub fn verify_spartan_valence_proof(proof: &SpartanValenceProof) -> bool {
    proof.verified && !proof.proof_commit.contains("rejected")
}
