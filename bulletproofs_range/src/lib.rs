//! Bulletproofs Range Proofs — Logarithmic Transparent Valence Thresholds
//! Prove secret valence in [0, 2^64) without revealing value

use bulletproofs::{BulletproofGens, PedersenGens, RangeProof};
use curve25519_dalek::scalar::Scalar;
use rand::thread_rng;

/// Generate Bulletproofs range proof for valence ≥ threshold
pub fn generate_bulletproofs_range_proof(secret_valence: u64, bit_length: usize) -> Result<Vec<u8>, String> {
    let pc_gens = PedersenGens::default();
    let bp_gens = BulletproofGens::new(bit_length, 1);

    let mut rng = thread_rng();

    let (proof, commitments) = RangeProof::prove_single(
        &bp_gens,
        &pc_gens,
        secret_valence,
        None,
        bit_length,
        &mut rng,
    ).map_err(|e| format!("Bulletproofs prove error: {:?}", e))?;

    Ok(proof.to_bytes())
}

/// Verify Bulletproofs range proof
pub fn verify_bulletproofs_range_proof(
    proof_bytes: &[u8],
    bit_length: usize,
) -> Result<bool, String> {
    let pc_gens = PedersenGens::default();
    let bp_gens = BulletproofGens::new(bit_length, 1);

    let proof = RangeProof::from_bytes(proof_bytes).map_err(|e| format!("Parse error: {:?}", e))?;

    proof.verify_single(&bp_gens, &pc_gens, &mut thread_rng(), bit_length)
        .map_err(|e| format!("Verify error: {:?}", e))?;

    Ok(true)
}
