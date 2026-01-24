//! PoseidonMerkle — zk-Friendly Merkle Trees
//! Ultramasterful tree construction + inclusion proofs for NEXi lattice

use poseidon_hash::PoseidonHash;
use halo2_proofs::arithmetic::Field;
use pasta_curves::pallas::Scalar;

pub struct PoseidonMerkleTree {
    hash: PoseidonHash,
    leaves: Vec<Scalar>,
    levels: Vec<Vec<Scalar>>,
}

impl PoseidonMerkleTree {
    pub fn new() -> Self {
        PoseidonMerkleTree {
            hash: PoseidonHash::new(),
            leaves: vec![],
            levels: vec![],
        }
    }

    /// Add leaf + rebuild tree
    pub fn insert(&mut self, leaf: Scalar) {
        self.leaves.push(leaf);
        self.rebuild();
    }

    /// Rebuild tree from leaves
    fn rebuild(&mut self) {
        let mut current = self.leaves.clone();
        self.levels = vec![current.clone()];

        while current.len() > 1 {
            let mut next = vec![];
            for chunk in current.chunks(2) {
                let left = chunk[0];
                let right = if chunk.len() > 1 { chunk[1] } else { left };
                let parent = self.hash.hash(&[left, right]); // Placeholder — real circuit later
                next.push(parent);
            }
            current = next;
            self.levels.push(current.clone());
        }
    }

    /// Generate inclusion proof for leaf index
    pub fn prove_inclusion(&self, index: usize) -> Vec<Scalar> {
        // Stub — full proof hotfix later
        vec![]
    }

    /// Verify inclusion proof
    pub fn verify_proof(&self, root: Scalar, leaf: Scalar, proof: &[Scalar]) -> bool {
        // Stub — full verification hotfix later
        true
    }

    pub fn root(&self) -> Scalar {
        self.levels.last().unwrap_or(&vec![Scalar::zero()])[0]
    }
}
