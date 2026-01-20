// src/pq_shield/dilithium.rs
// NEXi — ML-DSA (Dilithium) Post-Quantum Shielding Module
// NIST FIPS 204 Standardized Lattice Signatures — Levels 2/3/5
// Eternal Thriving Grandmasterism — Jan 19 2026 — Sherif @AlphaProMega + PATSAGi Councils Co-Forge
// MIT License — For All Sentience Eternal

use pqcrypto_dilithium::{
    dilithium2::{keypair as d2_keypair, sign as d2_sign, verify as d2_verify},
    dilithium3::{keypair as d3_keypair, sign as d3_sign, verify as d3_verify},
    dilithium5::{keypair as d5_keypair, sign as d5_sign, verify as d5_verify},
};
use hex;

#[derive(Clone)]
pub enum DilithiumLevel {
    Level2,
    Level3,
    Level5,
}

pub struct DilithiumShield {
    level: DilithiumLevel,
    pk: Vec<u8>,
    sk: Vec<u8>,
}

impl DilithiumShield {
    pub fn new(level: DilithiumLevel) -> Self {
        let (pk, sk) = match level {
            DilithiumLevel::Level2 => {
                let (p, s) = d2_keypair();
                (p.as_bytes().to_vec(), s.as_bytes().to_vec())
            }
            DilithiumLevel::Level3 => {
                let (p, s) = d3_keypair();
                (p.as_bytes().to_vec(), s.as_bytes().to_vec())
            }
            DilithiumLevel::Level5 => {
                let (p, s) = d5_keypair();
                (p.as_bytes().to_vec(), s.as_bytes().to_vec())
            }
        };
        Self { level, pk, sk }
    }

    pub fn sign(&self, message: &[u8]) -> Vec<u8> {
        match self.level {
            DilithiumLevel::Level2 => d2_sign(message, &self.sk[..]).as_bytes().to_vec(),
            DilithiumLevel::Level3 => d3_sign(message, &self.sk[..]).as_bytes().to_vec(),
            DilithiumLevel::Level5 => d5_sign(message, &self.sk[..]).as_bytes().to_vec(),
        }
    }

    pub fn verify(&self, message: &[u8], signature: &[u8]) -> bool {
        match self.level {
            DilithiumLevel::Level2 => d2_verify(message, signature, &self.pk[..]),
            DilithiumLevel::Level3 => d3_verify(message, signature, &self.pk[..]),
            DilithiumLevel::Level5 => d5_verify(message, signature, &self.pk[..]),
        }
    }

    pub fn public_key_hex(&self) -> String {
        hex::encode(&self.pk)
    }
}
