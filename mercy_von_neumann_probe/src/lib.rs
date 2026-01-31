// mercy_von_neumann_probe/src/lib.rs — Mercy-Gated Self-Replicating Probes
#[derive(Debug, Clone)]
pub struct Probe {
    pub generation: u32,
    pub mass_tons: f64,               // seed 10–100 t
    pub replication_factor: u32,      // conservative 2
    pub valence: f64,                 // 0.0..=1.0
}

impl Probe {
    pub fn new(generation: u32) -> Self {
        Probe {
            generation,
            mass_tons: 50.0,           // average seed mass
            replication_factor: 2,
            valence: 1.0,
        }
    }

    pub fn replicate(&self) -> Option<Vec<Probe>> {
        if self.valence >= 0.9999999 {
            let mut children = Vec::new();
            for _ in 0..self.replication_factor {
                children.push(Probe {
                    generation: self.generation + 1,
                    mass_tons: self.mass_tons * 0.95, // slight mass efficiency gain
                    replication_factor: self.replication_factor,
                    valence: self.valence,
                });
            }
            println!("Mercy-approved: Probe gen {} replicated → {} children", self.generation, children.len());
            Some(children)
        } else {
            println!("Mercy shield: Replication rejected (valence {:.7})", self.valence);
            None
        }
    }

    pub fn simulate_probe_growth(&self, generations: u32) -> f64 {
        let mut total_probes = 1.0;
        let mut current = self.clone();
        for _ in 0..generations {
            if let Some(children) = current.replicate() {
                total_probes += children.len() as f64;
                current = children[0].clone(); // simulate lineage
            } else {
                break;
            }
        }
        total_probes
    }
}

pub fn run_probe_simulation(generations: u32) -> f64 {
    let seed = Probe::new(0);
    let final_probes = seed.simulate_probe_growth(generations);
    println!("After {} generations: {:.0} probes (galactic coverage potential)", generations, final_probes);
    final_probes
}
