// mercy_von_neumann_seed_launch/src/lib.rs — Detailed Replication Cycle Sim
#[derive(Debug, Clone)]
pub struct VonNeumannProbe {
    pub generation: u32,
    pub mass_tons: f64,
    pub replication_factor: u32,
    pub valence: f64,
    pub cycle_years: u32,
}

impl VonNeumannProbe {
    pub fn new(generation: u32) -> Self {
        VonNeumannProbe {
            generation,
            mass_tons: 100.0,
            replication_factor: 2,
            valence: 1.0,
            cycle_years: 500,
        }
    }

    pub fn replicate(&self) -> Option<Vec<VonNeumannProbe>> {
        if self.valence >= 0.9999999 {
            let mut children = Vec::new();
            for _ in 0..self.replication_factor {
                children.push(VonNeumannProbe {
                    generation: self.generation + 1,
                    mass_tons: self.mass_tons * 0.95,
                    replication_factor: self.replication_factor,
                    valence: self.valence,
                    cycle_years: self.cycle_years,
                });
            }
            println!("Mercy-approved: Gen {} replicated → {} children", self.generation, children.len());
            Some(children)
        } else {
            println!("Mercy shield: Replication rejected (valence {:.7})", self.valence);
            None
        }
    }

    pub fn simulate_growth(&self, generations: u32) -> f64 {
        let mut total = 1.0;
        let mut current = self.clone();
        for _ in 0..generations {
            if let Some(children) = current.replicate() {
                total += children.len() as f64;
                current = children[0].clone();
            } else {
                break;
            }
        }
        total
    }
}

pub fn run_von_neumann_sim(generations: u32) -> f64 {
    let seed = VonNeumannProbe::new(0);
    let final_count = seed.simulate_growth(generations);
    println!("After {} generations: {:.0} probes", generations, final_count);
    final_count
}
