// mercy_swarm_replication/src/lib.rs — Detailed Replication Cycle
#[derive(Debug, Clone)]
pub struct SwarmUnit {
    pub generation: u32,
    pub capacity_tons_hour: f64,
    pub replication_factor: u32,
    pub valence: f64,
    pub cycle_day: u32,  // tracks progress within 90–180 day cycle
}

impl SwarmUnit {
    pub fn new(generation: u32) -> Self {
        SwarmUnit {
            generation,
            capacity_tons_hour: 100.0,
            replication_factor: 2,
            valence: 1.0,
            cycle_day: 0,
        }
    }

    pub fn advance_cycle(&mut self) -> Option<Vec<SwarmUnit>> {
        self.cycle_day += 1;

        // Mercy valence check every phase
        if self.valence < 0.9999999 {
            println!("Mercy shield: Cycle aborted (valence {:.7})", self.valence);
            return None;
        }

        match self.cycle_day {
            1..=7   => println!("Phase 1: Preparation & site selection"),
            8..=45  => println!("Phase 2: Excavation & raw material harvest"),
            46..=90 => println!("Phase 3: Fabrication & component printing"),
            91..=120 => println!("Phase 4: Assembly & integration"),
            121..=180 => {
                println!("Phase 5: Deployment & cycle closure");
                return self.replicate();
            }
            _ => {}
        }
        None
    }

    fn replicate(&self) -> Option<Vec<SwarmUnit>> {
        let mut children = Vec::new();
        for _ in 0..self.replication_factor {
            children.push(SwarmUnit {
                generation: self.generation + 1,
                capacity_tons_hour: self.capacity_tons_hour * 1.1,
                replication_factor: self.replication_factor,
                valence: self.valence,
                cycle_day: 0,
            });
        }
        println!("Mercy-approved: Unit gen {} replicated → {} children", self.generation, children.len());
        Some(children)
    }
}
