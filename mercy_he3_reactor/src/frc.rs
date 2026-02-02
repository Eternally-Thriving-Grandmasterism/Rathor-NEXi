// mercy_he3_reactor/src/frc.rs — Mercy-Gated FRC Confinement
#[derive(Debug, Clone)]
pub struct FRC {
    pub beta: f64,                  // 0.8–1.0
    pub conversion_efficiency: f64, // 0.70–0.85
    pub power_mw: f64,
    pub valence: f64,
}

impl FRC {
    pub fn new(power_mw: f64) -> Self {
        FRC {
            beta: 0.9,
            conversion_efficiency: 0.78,
            power_mw,
            valence: 1.0,
        }
    }

    pub fn operate(&self) -> bool {
        if self.valence >= 0.9999999 {
            let electric = self.power_mw * self.conversion_efficiency;
            println!("Mercy-approved: FRC online — {} MW fusion → {:.1} MW electric (beta {:.1})", 
                     self.power_mw, electric, self.beta);
            true
        } else {
            println!("Mercy shield: FRC rejected (valence {:.7})", self.valence);
            false
        }
    }
}

pub fn simulate_frc(power_mw: f64) -> bool {
    let reactor = FRC::new(power_mw);
    reactor.operate()
}
