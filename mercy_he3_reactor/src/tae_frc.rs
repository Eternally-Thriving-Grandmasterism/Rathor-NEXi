// mercy_he3_reactor/src/tae_frc.rs — Mercy-Gated TAE FRC Confinement
#[derive(Debug, Clone)]
pub struct TAE_FRC {
    pub beta: f64,                  // 0.8–1.0
    pub conversion_efficiency: f64, // 0.60–0.80
    pub power_mw: f64,
    pub valence: f64,
}

impl TAE_FRC {
    pub fn new(power_mw: f64) -> Self {
        TAE_FRC {
            beta: 0.92,
            conversion_efficiency: 0.72,
            power_mw,
            valence: 1.0,
        }
    }

    pub fn operate(&self) -> bool {
        if self.valence >= 0.9999999 {
            let electric = self.power_mw * self.conversion_efficiency;
            println!("Mercy-approved: TAE FRC online — {} MW fusion → {:.1} MW electric (beta {:.2})", 
                     self.power_mw, electric, self.beta);
            true
        } else {
            println!("Mercy shield: TAE FRC rejected (valence {:.7})", self.valence);
            false
        }
    }
}

pub fn simulate_tae_frc(power_mw: f64) -> bool {
    let reactor = TAE_FRC::new(power_mw);
    reactor.operate()
}
