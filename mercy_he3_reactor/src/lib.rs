// mercy_he3_reactor/src/lib.rs — Mercy-Gated He³ Fusion Reactor Blueprint
#[derive(Debug, Clone)]
pub struct He3ReactorBlueprint {
    pub power_mw: f64,              // 100–10000 MW
    pub q_factor: f64,              // fusion gain
    pub conversion_efficiency: f64, // 0.70–0.85
    pub mass_tons: f64,             // 300–1200 t
    pub valence: f64,
}

impl He3ReactorBlueprint {
    pub fn new(power_mw: f64) -> Self {
        He3ReactorBlueprint {
            power_mw,
            q_factor: 15.0,
            conversion_efficiency: 0.78,
            mass_tons: 800.0,
            valence: 1.0,
        }
    }

    pub fn operate(&self) -> bool {
        if self.valence >= 0.9999999 {
            let electric_output = self.power_mw * self.conversion_efficiency;
            println!("Mercy-approved: He³ reactor blueprint online — {} MW fusion → {:.1} MW electric (mass {:.0} t)", 
                     self.power_mw, electric_output, self.mass_tons);
            true
        } else {
            println!("Mercy shield: Reactor blueprint rejected (valence {:.7})", self.valence);
            false
        }
    }
}

pub fn simulate_he3_blueprint(power_mw: f64) -> bool {
    let reactor = He3ReactorBlueprint::new(power_mw);
    reactor.operate()
}
