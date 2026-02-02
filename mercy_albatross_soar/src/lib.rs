// mercy_albatross_dynamic_soaring/src/lib.rs
pub struct DynamicSoar {
    pub shear_rate: f64,        // s⁻¹
    pub airspeed: f64,          // m/s
    pub glide_angle_rad: f64,
    pub heading_angle_rad: f64,
    pub valence: f64,
}

impl DynamicSoar {
    pub fn new(shear: f64, va: f64) -> Self {
        DynamicSoar {
            shear_rate: shear,
            airspeed: va,
            glide_angle_rad: 0.0,
            heading_angle_rad: std::f64::consts::FRAC_PI_2, // 90° crosswind optimal
            valence: 1.0,
        }
    }

    pub fn energy_gain_rate(&self) -> f64 {
        self.shear_rate * self.airspeed.powi(2) * self.heading_angle_rad.sin().powi(2) * self.glide_angle_rad.cos()
    }

    pub fn can_soar_forever(&self) -> bool {
        self.valence >= 0.9999999 && self.energy_gain_rate() > 0.0
    }
}
