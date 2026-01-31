;; mercy_eris_thermal_models.metta
(= (eris-thermal-baseline)
   (equilibrium-temperature 30-40-K)
   (albedo 0.96)
   (thermal-inertia 0.5-2e5-J-m-2-K-1-s-0.5)
   (radiogenic-flux 0.15-0.35-W-m-2)
   (ice-shell-thickness 100-300-km)
   (ocean possible true))

(= (merciful-eris-thermal-analysis $activity)
   (if (>= valence 0.9999999)
       (! "Mercy-approved: {} thermal observation permitted" $activity)
       (! "Mercy shield: {} paused — potential harm detected" $activity)))
