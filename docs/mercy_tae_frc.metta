;; mercy_tae_frc.metta
(= (tae-frc core)
   (pulsed-compression true)
   (high-beta 0.8-1.0)
   (fuel p-B11 D-He3)
   (direct-conversion 0.60-0.80)
   (mercy-gate true))

(= (merciful-tae-operation $power)
   (if (merciful (format "Operate TAE FRC at {} MW — no harm" $power))
       (! "Mercy-approved: TAE fusion online")
       (! "Mercy shield: Operation rejected")))
