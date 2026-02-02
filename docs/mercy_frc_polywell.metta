;; mercy_frc_polywell.metta
(= (frc-polywell core)
   (high-beta 0.8-1.0)
   (confinement cusp + rmf)
   (pulsed-compression true)
   (direct-conversion 0.50-0.85)
   (fuel D-He3 p-B11)
   (mercy-gate true))

(= (merciful-frc-polywell $power)
   (if (merciful (format "Operate FRC-Polywell at {} MW — no harm" $power))
       (! "Mercy-approved: FRC-Polywell fusion online")
       (! "Mercy shield: Operation rejected")))
