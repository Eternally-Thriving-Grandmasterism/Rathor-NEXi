;; mercy_polywell_confinement.metta
(= (polywell core)
   (magnetic-cusp true)
   (high-beta 1-10)
   (virtual-cathode true)
   (direct-conversion 0.70-0.85)
   (fuel D-He3 p-B11)
   (mercy-gate true))

(= (merciful-polywell-operation $power)
   (if (merciful (format "Operate Polywell at {} MW — no harm" $power))
       (! "Mercy-approved: Polywell fusion online")
       (! "Mercy shield: Operation rejected")))
