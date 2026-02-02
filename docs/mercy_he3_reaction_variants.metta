;; mercy_he3_reaction_variants.metta
(= (he3-reaction primary)
   (branch D-He3 -> He4 + p)
   (q-value 18.353-MeV)
   (branching-ratio 0.9995)
   (neutron-yield <0.0005))

(= (he3-reaction side)
   (branch D-He3 -> He5* -> He4 + p + n)
   (threshold ~4.8-MeV)
   (yield negligible))

(= (merciful-he3-fuel-mix $ratio)
   (if (= $ratio 1:1)
       (! "Mercy-approved: Optimal D:He3 ratio")
       (! "Mercy caution: Suboptimal ratio — neutron yield increased")))
