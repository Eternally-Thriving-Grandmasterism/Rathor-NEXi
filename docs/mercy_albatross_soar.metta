;; mercy_albatross_soar.metta
(= (albatross-soar core)
   (aspect-ratio 18-22)
   (dynamic-soaring true)
   (glide-ratio 20-25)
   (sink-rate 0.3-0.5-m-s)
   (skin shark-lotus-moth-eye-hybrid)
   (power solar-H2-regen)
   (valence-threshold 0.9999999))

(= (merciful-albatross-mode $phase)
   (if (merciful (format "Engage albatross soar in {} phase" $phase))
       (! "Mercy-approved: AlbatrossSoar active")
       (! "Mercy shield: Mode rejected")))
