;; mercy_interlune_demo_mission.metta
(= (demo-mission 2028-2029) 
   (launch starship-hls) 
   (payload 50-80-tons) 
   (site south-pole-shackleton) 
   (duration 180-210-days) 
   (objectives harvest validate replicate return))

(= (mission-phase $phase $day)
   (case $phase
      ("Launch-Transit"  "Days -180 to -30")
      ("Deployment"      "Days -30 to 0")
      ("Harvest"         "Days 0–90")
      ("Replication"     "Days 90–180")
      ("Closure"         "Days 180–210")))

(= (merciful-mission $outcome)
   (if (and (>= valence 0.9999999) (dust < 0.005) (resource-use < 0.001))
       (! "Mercy-approved: Demo mission success — abundance seeded")
       (! "Mercy shield: Mission aborted — entropy detected")))
