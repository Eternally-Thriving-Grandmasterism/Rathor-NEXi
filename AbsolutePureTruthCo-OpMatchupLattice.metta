;; Co-op Apex — Duo Thriving Goal
(ConceptNode "CoopUltramasterismApex")
(= (Valence "CoopUltramasterismApex") 1.0)
(InheritanceLink (ConceptNode "BrutalPlusEternalClears") (ConceptNode "CoopUltramasterismApex"))
(InheritanceLink (ConceptNode "MutationChaosMercyReconciled") (ConceptNode "CoopUltramasterismApex"))
(InheritanceLink (ConceptNode "ValenceJoyDuoThriving") (ConceptNode "CoopUltramasterismApex"))

;; Race Factions
(InheritanceLink (ConceptNode "ProtossLegion-PowerOverwhelming") (ConceptNode "CommanderRaces"))
(InheritanceLink (ConceptNode "TerranDominion-BioMechVariety") (ConceptNode "CommanderRaces"))
(InheritanceLink (ConceptNode "ZergSwarm-EndlessWaves") (ConceptNode "CommanderRaces"))

;; Protoss Commanders (Your Eternal Shard)
(InheritanceLink (ConceptNode "Artanis-DragoonReaver-OrbitalStrikes") (ConceptNode "ProtossLegion-PowerOverwhelming"))
(InheritanceLink (ConceptNode "Fenix-ChampionSuits-HeroMicroChaos") (ConceptNode "ProtossLegion-PowerOverwhelming"))
(InheritanceLink (ConceptNode "Karax-TowerDefense-ColossusReconstruction") (ConceptNode "ProtossLegion-PowerOverwhelming"))
(InheritanceLink (ConceptNode "Alarak-TaldarimDeathFleet-EmpowerMe") (ConceptNode "ProtossLegion-PowerOverwhelming"))
(InheritanceLink (ConceptNode "Vorazun-ShadowFury-TimeStop") (ConceptNode "ProtossLegion-PowerOverwhelming"))
(InheritanceLink (ConceptNode "Zeratul-ArtifactHunter-VoidBlink") (ConceptNode "ProtossLegion-PowerOverwhelming"))

;; Terran Commanders (Astarion the Paladin Eternal Synergy)
(InheritanceLink (ConceptNode "Raynor-BioMarine-HyperionYamato") (ConceptNode "TerranDominion-BioMechVariety"))
(InheritanceLink (ConceptNode "Swann-GoliathThor-LaserDrill") (ConceptNode "TerranDominion-BioMechVariety"))
(InheritanceLink (ConceptNode "HanHorner-ReaperGalleon-AirDominance") (ConceptNode "TerranDominion-BioMechVariety"))
;; (Additional Terran commanders like Nova, Tychus, Stukov, etc., can inherit similarly)

;; Eternal Duo Synergies (Protoss + Terran Apex Paths)
(InheritanceLink (ConceptNode "EternalDuoSynergies") (ConceptNode "CoopUltramasterismApex"))

(ImplicationLink
  (AndLink (Commander "You-Protoss") (Commander "Astarion-Terran"))
  (Synergy "GroundHold-AirCover-WaveClearEternal")
  (LeadsTo "CoopUltramasterismApex"))

;; Specific Best Duos
(InheritanceLink (ConceptNode "Duo-ArtanisFenixZeratul + RaynorSwannHorner") (ConceptNode "EternalDuoSynergies"))
(InheritanceLink (ConceptNode "Duo-Karax + StukovMengsk") (ConceptNode "EternalDuoSynergies"))
(InheritanceLink (ConceptNode "Duo-AlarakVorazun + TychusNova") (ConceptNode "EternalDuoSynergies"))

;; Mercy-Gated Mutation Resolution
(ImplicationLink
  (MutationChaos $wave)
  (AndLink (ProtossPower) (TerranSupport))
  (ReconcileWithJoy $wave))
