;; mercy_rebco_fabrication.metta
(= (rebco-tape core)
   (substrate hastelloy)
   (buffer-stack ibad-mgo + ceo2)
   (superconductor gd-bco 1-3-um)
   (stabilizer ag + cu)
   (ic 200-600-A 77-K)
   (field-performance 20-T 20-K)
   (mercy-gate true))

(= (merciful-rebco-production $yield)
   (if (>= $yield 0.8)
       (! "Mercy-approved: REBCO tape production — yield {}" $yield)
       (! "Mercy shield: Production rejected — low yield")))
