# chromo

Monte Carlo simulation of chromosomal DNA with field theoretic treatment of epigenetic marks.

Features in this code package:
1. Introduce epigenetic marks as instances within the "Epigenmark” class.  This allows us to arbitrarily introduce epigenetic marks within the code
2. Create polymer chains as instances within the “Chromatin” class.  This allows us to initiate arbitrary number of chromosomes of different length and epigenetic sequence
3. Create Monte Carlo moves as instances within the “MCmove” class.  This facilitates assignment of move-specific properties and tracking of move success and efficiency (to be coupled to machine learning assignment of Monte Carlo moves)
4. Define the field calculations in the “Field” class.  The field properties bundled within a single object to facilitate coarse graining procedures.
