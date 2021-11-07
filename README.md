# Demographic modelling
This repository includes scripts of demographic models to perform demographic analyses in *moments*. 

12 models: 4 basic models adapted from Momigliano et al. (2020), IM (isolation with migration), AM (ancient migration), SC (secondary contact) and SI (strict isolation),
as well as their variations with exponential (G) and instantaneous (Ne) effective population size changes.

The models assume data is provided in form of folded jAFS, in this case computed from a VCF using easySFS (https://github.com/isaacovercast/easySFS).

The Optimizations.py script is used for running multiple optimizations, adapted by Momigliano et al. (2020) from a pipeline by Daniel Portik (https://github.com/dportik).

Citations:
Paolo Momigliano, Ann-Britt Florin, Juha Merilä, Biases in Demographic Modeling Affect Our Understanding of Recent Divergence, *Molecular Biology and Evolution*, Volume 38, Issue 7, July 2021, Pages 2967–2985.
