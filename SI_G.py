import os
import matplotlib
matplotlib.use('PDF')
import moments
import random
import pylab
import matplotlib.pyplot as plt
import numpy 
from numpy import array
from datetime import datetime
import Optimize_Functions

from moments import Misc,Spectrum,Numerics,Manips,Integration,Demographics1D,Demographics2D
import sys

''' 
Carolin Dahms, April 2021
Written for Python 3.9
Python modules required:
-Numpy
-Scipy
-moments
-------------------------
If you use these scripts for your work, also cite the original versions :
    Portik, D.M., Leach, A.D., Rivera, D., Blackburn, D.C., Rdel, M.-O.,
    Barej, M.F., Hirschfeld, M., Burger, M., and M.K.Fujita. 2017.
    Evaluating mechanisms of diversification in a Guineo-Congolian forest
    frog using demographic model selection. Molecular Ecology 26: 52455263.
    doi: 10.1111/mec.14266
-------------------------
Daniel Portik
daniel.portik@gmail.com
https://github.com/dportik
Updated May 2018

'''

# inumpy.t data dictionary, first argument
infile=sys.argv[1]
# pops id (as in data dictionary), aguments 2 and 3
pop_ids=[sys.argv[2],sys.argv[3]]
#projection sizes, in ALLELES not individuals, arguments 4 and 5
#projections=[int(sys.argv[4]),int(sys.argv[5])]

numpy.set_printoptions(precision=3)    
 
data = moments.Spectrum.from_file ( infile )
print("\n\n============================================================================\nData for site frequency spectrum\n============================================================================\n")
# print("projection", projections)
print("sample sizes", data.sample_sizes)

sfs_sum = numpy.around(data.S(), 2)
print("Sum of SFS = ", sfs_sum, '\n', '\n')

def SI_G(params, ns):
    """
    Strict isolation model with exponential growth

    T1: Time in the past of split (in units of 2*Na generations) 
    nu10, nu20: Pop sizes after split
    T2: Time of change in Ne
    nu1,nu2: Pop sizes after T2
    nu1b: Final size of pop 1
    nu2b: Final size of pop 2
    n1, n2: Sample sizes of resulting Spectrum
    """
    nu10,nu20,nu1,nu2,nu1b,nu2b,T1,T2 = params

    sts = moments.LinearSystem_1D.steady_state_1D(ns[0] + ns[1])
    fs = moments.Spectrum(sts)
    fs = moments.Manips.split_1D_to_2D(fs, ns[0], ns[1])

    nu1_func = lambda t: nu10 * (nu1 / nu10) ** (t / T1)
    nu2_func = lambda t: nu20 * (nu2 / nu20) ** (t / T1)
    nu1b_func = lambda t: nu1 * (nu1b / nu1) ** (t / T2)
    nu2b_func = lambda t: nu2 * (nu2b / nu2) ** (t / T2)
    nu_func = lambda t: [nu1_func(t), nu2_func(t)]
    nub_func = lambda t: [nu1b_func(t), nu2b_func(t)]

    fs.integrate(nu_func, T1, dt_fac=0.01, m=numpy.array([[0, 0], [0, 0]]))
    fs.integrate(nub_func, T2, dt_fac=0.01, m=numpy.array([[0, 0],[0, 0]]))
    return fs

p_labels = "nu10,nu20,nu1,nu2,nu1b,nu2b,T1,T2"
upper = [20,20,20,20,20,20,10,10]
lower = [1e-3,1e-3,1e-3,1e-3,1e-3,1e-3,1e-3,1e-3]

reps = [20,20,30,30,30]
maxiters = [20,20,30,30,30]
folds = [3,2,2,2,1]

for i in range(1,6):
    prefix = infile+"_OPTI_Number_{}".format(i)
    Optimize_Functions.Optimize_Routine(data, prefix, "SI_G", SI_G, 5, 8, data_folded=True, param_labels = p_labels, in_upper=upper, in_lower=lower, reps = reps, maxiters = maxiters, folds = folds)