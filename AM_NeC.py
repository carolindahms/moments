import sys
import os
import numpy
from numpy import array
import pylab
from datetime import datetime
import moments
from moments import Misc,Spectrum,Numerics,Manips,Integration,Demographics1D,Demographics2D,Model_2pop_folded, Optimize_Functions

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
# print "projection", projections
print("sample sizes", data.sample_sizes)

sfs_sum = numpy.around(data.S(), 2)
print("Sum of SFS = ", sfs_sum, '\n', '\n')

def AM_NeC(params, ns):
    """
    Instantaneous Ne changes with ancient migration

    nu1= pop size for pop 1
    nu2=pop size for pop 2 
    nu1b=pop size for pop 1 at T2
    nu2b=pop size for pop 2 at T2
    T1= time of population split with migration
    T2= time of speciaton
    m12= proportion of pop 1 made up of migrants from pop 2
    m21= proportion of pop 2 made up of migrants from pop 1
    """
    nu1,nu2,nu1b,nu2b,T1,T2,m12,m21 = params

    sts = moments.LinearSystem_1D.steady_state_1D(ns[0] + ns[1])
    fs = moments.Spectrum(sts)
    fs = moments.Manips.split_1D_to_2D(fs, ns[0], ns[1])
    fs.integrate([nu1, nu2], T1, dt_fac=0.01, m = numpy.array([[0, m12], [m21,0 ]]))
    fs.integrate([nu1b, nu2b], T2, dt_fac=0.01, m=numpy.array([[0, 0], [0, 0]]))
    return fs

p_labels = "nu1,nu2,nu1b,nu2b,T1,T2,m12,m21"
upper = [100,100,100, 100, 10,10,200,200]
lower = [1e-3,1e-3,1e-3,1e-3,1e-3,1e-2,1e-3,1e-3]

reps = [20,20,30,30]
maxiters = [20,20,30,30]
folds = [3,2,2,1]

for i in range(1,6):
    prefix = infile+"_OPTI_Number_{}".format(i)
    Optimize_Functions.Optimize_Routine(data, prefix, "AM_NeC", AM_NeC, 4, 8, data_folded=True, param_labels = p_labels, in_upper=upper, in_lower=lower, reps = reps, maxiters = maxiters, folds = folds)      