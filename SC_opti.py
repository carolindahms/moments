import sys
import os
import numpy
from numpy import array
import moments
import pylab
from datetime import datetime
import Optimize_Functions
from moments import Misc,Spectrum,Numerics,Manips,Integration,Demographics1D,Demographics2D

# inumpy.t data dictionary, first argument
infile=sys.argv[1]
# pops id (as in data dictionary), aguments 2 and 3
pop_ids=[sys.argv[2],sys.argv[3]]
#projection sizes, in ALLELES not individuals, arguments 4 and 5
#projections=[int(sys.argv[4]),int(sys.argv[5])]

'''

General workflow:
 The optimization routine runs a user-defined number of rounds, each with a user-defined
 or predefined number of replicates. The starting parameters are initially random, but after
 each round is complete the parameters of the best scoring replicate from that round are
 used to generate perturbed starting parameters for the replicates of the subsequent round.
 The arguments controlling steps of the optimization algorithm (maxiter) and perturbation
 of starting parameters (fold) can be supplied by the user for more control across rounds.
 The user can also supply their own set of initial parameters, or set custom bounds on the
 parameters (upper_bound and lower_bound) to meet specific model needs. This flexibility
 should allow these scripts to be generally useful for model-fitting with any data set.
 
Outputs:
 For each model run, there will be a log file showing the optimization steps per replicate
 and a summary file that has all the important information. Here is an example of the output
 from a summary file, which will be in tab-delimited format:
 
 Model	Replicate	log-likelihood	AIC	chi-squared	theta	optimized_params(nu1, nu2, m, T)
 sym_mig	Round_1_Replicate_1	-1684.99	3377.98	14628.4	383.04	0.2356,0.5311,0.8302,0.182
 sym_mig	Round_1_Replicate_2	-2255.47	4518.94	68948.93	478.71	0.3972,0.2322,2.6093,0.611
 sym_mig	Round_1_Replicate_3	-2837.96	5683.92	231032.51	718.25	0.1078,0.3932,4.2544,2.9936
 sym_mig	Round_1_Replicate_4	-4262.29	8532.58	8907386.55	288.05	0.3689,0.8892,3.0951,2.8496
 sym_mig	Round_1_Replicate_5	-4474.86	8957.72	13029301.84	188.94	2.9248,1.9986,0.2484,0.3688
Notes/Caveats:
 The likelihood and AIC returned represent the true likelihood only if the SNPs are
 unlinked across loci. For ddRADseq data where a single SNP is selected per locus, this
 is true, but if SNPs are linked across loci then the likelihood is actually a composite
 likelihood and using something like AIC is no longer appropriate for model comparisons.
 See the discussion group for more information on this subject. 
Citations:

For this modified version running on moments cite Momigliano et al 2020 

 If you use these scripts for your work, also cite theroiginal versions :
    Portik, D.M., Leach, A.D., Rivera, D., Blackburn, D.C., Rdel, M.-O.,
    Barej, M.F., Hirschfeld, M., Burger, M., and M.K.Fujita. 2017.
    Evaluating mechanisms of diversification in a Guineo-Congolian forest
    frog using demographic model selection. Molecular Ecology 26: 52455263.
    doi: 10.1111/mec.14266
-------------------------
Written for Python 2.7
Python modules required:
-Numpy
-Scipy
-moments
-------------------------
Daniel Portik
daniel.portik@gmail.com
https://github.com/dportik
Updated May 2018
'''

#===========================================================================
# Import data to create joint-site frequency spectrum
#===========================================================================

numpy.set_printoptions(precision=3)    
 
data = moments.Spectrum.from_file ( infile )
print ("\n\n============================================================================\nData for site frequency spectrum\n============================================================================\n")
# print ("projection", projections)
print ("sample sizes", data.sample_sizes)

sfs_sum = numpy.around(data.S(), 2)
print ("Sum of SFS = ", sfs_sum, '\n', '\n')

#================================================================================
# Here is an example of using a custom model within this script
#================================================================================
'''
 We will use a function from the Optimize_Functions.py script:
 
 Optimize_Routine(data, outfile, model_name, func, rounds, param_number, fs_folded=True, reps=None, maxiters=None, folds=None, in_params=None, in_upper=None, in_lower=None, param_labels=" ")
 
   Mandatory Arguments =
    data:  spectrum object name
    outfile:  prefix for output naming
    model_name: a label to slap on the output files; ex. "no_mig"
    func: access the model function from within 'dadi_Run_Optimizations.py' or from a separate python model script, ex. after importing Models_2D, calling Models_2D.no_mig
    rounds: number of optimization rounds to perform
    param_number: number of parameters in the model selected (can count in params line for the model)
    fs_folded: A Boolean value (True or False) indicating whether the empirical fs is folded (True) or not (False).
   Optional Arguments =
     reps: a list of integers controlling the number of replicates in each of the optimization rounds
     maxiters: a list of integers controlling the maxiter argument in each of the optimization rounds
     folds: a list of integers controlling the fold argument when perturbing input parameter values
     in_params: a list of parameter values 
     in_upper: a list of upper bound values
     in_lower: a list of lower bound values
     param_labels: list of labels for parameters that will be written to the output file to keep track of their order
'''

def SC(params, ns):
    """
    Model with secondary contact

    nu1: pop size for pop 1
    nu2: pop size for pop 1 
    T1: time of split
    T2: time of secondary contact
    m12: migration rate from pop 2 to pop 1
    m21: migration rate from pop 1 to pop 2
    """
    nu1,nu2,T1,T2,m12,m21 = params

    sts = moments.LinearSystem_1D.steady_state_1D(ns[0] + ns[1])
    fs = moments.Spectrum(sts)
    fs = moments.Manips.split_1D_to_2D(fs, ns[0], ns[1])
    fs.integrate([nu1, nu2], T1, dt_fac=0.01, m=numpy.array([[0, 0], [0, 0]]))
    fs.integrate([nu1, nu2], T2, dt_fac=0.01, m=numpy.array([[0, m12], [m21, 0]]))
    return fs

'''You can also be very explicit about the optimization routine, controlling what happens
across each round. Let's keep the three rounds, but change the number of replicates,
the maxiter argument, and fold argument each time. We'll need to create a list of values
for each of these, that has three values within (to match three rounds).
'''

p_labels = "nu1,nu2,T1,T2,m12,m21"
upper = [20,20,10,10,200,200]
lower = [1e-3,1e-3,1e-3,1e-3,1e-3,1e-3]

reps = [20,20,20]
maxiters = [20,20,20]
folds = [3,2,1]

for i in range(1,6):
    prefix = infile+"_OPTI_Number_{}".format(i)
    Optimize_Functions.Optimize_Routine(data, prefix, "SC", SC, 3, 6, data_folded=True, param_labels = p_labels, in_upper=upper, in_lower=lower, reps = reps, maxiters = maxiters, folds = folds)
