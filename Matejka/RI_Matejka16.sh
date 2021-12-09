#!/bin/bash
#$ -M mmulleri@nd.edu	 # Email address for job notification
#$ -m abe	 	           # Send mail when job begins, ends and aborts
#$ -q long
#$ -pe smp 4
#$ -N matejka16        # give the job a name

# [optional] The MATLABPATH variable is set in the Matlab script to add additional
# directories to the internal search paths.
export MATLABPATH=.:..

module load matlab

profile -memory on;

matlab -nosplash -r "maxNumCompThreads(4);" < RI_Matejka16_init.m
matlab -nosplash -r "maxNumCompThreads(4);" < RI_Matejka16_main.m
matlab -nosplash -r "maxNumCompThreads(4);" < RI_Matejka16_figures.m
matlab -nosplash -r "maxNumCompThreads(4);" < RI_Matejka16_probsets.m
matlab -nosplash -r "maxNumCompThreads(4);" < RI_Matejka16_Appendix.m
