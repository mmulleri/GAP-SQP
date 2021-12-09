#!/bin/bash
#$ -M mmulleri@nd.edu	 # Email address for job notification
#$ -m abe		           # Send mail when job begins, ends and aborts
#$ -q long
#$ -pe smp 6
#$ -N RI_pf_B     # give the job a name

module load matlab
# [optional] The MATLABPATH variable is set in the Matlab script to add additional
# directories to the internal search paths.
export MATLABPATH=.:..

matlab -nosplash -r "RI_portfolio_main('B',6)" < RI_portfolio_main.m
