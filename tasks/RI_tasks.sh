#!/bin/bash
#$ -M mmulleri@nd.edu	 # Email address for job notification
#$ -m abe		           # Send mail when job begins, ends and aborts
#$ -q debug
#$ -pe smp 9
#$ -N RI_tasks         # give the job a name


# [optional] The MATLABPATH variable is set in the Matlab script to add additional
# directories to the internal search paths.
export MATLABPATH=.:..

module load matlab

rm -r tasks_output
mkdir tasks_output

matlab -nosplash < tasks.m
