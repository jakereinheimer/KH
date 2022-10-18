#!/bin/sh -l
# FILENAME: submit_job.slurm

#SBATCH -A PHY220053
#SBATCH -p wholenode # the default queue is "wholenode" queue
#SBATCH --nodes=4
#SBATCH --ntasks=512 
#SBATCH --time=4:00:00
#SBATCH -J Virgo_one
#SBATCH -o cluster.o%j
#SBATCH -e cluster.e%j
#SBATCH --mail-user=trungha@my.unt.edu
#SBATCH --mail-type=all

module load hdf5

# Launch MPI code
mpirun -np $SLURM_NTASKS $HOME/athena_KH/athena_KH/bin/athena -i athinput.cluster-dev
