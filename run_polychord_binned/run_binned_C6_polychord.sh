#!/bin/bash
#SBATCH -p boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --cpus-per-task=2
#SBATCH -t 24:00:00
#SBATCH -J  pc_binned_C6
#SBATCH -o  pc_binned_C6.log
#SBATCH -A INF26_indark
#SBATCH --export=ALL
#SBATCH --mem=64000
#SBATCH --mail-type=ALL
#SBATCH --mail-user=gnsvnt@unife.it

# PolyChord si parallelizza sui punti vivi: 16 processi MPI (175 punti vivi).
# 16 x 2 CPU = 32 core = un nodo intero; 4 GB per processo come nelle run MCMC.

cd $SLURM_SUBMIT_DIR
export OMP_PROC_BIND=spread
export OMP_PLACES=threads
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK


srun --cpu-bind=cores cobaya-run run_polychord_binned/binned_C6_polychord.yaml -p /leonardo_work/INF26_indark/vgenesin/Reiolike/packages -r
