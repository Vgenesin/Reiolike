#!/bin/bash
#SBATCH -p boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=2
#SBATCH -t 24:00:00
#SBATCH -J  tanh_planck_dz_libero_soloCMB
#SBATCH -o  run_solo_cmb/log/log_tanh_planck_dz_libero_soloCMB_job%j.txt
#SBATCH -A INF26_indark
#SBATCH --export=ALL
#SBATCH --mem=16000
#SBATCH --mail-type=ALL
#SBATCH --mail-user=gnsvnt@unife.it



cd $SLURM_SUBMIT_DIR
export OMP_PROC_BIND=spread
export OMP_PLACES=threads
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK


srun --cpu-bind=cores cobaya-run run_solo_cmb/tanh_planck_dz_libero_soloCMB.yaml -p /leonardo_work/INF26_indark/vgenesin/Reiolike/packages -r
