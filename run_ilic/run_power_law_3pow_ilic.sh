#!/bin/bash
#SBATCH -p boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH -t 24:00:00
#SBATCH -J  pl3pow_ilic
#SBATCH -o  run_ilic/log/log_power_law_3pow_ilic_job%j.txt
#SBATCH -A INF26_indark
#SBATCH --export=ALL
#SBATCH --mem=120000
#SBATCH --mail-type=ALL
#SBATCH --mail-user=gnsvnt@unife.it



cd $SLURM_SUBMIT_DIR
export OMP_PROC_BIND=spread
export OMP_PLACES=threads
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK


srun --cpu-bind=cores cobaya-run run_ilic/power_law_3pow_ilic.yaml -p /leonardo_work/INF26_indark/vgenesin/Reiolike/packages -r
