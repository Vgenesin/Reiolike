#!/bin/bash
#SBATCH -p boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=2
#SBATCH -t 04:00:00
#SBATCH -J  profilo_3pow
#SBATCH -o  profilo_3pow/log/log_%a_job%A.txt
#SBATCH -A INF26_indark
#SBATCH --export=ALL
#SBATCH --mem=16000
#SBATCH --array=0-28%10

# Job array: il job N minimizza il punto del profilo alla riga N+1 di
# profilo_3pow/lista.txt (0 = libero, 1-16 = alpha fissato, 17-28 = z_beg
# fissato).  %10 = al massimo 10 job insieme.  Job piccoli e corti (4 x 2 CPU,
# 16 GB, 4 h, come le run MCMC): partono con il backfill.
# 4 processi MPI = 4 x 4 partenze del minimizzatore (best_of 16).
#
# Si lancia dalla radice del repository:
#   sbatch profilo_3pow/run_profilo.sh
# Per rilanciare solo alcuni punti (es. se uno va oltre le 4 h):
#   sbatch --array=5,12 profilo_3pow/run_profilo.sh

cd $SLURM_SUBMIT_DIR
export OMP_PROC_BIND=spread
export OMP_PLACES=threads
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

NOME=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" profilo_3pow/lista.txt)
echo "job $SLURM_ARRAY_TASK_ID: $NOME"

srun --cpu-bind=cores cobaya-run profilo_3pow/yaml/$NOME.yaml -p /leonardo_work/INF26_indark/vgenesin/Reiolike/packages -r
