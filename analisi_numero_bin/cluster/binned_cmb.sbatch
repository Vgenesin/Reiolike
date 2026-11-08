#!/bin/bash
# Una run dello screening con CMB sul cluster (Leonardo).
#
#   sbatch analisi_numero_bin/cluster/binned_cmb.sbatch C7
#
# L'argomento (C6, C7, C8, C9) sceglie lo YAML: binned_<schema>.yaml.
# Si lancia dalla RADICE del repository Reiolike.
# Se il job finisce il tempo prima di convergere, rilancia lo stesso comando:
# '-r' (resume) fa ripartire le catene da dove erano.
#
#SBATCH --job-name=binned_cmb
#SBATCH --account=INF26_indark
#SBATCH --partition=boost_usr_prod     # senza, finisce in lrd_all_serial e fallisce
#SBATCH --time=24:00:00                # massimo della QOS normale
#SBATCH --nodes=1
#SBATCH --ntasks=4                     # 4 catene MCMC
#SBATCH --cpus-per-task=2              # 2 core OpenMP a catena per CLASS
#SBATCH --mem=16G
#SBATCH --output=binned_cmb_%x_%j.out  # %j = numero del job

SCHEMA=$1
if [ -z "$SCHEMA" ]; then
    echo "uso: sbatch analisi_numero_bin/cluster/binned_cmb.sbatch C7"; exit 1
fi

# --- ambiente: CONTROLLA che siano le stesse righe del tuo sbatch della tanh --
source ~/.bashrc
conda activate act_env

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK   # i 2 core a catena li usa CLASS

# srun lancia i 4 processi MPI (= mpirun -n 4 dentro SLURM)
srun cobaya-run analisi_numero_bin/cluster/binned_${SCHEMA}.yaml -r
