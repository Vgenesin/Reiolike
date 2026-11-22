#!/bin/bash
#SBATCH -p boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH -t 18:00:00
#SBATCH --time-min=6:00:00
#SBATCH -J  csld_fisso05_noelica
#SBATCH -o  run_camspec_lens_desi/log/log_tanh_class_dz_fisso05_corecon_noelica_camspec_lens_desi_job%j.txt
#SBATCH -A INF26_indark
#SBATCH --export=ALL
#SBATCH --mem=120000
#SBATCH --mail-type=ALL
#SBATCH --mail-user=gnsvnt@unife.it

cd $SLURM_SUBMIT_DIR
export OMP_PROC_BIND=spread
export OMP_PLACES=threads
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
# in caso di segfault Python stampa dove si trovava (dentro CLASS o no)
export PYTHONFAULTHANDLER=1
# prima di ogni chiamata a CLASS scrive il punto (un file per processo): dopo un segfault
# Tests/riprova_punto_class.py lo ricalcola da solo
export REIO_DUMP_POINT=chains/tanh_class_dz_fisso05_corecon_noelica_camspec_lens_desi/punti_class

srun --cpu-bind=cores cobaya-run run_camspec_lens_desi/tanh_class_dz_fisso05_corecon_noelica_camspec_lens_desi.yaml -p /leonardo_work/INF26_indark/vgenesin/Reiolike/packages -r
