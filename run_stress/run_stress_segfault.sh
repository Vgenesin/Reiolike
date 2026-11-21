#!/bin/bash
# Test A/B sui segfault di CLASS con LCDM libera (Tests/stress_segfault_ilic.py).
# Un nodo, 8 processi indipendenti (niente MPI) da 4 thread, per 11 ore.
# Ogni processo che muore viene rilanciato; le uscite con codice 139 sono segfault.
# Si lancia dalla radice del repository, una volta per braccio:
#   sbatch -J stress_A --export=ALL,FONDO=punto run_stress/run_stress_segfault.sh
#   sbatch -J stress_B --export=ALL,FONDO=fisso run_stress/run_stress_segfault.sh
# Modello e catena di partenza (default flexknot su flex3_ilic) si cambiano con
#   --export=ALL,FONDO=punto,MODELLO=power_law,CATENA=chains/power_law_2pow_ilic/power_law_2pow_ilic_PR4
#SBATCH -p boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=4
#SBATCH -t 12:00:00
#SBATCH -o  run_stress/log/log_%x_job%j.txt
#SBATCH -A INF26_indark
#SBATCH --mem=120000

cd $SLURM_SUBMIT_DIR
export OMP_PROC_BIND=spread
export OMP_PLACES=threads
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTHONFAULTHANDLER=1
export COBAYA_NOMPI=1
MODELLO=${MODELLO:-flexknot}
CATENA=${CATENA:-chains/flexknot_3flex_ilic/flexknot_3flex_ilic_PR4}
USCITA=run_stress/esiti/${FONDO}_${MODELLO}_job${SLURM_JOB_ID}
export REIO_DUMP_POINT=$USCITA/punti_class
mkdir -p $USCITA
FINE=$(( $(date +%s) + 11*3600 ))

srun --cpu-bind=cores bash -c '
  while [ $(date +%s) -lt '$FINE' ]; do
    resto=$(( '$FINE' - $(date +%s) ))
    python Tests/stress_segfault_ilic.py --catena '$CATENA' --modello '$MODELLO' \
           --fondo '$FONDO' --secondi $resto --uscita '$USCITA'
    codice=$?
    echo "rank $SLURM_PROCID uscita $codice $(date +%FT%T)" >> '$USCITA'/uscite.txt
    [ $codice -ne 0 ] && cp '$USCITA'/punti_class/punto_class_rank$SLURM_PROCID.json \
        '$USCITA'/punti_class/crash_rank${SLURM_PROCID}_$(date +%s).json 2>/dev/null
  done'

echo "=== riassunto"; cat $USCITA/rapporto_rank*.txt
echo "segfault (139): $(grep -c "uscita 139" $USCITA/uscite.txt)   altre uscite non zero: $(grep -v "uscita 0\|uscita 139" $USCITA/uscite.txt | wc -l)"
