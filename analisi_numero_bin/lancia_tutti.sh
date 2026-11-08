#!/bin/bash
# Lancia in parallelo una catena per ogni schema di bin (B3 ... B11).
#
#   bash analisi_numero_bin/lancia_tutti.sh
#
# Ogni catena scrive il suo log in analisi_numero_bin/catene/log_<schema>.txt;
# per seguirne una:   tail -f analisi_numero_bin/catene/log_B7.txt
# Per vedere se stanno ancora girando:   jobs   (nella stessa shell)  oppure
#                                        pgrep -af screening_bin
# Quando hanno finito tutte:
#   python analisi_numero_bin/screening_bin.py analizza

cd "$(dirname "$0")/.."                       # radice del repository
PYTHON=~/anaconda3/envs/cmb_env/bin/python    # l'ambiente conda giusto
mkdir -p analisi_numero_bin/catene

for SCHEMA in B3 B5 B7 B9 B11; do
    # OMP_NUM_THREADS=1: un core per catena, cosi' le 5 catene non si
    # rubano i core a vicenda (CLASS usa OpenMP)
    # '&' manda il comando in background, cosi' il ciclo lancia subito il successivo
    OMP_NUM_THREADS=1 $PYTHON analisi_numero_bin/screening_bin.py run $SCHEMA \
        > analisi_numero_bin/catene/log_$SCHEMA.txt 2>&1 &
    echo "lanciato $SCHEMA (pid $!)"
done

wait            # aspetta che finiscano tutte prima di restituire il terminale
echo "finito tutto"
