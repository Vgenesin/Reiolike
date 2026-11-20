#!/usr/bin/env python
"""
Ricalcola con CLASS, da solo, il punto registrato prima di un segfault.

La run va lanciata con  export REIO_DUMP_POINT=<cartella>  nello script del job:
prima di ogni chiamata a CLASS ReioTheory scrive in <cartella>/punto_class_rank<N>.json
gli argomenti ESATTI che CLASS riceve (vedi ReioTheory._registra_punto_class).
Dopo il crash, il file del processo morto (il "task N" della riga
"srun: error: ... task N: Segmentation fault") contiene il punto colpevole.

Uso (sul cluster, su un nodo di calcolo, dalla radice del repository):
    srun -p boost_usr_prod -A INF26_indark -n 1 -c 4 --mem=8000 -t 00:20:00 \\
         python Tests/riprova_punto_class.py <cartella>/punto_class_rank0.json

Per ogni numero di thread OpenMP (1, 2, 4 ...) il punto si calcola tre volte in
un PROCESSO SEPARATO (un segfault non ferma gli altri).  Esiti possibili:
  - crasha sempre, con qualunque numero di thread -> e' il punto (la tabella x_e
    o la cosmologia): la stampa della tabella dice cosa ha di strano;
  - crasha solo con piu' thread, o non sempre -> e' CLASS in parallelo, non il punto;
  - non crasha mai -> il crash dipende dallo stato del processo nella run
    (memoria, punti precedenti), non dal punto da solo.
"""
import json
import os
import subprocess
import sys

import numpy as np

FIGLIO = r'''
import faulthandler, json, sys
faulthandler.enable()
from classy import Class
args = json.load(open(sys.argv[1]))
c = Class(); c.set(args); c.compute()
print("tau_reio = %.6f" % c.tau_reio())
'''


def descrivi(args):
    print("Cosmologia e opzioni passate a CLASS:")
    for k, v in sorted(args.items()):
        if k not in ("reio_inter_z", "reio_inter_xe"):
            print(f"   {k}: {v}")
    if "reio_inter_z" in args:
        z = np.array([float(s) for s in str(args["reio_inter_z"]).split(",")])
        xe = np.array([float(s) for s in str(args["reio_inter_xe"]).split(",")])
        dz = np.diff(z)
        print(f"Tabella x_e: {z.size} nodi (reio_inter_num = {args.get('reio_inter_num')}), "
              f"z in [{z[0]:.6g}, {z[-1]:.6g}], x_e in [{xe.min():.6g}, {xe.max():.6g}]")
        print(f"   passo minimo in z {dz.min():.3g} (fra z = {z[np.argmin(dz)]:.8g} e "
              f"{z[np.argmin(dz) + 1]:.8g}); z crescenti: {bool(np.all(dz > 0))}; "
              f"x_e finiti: {bool(np.all(np.isfinite(xe)))}")
        print(f"   lunghezze stringhe: z {len(str(args['reio_inter_z']))}, "
              f"x_e {len(str(args['reio_inter_xe']))} caratteri")


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    file_punto = sys.argv[1]
    args = json.load(open(file_punto))
    descrivi(args)
    massimo = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))
    thread = sorted({1, 2, min(4, massimo), massimo})
    print()
    for n in thread:
        for prova in range(3):
            env = dict(os.environ, OMP_NUM_THREADS=str(n), COBAYA_NOMPI="True")
            try:
                r = subprocess.run([sys.executable, "-c", FIGLIO, file_punto], env=env,
                                   capture_output=True, text=True, timeout=600)
                if r.returncode == 0:
                    esito = "OK      " + r.stdout.strip()
                elif r.returncode == -11:
                    esito = "SEGFAULT\n" + r.stderr[-1500:]
                else:
                    esito = f"ERRORE (codice {r.returncode})\n" + r.stderr[-1500:]
            except subprocess.TimeoutExpired:
                esito = "BLOCCATO (nessuna risposta in 600 s)"
            print(f"OMP_NUM_THREADS={n}, prova {prova + 1}: {esito}", flush=True)


if __name__ == "__main__":
    main()
