#!/usr/bin/env python
"""
DIAGNOSI DEL SEGFAULT DELLA RUN power_law_3pow_ilic (sul cluster).

La run si e' fermata con "Segmentation fault" sul processo 3 (catena .4) e si
ferma di nuovo a ogni ripartenza con -r.  Ripartendo, cobaya ricalcola per
prima cosa l'ULTIMO punto di ogni catena: se CLASS va in crash su uno di
quei punti, ogni ripartenza muore.  L'ultimo punto della .4 ha alpha = 1.0011,
z_end 8.58, z_beg 9.08 (una storia corta): vicino ad alpha = 1 la griglia
della power law ha nodi quasi doppi (passo 1e-5).  In locale quel punto NON
crasha (16 nov 2026, 13690 punti provati): se crasha qui, e' la CLASS del
cluster.

Cosa fa: per ogni punto da provare lancia un PROCESSO SEPARATO (cosi' un
segfault non ferma gli altri) che costruisce ReioTheory come nella run
(run_ilic/power_law_3pow_ilic.yaml: stessa theory, stessi extra_args) e
calcola la storia e i Cl fino a l = 2500 (quelli che chiede HiLLiPoP), senza
le likelihood di Planck (pesanti e non coinvolte: sono in Python).
Punti provati:
  1. l'ultimo punto di ciascuna delle 4 catene;
  2. il punto della .4 con alpha spostato: 1 esatto, 1 +- 1e-2, 1e-3, 1e-4, 1e-6.

Si lancia dalla radice del repository, sul cluster (pochi minuti):
    PYTHONFAULTHANDLER=1 python Tests/debug_segfault_3pow_ilic.py
oppure su un nodo di calcolo:
    srun -p boost_usr_prod -A INF26_indark -n 1 -c 2 --mem=8000 -t 00:30:00 \\
         python Tests/debug_segfault_3pow_ilic.py

Esito: una riga per punto, OK, CRASH (con il codice di uscita: -11 e' il
segfault) oppure BLOCCATO (nessuna risposta in TIMEOUT secondi: si stampa
dove era fermo, grazie a faulthandler).  Per i CRASH stampa anche il traceback di faulthandler, che
dice se il crash e' dentro CLASS (classy.compute).
"""
import glob
import json
import os
import subprocess
import sys

RADICE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
YAML = os.path.join(RADICE, "run_ilic", "power_law_3pow_ilic.yaml")
CATENE = os.path.join(RADICE, "chains", "power_law_3pow_ilic", "power_law_3pow_ilic_PR4")
PACKAGES = "/leonardo_work/INF26_indark/vgenesin/Reiolike/packages"
TIMEOUT = 300            # secondi per punto: in locale un punto ne richiede ~7
COSMO = ["reio_z_end", "reio_alpha", "reio_z_beg", "logA", "n_s", "omega_b", "omega_cdm", "H0"]

FIGLIO = r'''
import faulthandler, json, os, sys, contextlib, io
faulthandler.enable()
# se il calcolo si blocca: ogni 60 s stampa dove si trova (su stderr)
faulthandler.dump_traceback_later(60, repeat=True)
sys.path.insert(0, RADICE); os.chdir(RADICE)
import numpy as np
from cobaya.yaml import yaml_load_file
from cobaya.model import get_model
info = yaml_load_file(YAML)
pt = json.loads(PUNTO)
P = lambda: {"prior": {"min": -1e9, "max": 1e9}}
params = {k: P() for k in COSMO}
params["logA"]["drop"] = True
params["A_s"] = info["params"]["A_s"]
params["N_ur"] = info["params"]["N_ur"]
params["tau_reio"] = {}
def dummy(_self=None):
    _self.provider.get_Cl(ell_factor=True)
    return 0.0
lik = {"dummy": {"external": dummy, "requires": {"Cl": {"tt": 2500, "te": 2500, "ee": 2500}}}}
with contextlib.redirect_stdout(io.StringIO()):
    m = get_model({"theory": info["theory"], "likelihood": lik, "params": params,
                   "packages_path": PACKAGES}, stop_at_error=True)
r = m.logposterior(pt, as_dict=True)
print("RISULTATO", r["logpost"], r["derived"].get("tau_reio"), flush=True)
'''


def ultimi_punti():
    out = {}
    for f in sorted(glob.glob(CATENE + ".[0-9].txt")):
        colonne = open(f).readline().lstrip("#").split()
        ultima = [r for r in open(f) if r.strip() and not r.startswith("#")][-1].split()
        riga = dict(zip(colonne, map(float, ultima)))
        out[f"ultimo punto catena {f.split('.')[-2]}"] = {k: riga[k] for k in COSMO}
    return out


def prova(nome, punto):
    codice = (FIGLIO.replace("RADICE", repr(RADICE)).replace("YAML", repr(YAML))
              .replace("PACKAGES", repr(PACKAGES)).replace("COSMO", repr(COSMO))
              .replace("PUNTO", repr(json.dumps(punto))))
    # OMP_NUM_THREADS = CPU del job: senza, CLASS puo' aprire un thread per
    # ogni core del nodo (32) su 2 CPU assegnate e diventare lentissimo.
    env = dict(os.environ, PYTHONFAULTHANDLER="1",
               OMP_NUM_THREADS=os.environ.get("SLURM_CPUS_PER_TASK", "2"))
    descr = ", ".join(f"{k.replace('reio_', '')} {punto[k]:.6g}" for k in ("reio_z_end", "reio_alpha", "reio_z_beg"))
    try:
        p = subprocess.run([sys.executable, "-c", codice], capture_output=True, text=True,
                           env=env, timeout=TIMEOUT)
    except subprocess.TimeoutExpired as e:
        print(f"BLOCCATO {nome:32s} ({descr})  nessuna risposta in {TIMEOUT} s. Dove era:", flush=True)
        err = e.stderr.decode() if isinstance(e.stderr, bytes) else (e.stderr or "")
        coda = [r for r in err.splitlines() if r.strip()][-30:]
        print("       " + "\n       ".join(coda), flush=True)
        return
    ris = [r for r in p.stdout.splitlines() if r.startswith("RISULTATO")]
    if p.returncode == 0 and ris:
        print(f"OK     {nome:34s} ({descr})  logpost, tau = {ris[0].split(maxsplit=1)[1]}", flush=True)
    else:
        print(f"CRASH  {nome:34s} ({descr})  codice di uscita {p.returncode}", flush=True)
        coda = [r for r in p.stderr.splitlines() if r.strip()][-25:]
        print("       " + "\n       ".join(coda), flush=True)


if __name__ == "__main__":
    punti = ultimi_punti()
    if not punti:
        raise SystemExit(f"nessuna catena trovata in {CATENE}.*.txt")
    for nome, p in punti.items():
        prova(nome, p)
    base = punti.get("ultimo punto catena 4") or list(punti.values())[-1]
    for a in (1.0, 1.01, 0.99, 1.001, 0.999, 1.0001, 0.9999, 1.000001):
        prova(f"catena 4 con alpha = {a}", dict(base, reio_alpha=a))
