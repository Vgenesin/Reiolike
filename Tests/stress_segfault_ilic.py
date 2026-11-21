#!/usr/bin/env python
"""
Test A/B sui segfault di CLASS nelle run Ilic (LCDM libera, nov 2026).

Fatti di partenza:
  - i segfault (flex3_ilic, 3-pow Ilic) e il fondo NaN (pl2pow_ilic) compaiono
    SOLO nelle run con LCDM libera, dopo 10^4 - 10^5 punti;
  - il punto registrato prima del segfault, ricalcolato da solo, non crasha mai
    (Tests/riprova_punto_class.py, 1..14 thread): il guasto dipende dallo stato
    del processo, non dal punto.
Con la LCDM libera cambiano due cose rispetto alle run a cosmologia fissa:
  (a) il fondo di ricombinazione si ricalcola a ogni punto con una SECONDA
      istanza di CLASS (ReioTheory._calcola_fondo);
  (b) servono i Cl lensati fino a l = 2500 (lensing + hmcode).
Questo script valuta in ciclo punti presi da una catena con ReioTheory e un
requisito di Cl alti (come HiLLiPoP/CamSpec), senza likelihood CMB vere:

  --fondo punto   come nelle run: fondo ricalcolato a ogni cosmologia   (A)
  --fondo fisso   fondo calcolato una volta sola e riusato              (B)

Se A crasha e B no, la colpa e' della seconda istanza; se crashano entrambi,
e' il calcolo principale (lensing/hmcode/thread).  I numeri (punti valutati,
fondi anomali) finiscono in una riga per processo in --uscita; i crash li
conta lo script del job (ogni uscita con codice 139 = segfault).

Uso (dentro uno srun, dalla radice del repository):
  python Tests/stress_segfault_ilic.py --catena chains/flexknot_3flex_ilic/flexknot_3flex_ilic_PR4 \
         --modello flexknot --fondo punto --secondi 3600 --uscita run_stress/A
"""
import argparse
import faulthandler
import json
import os
import sys
import time

faulthandler.enable()
os.environ.setdefault("COBAYA_NOMPI", "1")
import numpy as np

RADICE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RADICE)

from cobaya.model import get_model
from cobaya.likelihood import Likelihood


class ClAlti(Likelihood):
    """Chiede i Cl lensati fino a l = 2500, come HiLLiPoP e CamSpec."""
    def get_requirements(self):
        return {"Cl": {"tt": 2500, "te": 2500, "ee": 2500}}

    def logp(self, **kw):
        cl = self.provider.get_Cl(ell_factor=True)
        return -1e-12 * float(np.sum(cl["tt"][2:]))


LCDM = {
    "logA": {"prior": {"min": 1.61, "max": 3.91}, "drop": True},
    "A_s": {"value": "lambda logA: 1e-10*np.exp(logA)"},
    "n_s": {"prior": {"min": 0.8, "max": 1.2}},
    "omega_b": {"prior": {"min": 0.005, "max": 0.1}},
    "omega_cdm": {"prior": {"min": 0.001, "max": 0.99}},
    "H0": {"prior": {"min": 40, "max": 100}},
}
MODELLI = {
    "flexknot": ({"reio_model": "flexknot", "reio_flex_interp": "pchip"},
                 {"reio_flex_z1": (5, 25), "reio_flex_z2": (5, 25),
                  "reio_flex_z3": (5, 25), "reio_flex_xHII2": (0, 1)}),
    "power_law": ({"reio_model": "power_law", "reio_z_beg": 20.0},
                  {"reio_z_end": (5, 20), "reio_alpha": (0, 20)}),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--catena", required=True, help="radice dei file .N.txt")
    ap.add_argument("--modello", choices=MODELLI, required=True)
    ap.add_argument("--fondo", choices=("punto", "fisso"), required=True)
    ap.add_argument("--secondi", type=float, required=True)
    ap.add_argument("--uscita", required=True)
    a = ap.parse_args()
    rank = os.environ.get("SLURM_PROCID", "0")

    opz, reio = MODELLI[a.modello]
    params = {k: {"prior": {"min": lo, "max": hi}} for k, (lo, hi) in reio.items()}
    params.update(LCDM)
    params.update({"N_ur": 0.00441, "tau_reio": None})
    info = {"params": params,
            "theory": {"Reiolike.ReioTheory": dict(
                opz, z_max=30.0, z_points=250,
                extra_args={"N_ncdm": 1, "deg_ncdm": 3, "m_ncdm": 0.02,
                            "T_ncdm": 0.71611, "non_linear": "hmcode",
                            "lensing": "yes"})},
            "likelihood": {"cl_alti": ClAlti}, "debug": False}
    model = get_model(info)
    th = model.theory["Reiolike.ReioTheory"]

    if a.fondo == "fisso":
        originale = th._pavimento_del_punto
        primo = {}

        def fondo_fisso(native_params):
            if not primo:
                originale(native_params)
                primo.update(fondo=th._fondo, controllo=th._fondo_controllo,
                             tabella=th._tabella_fondo, pav=th._pavimento)
            th._fondo, th._fondo_controllo = primo["fondo"], primo["controllo"]
            th._tabella_fondo, th._pavimento = primo["tabella"], primo["pav"]
        th._pavimento_del_punto = fondo_fisso

    nomi = list(reio) + ["logA", "n_s", "omega_b", "omega_cdm", "H0"]
    hdr = open(a.catena + ".1.txt").readline().split()[1:]
    blocchi = []
    for i in (1, 2, 3, 4):
        f = "%s.%d.txt" % (a.catena, i)
        if os.path.exists(f):
            blocchi.append(np.loadtxt(f, usecols=[hdr.index(p) for p in nomi]))
    punti = np.vstack(blocchi)
    rng = np.random.default_rng(int(time.time() * 1000) % 2**32 + int(rank))

    os.makedirs(a.uscita, exist_ok=True)
    rapporto = os.path.join(a.uscita, "rapporto_rank%s.txt" % rank)
    n, anomali, rifiutati, t0 = 0, 0, 0, time.time()
    while time.time() - t0 < a.secondi:
        p = dict(zip(nomi, map(float, punti[rng.integers(len(punti))])))
        r = model.logposterior(p)
        n += 1
        rifiutati += not np.isfinite(r.logpost)
        f = th._fondo
        if not (np.isfinite(f) and f > 0):
            anomali += 1
            print("FONDO ANOMALO", f, json.dumps(p), flush=True)
        if n % 200 == 0:
            print("[rank %s] %d punti, %.0f s" % (rank, n, time.time() - t0), flush=True)
    with open(rapporto, "a") as fh:
        fh.write("%s modello=%s fondo=%s punti=%d rifiutati=%d fondi_anomali=%d secondi=%.0f\n"
                 % (time.strftime("%Y-%m-%dT%H:%M:%S"), a.modello, a.fondo, n,
                    rifiutati, anomali, time.time() - t0))


if __name__ == "__main__":
    main()
