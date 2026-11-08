#!/usr/bin/env python
"""
QUANTI BIN AD ALTO z REGGE LA CMB?  Seconda parte dello screening.

Dallo screening senza CMB (screening_bin.py) si tiene la parte a basso z, dove
decidono i dati astrofisici:
    bordi 5.35, 6.05, 6.45, 6.75, 7.15, 7.45
Sopra 7.45 si provano schemi annidati, C6 < C7 < C8 < C9, con
    astrofisica (corecon_config.yaml) + ELiCA hybridHL + Planck lowl.TT
e A_s libero.

Due comandi:

  # 1. scrive uno YAML per schema in analisi_numero_bin/cluster/
  #    (da fare una volta, anche in locale, poi si copiano sul cluster
  #     oppure si fa git pull)
  python analisi_numero_bin/screening_cmb.py yaml

  # 2. a catene finite: tabella con chi2 per likelihood, d, tau e r per bin
  python analisi_numero_bin/screening_cmb.py analizza

Sul cluster ogni YAML si lancia con lo sbatch accanto:
  sbatch analisi_numero_bin/cluster/binned_cmb.sbatch C7
"""
import glob
import os
import sys

import numpy as np

QUI = os.path.dirname(os.path.abspath(__file__))       # .../analisi_numero_bin
RADICE = os.path.dirname(QUI)
CARTELLA_YAML = os.path.join(QUI, "cluster")

# ---------------------------------------------------------------------------
# DA CONTROLLARE SUL CLUSTER: dove sta il repository di ELiCA.
# Se ELiCA e' installato nell'ambiente (pip install -e), metti None.
# ---------------------------------------------------------------------------
ELICA_PATH = None    # es. "/leonardo_work/INF26_indark/vgenesin/ELICA"

# ---------------------------------------------------------------------------
# GLI SCHEMI: parte bassa fissa + parte alta che cambia
# ---------------------------------------------------------------------------
BASSO = [5.35, 6.05, 6.45, 6.75, 7.15, 7.45]
SCHEMI = {
    "C6": BASSO + [25.0],
    "C7": BASSO + [9.5, 25.0],
    "C8": BASSO + [9.5, 12.0, 25.0],
    "C9": BASSO + [9.5, 12.0, 16.0, 25.0],
}
# Punto di partenza delle catene: le medie dello screening astrofisico (B9)
# per i 5 bin bassi; 0.1 per quelli alti.  Una CMB costa ~3 s a punto: partire
# vicino al posterior risparmia ore di burn-in.
PARTENZA_BASSO = [0.96, 0.81, 0.82, 0.64, 0.28]
PARTENZA_ALTO = 0.10

SIGMA_PRIOR = 1.0 / np.sqrt(12.0)    # sigma di U(0, 1)
BURN_IN = 0.3


# ===========================================================================
# COMANDO 'yaml'
# ===========================================================================
# Lo YAML e' scritto come TESTO (non con yaml.dump) cosi' resta leggibile e
# con i commenti, uguale a quelli che scrivi a mano.
TESTA = """\
# Screening del numero di bin con la CMB a larga scala: schema {nome}, {n} bin.
# Generato da analisi_numero_bin/screening_cmb.py: per cambiarlo, cambia lo
# script e rigeneralo (python analisi_numero_bin/screening_cmb.py yaml).
#
# bordi: {bordi}
#
# Si lancia dalla radice del repository:
#   sbatch analisi_numero_bin/cluster/binned_cmb.sbatch {nome}
# oppure a mano:
#   mpirun -n 4 cobaya-run analisi_numero_bin/cluster/binned_{nome}.yaml -r

debug: False

params:
  # --- i bin: frazione ionizzata x_HII, prior piatto U(0, 1) ---------------
{bin}
  # --- ampiezza: libera, la CMB la vede ------------------------------------
  A_s:
    prior: {{min: 1.5e-9, max: 3.5e-9}}
    ref: {{dist: norm, loc: 2.120e-9, scale: 3.0e-11}}
    proposal: 2.0e-11
    latex: A_s

  # --- cosmologia fissa: Pagano et al. 2020, Tab. 3, TTTEEE+lowE-S2 ---------
  n_s: 0.9651
  omega_b: 0.02237
  omega_cdm: 0.1201
  H0: 67.32
  N_ur: 2.0328

  # --- derivato -------------------------------------------------------------
  tau_reio:
    latex: \\tau_\\mathrm{{reio}}

theory:
  Reiolike.ReioTheory:
    reio_model: binned
    reio_bin_fixed_scheme: True     # True = usa reio_bin_edges qui sotto
    reio_bin_edges: {bordi}
    z_max: 30.0
    z_points: 250
    extra_args:
      N_ncdm: 1
      m_ncdm: 0.06
      T_ncdm: 0.71611

likelihood:
  Reiolike.ReioLike:
    corecon_config_file: corecon_config.yaml
    print_sigma_table: False
{elica}
  planck_2018_lowl.TT: null

sampler:
  mcmc:
    Rminus1_stop: 0.02
    Rminus1_cl_stop: 0.2
    max_tries: 10000
    learn_proposal: True

output: chains/binned_cmb/{nome}/binned_{nome}_corecon_ELiCA_lowTT
"""

BLOCCO_BIN = """\
  reio_xHII_bin{i}:          # z in [{z1}, {z2}]
    prior: {{min: 0.0, max: 1.0}}
    ref: {{dist: norm, loc: {loc}, scale: 0.05}}
    proposal: 0.05
    latex: x_\\mathrm{{HII}}^{{({i})}}
"""


def scrivi_yaml():
    os.makedirs(CARTELLA_YAML, exist_ok=True)
    if ELICA_PATH:
        elica = f"  elica.hybridHL:\n    python_path: {ELICA_PATH}"
    else:
        elica = "  elica.hybridHL: null"
    for nome, bordi in SCHEMI.items():
        n = len(bordi) - 1
        partenze = PARTENZA_BASSO + [PARTENZA_ALTO] * (n - len(PARTENZA_BASSO))
        bin_txt = "".join(
            BLOCCO_BIN.format(i=i + 1, z1=bordi[i], z2=bordi[i + 1], loc=partenze[i])
            for i in range(n))
        testo = TESTA.format(nome=nome, n=n, bordi=bordi, bin=bin_txt, elica=elica)
        file = os.path.join(CARTELLA_YAML, f"binned_{nome}.yaml")
        with open(file, "w") as f:
            f.write(testo)
        print("scritto", os.path.relpath(file, RADICE))


# ===========================================================================
# COMANDO 'analizza'
# ===========================================================================
def leggi_catene(nome):
    """
    Legge TUTTE le catene di uno schema (con mpirun -n 4 cobaya scrive
    <prefisso>.1.txt ... <prefisso>.4.txt), toglie il burn-in da ognuna e le
    mette insieme.  Restituisce {nome_colonna: array}.
    """
    prefisso = os.path.join(RADICE, "chains", "binned_cmb", nome,
                            f"binned_{nome}_corecon_ELiCA_lowTT")
    files = sorted(glob.glob(prefisso + ".[0-9]*.txt"))   # .1.txt, .2.txt, ...
    if not files:
        return None
    with open(files[0]) as f:
        colonne = f.readline().lstrip("#").split()
    pezzi = []
    for file in files:
        d = np.loadtxt(file)
        pezzi.append(d[int(BURN_IN * len(d)):])
    dati = np.vstack(pezzi)
    return {c: dati[:, k] for k, c in enumerate(colonne)}, len(files)


def media_e_sigma(x, peso):
    m = np.average(x, weights=peso)
    return m, np.sqrt(np.average((x - m) ** 2, weights=peso))


def analizza():
    riepilogo = []
    for nome, bordi in SCHEMI.items():
        letto = leggi_catene(nome)
        if letto is None:
            print(f"{nome}: catene non trovate, salto\n")
            continue
        c, n_catene = letto
        peso = c["weight"]
        # chi2 di ogni likelihood: colonne 'chi2__<nome likelihood>'
        chi2_singoli = {k[len("chi2__"):]: v for k, v in c.items() if k.startswith("chi2__")}
        i_best = np.argmin(c["chi2"])                 # il punto migliore visitato
        d = media_e_sigma(c["chi2"], peso)[1] ** 2 / 2.0
        tau = media_e_sigma(c["tau_reio"], peso)

        print(f"=== {nome}: {len(bordi) - 1} bin, {n_catene} catene, {len(peso)} campioni")
        print(f"    chi2 al best fit = {c['chi2'][i_best]:.2f}  di cui  " +
              ",  ".join(f"{k} {v[i_best]:.2f}" for k, v in chi2_singoli.items()))
        print(f"    d = {d:.1f}    tau = {tau[0]:.4f} +- {tau[1]:.4f}")
        for i in range(1, len(bordi)):
            m, s = media_e_sigma(c[f"reio_xHII_bin{i}"], peso)
            nota = "  <- solo prior" if s / SIGMA_PRIOR > 0.8 else ""
            print(f"    bin {i:2d}  {bordi[i-1]:5.2f}-{bordi[i]:5.2f}  "
                  f"x_HII {m:.3f} +- {s:.3f}  r {s / SIGMA_PRIOR:.2f}{nota}")
        print()
        riepilogo.append((nome, len(bordi) - 1, c["chi2"][i_best], d, tau[0]))

    print(f"{'':12s} {'bin in piu':>10s} {'calo chi2':>10s} {'aumento d':>10s} {'spostamento tau':>16s}")
    for a, b in zip(riepilogo, riepilogo[1:]):
        print(f"{a[0]:>4s} -> {b[0]:4s}  {b[1] - a[1]:10d} {a[2] - b[2]:10.2f} "
              f"{b[3] - a[3]:10.1f} {b[4] - a[4]:+16.4f}")
    print("\nSe tau si sposta ma chi2 e d no, lo spostamento e' effetto del prior dei\n"
          "bin nuovi, non informazione dai dati.")


if __name__ == "__main__":
    comando = sys.argv[1] if len(sys.argv) > 1 else ""
    if comando == "yaml":
        scrivi_yaml()
    elif comando == "analizza":
        analizza()
    else:
        print(__doc__)
