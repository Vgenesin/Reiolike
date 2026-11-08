#!/usr/bin/env python
"""
QUANTI BIN SERVONO?  Screening del modello binned sui soli dati astrofisici.

Idea
----
Si prende una sequenza di schemi di bin ANNIDATI (ogni schema contiene tutti i
bordi del precedente piu' qualcuno nuovo): 3, 5, 7, 9, 11 bin.
Per ognuno si fa girare una catena MCMC con cobaya (solo corecon_config.yaml,
niente CMB) e poi si guardano tre numeri:

  1. chi2_min                il miglior fit.  Scende sempre aggiungendo bin;
                             conta QUANTO scende rispetto ai parametri aggiunti.
  2. r_i = sigma_post/sigma_prior,  bin per bin.
                             Prior x_HII ~ U(0,1)  ->  sigma_prior = 1/sqrt(12) = 0.289.
                             r ~ 1   : il bin lo decide il prior (dati muti)
                             r < 0.5 : il bin e' vincolato dai dati
  3. d = 2 Var(log L)        numero effettivo di parametri vincolati
                             (Handley & Lemos 2019, arXiv:1903.06682).
                             Se aggiungendo bin d non cresce, i bin nuovi sono
                             solo prior: si sta overfittando / sprecando parametri.

Lo schema giusto e' quello oltre il quale d smette di crescere e chi2_min
smette di scendere in modo significativo.

Come si usa (dalla radice del repository, ambiente conda cmb_env)
----------------------------------------------------------------
  # 1. una catena per UNO schema (es. 7 bin):
  python analisi_numero_bin/screening_bin.py run B7

  # 2. tutte le catene in parallelo (vedi lancia_tutti.sh):
  bash analisi_numero_bin/lancia_tutti.sh

  # 3. quando le catene hanno finito, la tabella riassuntiva:
  python analisi_numero_bin/screening_bin.py analizza

Le catene finiscono in analisi_numero_bin/catene/<schema>/.
Se una catena si interrompe, rilanciando il comando 'run' riparte da dove era
(e' l'opzione resume=True di cobaya, la stessa del '-r' da terminale).
"""
import os
import sys

import numpy as np

# ---------------------------------------------------------------------------
# Percorsi: lo script si lancia dalla radice del repository, cosi' cobaya trova
# il pacchetto Reiolike e corecon_config.yaml con percorsi relativi.
# ---------------------------------------------------------------------------
QUI = os.path.dirname(os.path.abspath(__file__))       # .../analisi_numero_bin
RADICE = os.path.dirname(QUI)                           # radice del repository
sys.path.insert(0, RADICE)
os.chdir(RADICE)
CARTELLA_CATENE = os.path.join(QUI, "catene")


# ===========================================================================
# GLI SCHEMI DI BIN  (bordi, non centri: N bin = N+1 bordi)
# ===========================================================================
# B11 e' lo schema che il pacchetto costruisce da solo con
# reio_bin_fixed_scheme: False (bordi nei vuoti fra i dati del catalogo
# attuale, piu' le code 12, 16, 25).  Gli altri si ottengono TOGLIENDO bordi,
# cosi' ogni schema e' contenuto nel successivo:  B3 < B5 < B7 < B9 < B11.
# Il primo bordo (5.35) e l'ultimo (25) restano sempre: sotto 5.35 il modello
# mette x_HII = 1, sopra 25 mette x_HII = 0.
#
# Se cambi il catalogo, ricontrolla B11 con:
#   python analisi_numero_bin/screening_bin.py bordi
SCHEMI = {
    "B3":  [5.35,                   6.75,                    9.5,             25.0],
    "B5":  [5.35, 6.05,             6.75,       7.45,        9.5,             25.0],
    "B7":  [5.35, 6.05,             6.75,       7.45,  8.3,  9.5, 12.0,       25.0],
    "B9":  [5.35, 6.05, 6.45, 6.75, 7.15, 7.45,        8.3,  9.5, 12.0,       25.0],
    "B11": [5.35, 6.05, 6.45, 6.75, 7.15, 7.45, 7.8,   8.3,  9.5, 12.0, 16.0, 25.0],
}

SIGMA_PRIOR = 1.0 / np.sqrt(12.0)    # deviazione standard di U(0, 1)
BURN_IN = 0.3                        # frazione iniziale di catena scartata


# ===========================================================================
# LA CONFIGURAZIONE DI COBAYA (e' lo YAML di sempre, scritto come dizionario)
# ===========================================================================
def configurazione_cobaya(nome_schema):
    """Restituisce il dizionario 'info' di cobaya per uno schema di bin."""
    bordi = SCHEMI[nome_schema]
    n_bin = len(bordi) - 1

    # --- parametri campionati: un x_HII per bin, prior piatto U(0, 1) -------
    params = {}
    for i in range(1, n_bin + 1):
        params[f"reio_xHII_bin{i}"] = {
            "prior": {"min": 0.0, "max": 1.0},
            "ref": {"dist": "uniform", "min": 0.2, "max": 0.8},  # punto di partenza
            "proposal": 0.1,
            "latex": rf"x_\mathrm{{HII}}^{{({i})}}",
        }

    # --- cosmologia fissa: Pagano et al. 2020, Tab. 3, TTTEEE+lowE-S2 ----------
    # (identica a run_tanh_class_cluster.yaml; ai dati astrofisici serve solo
    #  per Y_He e per calcolare tau, che salviamo come derivato)
    params.update({
        "A_s": 2.120e-9, "n_s": 0.9651, "omega_b": 0.02237,
        "omega_cdm": 0.1201, "H0": 67.32, "N_ur": 2.0328,
        "tau_reio": {"latex": r"\tau_\mathrm{reio}"},   # derivato
    })

    return {
        "params": params,
        "theory": {
            "Reiolike.ReioTheory": {
                "reio_model": "binned",
                "reio_bin_fixed_scheme": True,   # True = usa i bordi espliciti qui sotto
                "reio_bin_edges": bordi,
                "z_max": 30.0,
                "z_points": 250,
                "extra_args": {"N_ncdm": 1, "m_ncdm": 0.06, "T_ncdm": 0.71611},
            }
        },
        "likelihood": {
            "Reiolike.ReioLike": {
                "corecon_config_file": "corecon_config.yaml",
                "print_sigma_table": False,
            }
        },
        "sampler": {
            "mcmc": {
                "Rminus1_stop": 0.02,       # Gelman-Rubin sulle medie
                "Rminus1_cl_stop": 0.2,     # ... e sugli intervalli
                "max_tries": 10000,
                "learn_proposal": True,
            }
        },
        "output": os.path.join(CARTELLA_CATENE, nome_schema, nome_schema),
        "resume": True,                    # = cobaya-run -r
    }


# ===========================================================================
# COMANDO 'run': lancia la catena di uno schema
# ===========================================================================
def lancia_catena(nome_schema):
    from cobaya.run import run          # cobaya.run.run(info) = cobaya-run file.yaml
    run(configurazione_cobaya(nome_schema))


# ===========================================================================
# COMANDO 'analizza': legge le catene e stampa la tabella
# ===========================================================================
def leggi_catena(nome_schema):
    """
    Legge il file di catena scritto da cobaya (<schema>.1.txt) con numpy.

    Il file e' una tabella di testo: la prima riga e' l'intestazione con i nomi
    delle colonne preceduti da '#', poi una riga per campione.  Colonne utili:
      weight          quante volte la catena e' rimasta su quel punto
      reio_xHII_binN  i parametri
      tau_reio        derivato
      chi2            -2 log L totale (qui c'e' una sola likelihood)
    """
    file = os.path.join(CARTELLA_CATENE, nome_schema, f"{nome_schema}.1.txt")
    if not os.path.exists(file):
        return None
    with open(file) as f:
        nomi = f.readline().lstrip("#").split()      # intestazione
    dati = np.loadtxt(file)                           # i numeri
    dati = dati[int(BURN_IN * len(dati)):]            # via il burn-in
    return {nome: dati[:, k] for k, nome in enumerate(nomi)}


def media_e_sigma(x, peso):
    """Media e deviazione standard pesate con i 'weight' di cobaya."""
    m = np.average(x, weights=peso)
    return m, np.sqrt(np.average((x - m) ** 2, weights=peso))


def analizza():
    print(f"burn-in scartato: {BURN_IN:.0%} di ogni catena")
    print(f"sigma_prior di un bin = 1/sqrt(12) = {SIGMA_PRIOR:.3f}\n")

    riepilogo = []
    for nome, bordi in SCHEMI.items():
        c = leggi_catena(nome)
        if c is None:
            print(f"{nome}: catena non trovata, salto\n")
            continue
        peso = c["weight"]
        n_bin = len(bordi) - 1

        # 1. miglior fit: il chi2 piu' basso visitato dalla catena
        #    (approssimazione per eccesso del vero minimo, va bene per confronto)
        chi2_min = c["chi2"].min()

        # 3. dimensionalita' bayesiana  d = 2 Var(log L) = Var(chi2) / 2
        _, sigma_chi2 = media_e_sigma(c["chi2"], peso)
        d = sigma_chi2 ** 2 / 2.0

        tau, sigma_tau = media_e_sigma(c["tau_reio"], peso)
        print(f"=== {nome}: {n_bin} bin, {len(peso)} campioni dopo il burn-in")
        print(f"    chi2_min = {chi2_min:.2f}    d = {d:.1f}    "
              f"tau = {tau:.4f} +- {sigma_tau:.4f}")

        # 2. bin per bin: media, sigma e r = sigma / sigma_prior
        print(f"    {'bin':>4s} {'z':>13s} {'x_HII':>7s} {'sigma':>6s} {'r':>5s}")
        for i in range(1, n_bin + 1):
            m, s = media_e_sigma(c[f"reio_xHII_bin{i}"], peso)
            r = s / SIGMA_PRIOR
            nota = "  <- solo prior" if r > 0.8 else ""
            print(f"    {i:4d} {bordi[i-1]:5.2f} - {bordi[i]:5.2f} "
                  f"{m:7.3f} {s:6.3f} {r:5.2f}{nota}")
        print()
        riepilogo.append((nome, n_bin, chi2_min, d, tau, sigma_tau))

    # --- confronto fra schemi consecutivi ----------------------------------
    print("Confronto fra schemi successivi")
    print(f"{'da':>4s} -> {'a':4s} {'bin in piu':>10s} {'calo chi2':>10s} {'aumento d':>10s}")
    for prima, dopo in zip(riepilogo, riepilogo[1:]):
        print(f"{prima[0]:>4s} -> {dopo[0]:4s} {dopo[1] - prima[1]:10d} "
              f"{prima[2] - dopo[2]:10.2f} {dopo[3] - prima[3]:10.1f}")
    print("\nCome leggerlo: un bin in piu' 'vale' se il chi2 cala di almeno ~2 per\n"
          "bin aggiunto (criterio di Akaike) e se d cresce. Se d resta fermo, i bin\n"
          "nuovi sono decisi dal prior.")


# ===========================================================================
# COMANDO 'bordi': controlla che B11 sia ancora lo schema dai dati
# ===========================================================================
def controlla_bordi():
    from cobaya.model import get_model   # costruisce il modello senza campionare
    info = configurazione_cobaya("B11")
    info.pop("sampler"); info.pop("output"); info.pop("resume")
    tema = info["theory"]["Reiolike.ReioTheory"]
    tema["reio_bin_fixed_scheme"] = False      # bordi dai dati
    tema.pop("reio_bin_edges")
    modello = get_model(info)
    bordi = modello.theory["Reiolike.ReioTheory"]._binned_edges()
    print("bordi dai dati :", [round(float(b), 3) for b in bordi])
    print("B11 nello script:", SCHEMI["B11"])


# ===========================================================================
if __name__ == "__main__":
    comando = sys.argv[1] if len(sys.argv) > 1 else ""
    if comando == "run" and len(sys.argv) == 3 and sys.argv[2] in SCHEMI:
        lancia_catena(sys.argv[2])
    elif comando == "analizza":
        analizza()
    elif comando == "bordi":
        controlla_bordi()
    else:
        print(__doc__)
        print("Schemi disponibili:", ", ".join(SCHEMI))
