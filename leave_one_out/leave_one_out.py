#!/usr/bin/env python
"""
TEST LEAVE-ONE-OUT SUI DATI ASTROFISICI.

Si rifa' la stessa run togliendo UN dataset alla volta, e si guarda quanto si
spostano i parametri rispetto alla run con tutti i dati.  Un dataset che sposta
molto i risultati (o che da solo ha un chi2 enorme) va guardato da vicino.

Run di partenza: run_tanh/tanh_planck_dz_libero.yaml
  (tanh alla CLASS, z_re e dz liberi, A_s libero, astrofisica + ELiCA +
   Planck lowl.TT, 3 neutrini massivi).
Ogni run del test e' QUELLO YAML con due sole righe cambiate:
  corecon_config_file  ->  il file dei dataset senza il dataset tolto
  output               ->  una cartella sua

Tre comandi, dalla radice del repository:

  # 1. scrive i file (da fare una volta, poi git push / git pull sul cluster)
  python leave_one_out/leave_one_out.py prepara

  # 2. sul cluster: lancia tutte le run (una sbatch per run)
  bash leave_one_out/lancia_tutti.sh

  # 3. a catene finite: tabella degli spostamenti rispetto al riferimento
  python leave_one_out/leave_one_out.py analizza
"""
import glob
import os
import re
import sys

import numpy as np
import yaml

QUI = os.path.dirname(os.path.abspath(__file__))       # .../leave_one_out
RADICE = os.path.dirname(QUI)
YAML_BASE = os.path.join(RADICE, "run_tanh", "tanh_planck_dz_libero.yaml")
SBATCH_BASE = os.path.join(RADICE, "run_tanh", "run_tanh_planck_dz_libero.sh")
CONFIG_BASE = os.path.join(RADICE, "corecon_config.yaml")
CARTELLA_CONFIG = os.path.join(QUI, "config")
CARTELLA_LOG = os.path.join(QUI, "log")

PARAMETRI = ["reio_z_re", "reio_delta_z", "A_s", "tau_reio"]
BURN_IN = 0.3


def nome_breve(dataset):
    """'Totani et al. 2014 [variable_dz]' -> 'Totani2014_variable_dz'."""
    autore = dataset.split()[0]
    anno = re.search(r"(19|20)\d\d", dataset).group(0)
    etichetta = re.search(r"\[(.*)\]", dataset)
    s = autore + anno
    if etichetta:
        s += "_" + re.sub(r"[^A-Za-z0-9]+", "_", etichetta.group(1)).strip("_")
    return s


def leggi_dataset():
    """Il file dei dataset come {sezione: [dataset, ...]}."""
    with open(CONFIG_BASE) as f:
        return yaml.safe_load(f)


def lista_run():
    """[(nome della run, dataset tolto o None, sezioni dei dataset)]."""
    config = leggi_dataset()
    run = [("tutti", None, config)]
    for sezione, elenco in config.items():
        for dataset in elenco:
            senza = {s: [d for d in l if d != dataset] for s, l in config.items()}
            senza = {s: l for s, l in senza.items() if l}     # niente sezioni vuote
            run.append(("senza_" + nome_breve(dataset), dataset, senza))
    return run


def prefisso_catene(nome):
    return f"chains/leave_one_out/loo_{nome}/loo_{nome}_ELiCA_lowTT"


# ===========================================================================
# COMANDO 'prepara'
# ===========================================================================
def prepara():
    os.makedirs(CARTELLA_CONFIG, exist_ok=True)
    os.makedirs(CARTELLA_LOG, exist_ok=True)
    open(os.path.join(CARTELLA_LOG, ".gitkeep"), "a").close()
    testo_yaml = open(YAML_BASE).read()
    testo_sbatch = open(SBATCH_BASE).read()
    righe_lancio = ["#!/bin/bash",
                    "# Lancia tutte le run del leave-one-out, una sbatch per run.",
                    "# Dalla radice del repository:  bash leave_one_out/lancia_tutti.sh", ""]

    for nome, tolto, sezioni in lista_run():
        # 1. il file dei dataset senza il dataset tolto
        file_config = f"leave_one_out/config/corecon_{nome}.yaml"
        with open(os.path.join(RADICE, file_config), "w") as f:
            f.write(f"# Dataset per la run leave-one-out '{nome}'.\n")
            f.write(f"# Tolto: {tolto}\n" if tolto else "# Tutti i dataset (run di riferimento).\n")
            f.write("# Generato da leave_one_out/leave_one_out.py: non modificare a mano.\n\n")
            yaml.safe_dump(sezioni, f, sort_keys=False, allow_unicode=True)

        # 2. lo YAML di cobaya: quello di run_tanh con due righe cambiate
        nuovo = testo_yaml.replace("corecon_config_file: corecon_config.yaml",
                                   f"corecon_config_file: {file_config}")
        nuovo = re.sub(r"^output: .*$", f"output: {prefisso_catene(nome)}", nuovo, flags=re.M)
        assert nuovo.count(file_config) == 1 and prefisso_catene(nome) in nuovo
        testa = (f"# Leave-one-out '{nome}': "
                 + (f"tolto '{tolto}'.\n" if tolto else "tutti i dataset (riferimento).\n")
                 + "# Generato da leave_one_out/leave_one_out.py a partire da\n"
                 + "# run_tanh/tanh_planck_dz_libero.yaml: cambiano SOLO\n"
                 + "# corecon_config_file e output.\n#\n")
        file_yaml = os.path.join(QUI, f"loo_{nome}.yaml")
        with open(file_yaml, "w") as f:
            f.write(testa + nuovo)

        # 3. lo sbatch: quello di run_tanh con nome del job, log e YAML cambiati
        sb = testo_sbatch.replace("tanh_planck_dz_libero", f"loo_{nome}")
        sb = sb.replace("run_tanh/log/", "leave_one_out/log/")
        sb = sb.replace(f"run_tanh/loo_{nome}.yaml", f"leave_one_out/loo_{nome}.yaml")
        assert f"leave_one_out/loo_{nome}.yaml" in sb
        file_sb = os.path.join(QUI, f"run_loo_{nome}.sh")
        with open(file_sb, "w") as f:
            f.write(sb)
        righe_lancio.append(f"sbatch leave_one_out/run_loo_{nome}.sh")
        print(f"{nome:42s} <- tolto: {tolto}")

    with open(os.path.join(QUI, "lancia_tutti.sh"), "w") as f:
        f.write("\n".join(righe_lancio) + "\n")
    print(f"\nscritte {len(lista_run())} run in leave_one_out/ (+ lancia_tutti.sh)")


# ===========================================================================
# COMANDO 'analizza'
# ===========================================================================
def leggi(nome):
    """Catene di una run, burn-in tolto da ognuna, tutte insieme."""
    files = sorted(glob.glob(os.path.join(RADICE, prefisso_catene(nome)) + ".[0-9]*.txt"))
    if not files:
        return None
    colonne = open(files[0]).readline().lstrip("#").split()
    d = np.vstack([(lambda x: x[int(BURN_IN * len(x)):])(np.loadtxt(f, ndmin=2)) for f in files])
    return {c: d[:, i] for i, c in enumerate(colonne)}


def media_sigma(c, p):
    w = c["weight"]
    m = np.average(c[p], weights=w)
    return m, np.sqrt(np.average((c[p] - m) ** 2, weights=w))


def analizza():
    rif = leggi("tutti")
    if rif is None:
        print("manca la run di riferimento 'tutti'")
        return
    stat_rif = {p: media_sigma(rif, p) for p in PARAMETRI}
    chi2_rif = rif["chi2__Reiolike.ReioLike"][np.argmin(rif["chi2"])]
    print("Riferimento (tutti i dati):  " + "  ".join(
        f"{p} {m:.4g}+-{s:.2g}" for p, (m, s) in stat_rif.items()))
    print(f"chi2 astro al best fit: {chi2_rif:.2f}\n")
    print("Spostamento della media in unita' della sigma del riferimento;")
    print("dchi2 astro = quanto chi2 astrofisico in meno senza quel dataset.\n")
    print(f"{'dataset tolto':46s}" + "".join(f"{p[5:] if p.startswith('reio_') else p:>10s}" for p in PARAMETRI)
          + f"{'dchi2':>9s}")
    for nome, tolto, _ in lista_run()[1:]:
        c = leggi(nome)
        if c is None:
            print(f"{tolto:46s}   catene non trovate")
            continue
        sposta = [(media_sigma(c, p)[0] - stat_rif[p][0]) / stat_rif[p][1] for p in PARAMETRI]
        dchi2 = chi2_rif - c["chi2__Reiolike.ReioLike"][np.argmin(c["chi2"])]
        avviso = "  <--" if max(abs(x) for x in sposta) > 1 else ""
        print(f"{tolto:46s}" + "".join(f"{x:+10.2f}" for x in sposta) + f"{dchi2:9.2f}{avviso}")
    print("\n'<--': un parametro si sposta di piu' di 1 sigma togliendo quel dataset.")


if __name__ == "__main__":
    comando = sys.argv[1] if len(sys.argv) > 1 else ""
    if comando == "prepara":
        prepara()
    elif comando == "analizza":
        analizza()
    else:
        print(__doc__)
