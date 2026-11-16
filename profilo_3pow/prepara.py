#!/usr/bin/env python
"""
PROFILO DELLA LIKELIHOOD DELLA POWER LAW 3-POW (alpha e z_beg).

Perche': nella catena della 3-pow la posteriore marginale mette il massimo ad
alpha ~ 7 (z_beg alto, sulla "cresta" alpha-z_beg), ma il chi2 migliore della
catena sta ad alpha ~ 0.5, z_beg ~ 8 (Delta chi2 ~ 4.8 a favore).  La cresta e'
lunga e occupa molto volume di prior, la soluzione ad alpha piccolo e'
compatta: la marginale somma il volume.  Il profilo invece, per ogni valore
fissato di alpha (o di z_beg), prende il chi2 MINIMO sugli altri parametri:
dice cosa preferiscono i dati, senza il volume del prior.

Cosa fa questo script: scrive un YAML per ogni punto del profilo, a partire
da run_power_law/power_law_3pow.yaml (stesse likelihood, stessa cosmologia,
stessi prior), con
  - il parametro del profilo FISSATO a un valore della griglia;
  - il sampler 'minimize' di cobaya al posto dell'MCMC: 16 partenze casuali
    (ref larghi, per toccare sia la cresta sia la soluzione ad alpha piccolo),
    si tiene la migliore.
Piu' un YAML con tutti i parametri liberi (il minimo assoluto).

Si lancia dalla radice del repository:
    python profilo_3pow/prepara.py
Scrive profilo_3pow/yaml/*.yaml e profilo_3pow/lista.txt (un nome per riga:
la riga N e' il job N dell'array, vedi run_profilo.sh).
"""
import copy
import os

import yaml

QUI = os.path.dirname(os.path.abspath(__file__))
RADICE = os.path.dirname(QUI)
BASE = os.path.join(RADICE, "run_power_law", "power_law_3pow.yaml")

GRIGLIA_ALPHA = [0.1, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 12.0]
GRIGLIA_ZBEG = [7.0, 7.5, 8.0, 9.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 24.5]

# Punti di partenza: larghi apposta.  Con i ref stretti della catena (alpha ~ 3,
# z_beg ~ 15) il minimizzatore, che e' locale, resterebbe sulla cresta.
REF_Z_END = {"dist": "uniform", "loc": 5.2, "scale": 1.3}       # 5.2 - 6.5
REF_ALPHA = {"dist": "uniform", "loc": 0.05, "scale": 12.0}     # 0.05 - 12
REF_DURATA = {"dist": "uniform", "loc": 0.5, "scale": 17.0}     # z_beg - z_end: 0.5 - 17.5

# IL MINIMIZZATORE VUOLE UN RETTANGOLO.  BOBYQA lavora dentro limiti
# rettangolari; nella 3-pow meta' del rettangolo (z_end, z_beg) e' vietata
# (z_beg <= z_end: la theory scarta il punto, chi2 infinito) e quando ci finisce
# il suo modello interno si rompe ("Singular matrix", visto in prova).  Quindi,
# SOLO nei YAML del profilo, si cambia variabile:
#   - alpha fissato e minimo libero: si campionano z_end e la DURATA
#     Delta = z_beg - z_end > 0, e z_beg = z_end + Delta;
#   - z_beg fissato: z_end ha come massimo z_beg - 0.001, ed e' gia' un rettangolo.
# z_end e' ristretto a [5, 7]: i dati lo vogliono a 5.6 +- 0.06 (secondo modo a
# 5.35), il minimo non cambia, e cosi' z_beg = z_end + Delta <= 7 + 18 = 25
# come nel prior della catena (con z_end ~ 5.6 il massimo e' ~ 23.6: dove la
# cresta e' piatta, Delta chi2 ~ 0.1 fra z_beg 14 e 24).
Z_END_MAX = 7.0
DURATA_MAX = 18.0

SAMPLER = {"minimize": {
    "method": "bobyqa",
    "ignore_prior": True,       # minimizza il chi2 (i prior sono piatti: cambia solo la costante)
    "best_of": 16,              # 16 partenze, divise fra i processi MPI (4 a testa con 4).
                                # Con 1 sola partenza, in prova, z_beg = 8 si e' fermato in un
                                # minimo locale (chi2 astro 109 invece di ~61); ogni partenza
                                # dura pochi minuti, quindi tante costano poco.
    "override_bobyqa": {"rhoend": 0.05},   # tolleranza per likelihood un po' rumorose (default di cobaya)
}}


def yaml_per(nome, fissi):
    info = yaml.safe_load(open(BASE))
    P = info["params"]
    P["reio_z_end"]["prior"] = {"min": 5.0, "max": Z_END_MAX}
    P["reio_z_end"]["ref"] = copy.deepcopy(REF_Z_END)
    P["reio_alpha"]["ref"] = copy.deepcopy(REF_ALPHA)
    if "reio_z_beg" in fissi:
        zb = float(fissi["reio_z_beg"])
        P["reio_z_beg"] = {"value": zb, "latex": P["reio_z_beg"]["latex"]}
        P["reio_z_end"]["prior"]["max"] = min(Z_END_MAX, zb - 0.001)
        P["reio_z_end"]["ref"] = {"dist": "uniform", "loc": 5.2,
                                  "scale": min(1.3, zb - 0.001 - 5.2)}
    else:
        # z_beg = z_end + durata: la durata e' campionata (e non va a CLASS: drop)
        latex_zb = P["reio_z_beg"]["latex"]
        nuovi = {}
        for k, v in P.items():
            if k == "reio_z_beg":
                nuovi["reio_durata"] = {"prior": {"min": 0.001, "max": DURATA_MAX},
                                        "ref": copy.deepcopy(REF_DURATA), "proposal": 1.0,
                                        "latex": r"z_\mathrm{beg} - z_\mathrm{end}", "drop": True}
                nuovi["reio_z_beg"] = {"value": "lambda reio_z_end, reio_durata: reio_z_end + reio_durata",
                                       "derived": True, "latex": latex_zb}
            else:
                nuovi[k] = v
        info["params"] = P = nuovi
    if "reio_alpha" in fissi:
        P["reio_alpha"] = {"value": float(fissi["reio_alpha"]), "latex": P["reio_alpha"]["latex"]}
    info["sampler"] = copy.deepcopy(SAMPLER)
    info["output"] = f"chains/profilo_3pow/{nome}/{nome}"
    testa = (f"# Profilo della power law 3-pow: {nome}\n"
             f"# Generato da profilo_3pow/prepara.py da run_power_law/power_law_3pow.yaml;\n"
             f"# parametri fissati: {fissi if fissi else 'nessuno (minimo assoluto)'}\n\n")
    with open(os.path.join(QUI, "yaml", nome + ".yaml"), "w") as f:
        f.write(testa + yaml.safe_dump(info, sort_keys=False))
    return nome


if __name__ == "__main__":
    os.makedirs(os.path.join(QUI, "yaml"), exist_ok=True)
    nomi = [yaml_per("libero", {})]
    nomi += [yaml_per(f"alpha_{a:05.2f}", {"reio_alpha": a}) for a in GRIGLIA_ALPHA]
    nomi += [yaml_per(f"zbeg_{z:05.2f}", {"reio_z_beg": z}) for z in GRIGLIA_ZBEG]
    with open(os.path.join(QUI, "lista.txt"), "w") as f:
        f.write("\n".join(nomi) + "\n")
    print(f"scritti {len(nomi)} YAML in profilo_3pow/yaml/ e profilo_3pow/lista.txt")
    print(f"nel run_profilo.sh: #SBATCH --array=0-{len(nomi) - 1}")
