#!/usr/bin/env python
"""
ANALISI DEL PROFILO DELLA POWER LAW 3-POW (da lanciare in locale, dopo aver
copiato chains/profilo_3pow/ dal cluster).

Per ogni punto del profilo legge il minimo trovato da cobaya
(chains/profilo_3pow/<nome>/<nome>.bestfit.txt) e confronta:
  - il PROFILO: Delta chi2(alpha) = chi2 minimo con alpha fissato - chi2 minimo
    assoluto (stesso per z_beg).  Dice cosa preferiscono i dati;
  - la POSTERIORE MARGINALE della catena MCMC (chains/power_law_3pow/), che
    contiene anche il volume del prior.
Se le due curve hanno il massimo in posti diversi, la marginale e' spostata
dal volume del prior (effetto di proiezione).

Controllo di qualita': il minimo di ogni punto deve essere <= del miglior
punto della catena nello stesso intervallo (il minimizzatore cerca meglio di
una catena).  Se e' piu' alto, il minimizzatore si e' fermato su un minimo
locale: il punto e' segnalato con "!" e va rilanciato.

    python profilo_3pow/analizza.py

Scrive profilo_3pow/profilo.txt e profilo_3pow/profilo_alpha_zbeg.png
"""
import glob
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

QUI = os.path.dirname(os.path.abspath(__file__))
RADICE = os.path.dirname(QUI)
CATENA = os.path.join(RADICE, "chains", "power_law_3pow", "power_law_3pow_corecon_ELiCA_lowTT")
BURN_IN = 0.3


def leggi_minimo(nome):
    """
    Il minimo di un punto: chains/profilo_3pow/<nome>/<nome>.bestfit.txt, una
    tabella (intestazione con '#' e una riga) con anche la colonna chi2 e i
    chi2__<likelihood>.  NB: il '-log(Like)' dell'altro file (.bestfit) include
    il termine costante del prior, quindi qui si usa la colonna chi2.
    Il parametro fissato non e' nella tabella: si prende dal nome del punto.
    """
    f = os.path.join(RADICE, "chains", "profilo_3pow", nome, nome + ".bestfit.txt")
    if not os.path.exists(f):
        return None
    righe = [r for r in open(f) if r.strip()]
    nomi = righe[0].lstrip("#").split()
    valori = [float(x) for x in righe[1].split()]
    out = dict(zip(nomi, valori))
    if nome.startswith("alpha_"):
        out["reio_alpha"] = float(nome.split("_")[1])
    elif nome.startswith("zbeg_"):
        out["reio_z_beg"] = float(nome.split("_")[1])
    return out


def leggi_catena():
    fs = sorted(glob.glob(CATENA + ".[0-9].txt"))
    cols = open(fs[0]).readline().lstrip("#").split()
    d = np.vstack([(lambda x: x[int(BURN_IN * len(x)):])(np.loadtxt(f)) for f in fs])
    return {c: d[:, i] for i, c in enumerate(cols)}


if __name__ == "__main__":
    nomi = [r.strip() for r in open(os.path.join(QUI, "lista.txt")) if r.strip()]
    minimi = {n: leggi_minimo(n) for n in nomi}
    mancanti = [n for n, m in minimi.items() if m is None]
    trovati = {n: m for n, m in minimi.items() if m is not None}
    if not trovati:
        raise SystemExit("nessun risultato in chains/profilo_3pow/: copia la cartella dal cluster")
    chi2_min = min(m["chi2"] for m in trovati.values())
    c = leggi_catena()
    w, chi2_cat = c["weight"], c["chi2"]

    righe = [f"chi2 minimo assoluto (fra tutti i punti): {chi2_min:.2f}"
             + (f"   [libero: {trovati['libero']['chi2']:.2f}]" if "libero" in trovati else ""),
             f"chi2 minimo nella catena MCMC: {chi2_cat.min():.2f}",
             f"punti mancanti: {', '.join(mancanti) if mancanti else 'nessuno'}", ""]
    figura = {}
    for par, prefisso, larg in (("reio_alpha", "alpha_", 0.25), ("reio_z_beg", "zbeg_", 0.5)):
        punti = sorted((m[par], n, m) for n, m in trovati.items() if n.startswith(prefisso))
        righe.append(f"=== profilo in {par}")
        righe.append(f"  {'valore':>7s} {'Dchi2':>7s} {'astro':>7s} {'elica':>7s} {'lowTT':>7s} "
                     f"{'z_end':>6s} {'alpha':>6s} {'z_beg':>6s} {'tau':>7s}  catena(min chi2 vicino)")
        xs, dchi = [], []
        for x, n, m in punti:
            vicino = np.abs(c[par] - x) < larg
            cat = chi2_cat[vicino].min() if vicino.any() else np.nan
            avviso = " !" if np.isfinite(cat) and m["chi2"] > cat + 0.5 else ""
            righe.append(f"  {x:7.2f} {m['chi2'] - chi2_min:7.2f} {m.get('chi2__Reiolike.ReioLike', np.nan):7.2f} "
                         f"{m.get('chi2__elica', np.nan):7.2f} {m.get('chi2__planck_2018_lowl.TT', np.nan):7.2f} "
                         f"{m.get('reio_z_end', np.nan):6.2f} {m.get('reio_alpha', np.nan):6.2f} "
                         f"{m.get('reio_z_beg', np.nan):6.2f} {m.get('tau_reio', np.nan):7.4f}  "
                         f"{cat - chi2_min:6.2f}{avviso}")
            xs.append(x)
            dchi.append(m["chi2"] - chi2_min)
        righe.append("")
        figura[par] = (np.array(xs), np.array(dchi))
    righe.append("'!': minimo piu' alto del miglior punto della catena vicino: minimo locale, da rilanciare.")
    testo = "\n".join(righe)
    print(testo)
    open(os.path.join(QUI, "profilo.txt"), "w").write(testo + "\n")

    # --- figura: in alto Delta chi2, in basso exp(-Dchi2/2) contro la marginale
    fig, axs = plt.subplots(2, 2, figsize=(11, 7), sharex="col")
    for j, (par, etichetta, bordi) in enumerate((("reio_alpha", r"$\alpha$", np.linspace(0, 14, 29)),
                                                 ("reio_z_beg", r"$z_\mathrm{beg}$", np.linspace(5, 25, 21)))):
        xs, dchi = figura[par]
        alto, basso = axs[0, j], axs[1, j]
        alto.plot(xs, dchi, "o-", color="#D55E00", label="profilo (minimizzatore)")
        for k in (1, 4):
            alto.axhline(k, color="#999999", lw=0.8, ls=":")
        alto.set_ylabel(r"$\Delta\chi^2$")
        alto.set_ylim(-0.5, max(12, np.nanmax(dchi) + 1) if len(dchi) else 12)
        alto.legend(fontsize=8, frameon=False)
        h, _ = np.histogram(c[par], bins=bordi, weights=w)
        centri = 0.5 * (bordi[1:] + bordi[:-1])
        basso.step(centri, h / h.max(), where="mid", color="#0072B2", label="posteriore marginale (catena)")
        if len(xs):
            basso.plot(xs, np.exp(-dchi / 2), "o-", color="#D55E00", label=r"profilo: $e^{-\Delta\chi^2/2}$")
        basso.set_xlabel(etichetta)
        basso.legend(fontsize=8, frameon=False)
    fig.suptitle("Power law 3-pow: profilo della likelihood contro posteriore marginale", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(QUI, "profilo_alpha_zbeg.png"), dpi=200)
    print("\nscritti profilo_3pow/profilo.txt e profilo_3pow/profilo_alpha_zbeg.png")
