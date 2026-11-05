#!/usr/bin/env python
"""
Verifica che le analisi di corecon vengano LETTE e USATE correttamente.

COME SI USA
-----------
Modifica la lista ANALISI qui sotto, una riga per analisi, nella forma

    (sezione_yaml, "Nome dataset [etichette]")

Poi:

    PYTHONPATH=. python Tests/verifica_analisi.py

Opzioni:
    --no-plot          salta la figura
    --out FILE.png     dove salvare la figura (default Tests/verifica_analisi.png)
    --npoints N        risoluzione delle curve di logL (default 601)

COSA CONTROLLA
--------------
 1. LETTURA        ogni punto caricato viene riconfrontato con la riga grezza del
                   data file di corecon: z, value, err_up, err_down, err_left,
                   err_right, lower_lim, upper_lim, CL.  Intercetta i
                   disallineamenti introdotti dalla selezione delle righe.
 2. SIGMA          il sigma memorizzato viene ricalcolato da zero, in modo
                   indipendente dal codice della likelihood.
 3. FORMA          per ogni punto si confronta la logL calcolata dal CODICE con
                   quella calcolata da una formula scritta a mano qui.  Se
                   divergono, il bug e' nella likelihood, non nella lettura.
 4. CONTINUITA'    scansione di logL(x) in cerca di salti: un gradino nella
                   likelihood e' un muro per la catena MCMC.
 5. MONOTONIA      per un limite, logL deve essere monotona (non cresce
                   allontanandosi dalla regione permessa).
 6. FINESTRE       la media del modello sul bin in z viene confrontata con una
                   quadratura ad alta precisione su un modello non banale.
 7. FIGURA         logL(x_HII) punto per punto: le gaussiane e le mezze
                   gaussiane si vedono a occhio.

Le curve di logL si ottengono iniettando una storia COSTANTE x_HII(z) = c e
chiamando i metodi veri della likelihood su un analysis_dict ridotto a un
punto.  Con una storia costante la media sulla finestra coincide con il valore
puntuale, quindi il test 7 isola la forma della likelihood; la finestra viene
verificata a parte nel test 6.
"""

import argparse
import os
import sys
import tempfile

import numpy as np
import yaml
from scipy.stats import norm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import corecon as con
import Reiolike.reio_theory as rt
from Reiolike.reio_like import (
    ReioLike, model_at, model_window_mean, split_normal_chi2,
)
from Reiolike.reio_models import _F_HE


# ===========================================================================
# LE ANALISI DA VERIFICARE  —  modifica questa lista
# ===========================================================================
ANALISI = [
    # sezione YAML                 spec con le etichette
    ("HII_fraction",              "Greig et al. 2022 [joint]"),
    ("HII_fraction",              "Totani et al. 2014 [fixed_dz]"),
    ("HII_fraction",              "Durovcikova et al. 2024 [CROC]"),
    ("HII_fraction",              "Whitler et al. 2020 [yes]"),
    ("HII_fraction",              "Umeda et al. 2025 [ACF]"),
    ("HII_fraction",              "Umeda et al. 2026"),
    ("HII_fraction",              "Jung et al. 2020"),
    ("HII_fraction",              "Kageura et al. 2025"),
    ("lower_limit_HII_fraction",  "Totani et al. 2006 [CL=0.68]"),
    ("lower_limit_HII_fraction",  "Zhu et al. 2022"),
    ("lower_limit_HII_fraction",  "Wold et al. 2022 [McQuinn et al. (2007), CL=0.68]"),
    ("lower_limit_HII_fraction",  "Fausey et al. 2024 [0.60, McQuinn et al. 2008]"),
    ("lower_limit_HII_fraction",  "Davies et al. 2026 [negative]"),
    ("upper_limit_HII_fraction",  "Mason et al. 2019 [CL=0.95]"),
    ("mixed_HII_fraction",        "Bolan et al. 2022 [CL=0.68]"),
    # --- le scelte di corecon_config.yaml (aggiunte il 19 ott 2026) ---------
    ("HII_fraction",              "Totani et al. 2014 [variable_dz]"),
    ("HII_fraction",              "Durovcikova et al. 2024 [ATON]"),
    ("HII_fraction",              "Whitler et al. 2020 [no]"),
    ("upper_limit_HII_fraction",  "Mason et al. 2019 [CL=0.68]"),
    ("lower_limit_HII_fraction",  "Fausey et al. 2024 [free, McQuinn et al. 2008]"),
    ("mixed_HII_fraction",        "Umeda et al. 2025 [LF]"),
    # --- limite con errore: Zhu e Davies [negative] sopra prendono gia'
    # l'integrale da soli (su err_up / err_down); Davies [threshold] e' quello
    # della configurazione Kageura
    ("lower_limit_HII_fraction",  "Davies et al. 2026 [threshold]"),
    # integrale sugli errori al 95% (sigma = err2 / 1.96)
    ("lower_limit_HII_fraction",  "Davies et al. 2026 [negative, integral_errxhi_95]"),
    ("lower_limit_HII_fraction",  "Davies et al. 2026 [threshold, integral_errxhi_95]"),
]

# ===========================================================================
# COME SI LEGGE IL CL DI UN LIMITE  --  riscritto qui, NON importato dal codice
# ===========================================================================
# Il controllo deve essere indipendente: se la funzione del codice che
# converte il CL in un numero di sigma fosse sbagliata e la importassimo,
# il test sbaglierebbe allo stesso modo e passerebbe lo stesso.
#
#   l'area della mezza gaussiana fra il bordo e il limite e' CL:
#       2 Phi(z) - 1 = CL   ->   z = Phi^-1((1+CL)/2)        (68% -> 0.994)
#   E' l'unica lettura del pacchetto (dal 21 ott 2026 non c'e' piu' scelta).
CONVENZIONE = "mezza_gaussiana"     # solo per le stampe
CL_ASSUNTO = 0.95            # CL usato quando il dato non lo dichiara


def errori_del_limite(r):
    """(sotto, sopra) in x_HII: err_down / err_up, oppure, con
    integral_errxhi_95, err_down2 / err_up2 divisi per 1.96 (il 95% di una
    gaussiana sta entro +-1.95996 sigma)."""
    if r.get("err95"):
        z95 = norm.ppf(0.975)
        return r["err_down2"] / z95, r["err_up2"] / z95
    return r["err_down"], r["err_up"]


def errore_sul_limite(r):
    """Incertezza pubblicata sul limite, dal lato del vincolo (e se manca,
    dall'altro): per un lower limit su x_HII il limite si sposta in basso."""
    giu, su = errori_del_limite(r)
    lato, altro = (giu, su) if r["lower"] else (su, giu)
    for e in (lato, altro):
        if np.isfinite(e) and e > 0:
            return float(e)
    return np.nan


def sigma_attesa(r):
    """sigma del limite, ricalcolata da zero secondo come va letto.

    normale         sigma = d / z(CL),  d = distanza del limite dal bordo
    con errore      sigma = e           (errore sul limite, lato del vincolo;
                                         serve solo alle stampe)
    """
    d = (1.0 - r["value"]) if r["lower"] else r["value"]
    if r["incerto"]:
        return errore_sul_limite(r)
    cl = r["cl"] if np.isfinite(r["cl"]) else CL_ASSUNTO
    return d / z_atteso(cl) if d > 0 else np.nan


def z_atteso(cl):
    """Numero di sigma di un limite al livello cl: 2 Phi(z) - 1 = cl."""
    return float(norm.ppf(0.5 * (1.0 + cl)))


TOL = 1e-9          # tolleranza sui confronti esatti
TOL_SHAPE = 1e-8    # tolleranza fra logL del codice e formula a mano


# ===========================================================================
# utilita'
# ===========================================================================
class Esito:
    def __init__(self):
        self.ok, self.ko = 0, []

    def check(self, cond, msg):
        if cond:
            self.ok += 1
        else:
            self.ko.append(msg)
        return cond

    def report(self, titolo):
        if not self.ko:
            print(f"  [OK]   {titolo}: {self.ok} controlli superati")
        else:
            print(f"  [FAIL] {titolo}: {len(self.ko)} su {self.ok + len(self.ko)} falliti")
            for m in self.ko:
                print(f"           - {m}")


def carica(sezione, spec, quiet=True):
    """Carica una singola analisi attraverso ReioLike, come farebbe cobaya."""
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        yaml.safe_dump({sezione: [spec]}, f,
                       allow_unicode=True)
        path = f.name
    try:
        lk = ReioLike.__new__(ReioLike)
        lk.corecon_config_file = path
        lk.print_sigma_table = not quiet
        lk.debug_points = False
        if quiet:
            import io, contextlib
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                lk.initialize()
            log = buf.getvalue()
        else:
            lk.initialize()
            log = ""
        return lk, lk.analyses[0], log
    finally:
        os.unlink(path)


def righe_grezze(spec):
    """
    Le righe del data file di corecon che corrispondono a questo spec, come
    lista di dizionari.  Rifatto a mano, senza passare dal loader.
    """
    nome, label, cl_req, joint, _indip, err95 = ReioLike._parse_dataset_spec(spec)
    o = con.get("x_HII")[nome]
    n = len(o.axes)

    def col(attr, i, default=np.nan):
        v = getattr(o, attr, None)
        if v is None:
            return default
        a = np.asarray(v)
        # err_left/err_right possono essere (n_entry, n_dim): per ogni riga
        # danno [errore sul redshift, errore sulle colonne-etichetta].  Quello
        # che serve e' la colonna 0, ed e' esattamente quello che fa
        # _extract_realization con el[0].  Appiattire sarebbe sbagliato:
        # mescolerebbe le due colonne.
        if a.ndim > 1:
            a = a[:, 0]
        a = np.atleast_1d(a.ravel())
        if a.size == 1:
            return a[0]
        return a[i] if i < a.size else default

    out = []
    for i in range(n):
        a = o.axes[i]
        if hasattr(a, "__len__"):
            z = float(a[0])
            labs = [str(a[k]) for k in range(1, len(a))]
        else:
            z, labs = float(a), []
        if label is not None and labs[-len(label):] != label:
            continue
        lo = bool(col("lower_lim", i, False))
        up = bool(col("upper_lim", i, False))
        jc = bool(col("joint_constraint", i, False))
        if getattr(o, "joint_constraint", None) is not None and jc != bool(joint):
            continue
        cl = float(col("limit_confidence_level", i))
        if cl_req is not None and (lo or up) and not np.isclose(cl, cl_req, atol=1e-9):
            continue
        out.append(dict(
            z=z, value=float(col("values", i)),
            err_up=float(col("err_up", i)), err_down=float(col("err_down", i)),
            err_up2=float(col("err_up2", i)), err_down2=float(col("err_down2", i)),
            err_left=float(col("err_left", i, 0.0)),
            err_right=float(col("err_right", i, 0.0)),
            lower=lo, upper=up, cl=cl, labels=labs,
            # regola, riscritta qui da zero: un limite con errore pubblicato
            # sul limite si integra su quell'errore;
            # con integral_errxhi_95 sull'errore al 95%, che e' 1.96 sigma
            incerto=bool((lo or up) and any(
                np.isfinite(e) and e > 0
                for e in ((float(col("err_up2", i)), float(col("err_down2", i))) if err95
                          else (float(col("err_up", i)), float(col("err_down", i)))))),
            err95=bool(err95),
        ))
    return out, cl_req


def storia_costante(c, zmax=30.0, n=601):
    """Inietta x_HII(z) = c costante nel deposito condiviso."""
    z = np.linspace(0.0, zmax, n)
    x = np.full_like(z, float(c))
    rt._shared_reio_history.update(
        z=z, xe=x * (1.0 + _F_HE), xe_HeII=np.zeros_like(z), f_He=_F_HE)
    return z, x


def un_punto(analysis, i):
    """analysis_dict ridotto alla sola riga i, per chiamare i compute_* veri."""
    d = {}
    for k, v in analysis.items():
        if isinstance(v, np.ndarray):
            d[k] = v[i:i + 1]
        else:
            d[k] = v
    return d


def logL_dal_codice(lk, analysis, i, xs):
    """logL del punto i in funzione di x_HII, chiamando i metodi veri."""
    d = un_punto(analysis, i)
    tipo = analysis["type"]
    metodo = {
        "HII_fraction": lk.compute_gaussian_loglike,
        "upper_limit_HII_fraction": lk.compute_upper_limit_loglike,
        "lower_limit_HII_fraction": lk.compute_lower_limit_loglike,
        "mixed_HII_fraction": lk.compute_mixed_loglike,
    }[tipo]
    out = np.empty_like(xs)
    for k, c in enumerate(xs):
        z, x = storia_costante(c)
        try:
            out[k] = metodo(d, z, x)
        except Exception:
            out[k] = np.nan
    return out


def logL_a_mano(r, cl_req, x):
    """
    logL del punto, riscritta da zero qui, in base a come il dato e' fatto.
    Serve da controllo indipendente: se non coincide con il codice, il
    problema e' nella likelihood.
    """
    g = r["value"]
    if r["incerto"]:
        # integrale sull'errore del limite: B split-normal centrata in g, larga
        # err_down sotto ed err_up sopra (in x_HII; un lato mancante prende
        # l'altro).  Probabilita' che il modello stia dal lato giusto di B,
        # cioe' la funzione di ripartizione di B, scritta qui da zero:
        #     P(B <= x) = 2 lo/(lo+hi) Phi((x-g)/lo)               x <= g
        #               = lo/(lo+hi) + 2 hi/(lo+hi) (Phi((x-g)/hi) - 1/2)
        def _ok(e):
            return np.isfinite(e) and e > 0
        giu, su = errori_del_limite(r)
        lo = giu if _ok(giu) else su
        hi = su if _ok(su) else giu
        x = np.asarray(x, dtype=float)
        F = np.where(x <= g, 2 * lo / (lo + hi) * norm.cdf((x - g) / lo),
                     lo / (lo + hi) + 2 * hi / (lo + hi) * (norm.cdf((x - g) / hi) - 0.5))
        with np.errstate(divide="ignore"):
            return np.log(F) if r["lower"] else np.log(1.0 - F)
    if r["lower"]:
        s = sigma_attesa(r)
        # mezza gaussiana appoggiata al bordo fisico x_HII = 1, speculare al
        # ramo upper limit (bordo 0).  Il taglio e' in 1, non in g.
        return np.where(x < 1.0, -0.5 * ((x - 1.0) / s) ** 2, 0.0)
    if r["upper"]:
        s = sigma_attesa(r)
        return np.where(x > 0, -0.5 * (x / s) ** 2, 0.0)
    # detection: split normal, con le barre al livello di confidenza scelto
    s_hi, s_lo = r["err_up"], r["err_down"]
    if cl_req is not None and cl_req > 0.9:
        ratio = norm.ppf((1 + 0.95) / 2) / norm.ppf((1 + 0.68) / 2)
        if np.isfinite(r["err_up2"]):
            s_hi = r["err_up2"] / ratio
        if np.isfinite(r["err_down2"]):
            s_lo = r["err_down2"] / ratio
    return -0.5 * split_normal_chi2(x, g, s_lo, s_hi)


# ===========================================================================
# i test
# ===========================================================================
def test_lettura(analysis, grezze, e):
    """1. Ogni punto caricato combacia con la riga grezza di corecon."""
    e.check(len(analysis["z"]) == len(grezze),
            f"numero di punti: caricati {len(analysis['z'])}, attesi {len(grezze)}")
    n = min(len(analysis["z"]), len(grezze))
    for i in range(n):
        r = grezze[i]
        for chiave, atteso, letto in (
            ("z",         r["z"],      analysis["z"][i]),
            ("value",     r["value"],  analysis["values"][i]),
            ("err_up",    r["err_up"], analysis["errors"][i]),
            ("err_down",  r["err_down"], analysis["errors_down"][i]),
            ("lower_lim", r["lower"],  bool(analysis["lower_lim"][i])),
            ("upper_lim", r["upper"],  bool(analysis["upper_lim"][i])),
        ):
            if chiave in ("err_up", "err_down") and analysis.get("_barre_95"):
                continue          # sostituite di proposito, vedi test_sigma
            if isinstance(atteso, bool):
                e.check(atteso == letto, f"riga {i} {chiave}: {letto} invece di {atteso}")
            else:
                ok = (np.isnan(atteso) and np.isnan(letto)) or \
                     (np.isfinite(atteso) and abs(atteso - letto) < TOL)
                e.check(ok, f"riga {i} {chiave}: {letto} invece di {atteso}")
        cl_letto = analysis["limit_cl"][i]
        ok = (np.isnan(r["cl"]) and np.isnan(cl_letto)) or \
             (np.isfinite(r["cl"]) and abs(r["cl"] - cl_letto) < TOL)
        e.check(ok, f"riga {i} CL: {cl_letto} invece di {r['cl']}")
        for chiave, atteso, arr in (("err_left", r["err_left"], analysis["err_z_left"]),
                                    ("err_right", r["err_right"], analysis["err_z_right"])):
            if arr is None:
                e.check(not np.isfinite(atteso) or atteso == 0.0,
                        f"riga {i} {chiave}: il loader ha None ma corecon ha {atteso}")
            else:
                att = 0.0 if not np.isfinite(atteso) else atteso
                e.check(abs(att - arr[i]) < TOL,
                        f"riga {i} {chiave}: {arr[i]} invece di {att}")


def test_sigma(analysis, grezze, e):
    """2. Il sigma dei limiti, ricalcolato da zero."""
    for i in range(min(len(analysis["z"]), len(grezze))):
        r = grezze[i]
        s = analysis["limit_sigma"][i]
        if not (r["lower"] or r["upper"]):
            e.check(not np.isfinite(s),
                    f"riga {i}: detection ma limit_sigma = {s} (dovrebbe essere nan)")
            continue
        atteso = sigma_attesa(r)
        ok = (np.isnan(atteso) and np.isnan(s)) or \
             (np.isfinite(atteso) and abs(atteso - s) < 1e-12)
        e.check(ok, f"riga {i}: limit_sigma = {s}, ricalcolato {atteso}")


def test_forma(lk, analysis, grezze, cl_req, xs, e, curve):
    """3. logL dal codice vs logL riscritta a mano."""
    for i in range(min(len(analysis["z"]), len(grezze))):
        y_cod = logL_dal_codice(lk, analysis, i, xs)
        y_man = np.asarray(logL_a_mano(grezze[i], cl_req, xs), dtype=float)
        curve.append((analysis["name"], grezze[i], y_cod, y_man))
        fin = np.isfinite(y_cod) & np.isfinite(y_man)
        e.check(fin.sum() > 0, f"riga {i}: logL dal codice tutta non finita")
        if fin.sum():
            dmax = float(np.max(np.abs(y_cod[fin] - y_man[fin])))
            e.check(dmax < TOL_SHAPE,
                    f"riga {i}: logL codice vs formula differiscono di {dmax:.3e}")


def test_continuita(lk, analysis, xs, curve, e):
    """
    4. Salti in logL(x): un gradino e' un muro per la catena MCMC.

    Non si distingue un salto da una pendenza ripida guardando il passo su una
    griglia fissa: con err_up = 1e-4 (Durovcikova CROC a z=6.1) la logL scende
    di 5e5 restando perfettamente liscia, e un'euristica sul passo la
    scambierebbe per una discontinuita'.

    Il criterio giusto e' il RAFFINAMENTO: attorno al passo piu' grande si
    infittisce la griglia di 100 volte.  Se la funzione e' liscia il salto si
    riduce in proporzione; se e' discontinua resta della stessa altezza.
    """
    for idx, (nome, r, y_cod, _) in enumerate(curve):
        if nome != analysis["name"]:
            continue
        fin = np.isfinite(y_cod)
        if fin.sum() < 3:
            continue
        xf, yf = xs[fin], y_cod[fin]
        salti = np.abs(np.diff(yf))
        imax = int(np.argmax(salti))
        if salti[imax] < 1e-12:
            e.check(True, "")
            continue
        i = [k for k, (nm, rr, _, _) in enumerate(curve)
             if nm == nome and rr is r][0]
        riga = [k for k in range(len(analysis["z"]))
                if abs(analysis["z"][k] - r["z"]) < 1e-12
                and abs(analysis["values"][k] - r["value"]) < 1e-12][0]
        lo, hi = xf[imax], xf[imax + 1]
        xr = np.linspace(lo, hi, 101)
        yr = logL_dal_codice(lk, analysis, riga, xr)
        fr = np.isfinite(yr)
        if fr.sum() < 2:
            e.check(True, "")
            continue
        salto_raffinato = float(np.max(np.abs(np.diff(yr[fr]))))
        # liscia: il passo scala come dx, quindi /100.  Tolleranza larga (1/10).
        liscia = salto_raffinato < max(salti[imax] / 10.0, 1e-9)
        e.check(liscia,
                f"z={r['g'] if 'g' in r else r['z']:g}: salto di "
                f"{salto_raffinato:.4f} in logL che NON si riduce raffinando, "
                f"attorno a x_HII in [{lo:.4f},{hi:.4f}] (g={r['value']:g}) "
                f"-> discontinuita' vera")


def test_monotonia(analysis, xs, curve, e):
    """5. Per un limite, logL non cresce allontanandosi dalla zona permessa."""
    for nome, r, y_cod, _ in curve:
        if nome != analysis["name"] or not (r["lower"] or r["upper"]):
            continue
        fin = np.isfinite(y_cod)
        y = y_cod[fin]
        d = np.diff(y)
        if r["lower"]:      # permesso in alto: logL non decrescente in x
            e.check(np.all(d >= -1e-9),
                    f"z={r['z']:g}: lower limit ma logL decresce verso x_HII alti")
        else:               # permesso in basso: logL non crescente in x
            e.check(np.all(d <= 1e-9),
                    f"z={r['z']:g}: upper limit ma logL cresce verso x_HII alti")


def test_finestre(e):
    """
    6. La media sul bin in z contro un riferimento indipendente.

    Il riferimento e' un trapezio molto fitto su model_at, non scipy.quad:
    quad campiona l'interpolante in modo adattivo e su una spezzata con un
    gradino ripido il suo errore di roundoff (~1e-9) e' piu' grande della
    precisione che si vuole verificare, quindi darebbe falsi fallimenti.
    """
    z = np.linspace(0.0, 30.0, 1201)
    x = 0.5 * (1.0 - np.tanh((z - 7.0) / 0.4))      # transizione ripida
    for zc, l, r in ((5.9, 0.40, 0.49), (7.0, 0.42, 0.52),
                     (8.41, 0.90, 1.03), (11.0, 1.39, 3.18)):
        lo, hi = zc - l, zc + r
        mio = model_window_mean(z, x, lo, hi)
        t = np.linspace(lo, hi, 2_000_001)
        rif = float(np.trapz(model_at(z, x, t), t) / (hi - lo))
        e.check(abs(mio - rif) < 1e-9,
                f"finestra [{lo:.2f},{hi:.2f}]: media {mio:.12f} vs "
                f"riferimento {rif:.12f}  (diff {abs(mio-rif):.2e})")


# ===========================================================================
# figura
# ===========================================================================
def figura(tutte_curve, xs, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    per_ds = {}
    for nome, r, y_cod, y_man in tutte_curve:
        per_ds.setdefault(nome, []).append((r, y_cod, y_man))

    n = len(per_ds)
    ncol = 3
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.6 * ncol, 3.4 * nrow),
                             squeeze=False)
    for ax in axes.ravel()[n:]:
        ax.axis("off")

    for ax, (nome, righe) in zip(axes.ravel(), per_ds.items()):
        for r, y_cod, y_man in righe:
            if r["lower"]:
                tipo, colore = "lower lim", "tab:red"
            elif r["upper"]:
                tipo, colore = "upper lim", "tab:orange"
            else:
                tipo, colore = "detection", "tab:blue"
            ax.plot(xs, y_cod, lw=1.8, color=colore, alpha=0.9,
                    label=f"z={r['z']:.2f}  {tipo}")
            ax.plot(xs, y_man, lw=0.8, ls=":", color="k", alpha=0.7)
            ax.axvline(r["value"], color=colore, lw=0.7, ls="--", alpha=0.5)
        ax.set_title(nome, fontsize=8.5)
        ax.set_xlabel(r"$x_\mathrm{HII}$ del modello", fontsize=8)
        ax.set_ylabel(r"$\log L$", fontsize=8)
        ax.set_xlim(0, 1)
        ax.set_ylim(-25, 1.5)
        ax.tick_params(labelsize=7.5)
        ax.legend(fontsize=6.2, loc="lower left", framealpha=0.85)
        ax.grid(alpha=0.25, lw=0.5)

    fig.suptitle("logL punto per punto.  Linea piena = codice, "
                 "punteggiata nera = formula indipendente.  "
                 "Tratteggio verticale = valore del dato.", fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out, dpi=135)
    print(f"\nFigura salvata in {out}")


# ===========================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-plot", action="store_true")
    ap.add_argument("--out", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "verifica_analisi.png"))
    ap.add_argument("--npoints", type=int, default=601)
    args = ap.parse_args()
    global CONVENZIONE
    convenzioni = ("mezza_gaussiana",)      # l'unica lettura del CL

    xs = np.linspace(0.0, 1.0, args.npoints)
    tutte_curve = []
    totali = Esito()

    print("=" * 78)
    print("VERIFICA DELLA LETTURA DELLE ANALISI")
    print("=" * 78)

    e = Esito()
    test_finestre(e)
    print("\nFinestre in redshift (indipendente dai dataset)")
    e.report("media sul bin vs quadratura ad alta precisione")
    totali.ok += e.ok; totali.ko += e.ko

    for CONVENZIONE in convenzioni:
      tutte_curve = []
      print()
      print("#" * 78)
      print(f"# CONVENZIONE PER IL CL: {CONVENZIONE}")
      print("#" * 78)
      for sezione, spec in ANALISI:
        print()
        print("-" * 78)
        print(f"{spec}   [{CONVENZIONE}]")
        print(f"   sezione: {sezione}")
        try:
            lk, analysis, _ = carica(sezione, spec)
        except Exception as exc:
            print(f"  [FAIL] caricamento: {type(exc).__name__}: {exc}")
            totali.ko.append(f"{spec}: caricamento fallito")
            continue

        grezze, cl_req = righe_grezze(spec)
        print(f"   {len(analysis['z'])} punto/i caricato/i, {len(grezze)} atteso/i da corecon")

        if cl_req is not None and cl_req > 0.9:
            analysis["_barre_95"] = True

        curve = []
        for nome, fn in (("1. lettura vs corecon", lambda ee: test_lettura(analysis, grezze, ee)),
                         ("2. sigma ricalcolato",  lambda ee: test_sigma(analysis, grezze, ee)),
                         ("3. forma: codice vs formula",
                          lambda ee: test_forma(lk, analysis, grezze, cl_req, xs, ee, curve)),
                         ("4. continuita' di logL", lambda ee: test_continuita(lk, analysis, xs, curve, ee)),
                         ("5. monotonia dei limiti", lambda ee: test_monotonia(analysis, xs, curve, ee))):
            e = Esito()
            fn(e)
            e.report(nome)
            totali.ok += e.ok
            totali.ko += [f"[{CONVENZIONE}] {spec} / {nome}: {m}" for m in e.ko]

        tutte_curve += curve

    print()
    print("=" * 78)
    if totali.ko:
        print(f"TOTALE: {len(totali.ko)} controlli falliti su "
              f"{totali.ok + len(totali.ko)}")
        for m in totali.ko:
            print(f"   - {m}")
    else:
        print(f"TOTALE: tutti i {totali.ok} controlli superati")
    print("=" * 78)

    if not args.no_plot and tutte_curve:
        figura(tutte_curve, xs, args.out)

    return 1 if totali.ko else 0


if __name__ == "__main__":
    sys.exit(main())
