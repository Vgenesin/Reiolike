#!/usr/bin/env python
"""
Test di affidabilita' del modello flexknot (Ilic et al. 2025,
arXiv:2504.13254, Sect. 2.4 e Tab. 1; Millea & Bouchet 2018) con
l'astrofisica di corecon_config.yaml.  Stesso schema di
test_affidabilita_power_law.py: il pacchetto contro un riferimento scritto da
zero (PCHIP di scipy chiamata direttamente, integrale di tau fatto qui).

    cd cobaya_corecon
    /home/valentina/anaconda3/envs/cmb_env/bin/python -m pytest -v Tests/test_affidabilita_flexknot.py
    (aggiungere  -m "not lento"  per saltare le scansioni lunghe)

Le varianti del paper (Tab. 1), con i loro prior:
    2-flex / 2-knots : z1, z2 ~ U(5, 25)
    3-flex / 3-knots : z1, z2, z3 ~ U(5, 25),  x_e,2 ~ U(0, 1 + f_He)
nodo 1 (il piu' basso) a x_e = 1 + f_He, nodo N al valore residuo.

Cosa si cerca:
  - i nodi del modello devono essere nodi ESATTI della tabella, anche dopo la
    formattazione per CLASS (il primo e l'ultimo sono kink);
  - la PCHIP campionata dalla griglia deve coincidere con la PCHIP vera;
  - con prior indipendenti la maggior parte del volume ha nodi non ordinati:
    likelihood nulla, mai un'eccezione (stop_at_error fermerebbe la catena);
  - i parametri reio_flex_* non devono arrivare a CLASS;
  - schema dei parametri sbagliato o prior fuori dagli intervalli: errore
    all'avvio, non punti scartati in silenzio.
"""
import logging
import math
import os
import sys
import warnings

import numpy as np
import pytest
import yaml
from scipy import integrate
from scipy.interpolate import PchipInterpolator

RADICE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RADICE)
sys.path.insert(0, os.path.join(RADICE, "Tests"))
os.chdir(RADICE)
warnings.filterwarnings("ignore")

from test_affidabilita_tanh import pavimento_indipendente, Riferimento, _silenzioso   # noqa: E402
from Reiolike.reio_models import flexknot_model                # noqa: E402

YAML_RUN = "test_elica_reiolike.yaml"
PRIOR_Z = (5.0, 25.0)
TOL_LOGL = 3e-2                  # come per tanh e power law


# =============================================================================
# Modelli cobaya
# =============================================================================
def _info(n, interp):
    i = yaml.safe_load(open(YAML_RUN))
    i.pop("sampler", None)
    i.pop("output", None)
    i["debug"] = False
    p = i["params"]
    p.pop("reio_z_re")
    p.pop("reio_delta_z")
    p["A_s"] = 2.1e-9                # fisso: senza CMB non lo vincola nessuno
    for k in range(1, n + 1):
        p[f"reio_flex_z{k}"] = {"prior": {"min": PRIOR_Z[0], "max": PRIOR_Z[1]}}
    for k in range(2, n):
        p[f"reio_flex_xHII{k}"] = {"prior": {"min": 0.0, "max": 1.0}}
    p["tau_reio"] = {"derived": True}
    t = i["theory"]["Reiolike.ReioTheory"]
    t["reio_model"] = "flexknot"
    t["reio_flex_interp"] = interp
    return i


def _modello(n, interp):
    from cobaya.model import get_model
    return _silenzioso(get_model, _info(n, interp))


@pytest.fixture(scope="module")
def m3():
    return _modello(3, "pchip")


@pytest.fixture(scope="module")
def m2():
    return _modello(2, "pchip")


@pytest.fixture(scope="module")
def k2():
    return _modello(2, "linear")


@pytest.fixture(scope="module")
def k3():
    return _modello(3, "linear")


@pytest.fixture(scope="module")
def like3(m3):
    return m3.likelihood["Reiolike.ReioLike"]


@pytest.fixture(scope="module")
def theory3(m3):
    return m3.theory["Reiolike.ReioTheory"]


def _punto(zk, xh=()):
    pt = {f"reio_flex_z{i + 1}": float(z) for i, z in enumerate(zk)}
    pt.update({f"reio_flex_xHII{i + 2}": float(x) for i, x in enumerate(xh)})
    return pt


def valuta(model, zk, xh=()):
    r = _silenzioso(model.logposterior, _punto(zk, xh))
    lp = float(r.loglikes[0]) if r.loglikes is not None else -np.inf
    der = dict(zip(model.parameterization.derived_params(), r.derived or []))
    return lp, der


# Punti del 3-flex: transizioni rapide e lente, nodo interno alto e basso,
# nodi vicini, bordi del prior
PUNTI_3 = [((5.5, 7.0, 9.0), (0.5,)), ((6.0, 6.5, 12.0), (0.2,)),
           ((5.0, 10.0, 25.0), (0.05,)), ((5.2, 5.4, 6.0), (0.9,)),
           ((6.0, 8.0, 20.0), (0.0,)), ((6.0, 8.0, 20.0), (1.0,)),
           ((7.0, 7.002, 7.004), (0.5,)), ((5.0, 15.0, 25.0), (0.3,)),
           ((5.6, 6.2, 15.0), (0.15,)), ((8.0, 12.0, 14.0), (0.6,))]
PUNTI_2 = [(5.5, 8.0), (5.0, 25.0), (6.0, 6.01), (5.8, 12.0), (10.0, 20.0)]


# =============================================================================
# Riferimento: PCHIP di scipy chiamata qui, sulle stesse convenzioni del paper
# =============================================================================
class RifFlexknot(Riferimento):
    """Stessa statistica del riferimento tanh, storia flexknot ricostruita da
    zero: x_e = PCHIP sui nodi {(z1, 1+f), (z_i, x_i (1+f)), (zN, pavimento)},
    costante fuori; poi elio HeII e pavimento come in ReioTheory."""

    def __init__(self, *a, interp="pchip", **k):
        super().__init__(*a, **k)
        self.interp = interp

    def xe_h(self, z, zk, xh):
        f = self.fHe
        z = np.asarray(z, dtype=float)
        xk = np.concatenate([[1 + f], np.asarray(xh) * (1 + f), [self.floor]])
        zc = np.clip(z, zk[0], zk[-1])
        if self.interp == "linear":
            return np.interp(zc, zk, xk)
        return PchipInterpolator(np.asarray(zk), xk)(zc)

    def x_HII(self, z, zk, xh):
        f = self.fHe
        he = f / 2 * (1 + np.tanh((3.5 - np.asarray(z, dtype=float)) / 0.5))
        return (np.maximum(self.xe_h(z, zk, xh) + he, self.floor) - he) / (1 + f)

    def media(self, zlo, zhi, zk, xh):
        if not zhi > zlo:
            return float(self.x_HII(zlo, zk, xh))
        pts = [p for p in zk if zlo < p < zhi]
        v, _ = integrate.quad(self.x_HII, zlo, zhi, args=(zk, xh),
                              points=pts or None, limit=400,
                              epsabs=1e-11, epsrel=1e-11)
        return v / (zhi - zlo)

    def logL_dataset(self, a, *par):
        return sum(self.logL_riga(a, i, self.media(*self.finestra(a, i), *par))
                   for i in range(len(a["z"])))

    def logL(self, *par):
        return sum(self.logL_dataset(a, *par) for a in self.analyses)


@pytest.fixture(scope="module")
def rif(m3, like3, theory3):
    valuta(m3, (6.0, 7.0, 9.0), (0.5,))
    return RifFlexknot(like3.analyses, theory3.classy.pars["YHe"])


def _x_ai_dati(like, h):
    x = (h["xe"] - h["xe_HeII"]) / (1 + h["f_He"])
    return [like._model_on_windows(a, h["z"], x)[0] for a in like.analyses]


def _stringhe(theory):
    p = theory.classy.pars
    z = np.array([float(v) for v in p["reio_inter_z"].split(",")])
    x = np.array([float(v) for v in p["reio_inter_xe"].split(",")])
    return z, x, max(len(p["reio_inter_z"]), len(p["reio_inter_xe"]))


# =============================================================================
# A. La funzione del modello (niente CLASS)
# =============================================================================
@pytest.mark.parametrize("interp", ["pchip", "linear"])
def test_A1_nodi_esatti_ed_estensione_costante(interp):
    zk, xk = [5.5, 7.0, 9.0], [1.08, 0.4, 2e-4]
    z = np.array([0.0, 3.0, 5.5, 7.0, 9.0, 12.0, 30.0])
    xe = flexknot_model(z, zk, xk, interp)
    assert np.array_equal(xe, [1.08, 1.08, 1.08, 0.4, 2e-4, 2e-4, 2e-4])


def test_A2_due_nodi_pchip_uguale_lineare():
    """Con 2 nodi la PCHIP e' la retta: 2-flex e 2-knots sono lo stesso modello
    (Tab. 2 del paper: tau 0.0587 contro 0.0586)."""
    rng = np.random.default_rng(1)
    for _ in range(100):
        a, b = np.sort(rng.uniform(5, 25, 2))
        z = rng.uniform(0, 30, 200)
        p = flexknot_model(z, [a, b], [1.08, 2e-4], "pchip")
        lin = flexknot_model(z, [a, b], [1.08, 2e-4], "linear")
        assert np.allclose(p, lin, rtol=0, atol=1e-14)


def test_A3_monotona_e_senza_overshoot():
    """PCHIP con nodi monotoni: storia monotona, mai fuori dall'intervallo dei
    due nodi adiacenti (una spline cubica normale qui oscillerebbe)."""
    rng = np.random.default_rng(2)
    for _ in range(300):
        n = rng.integers(3, 7)
        zk = np.sort(rng.uniform(5, 25, n))
        if np.any(np.diff(zk) < 1e-3):
            continue
        xk = np.concatenate([[1.08], np.sort(rng.uniform(0, 1.08, n - 2))[::-1], [2e-4]])
        z = np.linspace(0, 30, 3001)
        xe = flexknot_model(z, zk, xk, "pchip")
        assert np.all(np.diff(xe) <= 1e-15), "non monotona"
        for (a, b), (xa, xb) in zip(zip(zk[:-1], zk[1:]), zip(xk[:-1], xk[1:])):
            m = (z >= a) & (z <= b)
            assert np.all(xe[m] <= max(xa, xb) + 1e-15) and np.all(xe[m] >= min(xa, xb) - 1e-15)


def test_A4_pchip_e_C1_nei_nodi_interni():
    zk, xk = [5.5, 7.0, 9.0], [1.08, 0.4, 2e-4]
    e = 1e-6
    d = lambda z0: (flexknot_model(z0 + e, zk, xk) - flexknot_model(z0 - e, zk, xk)) / (2 * e)
    sx = (flexknot_model(7.0, zk, xk) - flexknot_model(7.0 - e, zk, xk)) / e
    dx = (flexknot_model(7.0 + e, zk, xk) - flexknot_model(7.0, zk, xk)) / e
    assert sx == pytest.approx(dx, rel=1e-4)
    assert d(7.0) < 0
    # agli estremi invece c'e' un kink: da 0 (plateau) a una pendenza finita
    k = (flexknot_model(5.5 + e, zk, xk) - flexknot_model(5.5, zk, xk)) / e
    assert k < -1e-2


def test_A5_ingressi_sbagliati():
    z = np.linspace(0, 30, 10)
    for zk, xk, it in (([7, 5], [1, 0], "pchip"), ([5, 5], [1, 0], "pchip"),
                       ([5, 7, 9], [1, 0], "pchip"), ([5], [1], "pchip"),
                       ([5, np.nan], [1, 0], "pchip"), ([5, 7], [1, 0], "cubic")):
        with pytest.raises(ValueError):
            flexknot_model(z, zk, xk, it)


# =============================================================================
# B. La griglia e la tabella passata a CLASS
# =============================================================================
def test_B1_nodi_del_modello_sono_nodi_esatti_anche_per_CLASS(m3, theory3):
    lim = theory3._class_arg_limit()
    for zk, xh in PUNTI_3:
        valuta(m3, zk, xh)
        h = theory3.current_state["reio_history"]
        for z0 in zk:
            assert np.any(h["z"] == z0), (zk, xh, z0)
        zs, xs, L = _stringhe(theory3)
        assert L < lim
        assert len(zs) == len(h["z"]) and np.all(np.diff(zs) > 0), (zk, xh)
        for z0 in zk:
            assert np.any(np.abs(zs - z0) < 1e-7), (zk, xh, z0)
        assert np.max(np.abs(xs - h["xe"])) < 1e-5


def test_B2_tabella_contro_pchip_vera(m3, theory3, rif):
    """La tabella che CLASS interpola linearmente contro la PCHIP valutata
    fitta: e' l'errore di discretizzazione del modello."""
    peggio, dove = 0.0, None
    for zk, xh in PUNTI_3:
        valuta(m3, zk, xh)
        h = theory3.current_state["reio_history"]
        zf = np.linspace(0, theory3.z_max * 0.999, 30001)
        tab = np.interp(zf, h["z"], h["xe"])
        f = h["f_He"]
        esatto = np.maximum(rif.xe_h(zf, np.array(zk), xh)
                            + f / 2 * (1 + np.tanh((3.5 - zf) / 0.5)), rif.floor)
        d = np.max(np.abs(tab - esatto))
        if d > peggio:
            peggio, dove = d, (zk, xh)
    print(f"\n  max |tabella - PCHIP| = {peggio:.2e} a {dove}")
    assert peggio < 2e-3, dove


def test_B3_lineare_esatta(k3):
    th = k3.theory["Reiolike.ReioTheory"]
    for zk, xh in PUNTI_3:
        valuta(k3, zk, xh)
        h = th.current_state["reio_history"]
        f = h["f_He"]
        rl = RifFlexknot([], th.classy.pars["YHe"], interp="linear")
        esatto = np.maximum(rl.xe_h(h["z"], np.array(zk), xh) + h["xe_HeII"], rl.floor)
        esatto[-1] = 0.0                           # convenzione reio_inter
        assert np.max(np.abs(h["xe"] - esatto)) < 1e-12, (zk, xh)
        # e fra i nodi della tabella CLASS ricostruisce la spezzata giusta
        zf = np.linspace(0, th.z_max * 0.99, 20001)
        d = np.interp(zf, h["z"], h["xe"]) - np.maximum(
            rl.xe_h(zf, np.array(zk), xh) + f / 2 * (1 + np.tanh((3.5 - zf) / 0.5)), rl.floor)
        assert np.max(np.abs(d)) < 1e-3, (zk, xh, np.max(np.abs(d)))


def test_B4_valori_nei_nodi(m3, theory3, like3):
    """Nodo 1 a 1 + f_He, nodo interno a x_HII campionato (letto come lo legge
    la likelihood), nodo N al pavimento."""
    zk, xh = (5.5, 7.0, 9.0), (0.37,)
    valuta(m3, zk, xh)
    h = theory3.current_state["reio_history"]
    f = h["f_He"]
    x_hii = (h["xe"] - h["xe_HeII"]) / (1 + f)
    i1, i2, i3 = (int(np.flatnonzero(h["z"] == z0)[0]) for z0 in zk)
    assert x_hii[i1] == pytest.approx(1.0, abs=1e-12)
    assert x_hii[i2] == pytest.approx(0.37, abs=1e-12)
    assert h["xe"][i3] == pytest.approx(theory3._pavimento + h["xe_HeII"][i3], abs=1e-15)


def test_B5_parametri_non_arrivano_a_CLASS(m3, theory3):
    valuta(m3, (5.5, 7.0, 9.0), (0.4,))
    assert not [k for k in theory3.classy.pars if k.startswith("reio_flex")]


# =============================================================================
# C. CLASS e tau
# =============================================================================
@pytest.mark.parametrize("zk,xh", PUNTI_3[:5])
def test_C1_CLASS_legge_la_tabella(m3, theory3, zk, xh):
    valuta(m3, zk, xh)
    h = theory3.current_state["reio_history"]
    th = theory3.classy.get_thermodynamics()
    z, xe = np.asarray(th["z"]), np.asarray(th["x_e"])
    m = (z > 0.05) & (z < 0.98 * theory3.z_max)
    tab = np.interp(z[m], h["z"], h["xe"])
    assert np.max(np.abs(xe[m] - tab)) < 2e-4, np.max(np.abs(xe[m] - tab))


@pytest.mark.parametrize("zk,xh", [((5.5, 7.0, 9.0), (0.5,)), ((6.0, 6.5, 12.0), (0.2,)),
                                   ((5.0, 15.0, 25.0), (0.3,)), ((5.2, 5.4, 6.0), (0.9,))])
def test_C2_tau_contro_integrale_indipendente(m3, theory3, rif, zk, xh):
    """tau di CLASS contro sigma_T c int n_e dt, con la PCHIP ANALITICA (non la
    tabella): misura insieme CLASS e la discretizzazione."""
    sT, Mpc, mH, G = 6.6524616e-29, 3.085677581282e22, 1.673575e-27, 6.67428e-11
    _, der = valuta(m3, zk, xh)
    cl = theory3.classy
    f = theory3.current_state["reio_history"]["f_He"]
    YHe = cl.get_current_derived_parameters(["YHe"])["YHe"]
    rho_b = 3 * (cl.Hubble(0) / Mpc * 2.99792458e8) ** 2 / (8 * np.pi * G) * cl.Omega_b()
    nH0 = (1 - YHe) * rho_b / mH
    zg = np.unique(np.concatenate([np.linspace(0, 30, 60001), zk]))
    xe = np.maximum(rif.xe_h(zg, np.array(zk), xh)
                    + f / 2 * (1 + np.tanh((3.5 - zg) / 0.5)), pavimento_indipendente())
    zh = np.linspace(0, 30, 601)
    Hz = np.interp(zg, zh, [cl.Hubble(z) for z in zh])
    tau = sT * nH0 * Mpc * integrate.trapezoid(xe * (1 + zg) ** 2 / Hz, zg)
    assert der["tau_reio"] == pytest.approx(tau, rel=3e-3), (der["tau_reio"], tau)


def test_C3_due_flex_uguale_due_knots(m2, k2):
    for zk in PUNTI_2:
        lp_f, df = valuta(m2, zk)
        lp_k, dk = valuta(k2, zk)
        assert df["tau_reio"] == pytest.approx(dk["tau_reio"], rel=1e-6), zk
        assert lp_f == pytest.approx(lp_k, abs=1e-6), zk


def test_C4_tre_flex_diverso_da_tre_knots(m3, k3):
    """Con un nodo interno le due interpolazioni sono modelli diversi:
    controllo che l'opzione arrivi davvero al calcolo."""
    t_f = valuta(m3, (5.5, 7.0, 12.0), (0.5,))[1]["tau_reio"]
    t_k = valuta(k3, (5.5, 7.0, 12.0), (0.5,))[1]["tau_reio"]
    assert abs(t_f - t_k) > 1e-4, (t_f, t_k)


# =============================================================================
# D. Robustezza
# =============================================================================
class _Avvisi(logging.Handler):
    def __init__(self):
        super().__init__(logging.WARNING)
        self.msg = []

    def emit(self, r):
        self.msg.append(r.getMessage().split("\n")[0][:120])


def test_D1_nodi_non_ordinati_likelihood_nulla_senza_eccezioni(m3, m2):
    for zk, xh in (((9.0, 7.0, 5.5), (0.5,)), ((7.0, 5.5, 9.0), (0.5,)),
                   ((5.5, 9.0, 7.0), (0.5,)), ((7.0, 7.0, 9.0), (0.5,)),
                   ((7.0, 7.0005, 9.0), (0.5,)), ((24.9, 5.1, 6.0), (0.1,))):
        try:
            lp, _ = valuta(m3, zk, xh)
        except Exception as e:
            pytest.fail(f"{zk}: {type(e).__name__}: {str(e)[:120]}")
        assert lp == -np.inf, zk
    assert valuta(m2, (12.0, 6.0))[0] == -np.inf
    # appena sopra la distanza minima: storia valida
    assert np.isfinite(valuta(m3, (7.0, 7.002, 9.0), (0.5,))[0])


def test_D2_cache(m3):
    a1 = valuta(m3, (5.6, 7.0, 10.0), (0.3,))[0]
    b = valuta(m3, (6.0, 8.0, 12.0), (0.6,))[0]
    a2 = valuta(m3, (5.6, 7.0, 10.0), (0.3,))[0]
    assert a1 == a2 != b


def test_D3_frazione_di_prior_accettata(m3):
    """Con prior indipendenti U(5, 25) sui tre nodi solo 1/3! = 1/6 del volume
    ha i nodi ordinati: e' la frazione di punti con likelihood finita."""
    rng = np.random.default_rng(3)
    N = 600
    ok = 0
    for _ in range(N):
        zk = rng.uniform(*PRIOR_Z, 3)
        lp, der = valuta(m3, zk, (rng.uniform(),))
        if np.isfinite(lp):
            ok += 1
            assert np.all(np.diff(zk) > 0) and np.isfinite(der["tau_reio"])
        else:
            assert not np.all(np.diff(zk) >= 1e-3)
    p = ok / N
    assert abs(p - 1 / 6) < 4 * math.sqrt(1 / 6 * 5 / 6 / N), p


@pytest.mark.lento
def test_D4_prior_3flex_valido_intero(m3):
    """Tutto il prior ordinato, angoli compresi: likelihood e tau finiti, nessun
    avviso tranne quello (eventuale) del budget di nodi."""
    h = _Avvisi()
    logging.getLogger().addHandler(h)
    try:
        rng = np.random.default_rng(4)
        punti = [np.sort(rng.uniform(*PRIOR_Z, 3)) for _ in range(250)]
        punti += [np.array(p) for p in ((5.0, 5.01, 5.02), (24.98, 24.99, 25.0),
                                         (5.0, 15.0, 25.0), (5.0, 24.99, 25.0))]
        for zk in punti:
            if np.any(np.diff(zk) < 1e-3):
                continue
            for x in (0.0, rng.uniform(), 1.0):
                lp, der = valuta(m3, zk, (x,))
                assert np.isfinite(lp) and np.isfinite(der["tau_reio"]), (zk, x)
                assert 0.03 < der["tau_reio"] < 0.3, (zk, x, der["tau_reio"])
    finally:
        logging.getLogger().removeHandler(h)
    altri = [m for m in h.msg if "ridotta a" not in m]
    print(f"\n  avvisi: {len(h.msg)} ({sorted(set(h.msg))})")
    assert not altri, altri


def _avvio_fallisce(info, frase):
    from cobaya.model import get_model
    with pytest.raises(Exception) as e:
        _silenzioso(get_model, info)
    assert frase in str(e.value), str(e.value)[:300]


def test_D5_schema_dei_parametri_sbagliato_ferma_l_avvio():
    i = _info(3, "pchip")
    i["params"].pop("reio_flex_xHII2")
    _avvio_fallisce(i, "reio_flex_xHII2")
    i = _info(3, "pchip")
    i["params"].pop("reio_flex_z2")
    _avvio_fallisce(i, "senza buchi")
    i = _info(2, "pchip")
    i["params"]["reio_flex_xHII2"] = {"prior": {"min": 0, "max": 1}}
    _avvio_fallisce(i, "nessun reio_flex_xHII")
    i = _info(2, "pchip")
    i["params"]["reio_flex_x2"] = 0.5
    _avvio_fallisce(i, "non riconosciuti")
    i = _info(1, "pchip")
    _avvio_fallisce(i, "N >= 2")


def test_D6_prior_fuori_intervallo_ferma_l_avvio():
    i = _info(3, "pchip")
    i["params"]["reio_flex_z3"] = {"prior": {"min": 5, "max": 35}}
    _avvio_fallisce(i, "reio_flex_z3 in [5, 35]")
    i = _info(3, "pchip")
    i["params"]["reio_flex_xHII2"] = {"prior": {"min": 0, "max": 1.0819}}
    _avvio_fallisce(i, "reio_flex_xHII2 in [0, 1.0819]")
    i = _info(3, "pchip")
    i["params"]["reio_flex_z1"] = {"prior": {"min": 0, "max": 25}}
    _avvio_fallisce(i, "reio_flex_z1 in [0, 25]")


def test_D7_interpolazione_sconosciuta_ferma_l_avvio():
    _avvio_fallisce(_info(3, "cubic"), "reio_flex_interp='cubic'")


# =============================================================================
# E. La likelihood astrofisica contro il riferimento
# =============================================================================
def test_E1_x_HII_ai_dati_contro_analitico(m3, like3, rif):
    peggio, dove = 0.0, None
    for zk, xh in PUNTI_3:
        valuta(m3, zk, xh)
        for an, xa in zip(like3.analyses, _x_ai_dati(like3, like3._reio_history())):
            for i in range(len(an["z"])):
                d = abs(xa[i] - rif.media(*rif.finestra(an, i), np.array(zk), xh))
                if d > peggio:
                    peggio, dove = d, (an["name"], an["z"][i], zk, xh)
    print(f"\n  max |x_HII pacchetto - analitico| = {peggio:.2e} a {dove}")
    assert peggio < 1e-3, dove


def test_E2_logL_contro_riferimento(m3, rif):
    vals = [(zk, xh, valuta(m3, zk, xh)[0], rif.logL(np.array(zk), xh)) for zk, xh in PUNTI_3]
    best = max(v[3] for v in vals)
    peggio = 0.0
    for zk, xh, lp, lr in vals:
        d = abs(lp - lr)
        if lr > best - 50:
            peggio = max(peggio, d)
            assert d < TOL_LOGL, (zk, xh, lp, lr)
        else:
            assert d < 1e-3 * abs(lr) + TOL_LOGL, (zk, xh, lp, lr)
    print(f"\n  max |dlogL| entro 50 dal massimo = {peggio:.2e}  (massimo dei punti {best:.2f})")
