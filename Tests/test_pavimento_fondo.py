#!/usr/bin/env python
"""
Il pavimento di x_e e' il fondo di ricombinazione di CLASS (+5%), per ogni
cosmologia: test rigidi.

    cd cobaya_corecon
    /home/valentina/anaconda3/envs/cmb_env/bin/python -m pytest -v Tests/test_pavimento_fondo.py

COSA SI VERIFICA
  P1  il fondo usato e' quello di CLASS senza reionizzazione (reio_none),
      calcolato qui in modo indipendente, per cinque cosmologie diverse;
  P2  CLASS, nella run vera, usa lo STESSO fondo (x_e a z_max);
  P3  per ogni modello (tanh, pinned, power law, bin con buchi, Gompertz, PCA)
      la tabella non scende mai sotto il pavimento, che sta sopra il fondo;
  P4  tau_reio di CLASS NON viene mai troncata: coincide con l'integrale di
      x_e fatto qui (che non usa la regola del minimo globale), su storie a
      bin con buchi e su punti PCA;
  P5  sabotaggio: con il pavimento SOTTO il fondo, P4 se ne accorge (cioe'
      il test e' capace di vedere il troncamento);
  P6  il bias su tau della tanh e' sparito: tau scende rispetto al vecchio
      pavimento 1e-3, della quantita' attesa;
  P7  il fondo si calcola una volta per cosmologia (cache), e si ricalcola
      quando la cosmologia cambia;
  P8  la likelihood vede x_HII = pavimento/(1+f_He) prima della reionizzazione;
  P9  'reio_xe_floor' nello YAML non e' piu' un'opzione: ferma l'avvio;
  P10 _finalize_xe senza pavimento calcolato si ferma (niente ritorno
      silenzioso a un numero fisso).
"""
import contextlib
import copy
import io
import logging
import os
import sys
import warnings

import numpy as np
import pytest
import yaml

warnings.filterwarnings("ignore")
RADICE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RADICE)
YAML_RUN = os.path.join(RADICE, "test_elica_reiolike.yaml")
BASE_PCA = os.path.join(RADICE, "eigenfunctions_PCA.npz")

from classy import Class                                       # noqa: E402

MARGINE = 1.05
# costanti di CLASS (include/background.h, include/thermodynamics.h)
SIGMA_T = 6.6524616e-29
M_H = 1.673575e-27
G_NEWTON = 6.67428e-11
MPC = 3.085677581282e22

COSMO_PLANCK = {"omega_b": 0.02237, "omega_cdm": 0.1201, "H0": 67.32}
COSMOLOGIE = [COSMO_PLANCK,
              {"omega_b": 0.0200, "omega_cdm": 0.1201, "H0": 67.32},
              {"omega_b": 0.0250, "omega_cdm": 0.1100, "H0": 72.00},
              {"omega_b": 0.02237, "omega_cdm": 0.1300, "H0": 62.00},
              {"omega_b": 0.0230, "omega_cdm": 0.1201, "H0": 67.32, "N_ur": 3.0}]


def _silenzioso(f, *a, **k):
    with contextlib.redirect_stdout(io.StringIO()):
        return f(*a, **k)


def _info(modello="tanh", params=None, theory=None, cosmo=None):
    i = yaml.safe_load(open(YAML_RUN))
    for k in ("sampler", "output", "force"):
        i.pop(k, None)
    i["debug"] = False
    i["likelihood"] = {"one": None}
    p = i["params"]
    p["A_s"] = 2.1e-9
    p.update(cosmo or {})
    p["tau_reio"] = {"derived": True}
    if modello != "tanh":
        p.pop("reio_z_re"); p.pop("reio_delta_z")
    p.update(params or {})
    t = i["theory"]["Reiolike.ReioTheory"]
    t["reio_model"] = modello
    t.update(theory or {})
    return i


def _modello(**kw):
    from cobaya.model import get_model
    return _silenzioso(get_model, _info(**kw))


def _teoria(m):
    return m.theory["Reiolike.ReioTheory"]


def _punto(m, pt):
    r = _silenzioso(m.logposterior, pt)
    assert np.isfinite(r.logpost), f"punto rifiutato: {pt}"
    der = dict(zip(m.parameterization.derived_params(), r.derived))
    return der["tau_reio"], _teoria(m).current_state["reio_history"]


def fondo_indipendente(cosmo, z_max=30.0):
    """x_e(z_max) di CLASS senza reionizzazione: il fondo da ricombinazione."""
    i = yaml.safe_load(open(YAML_RUN))
    base = {k: v for k, v in i["params"].items() if not isinstance(v, dict)}
    base.update(i["theory"]["Reiolike.ReioTheory"].get("extra_args", {}))
    base.update(cosmo)
    c = Class()
    c.set({**base, "A_s": 2.1e-9, "reio_parametrization": "reio_none",
           "YHe": "BBN"})
    try:
        c.compute(level=["thermodynamics"])
        return float(c.ionization_fraction(z_max))
    finally:
        c.struct_cleanup(); c.empty()


def tau_integrale(theory, h):
    """
    tau come integrale di x_e(z) della tabella su [0, z_max], con il background
    della run.  NON usa la regola del minimo globale di CLASS: e' il valore
    che tau_reio deve avere se non e' troncata.  L'ultimo nodo lo sostituisce
    CLASS con xe_before, come qui.
    """
    c = theory.classy
    z = np.asarray(h["z"], float); xe = np.asarray(h["xe"], float).copy()
    xe[-1] = c.ionization_fraction(float(z[-1]))
    zf = np.linspace(0.0, z[-1], 60001)
    xf = np.interp(zf, z, xe)
    H = np.array([c.Hubble(v) for v in zf[::50]])                   # 1/Mpc
    H = np.interp(zf, zf[::50], H)
    ob = c.omega_b(); Y = theory.classy.get_current_derived_parameters(["YHe"])["YHe"]
    rho_b = ob * 3 * (1e5 / MPC) ** 2 / (8 * np.pi * G_NEWTON)
    nH0 = (1 - Y) * rho_b / M_H
    integrando = SIGMA_T * nH0 * (1 + zf) ** 2 * xf / (H / MPC)
    return float(np.trapz(integrando, zf))


# =============================================================================
# fixture
# =============================================================================
@pytest.fixture(scope="module")
def mtanh():
    return _modello()


def _bin_params():
    return {f"reio_xHII_bin{b}": {"prior": {"min": 0.0, "max": 1.0}} for b in range(1, 11)}


@pytest.fixture(scope="module")
def mbin():
    return _modello(modello="binned", params=_bin_params())


@pytest.fixture(scope="module")
def mpca():
    from Reiolike.pca_basis import load_basis
    b = load_basis(BASE_PCA)
    lo, hi = b.mode_bounds(5)
    pr = {f"reio_pca_m{a + 1}": {"prior": {"min": float(lo[a]), "max": float(hi[a])}} for a in range(5)}
    m = _modello(modello="pca", params=pr,
                 theory={"reio_pca_file": BASE_PCA, "reio_pca_n_modes": 5,
                         "z_max": max(30.0, b.z_max)})
    return m, b


# =============================================================================
# P1, P2: il fondo e' quello di CLASS, per ogni cosmologia
# =============================================================================
@pytest.mark.parametrize("cosmo", COSMOLOGIE, ids=lambda c: ",".join(f"{k}={v}" for k, v in c.items()))
def test_P1_P2_fondo_e_pavimento_per_cosmologia(cosmo):
    m = _modello(cosmo=cosmo)
    th = _teoria(m)
    _punto(m, {"reio_z_re": 7.2, "reio_delta_z": 0.9})
    atteso = fondo_indipendente(cosmo)
    # 1e-5: Y_He del pacchetto (tabella BBN propria) e YHe='BBN' di CLASS
    # differiscono di ~2e-6; un fondo sbagliato sarebbe fuori di percentuali.
    assert th._fondo == pytest.approx(atteso, rel=1e-5), (th._fondo, atteso)            # P1
    # pavimento = fondo x 1.05 arrotondato PER ECCESSO a 2 cifre significative
    assert MARGINE * th._fondo <= th._pavimento <= 1.12 * th._fondo, th._pavimento / th._fondo
    assert len(f"{th._pavimento:.6g}") <= 7, f"{th._pavimento:.6g}"
    # P2: la ricombinazione della run vera e' la stessa del calcolo del fondo.
    # Si legge a z_max + 1: a z_max la lettura e' contaminata dalla tabella.
    x_sopra = th.classy.ionization_fraction(th.z_max + 1.0)
    assert x_sopra == pytest.approx(th._fondo_controllo, rel=1e-6), (x_sopra, th._fondo_controllo)
    assert th._pavimento / x_sopra > 1.04, th._pavimento / x_sopra               # margine
    assert 1.5e-4 < th._fondo < 3.5e-4, th._fondo     # ordine di grandezza fisico


# =============================================================================
# P3: la tabella non scende mai sotto il pavimento, per nessun modello
# =============================================================================
def _controlla_tabella(th, h):
    xe = np.asarray(h["xe"])[:-1]
    assert np.all(xe >= th._pavimento * (1 - 1e-6)), float(xe.min())
    assert np.all(xe > th._fondo), float(xe.min())


@pytest.mark.parametrize("pt", [(5.0, 0.1), (7.2, 0.9), (12.0, 3.0), (9.0, 0.1)])
def test_P3_tanh(mtanh, pt):
    _, h = _punto(mtanh, {"reio_z_re": pt[0], "reio_delta_z": pt[1]})
    _controlla_tabella(_teoria(mtanh), h)


def test_P3_pinned_power_law_gompertz():
    casi = [
        ("tanh_pinned_end", {"reio_z_re": {"prior": {"min": 6.0, "max": 12.0}}}, {}, {"reio_z_re": 7.5}),
        ("power_law", {"reio_z_end": {"prior": {"min": 5.0, "max": 9.0}},
                       "reio_alpha": {"prior": {"min": 0.5, "max": 10.0}}},
         {"reio_z_beg": 20.0}, {"reio_z_end": 6.0, "reio_alpha": 4.0}),
    ]
    for mod, par, th_opt, pt in casi:
        m = _modello(modello=mod, params=par, theory=th_opt)
        _, h = _punto(m, pt)
        _controlla_tabella(_teoria(m), h)
    i = _info(modello="gompertz_sr", params={"sigma8": {"prior": {"min": 0.74, "max": 0.90}},
                                             "zeta_uv": {"prior": {"min": 20.0, "max": 35.0}}})
    i["params"].pop("A_s")
    from cobaya.model import get_model
    m = _silenzioso(get_model, i)
    _, h = _punto(m, {"sigma8": 0.81, "zeta_uv": 27.0})
    _controlla_tabella(_teoria(m), h)


# =============================================================================
# P4, P5: tau_reio mai troncata (bin con buchi, PCA)
# =============================================================================
STORIE_BIN = [
    [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [1.0, 0.02, 0.0, 0.02, 0.16, 0.16, 0.02, 0.0, 0.0, 0.0],   # buchi in mezzo
    [1.0, 0.0, 0.5, 0.0, 0.3, 0.0, 0.2, 0.0, 0.1, 0.0],        # pettine
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5],        # solo in cima
    [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
]


def _bin_pt(v):
    return {f"reio_xHII_bin{b + 1}": float(x) for b, x in enumerate(v)}


@pytest.mark.parametrize("storia", range(len(STORIE_BIN) + 25))
def test_P4_bin_tau_non_troncata(mbin, storia):
    if storia < len(STORIE_BIN):
        v = STORIE_BIN[storia]
    else:                                   # storie casuali, con zeri sparsi
        rng = np.random.default_rng(storia)
        v = rng.uniform(0, 1, 10) * (rng.uniform(0, 1, 10) > 0.45)
    tau, h = _punto(mbin, _bin_pt(v))
    th = _teoria(mbin)
    _controlla_tabella(th, h)
    assert tau == pytest.approx(tau_integrale(th, h), rel=1e-3), (v, tau)


def test_P4_pca_tau_non_troncata(mpca):
    m, b = mpca
    lo, hi = b.mode_bounds(5)
    rng = np.random.default_rng(1)
    fatti = 0
    for _ in range(400):
        pt = {f"reio_pca_m{a + 1}": float(rng.uniform(lo[a], hi[a])) for a in range(5)}
        r = _silenzioso(m.logposterior, pt)
        if not np.isfinite(r.logpost):
            continue                         # scartato dalla fisicita' (mh08)
        tau = dict(zip(m.parameterization.derived_params(), r.derived))["tau_reio"]
        th = _teoria(m); h = th.current_state["reio_history"]
        _controlla_tabella(th, h)
        assert tau == pytest.approx(tau_integrale(th, h), rel=1e-3), (pt, tau)
        fatti += 1
        if fatti == 15:
            break
    assert fatti >= 10, f"solo {fatti} punti PCA accettati"


def test_P5_sabotaggio_pavimento_sotto_il_fondo(mbin, monkeypatch):
    """Con il pavimento all'1% SOTTO il fondo, la storia con i buchi deve dare
    una tau_reio troncata, e il confronto di P4 deve accorgersene."""
    from Reiolike.reio_theory import ReioTheory
    monkeypatch.setattr(ReioTheory, "_MARGINE_SUL_FONDO", 0.99)
    th = _teoria(mbin)
    th._cache_fondo = {}                     # ricalcolo con il margine sabotato
    tau, h = _punto(mbin, _bin_pt(STORIE_BIN[1]))
    vero = tau_integrale(th, h)
    assert abs(tau / vero - 1) > 0.05, ("il troncamento non si vede", tau, vero)
    monkeypatch.undo()
    th._cache_fondo = {}


# =============================================================================
# P6: il bias della tanh e' sparito
# =============================================================================
def test_P6_bias_tanh_sparito(mtanh):
    tau, h = _punto(mtanh, {"reio_z_re": 7.16, "reio_delta_z": 0.94})
    th = _teoria(mtanh)
    assert tau == pytest.approx(tau_integrale(th, h), rel=1e-3)
    # la stessa storia col vecchio pavimento 1e-3, integrata allo stesso modo
    vecchio = dict(h); xe = np.asarray(h["xe"]).copy()
    he = np.asarray(h["xe_HeII"]); zz = np.asarray(h["z"])
    from Reiolike.reio_models import tanh_model
    xe_v = np.maximum(tanh_model(zz, 7.16, 0.94, xe_max=1 + h["f_He"]) + he, 1e-3); xe_v[-1] = 0
    vecchio["xe"] = xe_v
    d = tau_integrale(th, vecchio) - tau
    assert 1.0e-4 < d < 3.0e-4, d            # il vecchio bias, ora tolto


# =============================================================================
# P7: cache del fondo
# =============================================================================
def test_P7_cache_per_cosmologia(monkeypatch):
    from Reiolike.reio_theory import ReioTheory
    chiamate = []
    vera = ReioTheory._calcola_fondo

    def conta(self, args):
        chiamate.append(1)
        return vera(self, args)
    monkeypatch.setattr(ReioTheory, "_calcola_fondo", conta)
    i = _info(); i["params"]["omega_b"] = {"prior": {"min": 0.02, "max": 0.025}}
    from cobaya.model import get_model
    m = _silenzioso(get_model, i)
    for zr in np.linspace(6, 9, 12):
        _punto(m, {"reio_z_re": float(zr), "reio_delta_z": 0.8, "omega_b": 0.02237})
    assert len(chiamate) == 1, len(chiamate)
    f1 = _teoria(m)._fondo
    _punto(m, {"reio_z_re": 7.0, "reio_delta_z": 0.8, "omega_b": 0.0240})
    assert len(chiamate) == 2 and _teoria(m)._fondo != f1


# =============================================================================
# P8: la likelihood astrofisica
# =============================================================================
def test_P8_x_HII_prima_della_reionizzazione():
    i = _info(); i["likelihood"] = {"Reiolike.reio_like.ReioLike": {
        "corecon_config_file": os.path.join(RADICE, "corecon_config.yaml"), "print_sigma_table": False}}
    from cobaya.model import get_model
    m = _silenzioso(get_model, i)
    _punto(m, {"reio_z_re": 7.16, "reio_delta_z": 0.94})
    th = _teoria(m); h = th.current_state["reio_history"]
    x = (np.asarray(h["xe"]) - np.asarray(h["xe_HeII"])) / (1 + h["f_He"])
    atteso = th._pavimento / (1 + h["f_He"])
    assert np.interp(20.0, h["z"], x) == pytest.approx(atteso, rel=1e-9)
    assert 1.8e-4 < atteso < 2.6e-4


# =============================================================================
# P9, P10: niente vecchia opzione, niente ritorno silenzioso
# =============================================================================
def test_P9_reio_xe_floor_non_e_piu_un_opzione():
    from cobaya.model import get_model
    with pytest.raises(Exception) as e:
        _silenzioso(get_model, _info(theory={"reio_xe_floor": 1e-3}))
    assert "reio_xe_floor" in str(e.value), str(e.value)


def test_P10_finalize_senza_pavimento_si_ferma():
    from Reiolike.reio_theory import ReioTheory
    th = ReioTheory.__new__(ReioTheory)
    th.reio_include_HeII = False
    th.log = logging.getLogger("p10")
    with pytest.raises(Exception) as e:
        th._finalize_xe(np.linspace(0, 30, 10), np.ones(10), {})
    assert "pavimento" in str(e.value).lower(), str(e.value)


def test_P11_sabotaggio_cosmologia_diversa_nel_fondo(monkeypatch):
    """Se il calcolo del fondo vedesse una cosmologia diversa da quella della
    run (qui omega_b alterato dell'1%), il controllo dopo la run deve fermare
    tutto invece di usare un pavimento sbagliato."""
    from cobaya.log import LoggedError
    from Reiolike.reio_theory import ReioTheory
    vera = ReioTheory._argomenti_fondo

    def sbagliata(self, native_params):
        a = vera(self, native_params)
        a["omega_b"] = 1.01 * a["omega_b"]
        return a
    monkeypatch.setattr(ReioTheory, "_argomenti_fondo", sbagliata)
    m = _modello()
    with pytest.raises(LoggedError) as e:
        _silenzioso(m.logposterior, {"reio_z_re": 7.2, "reio_delta_z": 0.9})
    assert "ricombinazione" in str(e.value), str(e.value)
