#!/usr/bin/env python
"""
reio_model: tanh_class, la tanh di CLASS (reio_camb) dentro Reiolike: test
rigidi contro CLASS nativo.

    cd cobaya_corecon
    /home/valentina/anaconda3/envs/cmb_env/bin/python -m pytest -v Tests/test_tanh_class.py

Riferimento sempre CLASS con reio_parametrization = reio_camb, z_reio = z_re,
reionization_width = delta_z, stesso Y_He del pacchetto.  Soglie fissate sui
valori misurati il 6 nov 2026 (cosmologia di Pagano et al. 2020, Tab. 3):

  C1  la formula di tanh_class_model + step HeII, con x_noreio(z) dal run senza
      reionizzazione, riproduce x_e di CLASS nativo (misurato 3e-6; soglia 1e-5);
  C2  tau_reio della pipeline = tau_reio nativo su tutto il prior, z_re 5.5-12,
      delta_z 0.1-3 (misurato 2.2e-6; soglia 5e-6, ~1e-3 sigma di Planck);
  C3  la tabella passata a CLASS e alla likelihood coincide ai nodi con x_e
      nativo (soglia 1e-5);
  C4  la logL astrofisica e' quella che si ottiene dalla storia nativa;
  C5  niente pavimento x 1.05: fra z_start e z_max la tabella e' x_noreio(z);
  C6  sabotaggio 1: x_noreio costante = pavimento -> C2 se ne accorge;
  C7  sabotaggio 2: tanh lineare al posto di quella in y -> C3 se ne accorge;
  C8  la tanh lineare (reio_model: tanh) resta com'era: pavimento applicato;
  C9  il fondo e la tabella x_noreio si calcolano una volta per cosmologia;
  C10 senza reio_delta_z la catena si ferma; fuori da cobaya _x_noreio senza
      tabella si ferma.
"""
import contextlib
import io
import os
import sys
import warnings

import numpy as np
import pytest

warnings.filterwarnings("ignore")
RADICE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RADICE)

from classy import Class                                       # noqa: E402

COSMO = {"omega_b": 0.02237, "omega_cdm": 0.1201, "H0": 67.32, "A_s": 2.120e-9,
         "n_s": 0.9651, "N_ur": 2.0328}
EXTRA = {"N_ncdm": 1, "m_ncdm": 0.06, "T_ncdm": 0.71611}
PUNTI = [(zr, dz) for zr in (5.5, 7.2, 8.14, 10.0, 12.0) for dz in (0.1, 0.5, 1.0, 3.0)]
SOGLIA_TAU = 5e-6
SOGLIA_XE = 1e-5


def _silenzio(f, *a, **k):
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return f(*a, **k)


def _info(modello="tanh_class", likelihood=None, params=None, cosmo=None):
    p = dict(COSMO)
    p.update(cosmo or {})
    p.update({"reio_z_re": {"prior": {"min": 4.0, "max": 13.0}},
              "reio_delta_z": {"prior": {"min": 0.05, "max": 3.5}},
              "tau_reio": None})
    p.update(params or {})
    return {"theory": {"Reiolike.ReioTheory": {
                "reio_model": modello, "z_max": 30.0, "z_points": 250, "extra_args": EXTRA}},
            "likelihood": likelihood or {"one": None},
            "params": {k: v for k, v in p.items() if v is not False}}


def _modello(**kw):
    from cobaya.model import get_model
    os.chdir(RADICE)
    return _silenzio(get_model, _info(**kw))


def _teoria(m):
    return m.theory["Reiolike.ReioTheory"]


def _punto(m, zr, dz):
    r = _silenzio(m.logposterior, {"reio_z_re": zr, "reio_delta_z": dz})
    assert np.isfinite(r.logpost), (zr, dz)
    der = dict(zip(m.parameterization.derived_params(), r.derived))
    return r, der["tau_reio"], _teoria(m).current_state["reio_history"]


def _nativo(zr, dz, YHe, **extra):
    c = Class()
    c.set({**COSMO, **EXTRA, "YHe": YHe, "reio_parametrization": "reio_camb",
           "z_reio": zr, "reionization_width": dz, **extra})
    c.compute(level=["thermodynamics"])
    return c


def _chiudi(c):
    c.struct_cleanup(); c.empty()


@pytest.fixture(scope="module")
def mclass():
    return _modello()


@pytest.fixture(scope="module")
def fondo():
    c = Class()
    c.set({**COSMO, **EXTRA, "reio_parametrization": "reio_none"})
    c.compute(level=["thermodynamics"])
    yield c
    _chiudi(c)


# =============================================================================
# C1: la formula
# =============================================================================
@pytest.mark.parametrize("zr,dz", PUNTI)
def test_C1_formula_come_class(zr, dz, fondo):
    from Reiolike.reio_models import tanh_class_model, class_reio_start, helium_fullreio_step
    YHe = fondo.get_current_derived_parameters(["YHe"])["YHe"]
    fhe = YHe / (3.9715 * (1 - YHe))
    c = _nativo(zr, dz, YHe)
    try:
        z_start = class_reio_start(zr, dz)
        z = np.linspace(0.0, min(z_start, 30.0) - 1e-6, 1500)
        xnr = np.array([fondo.ionization_fraction(q) for q in z])
        x = tanh_class_model(z, zr, dz, xnr, 1 + fhe, z_start=z_start) \
            + helium_fullreio_step(z, 3.5, 0.5, YHe)
        xn = np.array([c.ionization_fraction(q) for q in z])
        assert np.max(np.abs(x - xn)) < SOGLIA_XE, np.max(np.abs(x - xn))
    finally:
        _chiudi(c)


# =============================================================================
# C2, C3: tau e tabella contro CLASS nativo
# =============================================================================
@pytest.mark.parametrize("zr,dz", PUNTI)
def test_C2_C3_tau_e_tabella_come_class(mclass, zr, dz):
    _, tau, h = _punto(mclass, zr, dz)
    c = _nativo(zr, dz, float(_teoria(mclass)._last_YHe))
    try:
        tn = c.get_current_derived_parameters(["tau_reio"])["tau_reio"]
        assert abs(tau - tn) < SOGLIA_TAU, (tau, tn, tau - tn)                    # C2
        z = np.asarray(h["z"])[:-1]           # l'ultimo nodo lo sostituisce CLASS
        xn = np.array([c.ionization_fraction(q) for q in z])
        dx = np.max(np.abs(np.asarray(h["xe"])[:-1] - xn))
        assert dx < SOGLIA_XE, dx                                                   # C3
    finally:
        _chiudi(c)


# =============================================================================
# C4: la logL astrofisica
# =============================================================================
@pytest.fixture(scope="module")
def mlike():
    return _modello(likelihood={"Reiolike.ReioLike": {
        "corecon_config_file": "corecon_config.yaml", "print_sigma_table": False}})


def _logL_su_storia(like, h, xe):
    x = (np.asarray(xe) - np.asarray(h["xe_HeII"])) / (1 + h["f_He"])
    tot = 0.0
    for a in like.analyses:
        f = {"HII_fraction": like.compute_gaussian_loglike,
             "upper_limit_HII_fraction": like.compute_upper_limit_loglike,
             "lower_limit_HII_fraction": like.compute_lower_limit_loglike,
             "mixed_HII_fraction": like.compute_mixed_loglike}[a["type"]]
        args = (a, h["z"], x, None) if a["type"] == "HII_fraction" else (a, h["z"], x)
        tot += _silenzio(f, *args)
    return float(tot)


@pytest.mark.parametrize("zr,dz", [(7.2, 0.5), (7.2, 1.0), (6.5, 0.5), (8.14, 2.0), (10.0, 1.0)])
def test_C4_logL_come_storia_nativa(mlike, zr, dz):
    r, _, h = _punto(mlike, zr, dz)
    like = list(mlike.likelihood.values())[0]
    c = _nativo(zr, dz, float(_teoria(mlike)._last_YHe))
    try:
        xe = np.array([c.ionization_fraction(q) for q in h["z"]])
        xe[-1] = h["xe"][-1]
        atteso = _logL_su_storia(like, h, xe)
        assert float(r.loglikes[0]) == pytest.approx(atteso, abs=1e-3), (r.loglikes[0], atteso)
    finally:
        _chiudi(c)


# =============================================================================
# C5: niente pavimento
# =============================================================================
@pytest.mark.parametrize("zr,dz", [(7.2, 0.5), (7.2, 1.0)])
def test_C5_coda_e_x_noreio_non_pavimento(mclass, fondo, zr, dz):
    from Reiolike.reio_models import class_reio_start
    _, _, h = _punto(mclass, zr, dz)
    th = _teoria(mclass)
    z = np.asarray(h["z"])[:-1]; xe = np.asarray(h["xe"])[:-1]
    sopra = z > class_reio_start(zr, dz)
    assert sopra.sum() > 5
    xnr = np.array([fondo.ionization_fraction(q) for q in z[sopra]])
    assert np.allclose(xe[sopra], xnr, rtol=1e-6, atol=0), np.max(np.abs(xe[sopra] / xnr - 1))
    assert np.all(xe[sopra] < th._pavimento)          # il pavimento non e' stato applicato


# =============================================================================
# C6, C7: sabotaggi (i test sono capaci di vedere l'errore)
# =============================================================================
def test_C6_sabotaggio_x_noreio_costante(monkeypatch):
    from Reiolike.reio_theory import ReioTheory
    m = _modello()
    th = _teoria(m)
    monkeypatch.setattr(ReioTheory, "_x_noreio",
                        lambda self, z: np.full(np.shape(z), self._pavimento))
    _, tau, _ = _punto(m, 7.2, 0.5)
    c = _nativo(7.2, 0.5, float(th._last_YHe))
    try:
        d = tau - c.get_current_derived_parameters(["tau_reio"])["tau_reio"]
    finally:
        _chiudi(c)
    assert abs(d) > 5 * SOGLIA_TAU, d


def test_C7_sabotaggio_tanh_lineare(monkeypatch):
    import Reiolike.reio_theory as RT
    from Reiolike.reio_models import tanh_model
    m = _modello()
    monkeypatch.setattr(RT, "tanh_class_model",
                        lambda z, zr, dz, x_noreio, xe_max, z_start: np.maximum(
                            tanh_model(z, zr, dz, xe_max=xe_max), x_noreio))
    _, _, h = _punto(m, 7.2, 1.0)
    c = _nativo(7.2, 1.0, float(_teoria(m)._last_YHe))
    try:
        z = np.asarray(h["z"])[:-1]
        dx = np.max(np.abs(np.asarray(h["xe"])[:-1] - [c.ionization_fraction(q) for q in z]))
    finally:
        _chiudi(c)
    assert dx > 100 * SOGLIA_XE, dx


# =============================================================================
# C8: la tanh lineare non e' cambiata
# =============================================================================
def test_C8_tanh_lineare_invariata():
    m = _modello(modello="tanh")
    th = _teoria(m)
    _, tau, h = _punto(m, 7.2, 0.5)
    xe = np.asarray(h["xe"])[:-1]
    assert np.all(xe >= th._pavimento * (1 - 1e-12)), xe.min()
    assert getattr(th, "_tabella_fondo", None) is None
    # stesso z_re e delta_z, modelli diversi: le storie devono differire
    mc = _modello()
    _, tauc, hc = _punto(mc, 7.2, 0.5)
    assert abs(tau - tauc) > 3e-5, (tau, tauc)


# =============================================================================
# C9: cache
# =============================================================================
def test_C9_cache_per_cosmologia(monkeypatch):
    from Reiolike.reio_theory import ReioTheory
    chiamate = []
    vera = ReioTheory._calcola_fondo

    def conta(self, args):
        chiamate.append(1)
        return vera(self, args)
    monkeypatch.setattr(ReioTheory, "_calcola_fondo", conta)
    m = _modello(cosmo={"omega_b": {"prior": {"min": 0.02, "max": 0.025}}})
    for zr in (6.5, 7.0, 7.5, 8.0):
        _silenzio(m.logposterior, {"reio_z_re": zr, "reio_delta_z": 0.7, "omega_b": 0.02237})
    assert len(chiamate) == 1, len(chiamate)
    assert _teoria(m)._tabella_fondo is not None
    _silenzio(m.logposterior, {"reio_z_re": 7.0, "reio_delta_z": 0.7, "omega_b": 0.0230})
    assert len(chiamate) == 2, len(chiamate)


# =============================================================================
# C10: errori
# =============================================================================
def test_C10_senza_delta_z_si_ferma():
    from cobaya.log import LoggedError
    m = _silenzio(__import__("cobaya.model", fromlist=["get_model"]).get_model,
                  _info(params={"reio_delta_z": False}))
    with pytest.raises((ValueError, LoggedError)):
        _silenzio(m.logposterior, {"reio_z_re": 7.2})


def test_C10_x_noreio_senza_tabella_si_ferma():
    from Reiolike.reio_theory import ReioTheory
    th = ReioTheory.__new__(ReioTheory)
    th.reio_model = "tanh_class"
    with pytest.raises(RuntimeError):
        th._x_noreio(np.array([1.0, 2.0]))
