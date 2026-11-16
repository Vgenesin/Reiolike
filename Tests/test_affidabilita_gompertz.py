#!/usr/bin/env python
"""
Stress test del modello gompertz_sr (Montero-Camacho, Li & Cranmer 2024,
arXiv:2405.13680) e della sua griglia.

    /home/valentina/anaconda3/envs/cmb_env/bin/python -m pytest Tests/test_affidabilita_gompertz.py -s

Il gompertz e' diverso dagli altri modelli in due punti che toccano la griglia:
  - non ha un z_re: calculate() fa una passata di sonda su 600 punti
    uniformi in [0, z_max], ne stima (z_mid, delta_z) con estimate_transition
    (punti al 25% e 75%, come per una tanh) e centra li' la finestra fitta;
  - la forma e' ASIMMETRICA (Gompertz), mentre la finestra e' simmetrica,
    +-z_window_widths*delta_z: le code possono uscirne.

Run: astrofisica di corecon_config.yaml, cosmologia fissata a quella di
test_elica_reiolike.yaml, campionati sigma8 (gompertz_sigma8_source='sample')
e zeta_uv, con prior = intervallo di validita' del paper (Eq. 2.1).

  A. le formule del pacchetto contro il paper (Eqs. 2.2-2.6, 3.1, 3.2)
  B. la griglia, senza CLASS, su tutto il box di validita' a 6 parametri
  C. dentro cobaya: dati, logL, tau, output
  D. rumore della logL lungo sigma8 e zeta_uv
  E. robustezza: prior intero, fuori dal box di validita', cache
  F. posteriore su griglia 2D, pacchetto contro riferimento

Il riferimento e' scritto qui da zero dal testo del paper.  Fino al 28 ott
2026 l'Eq. 3.2 del pacchetto era diversa da quella del paper (A2): corretta,
ora i gruppi B-F confrontano con il riferimento del paper.
"""
import contextlib
import io
import os
import sys
import time
import warnings

import numpy as np
import pytest
import yaml
from scipy import integrate

RADICE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RADICE)
sys.path.insert(0, os.path.join(RADICE, "Tests"))
os.chdir(RADICE)
warnings.filterwarnings("ignore")

from test_affidabilita_tanh import Riferimento, _silenzioso, TOL_LOGL, YAML_RUN  # noqa: E402
from Reiolike.reio_models import (gompertz_sr_model, _sr_ln_alpha, _sr_beta,  # noqa: E402
                                  _gompertz_xhi, f_he)

# Eq. 2.1: intervallo di validita' (training) delle formule SR
BOX = {"sigma8": (0.74, 0.90), "ns": (0.92, 1.00), "h": (0.61, 0.73),
       "Omega_b": (0.04, 0.06), "Omega_m": (0.24, 0.40), "zeta_uv": (20.0, 35.0)}
PRIOR_S8, PRIOR_ZETA = BOX["sigma8"], BOX["zeta_uv"]
C_P5 = [0.0, 1.0, 0.1130, 0.02600, 0.0005491, -0.00006518]     # Eq. 2.3
X_REC = 1e-4                                                   # Eq. 2.5


# =============================================================================
# Riferimento dal paper
# =============================================================================
def ln_alpha_paper(s8, ns, h, Ob, Om, zeta):
    """Eq. 3.1."""
    return (Ob / Om) ** h - (s8 - 0.04835) * (ns + 0.3558 * np.log(0.1123 * zeta)) - Om - ns


def beta_paper(s8, ns, h, Ob, Om, zeta, variante="paper"):
    """Eq. 3.2.  Nel paper il primo termine e' 0.005660^Omega_m / 0.6015
    (letto dall'immagine dell'equazione, pag. 12 della v3).  variante='vecchia'
    e' la forma che il pacchetto usava fino al 28 ott 2026,
    0.005660 / Omega_m^0.6015: serve solo ad A2b."""
    t1 = (0.005660 ** Om / 0.6015 if variante == "paper"
          else 0.005660 / Om ** 0.6015)
    return (t1 - np.log(zeta - (Om + ns * h) ** 15.05) + h) * np.log(Ob) + h / s8


def x_HI_paper(z, la, b):
    """Eqs. 2.2-2.4: x_HI = exp(-exp(P5(ln a~))), ln a~ = beta (ln a - ln alpha)."""
    u = b * (-np.log1p(np.asarray(z, dtype=float)) - la)
    p5 = sum(c * u ** k for k, c in enumerate(C_P5))
    return np.clip(np.exp(-np.exp(p5)), 0.0, 1.0)


class RifGompertz(Riferimento):
    """x_HII(z) e logL scritti dal paper.  Parametri del punto: (sigma8, zeta)."""

    def __init__(self, analyses, YHe, cosmo, variante="paper", floor=None):
        super().__init__(analyses, YHe, floor)
        self.cosmo, self.variante = cosmo, variante

    def _ab(self, s8, zeta):
        c = self.cosmo
        a = (s8, c["ns"], c["h"], c["Omega_b"], c["Omega_m"], zeta)
        return ln_alpha_paper(*a), beta_paper(*a, variante=self.variante)

    def x_HII(self, z, s8, zeta):
        f = self.fHe
        la, b = self._ab(s8, zeta)
        xe = (1 + f - X_REC) * (1 - x_HI_paper(z, la, b)) + X_REC          # Eq. 2.5
        he = f / 2 * (1 + np.tanh((3.5 - np.asarray(z, float)) / 0.5))       # Eq. 2.6
        return (np.maximum(xe + he, self.floor) - he) / (1 + f)

    def media(self, zlo, zhi, s8, zeta):
        if not zhi > zlo:
            return float(self.x_HII(zlo, s8, zeta))
        v, _ = integrate.quad(self.x_HII, zlo, zhi, args=(s8, zeta), limit=400,
                              epsabs=1e-11, epsrel=1e-11)
        return v / (zhi - zlo)


# =============================================================================
# Modello cobaya
# =============================================================================
def _info(prior_s8=PRIOR_S8, prior_zeta=PRIOR_ZETA, theory=None):
    i = yaml.safe_load(open(YAML_RUN))
    i.pop("sampler", None)
    i.pop("output", None)
    i["debug"] = False
    p = i["params"]
    for k in ("reio_z_re", "reio_delta_z", "A_s"):
        p.pop(k)
    p["sigma8"] = {"prior": {"min": prior_s8[0], "max": prior_s8[1]}}
    p["zeta_uv"] = {"prior": {"min": prior_zeta[0], "max": prior_zeta[1]}}
    p["tau_reio"] = {"derived": True}
    t = i["theory"]["Reiolike.ReioTheory"]
    t["reio_model"] = "gompertz_sr"
    t.update(theory or {})
    return i


def _modello(**kw):
    from cobaya.model import get_model
    return _silenzioso(get_model, _info(**kw))


@pytest.fixture(scope="module")
def mg():
    return _modello()


@pytest.fixture(scope="module")
def like(mg):
    return mg.likelihood["Reiolike.ReioLike"]


@pytest.fixture(scope="module")
def theory(mg):
    return mg.theory["Reiolike.ReioTheory"]


def valuta(model, s8, zeta):
    r = _silenzioso(model.logposterior, {"sigma8": s8, "zeta_uv": zeta})
    lp = float(r.loglikes[0]) if r.loglikes is not None and len(r.loglikes) else -np.inf
    der = dict(zip(model.parameterization.derived_params(), r.derived or []))
    return lp, der


@pytest.fixture(scope="module")
def cosmo(mg):
    p = mg.parameterization.constant_params()
    h = p["H0"] / 100
    return {"ns": p["n_s"], "h": h, "Omega_b": p["omega_b"] / h ** 2,
            "Omega_m": (p["omega_b"] + p["omega_cdm"]) / h ** 2}


@pytest.fixture(scope="module")
def rif(mg, like, theory, cosmo):
    valuta(mg, 0.81, 27.0)
    return RifGompertz(like.analyses, theory.classy.pars["YHe"], cosmo)


def _x_ai_dati(like, h):
    x = (h["xe"] - h["xe_HeII"]) / (1 + h["f_He"])
    return [like._model_on_windows(a, h["z"], x)[0] for a in like.analyses]


PUNTI = [(s8, ze) for s8 in np.linspace(*PRIOR_S8, 5) for ze in np.linspace(*PRIOR_ZETA, 5)]


# =============================================================================
# A. Formule contro il paper
# =============================================================================
@pytest.fixture(scope="module")
def box_casuale():
    rng = np.random.default_rng(1)
    nomi = list(BOX)
    pts = rng.uniform([BOX[k][0] for k in nomi], [BOX[k][1] for k in nomi], size=(3000, 6))
    angoli = np.array(np.meshgrid(*[BOX[k] for k in nomi])).T.reshape(-1, 6)
    return np.vstack([pts, angoli])           # (s8, ns, h, Ob, Om, zeta)


def test_A1_ln_alpha_Eq_3_1(box_casuale):
    for s8, ns, h, Ob, Om, ze in box_casuale:
        assert _sr_ln_alpha(s8, ns, h, Ob, Om, ze) == pytest.approx(
            ln_alpha_paper(s8, ns, h, Ob, Om, ze), rel=1e-12, abs=1e-12)


def test_A2_beta_Eq_3_2(box_casuale):
    """Il pacchetto usava 0.005660/Omega_m^0.6015, il paper ha
    0.005660^Omega_m/0.6015 (differenza fino a 1.5 nel box).  Corretto il
    28 ott 2026."""
    d = [abs(_sr_beta(*p) - beta_paper(*p)) for p in box_casuale]
    print(f"\n  |beta pacchetto - beta paper|: max {max(d):.3f}, mediana {np.median(d):.3f}")
    assert max(d) < 1e-10


def test_A2b_la_lettura_del_paper_riproduce_il_suo_best_fit():
    """Prova che la lettura 0.005660^Omega_m/0.6015 e' quella giusta: con i
    parametri di best fit di gomp+SRFull (Tab. 1: sigma8 0.8100, ns 0.9636,
    H0 67.26, Omega_b h^2 0.02234, Omega_m 0.3166, zeta 26.9) il paper
    riporta z_re = 6.98 (x_HI = 0.5).  Con la forma che il pacchetto usava
    prima viene 6.94; il pacchetto ora deve coincidere con la lettura giusta."""
    h, Om = 0.6726, 0.3166
    a = (0.8100, 0.9636, h, 0.02234 / h ** 2, Om, 26.9)
    z = np.linspace(4, 12, 400001)
    zre = {v: np.interp(0.5, x_HI_paper(z, ln_alpha_paper(*a), beta_paper(*a, variante=v)), z)
           for v in ("paper", "vecchia")}
    zre["pacchetto"] = np.interp(0.5, _gompertz_xhi(z, _sr_ln_alpha(*a), _sr_beta(*a)), z)
    print(f"\n  z_re best fit: lettura paper {zre['paper']:.3f}, forma vecchia "
          f"{zre['vecchia']:.3f}, pacchetto {zre['pacchetto']:.3f}; paper Tab. 1: 6.98(17)")
    assert abs(zre["paper"] - 6.98) < 0.01
    assert abs(zre["vecchia"] - 6.98) > 0.02
    assert zre["pacchetto"] == pytest.approx(zre["paper"], abs=1e-9)


def test_A3_forma_Gompertz_Eqs_2_2_2_4(box_casuale):
    z = np.linspace(0, 30, 601)
    for p in box_casuale[::50]:
        la, b = _sr_ln_alpha(*p), _sr_beta(*p)
        assert np.allclose(_gompertz_xhi(z, la, b), x_HI_paper(z, la, b), atol=1e-13)


def test_A4_xe_Eq_2_5(box_casuale):
    """Eq. 2.5 con n_He/n_H = Y/(3.9715 (1-Y)) (CLASS) invece di C = 4: e'
    la scelta dichiarata nel pacchetto.  Lo step HeII lo aggiunge ReioTheory."""
    z = np.linspace(0, 30, 601)
    Y = 0.2454
    f = f_he(Y)
    for p in box_casuale[::50]:
        s8, ns, h, Ob, Om, ze = p
        xe = gompertz_sr_model(z, s8, ns, h, Ob, Om, ze, include_HeII=False, Y_He=Y)
        xhi = x_HI_paper(z, _sr_ln_alpha(*p), _sr_beta(*p))
        assert np.allclose(xe, (1 + f - X_REC) * (1 - xhi) + X_REC, atol=1e-13)


def test_A5_nessun_errore_nel_box_di_validita(box_casuale):
    """Dentro l'Eq. 2.1 (angoli compresi) le formule non devono mai
    sollevare (beta <= 0, log di argomento negativo)."""
    z = np.linspace(0, 30, 61)
    for p in box_casuale:
        gompertz_sr_model(z, *p, include_HeII=False)


# =============================================================================
# B. La griglia, senza CLASS, su tutto il box a 6 parametri
# =============================================================================
def griglia_come_calculate(th, p):
    """Esattamente i passi di calculate() per gompertz_sr, senza CLASS."""
    kw = dict(zip(("sigma8", "ns", "h", "Omega_b", "Omega_m", "zeta_uv"), p))
    kw.update(include_HeII=False, Y_He=th._last_YHe)
    zp = np.linspace(0.0, th.z_max, 600)
    z_mid, dz = th.estimate_transition(zp, gompertz_sr_model(zp, **kw))
    zg = th._build_grid(z_mid, delta_z=dz)
    xe = th._finalize_xe(zg, gompertz_sr_model(zg, **kw), {})
    return zg, xe, z_mid, dz, kw


def _x_esatto(th, z, kw):
    f = f_he(th._last_YHe)
    he = f / 2 * (1 + np.tanh((th.reio_z_HeII - z) / th.reio_delta_z_HeII))
    xe = np.maximum(gompertz_sr_model(z, **kw) + he, th._pavimento)
    return (xe - he) / (1 + f)


@pytest.fixture(scope="module")
def misure_box(mg, theory, box_casuale):
    """Per ogni punto del box: errore punto per punto della tabella su
    [0, z_max - 1] (l'ultimo nodo e' azzerato per convenzione), dove cade,
    e se la transizione vera (x_HI fra 0.001 e 0.999) sta nella finestra fitta."""
    th = theory
    valuta(mg, 0.81, 27.0)
    zf = np.linspace(0, th.z_max - 1.0, 20001)
    out = []
    for p in box_casuale:
        with contextlib.redirect_stdout(io.StringIO()):
            zg, xe, z_mid, dz, kw = griglia_come_calculate(th, p)
        f = f_he(th._last_YHe)
        he = f / 2 * (1 + np.tanh((th.reio_z_HeII - zg) / th.reio_delta_z_HeII))
        x_tab = np.interp(zf, zg, (xe - he) / (1 + f))
        d = np.abs(x_tab - _x_esatto(th, zf, kw))
        la, b = _sr_ln_alpha(*p), _sr_beta(*p)
        xhi = x_HI_paper(zf, la, b)
        # l'artefatto si cerca su tutta la tabella, fino a z_max
        x30 = x_HI_paper(np.linspace(0, th.z_max, 3001), la, b)
        k = np.argmax(x30)
        artefatto = bool(x30[k] - x30[k:].min() > 0.01)
        trans = zf[(xhi > 1e-3) & (xhi < 0.999)]
        half = th.z_window_widths * dz
        z_vero = np.interp(0.5, xhi, zf)
        out.append(dict(p=p, artefatto=artefatto, xe=xe, zg=zg,
                        err=d.max(), z_err=zf[np.argmax(d)], n=len(zg), z_mid=z_mid,
                        z_vero=z_vero, dz=dz, fuori=(trans.min() < z_mid - half or trans.max() > z_mid + half),
                        trans=(trans.min(), trans.max()), win=(z_mid - half, z_mid + half)))
    return out


def test_B0_artefatto_del_polinomio_ad_alto_z(misure_box, cosmo, theory):
    """P5 (Eq. 2.3) ha c5 < 0: dP5/du si annulla a u = ln a~ = -11.23, e per
    u piu' negativo (z piu' alto) P5 risale, quindi x_HI = exp(-exp(P5))
    RISCENDE verso 0: una seconda "ionizzazione" finta ad alto z, fuori
    dall'intervallo in cui il polinomio e' stato fittato.

    Con la beta corretta (28 ott 2026) nel box dell'Eq. 2.1 non succede mai
    sotto z = 30 (comincia sopra z ~ 32); nella fetta della run sopra
    z = 38.7.  Con la beta sbagliata di prima succedeva nel 2% del box.
    Fuori dall'Eq. 2.1 (zeta_uv > 35) puo' cominciare anche a z < 10: per
    questo z_max > 30 e' vietato (E4) e i prior vanno tenuti nell'Eq. 2.1."""
    n = sum(m["artefatto"] for m in misure_box)
    print(f"\n  box 6D: {n}/{len(misure_box)} storie con x_HI che riscende entro z = 30")
    c = cosmo
    zf = np.linspace(0, 60, 30001)
    primo = np.inf
    for s8 in np.linspace(*PRIOR_S8, 21):
        for ze in np.linspace(*PRIOR_ZETA, 21):
            a = (s8, c["ns"], c["h"], c["Omega_b"], c["Omega_m"], ze)
            x = x_HI_paper(zf, ln_alpha_paper(*a), beta_paper(*a))
            k = np.argmax(x)
            if np.any(x[k:] < 0.99):
                primo = min(primo, zf[k + np.argmax(x[k:] < 0.99)])
    print(f"  fetta della run: x_HI riscende sotto 0.99 al piu' presto a z = {primo:.1f}; "
          f"z_max = {theory.z_max:g}")
    assert primo > theory.z_max


def test_B1_stima_della_transizione(misure_box):
    """La sonda a 600 punti (passo 0.05) deve trovare il midpoint vero.
    Solo storie monotone: con l'artefatto (B0) il "midpoint" non e' definito."""
    buone = [m for m in misure_box if not m["artefatto"]]
    e = np.array([abs(m["z_mid"] - m["z_vero"]) for m in buone])
    zv = np.array([m["z_vero"] for m in buone])
    print(f"\n  midpoint vero nel box: {zv.min():.2f} - {zv.max():.2f}; "
          f"|z_mid stimato - vero| max {e.max():.1e}")
    assert e.max() < 0.01


def test_B2_la_transizione_sta_nella_finestra_fitta(misure_box):
    """La finestra fitta e' simmetrica, +-6 delta_z attorno a z_mid, con
    delta_z stimato dai punti al 25% e al 75%: e' la ricetta della tanh.  Il
    Gompertz e' asimmetrico, con la coda ad alto z ~5 volte piu' lunga, e
    con la beta corretta la coda esce dalla finestra nel ~9% del box (275
    storie su 3064, 28 ott 2026): li' cade sulla griglia di fondo.  Conta
    quanto si sbaglia in quei casi, non la copertura: l'errore deve restare
    sotto 1e-3 (misurato 4.3e-4 su tutto il box).  Una finestra asimmetrica
    dai quantili della sonda lo ridurrebbe di ~1/3 (prototipo gomp_asim.py):
    non serve, la posteriore si sposta di 0.001 sigma (F1)."""
    buone = [m for m in misure_box if not m["artefatto"]]
    fuori = [m for m in buone if m["fuori"]]
    e_fuori = max([m["err"] for m in fuori] or [0.0])
    print(f"\n  transizione (x_HI 0.001-0.999) fuori dalla finestra: {len(fuori)}/{len(buone)}; "
          f"errore max in quei casi {e_fuori:.1e}")
    assert len(fuori) < 0.15 * len(buone)
    assert e_fuori < 1e-3


def test_B3_errore_della_tabella_su_tutto_il_box(misure_box):
    """Errore punto per punto della spezzata su x_HII, su tutto z, storie
    monotone.  La tanh della run sta a 1e-4 (test C0 della tanh); il
    Gompertz (beta corretta, 28 ott 2026) sta a 2.4e-4 di mediana, fino a
    4.3e-4, e in ~20% dei punti il massimo e' FUORI dalla finestra fitta (coda
    ad alto z, vedi B2).  Sui dati della run resta 2e-4 (C0) e la posteriore
    non si sposta (F1)."""
    buone = [m for m in misure_box if not m["artefatto"]]
    e = np.array([m["err"] for m in buone])
    fuori = sum(not (m["win"][0] <= m["z_err"] <= m["win"][1]) for m in buone)
    peg = max(buone, key=lambda m: m["err"])
    print(f"\n  errore max {e.max():.1e} (mediana {np.median(e):.1e}) a z={peg['z_err']:.2f}; "
          f"massimo fuori dalla finestra in {fuori}/{len(buone)} punti; "
          f"nodi {min(m['n'] for m in buone)}-{max(m['n'] for m in buone)}")
    assert e.max() < 1e-3


def test_B4_griglia_valida_e_nel_budget(theory, misure_box):
    th = theory
    for m in misure_box[::10]:
        if m["artefatto"]:
            continue
        zg, xe = m["zg"], m["xe"]
        assert zg[0] == 0.0 and zg[-1] == th.z_max and np.all(np.diff(zg) >= th.z_min_spacing * (1 - 1e-9))
        assert len(zg) <= th._node_budget()
        assert np.all(np.isin(th._z_nodi_dati, zg))
        assert np.all(xe[:-1] >= th._pavimento) and xe[-2] < 10 * th._pavimento


# =============================================================================
# C. Dentro cobaya
# =============================================================================
def test_C0_x_HII_ai_dati(mg, like, rif):
    ep = ef = 0.0
    for s8, ze in PUNTI:
        valuta(mg, s8, ze)
        xp = _x_ai_dati(like, mg.theory["Reiolike.ReioTheory"].current_state["reio_history"])
        for a, xa in zip(like.analyses, xp):
            for i in range(len(a["z"])):
                lo, hi = rif.finestra(a, i)
                d = abs(xa[i] - rif.media(lo, hi, s8, ze))
                if hi > lo:
                    ef = max(ef, d)
                else:
                    ep = max(ep, d)
    print(f"\n  errore max su x_HII: dati puntuali {ep:.1e}, con finestra {ef:.1e}")
    assert ep < 1e-9 and ef < 3e-4


def test_C1_logL_contro_riferimento(mg, rif):
    d = []
    for s8, ze in PUNTI:
        r = rif.logL(s8, ze)
        lp = valuta(mg, s8, ze)[0]
        d.append((abs(lp - r), r, s8, ze))
    vicini = [x for x in d if x[1] > -50]
    print(f"\n  |dlogL| max su tutto il prior {max(x[0] for x in d):.1e}; "
          f"dove logL > -50 ({len(vicini)} punti): {max([x[0] for x in vicini] or [0]):.1e}")
    assert all(x[0] < TOL_LOGL for x in vicini)
    assert all(x[0] < max(TOL_LOGL, 1e-3 * abs(x[1])) for x in d)


def test_C2_tau_contro_integrale_indipendente(mg, theory, rif):
    """tau di CLASS contro sigma_T c int n_e dt, con xe ANALITICO (non la
    tabella): misura insieme la lettura di CLASS e l'errore di griglia."""
    sT, Mpc, mH, G = 6.6524616e-29, 3.085677581282e22, 1.673575e-27, 6.67428e-11
    for s8, ze in ((0.74, 20.0), (0.81, 27.0), (0.90, 35.0), (0.74, 35.0), (0.90, 20.0)):
        _, der = valuta(mg, s8, ze)
        cl = theory.classy
        YHe = cl.get_current_derived_parameters(["YHe"])["YHe"]
        H0 = cl.Hubble(0)
        rho_b = 3 * (H0 / Mpc * 2.99792458e8) ** 2 / (8 * np.pi * G) * cl.Omega_b()
        nH0 = (1 - YHe) * rho_b / mH
        zg = np.linspace(0, theory.z_max, 60001)
        f = rif.fHe
        he = f / 2 * (1 + np.tanh((3.5 - zg) / 0.5))
        xe = rif.x_HII(zg, s8, ze) * (1 + f) + he
        Hz = np.interp(zg, zg[::100], np.array([cl.Hubble(z) for z in zg[::100]]))
        # CLASS integra la storia di reionizzazione solo fin dove la tabella
        # arriva (z_max), oltre usa la ricombinazione: stesso dominio qui
        tau = sT * nH0 * Mpc * integrate.trapezoid(xe * (1 + zg) ** 2 / Hz, zg)
        print(f"\n  sigma8={s8}, zeta={ze}: tau CLASS {der['tau_reio']:.5f}, integrale {tau:.5f}")
        assert der["tau_reio"] == pytest.approx(tau, rel=3e-3)


def test_C2b_riproduce_la_Tab_1_del_paper():
    """Il pacchetto completo (cobaya + CLASS) con i parametri di gomp+SRFull
    della Tab. 1 (sigma8 0.8100, ns 0.9636, H0 67.26, Omega_b h^2 0.02234,
    Omega_c h^2 0.1202, zeta 26.9): il paper da' tau = 0.0526(15) e
    z_re = 6.98(17).  Misurato dopo la correzione di beta: tau 0.0526,
    z_re 6.961 (Omega_m senza neutrini, scelta voluta: con i neutrini 6.979).
    Prima della correzione: tau 0.0514, z_re 6.917."""
    from cobaya.model import get_model
    info = _info()
    info["params"].update({"n_s": 0.9636, "omega_b": 0.02234, "omega_cdm": 0.1202, "H0": 67.26})
    m = _silenzioso(get_model, info)
    _, der = valuta(m, 0.8100, 26.9)
    h = m.theory["Reiolike.ReioTheory"].current_state["reio_history"]
    x = (h["xe"] - h["xe_HeII"]) / (1 + h["f_He"])
    zf = np.linspace(4, 15, 110001)
    zre = np.interp(0.5, 1 - np.interp(zf, h["z"], x), zf)
    print(f"\n  tau {der['tau_reio']:.4f} (paper 0.0526 +- 0.0015), z_re {zre:.3f} (paper 6.98 +- 0.17)")
    assert abs(der["tau_reio"] - 0.0526) < 0.2 * 0.0015
    assert abs(zre - 6.98) < 0.2 * 0.17


def test_C3_nessuna_stampa_a_ogni_valutazione(mg):
    """In una catena calculate() gira milioni di volte: niente print su
    stdout (i messaggi di diagnostica vanno nel logger, a livello debug)."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        mg.logposterior({"sigma8": 0.80, "zeta_uv": 26.0})
    out = buf.getvalue()
    print(f"\n  stdout di una valutazione: {out.strip()[:150]!r}")
    assert "DEBUG" not in out


def test_C4_tempo(mg, theory, cosmo):
    """La passata di sonda (600 punti) e la griglia non devono pesare.  Il
    tempo TOTALE (~0.55 s contro ~0.04 s della tanh) e' quasi tutto di CLASS:
    con gompertz_sigma8_source='sample' riceve sigma8 invece di A_s e deve
    ricavare A_s calcolando P(k).  Si stampa, non si fa una condizione."""
    c = cosmo
    p = (0.81, c["ns"], c["h"], c["Omega_b"], c["Omega_m"], 27.0)
    t0 = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()):
        for _ in range(50):
            griglia_come_calculate(theory, p)
    t_griglia = (time.perf_counter() - t0) / 50
    t = []
    for k in range(10):
        t0 = time.perf_counter()
        valuta(mg, 0.76 + 0.01 * k, 21.0 + 0.9 * k)
        t.append(time.perf_counter() - t0)
    print(f"\n  sonda + griglia {1e3 * t_griglia:.2f} ms; valutazione completa "
          f"{1e3 * np.median(t):.0f} ms")
    assert t_griglia < 5e-3


# =============================================================================
# D. Rumore
# =============================================================================
def _rumore(res):
    res = np.asarray(res)
    return res[1:-1] - 0.5 * (res[:-2] + res[2:])


@pytest.fixture(scope="module")
def massimo(mg, rif):
    """Massimo della logL di riferimento (posteriore piatta nel prior)."""
    from scipy.optimize import minimize
    best = max(PUNTI, key=lambda p: rif.logL(*p))
    r = minimize(lambda v: -rif.logL(*v), best, method="Nelder-Mead",
                 bounds=[PRIOR_S8, PRIOR_ZETA], options={"xatol": 1e-4, "fatol": 1e-4})
    return tuple(r.x), -r.fun


@pytest.mark.parametrize("asse", ["sigma8", "zeta_uv"])
def test_D1_rumore_nella_posteriore(mg, rif, massimo, asse):
    (s8, ze), _ = massimo
    if asse == "sigma8":
        pts = [(s, ze) for s in np.linspace(max(s8 - 0.03, PRIOR_S8[0]), min(s8 + 0.03, PRIOR_S8[1]), 121)]
    else:
        pts = [(s8, z) for z in np.linspace(max(ze - 3, PRIOR_ZETA[0]), min(ze + 3, PRIOR_ZETA[1]), 121)]
    res = np.array([valuta(mg, *p)[0] - rif.logL(*p) for p in pts])
    n = np.abs(_rumore(res)).max()
    print(f"\n  massimo sigma8={s8:.4f}, zeta={ze:.3f}; lungo {asse}: residuo max "
          f"{np.abs(res).max():.1e}, rumore {n:.1e}")
    assert n < 1e-2 and np.abs(res).max() < 0.05


# =============================================================================
# E. Robustezza
# =============================================================================
def test_E1_prior_intero_fitto(mg):
    for s8 in np.linspace(*PRIOR_S8, 13):
        for ze in np.linspace(*PRIOR_ZETA, 13):
            lp, der = valuta(mg, s8, ze)
            assert np.isfinite(lp) and np.isfinite(der["tau_reio"]), (s8, ze)


def test_E2_fuori_dal_box_non_ferma_la_catena():
    """Fuori dall'Eq. 2.1 le formule SR possono sollevare (log di un numero
    negativo, beta <= 0): con stop_at_error=True fermerebbero la catena.  Dal
    16 nov 2026 calculate() scarta ogni punto fuori dal box (likelihood nulla).
    Prior gaussiani larghi (illimitati: ammessi, con avviso), punti dentro e
    fuori dal box: nessuna eccezione; fuori -> -inf, dentro -> finito."""
    info = _info()
    info["params"]["sigma8"] = {"prior": {"dist": "norm", "loc": 0.81, "scale": 0.3}}
    info["params"]["zeta_uv"] = {"prior": {"dist": "norm", "loc": 27.0, "scale": 40.0}}
    from cobaya.model import get_model
    m = _silenzioso(get_model, info)
    problemi, sbagliati = [], []
    for s8 in np.linspace(0.5, 1.1, 7):
        for ze in (5.0, 10.0, 15.0, 20.0, 27.0, 35.0, 50.0, 100.0):
            try:
                lp = valuta(m, s8, ze)[0]
            except Exception as e:
                problemi.append((s8, ze, type(e).__name__, str(e)[:60]))
                continue
            dentro = BOX["sigma8"][0] <= s8 <= BOX["sigma8"][1] and BOX["zeta_uv"][0] <= ze <= BOX["zeta_uv"][1]
            if np.isfinite(lp) != dentro:
                sbagliati.append((s8, ze, lp))
    print(f"\n  punti che fermano la catena: {len(problemi)} su 56; {problemi[:3]}")
    print(f"  punti con esito sbagliato (fuori ma finiti o dentro ma -inf): {sbagliati[:3]}")
    assert not problemi and not sbagliati


def test_E3_cache(mg):
    a1 = valuta(mg, 0.80, 25.0)[0]
    valuta(mg, 0.88, 33.0)
    a2 = valuta(mg, 0.80, 25.0)[0]
    assert a1 == a2


# =============================================================================
# F. Posteriore
# =============================================================================
@pytest.mark.lento
def test_F1_posteriore_su_griglia(mg, rif):
    """Momenti della posteriore (prior piatto sull'Eq. 2.1) su griglia 2D,
    pacchetto contro riferimento."""
    ss = np.linspace(*PRIOR_S8, 33)
    zz = np.linspace(*PRIOR_ZETA, 41)
    S, Z = np.meshgrid(ss, zz, indexing="ij")

    def mom(L):
        w = np.exp(L - L.max())
        w /= w.sum()
        m = np.array([np.sum(w * X) for X in (S, Z)])
        s = np.array([np.sqrt(np.sum(w * (X - mi) ** 2)) for X, mi in zip((S, Z), m)])
        return m, s

    Lp = np.array([[valuta(mg, s, z)[0] for z in zz] for s in ss])
    Lr = np.array([[rif.logL(s, z) for z in zz] for s in ss])
    (mp, sp), (mr, sr) = mom(Lp), mom(Lr)
    print(f"\n  riferimento: sigma8 {mr[0]:.4f} +- {sr[0]:.4f}, zeta {mr[1]:.3f} +- {sr[1]:.3f}")
    print(f"  pacchetto  : sigma8 {mp[0]:.4f} +- {sp[0]:.4f}, zeta {mp[1]:.3f} +- {sp[1]:.3f}")
    print(f"  spostamenti {np.abs(mp - mr) / sr} sigma, sigma relative {np.abs(sp / sr - 1)}")
    assert np.all(np.abs(mp - mr) < 0.02 * sr) and np.all(np.abs(sp / sr - 1) < 0.02)


def test_E4_z_max_oltre_30_fermato_all_avvio():
    """Oltre z = 30 l'artefatto di P5 (B0) entrerebbe nella tabella: con
    gompertz_sr z_max > 30 deve fermare il run SUBITO, spiegando perche'.
    z_max = 30 (il default) funziona; gli altri modelli non sono toccati."""
    with pytest.raises(Exception) as e:
        _modello(theory={"z_max": 35.0})
    msg = str(e.value)
    assert "z_max=35" in msg and "P5" in msg, msg
    m = _modello(theory={"z_max": 30.0})
    assert np.isfinite(valuta(m, 0.81, 27.0)[0])


def _avvisi_alla_costruzione(**kw):
    import logging

    class H(logging.Handler):
        def __init__(self):
            super().__init__(logging.WARNING)
            self.msg = []

        def emit(self, r):
            self.msg.append(r.getMessage())
    h = H()
    logging.getLogger().addHandler(h)
    try:
        m = _modello(**kw)
    finally:
        logging.getLogger().removeHandler(h)
    return m, [x for x in h.msg if "gompertz_sr: i prior" in x]


def test_E5_errore_se_i_prior_escono_dall_Eq_2_1():
    """Il prior deve stare dentro l'Eq. 2.1.  Dentro (quelli della suite,
    cosmologia di Planck fissata): nessun avviso, nessun errore.  zeta_uv fino
    a 100, sigma8 fino a 1.1, H0 fra 55 e 85: ERRORE all'avvio, che nomina il
    parametro e l'intervallo del paper."""
    _, a = _avvisi_alla_costruzione()
    assert not a, a
    casi = [(dict(prior_zeta=(10.0, 100.0)), "zeta_uv in [10, 100]"),
            (dict(prior_s8=(0.6, 1.1)), "sigma8 in [0.6, 1.1]")]
    for kw, atteso in casi:
        with pytest.raises(Exception) as e:
            _modello(**kw)
        print(f"\n  {kw}: {str(e.value)[:160]}")
        assert atteso in str(e.value) and "Eq. 2.1" in str(e.value)
    info = _info()
    info["params"]["H0"] = {"prior": {"min": 55.0, "max": 85.0}}
    from cobaya.model import get_model
    with pytest.raises(Exception) as e:
        _silenzioso(get_model, info)
    print(f"  H0 55-85: {str(e.value)[:160]}")
    assert "H0 in [55, 85]" in str(e.value)


def test_E6_prior_gaussiano_ammesso_con_avviso():
    """Un prior gaussiano non puo' stare dentro il box (e' illimitato): il
    modello si costruisce, con un avviso che dice che le code vengono
    scartate; il punto centrale resta finito, la coda fuori -> -inf."""
    info = _info()
    info["params"]["zeta_uv"] = {"prior": {"dist": "norm", "loc": 27.0, "scale": 1.0}}
    import logging
    from cobaya.model import get_model

    class H(logging.Handler):
        def __init__(self):
            super().__init__(logging.WARNING)
            self.msg = []

        def emit(self, r):
            self.msg.append(r.getMessage())
    h = H()
    logging.getLogger().addHandler(h)
    try:
        m = _silenzioso(get_model, info)
    finally:
        logging.getLogger().removeHandler(h)
    a = [x for x in h.msg if "prior illimitato su zeta_uv" in x]
    assert a, h.msg
    assert np.isfinite(valuta(m, 0.81, 27.0)[0])
    assert not np.isfinite(valuta(m, 0.81, 36.0)[0])


def test_E7_Omega_b_Omega_m_tagliati_punto_per_punto():
    """omega_b, omega_cdm, H0 campionati con prior rettangolari ragionevoli:
    gli angoli escono dal box in Omega_m (combinazione dei tre).  Il modello si
    costruisce con un avviso che dice QUANTO del prior viene tagliato; un
    punto con Omega_m fuori -> -inf, uno dentro -> finito.  Se nessun punto del
    prior sta nel box: errore."""
    import logging
    from cobaya.model import get_model

    def costruisci(ob, oc, H0):
        info = _info()
        info["params"]["omega_b"] = {"prior": {"min": ob[0], "max": ob[1]}}
        info["params"]["omega_cdm"] = {"prior": {"min": oc[0], "max": oc[1]}}
        info["params"]["H0"] = {"prior": {"min": H0[0], "max": H0[1]}}

        class H(logging.Handler):
            def __init__(self):
                super().__init__(logging.WARNING)
                self.msg = []

            def emit(self, r):
                self.msg.append(r.getMessage())
        h = H()
        logging.getLogger().addHandler(h)
        try:
            m = _silenzioso(get_model, info)
        finally:
            logging.getLogger().removeHandler(h)
        return m, [x for x in h.msg if "fuori dall'Eq. 2.1" in x]

    m, a = costruisci((0.019, 0.026), (0.10, 0.14), (61.0, 73.0))
    print(f"\n  {a[0][:200] if a else a}")
    assert len(a) == 1 and "% del prior" in a[0]
    base = {"sigma8": 0.81, "zeta_uv": 27.0}
    dentro = _silenzioso(m.logposterior, dict(base, omega_b=0.0224, omega_cdm=0.120, H0=67.0))
    fuori = _silenzioso(m.logposterior, dict(base, omega_b=0.0224, omega_cdm=0.140, H0=61.5))  # Omega_m 0.43
    assert np.isfinite(dentro.logpost) and not np.isfinite(fuori.logpost)

    with pytest.raises(Exception) as e:
        costruisci((0.019, 0.026), (0.20, 0.25), (61.0, 73.0))      # Omega_m sempre > 0.41
    assert "nessun punto" in str(e.value)


# =============================================================================
# G. gompertz_sigma8_source = 'class_mPk' (A_s campionato, sigma8 da CLASS)
# =============================================================================
def _modello_class_mpk(nu):
    from cobaya.model import get_model
    P = lambda lo, hi: {"prior": {"min": lo, "max": hi}}
    params = {"logA": dict(P(2.9, 3.2), drop=True), "A_s": {"value": "lambda logA: 1e-10*np.exp(logA)"},
              "zeta_uv": P(20, 35), "n_s": 0.9636, "omega_b": 0.02234, "omega_cdm": 0.1202, "H0": 67.26,
              "N_ur": nu["N_ur"], "tau_reio": {}}
    th = {"reio_model": "gompertz_sr", "gompertz_sigma8_source": "class_mPk", "z_max": 30.0,
          "z_points": 250, "extra_args": {k: v for k, v in nu.items() if k != "N_ur"}}

    def dummy(_self=None):
        _self.provider.get_Cl(ell_factor=True)
        return 0.0
    lik = {"dummy": {"external": dummy, "requires": {"Cl": {"tt": 2500, "te": 2500, "ee": 2500}}}}
    return _silenzioso(get_model, {"params": params, "theory": {"Reiolike.ReioTheory": th},
                                   "likelihood": lik})


def test_G1_class_mPk_sigma8_con_i_neutrini_della_run():
    """La run di CLASS che ricava sigma8 da A_s deve avere la STESSA cosmologia
    della run vera, neutrini massivi compresi.  Fino al 16 nov 2026 riceveva
    solo omega_b, omega_cdm, H0, n_s, A_s: neutrini senza massa, sigma8 +1.3%
    con 3 nu da 0.02 eV e insensibile a m_nu.  Confronto con una run di CLASS
    indipendente, per tre configurazioni di neutrini; con i Cl richiesti
    (come HiLLiPoP) il punto deve essere valido.  Con 1 nu da 0.06 eV e A_s
    del best fit del paper, sigma8 deve tornare quello della Tab. 1."""
    from classy import Class
    logA = np.log(2.094e-9 * 1e10)                       # Tab. 1, gomp + SRFull
    configs = {"3 x 0.02 eV": dict(N_ncdm=1, deg_ncdm=3, m_ncdm=0.02, T_ncdm=0.71611, N_ur=0.00441),
               "1 x 0.06 eV": dict(N_ncdm=1, m_ncdm=0.06, N_ur=2.0328),
               "1 x 0.30 eV": dict(N_ncdm=1, m_ncdm=0.30, N_ur=2.0328)}
    s8 = {}
    for nome, nu in configs.items():
        m = _modello_class_mpk(nu)
        r = _silenzioso(m.logposterior, {"logA": logA, "zeta_uv": 26.9})
        assert np.isfinite(r.logpost), nome
        th = m.theory["Reiolike.ReioTheory"]
        punto = {"omega_b": 0.02234, "omega_cdm": 0.1202, "H0": 67.26, "n_s": 0.9636,
                 "N_ur": nu["N_ur"], "A_s": 1e-10 * np.exp(logA)}
        s8[nome] = th._compute_sigma8_from_As(punto, logA=logA)
        c = Class()
        c.set(dict(output="mPk", **{"P_k_max_h/Mpc": 1.0}, omega_b=0.02234, omega_cdm=0.1202,
                   H0=67.26, n_s=0.9636, A_s=1e-10 * np.exp(logA), **nu))
        c.compute()
        rif = c.sigma8()
        c.struct_cleanup()
        c.empty()
        print(f"\n  {nome}: sigma8 pacchetto {s8[nome]:.5f}, CLASS indipendente {rif:.5f}")
        assert abs(s8[nome] / rif - 1) < 1e-4, nome
    # sente m_nu, nel verso giusto (piu' massa -> meno struttura)
    assert s8["1 x 0.30 eV"] < s8["1 x 0.06 eV"] - 0.04
    # Tab. 1 del paper: sigma8 = 0.8100(44) con A_s = 2.094e-9
    assert abs(s8["1 x 0.06 eV"] - 0.8100) < 0.0044


def test_G2_A_s_e_sigma8_insieme_fermano_l_avvio():
    """'sample' (sigma8 campionato) con A_s o logA nei parametri: A_s verrebbe
    ignorato in silenzio (misurato: A_s 1.6, 2.1, 3.4e-9 -> stessa logpost),
    quindi ERRORE all'avvio che indica class_mPk.  E viceversa: 'class_mPk'
    con sigma8 nei parametri -> errore.  Le due configurazioni giuste partono."""
    from cobaya.model import get_model
    # sample + A_s
    info = _info()
    info["params"]["A_s"] = {"prior": {"min": 1.5e-9, "max": 3.5e-9}}
    with pytest.raises(Exception) as e:
        _silenzioso(get_model, info)
    print(f"\n  sample + A_s: {str(e.value)[:150]}")
    assert "class_mPk" in str(e.value) and "A_s" in str(e.value)
    # sample + logA
    info = _info()
    info["params"]["logA"] = {"prior": {"min": 2.9, "max": 3.2}, "drop": True}
    info["params"]["A_s"] = {"value": "lambda logA: 1e-10*np.exp(logA)"}
    with pytest.raises(Exception) as e:
        _silenzioso(get_model, info)
    assert "logA" in str(e.value)
    # class_mPk + sigma8
    nu = dict(N_ncdm=1, m_ncdm=0.06, N_ur=2.0328)
    from copy import deepcopy
    m_ok = _modello_class_mpk(nu)                     # class_mPk + logA: parte
    assert m_ok is not None
    with pytest.raises(Exception) as e:
        P = lambda lo, hi: {"prior": {"min": lo, "max": hi}}
        params = {"logA": dict(P(2.9, 3.2), drop=True), "A_s": {"value": "lambda logA: 1e-10*np.exp(logA)"},
                  "sigma8": P(0.74, 0.90), "zeta_uv": P(20, 35), "n_s": 0.9636, "omega_b": 0.02234,
                  "omega_cdm": 0.1202, "H0": 67.26, "N_ur": 2.0328}
        th = {"reio_model": "gompertz_sr", "gompertz_sigma8_source": "class_mPk",
              "extra_args": {"N_ncdm": 1, "m_ncdm": 0.06}}
        _silenzioso(get_model, {"params": params, "theory": {"Reiolike.ReioTheory": th},
                                "likelihood": {"one": None}})
    print(f"  class_mPk + sigma8: {str(e.value)[:150]}")
    assert "togli sigma8" in str(e.value)
