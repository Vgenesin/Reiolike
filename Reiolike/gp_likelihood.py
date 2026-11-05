from __future__ import annotations
# re, yaml, corecon ed erfinv servivano al lettore locale, ora rimosso.
import warnings
import numpy as np
from cobaya.likelihood import Likelihood
from scipy.optimize import minimize
from scipy.special import expit  # sigmoide stabile

# l'integrale sui limiti con errore e' quello della likelihood astrofisica:
# stesse funzioni, una sola definizione (reio_like.py)
try:
    from .reio_like import (logL_limite_inferiore_con_errore,
                            logL_limite_superiore_con_errore,
                            dlogL_limite_inferiore_con_errore,
                            dlogL_limite_superiore_con_errore)
except ImportError:                      # eseguito come script / da Tests
    from Reiolike.reio_like import (logL_limite_inferiore_con_errore,
                                    logL_limite_superiore_con_errore,
                                    dlogL_limite_inferiore_con_errore,
                                    dlogL_limite_superiore_con_errore)

# ──────────────────────────────────────────────────────────────────────────────
# Low-level helpers
# ──────────────────────────────────────────────────────────────────────────────

def _sigmoid(f: np.ndarray) -> np.ndarray:
    return expit(f)

def _sigmoid_deriv(f: np.ndarray) -> np.ndarray:
    s = _sigmoid(f)
    return s * (1.0 - s)

def _logit(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    x = np.clip(x, eps, 1.0 - eps)
    return np.log(x / (1.0 - x))

def _rbf_kernel(z1: np.ndarray, z2: np.ndarray, s: float, r: float) -> np.ndarray:
    x1 = np.log10(1.0 + np.asarray(z1, dtype=float))
    x2 = np.log10(1.0 + np.asarray(z2, dtype=float))
    diff = x1[:, None] - x2[None, :]
    return s**2 * np.exp(-0.5 * diff**2 / r**2)

def _kernel_matrix(z: np.ndarray, s: float, r: float, nugget: float = 1e-6) -> np.ndarray:
    K = _rbf_kernel(z, z, s, r)
    K += nugget * np.eye(len(z))
    return K


# ──────────────────────────────────────────────────────────────────────────────
# Data loading from corecon
# ──────────────────────────────────────────────────────────────────────────────

def _valori_xHI(a, sigma, is_lim, bordi_come_bound):
    """x_HI = 1 - x_HII, con i bordi spostati di sigma se richiesto."""
    x = 1.0 - np.asarray(a["values"], dtype=float)
    if bordi_come_bound:
        al_massimo = (~is_lim) & (x >= 1.0 - 1e-12)
        al_minimo = (~is_lim) & (x <= 1e-12)
        x = np.where(al_massimo, 1.0 - sigma, x)
        x = np.where(al_minimo, sigma, x)
    return x


def load_xHI_data(corecon_config_file: str, verbose: bool = False,
                  z_min: float | None = None,
                  errore_dichiarato: bool = False,
                  bordi_come_bound: bool = True) -> list[dict]:
    """
    Dati astrofisici per il GP, letti ATTRAVERSO ReioLike.

    Perche' non c'e' piu' un lettore qui.  Questo file aveva una seconda
    implementazione del caricamento (_parse_spec, _extract_labeled,
    load_xHI_data), indipendente da quella di reio_like.py e rimasta al vecchio
    formato di corecon.  Sul branch HII_fraction_updates si rompeva subito
    ("Mason et al. 2019 [0.68]" -> TypeError: object of type 'float' has no
    len()) e, quando non si rompeva, sbagliava in silenzio:

      - non capiva [CL=0.68], quindi non poteva scegliere fra due righe-limite
        quotate a confidence level diversi (Totani 2006, Mason 2019, Bolan 2022
        ne hanno due ciascuno: le caricava entrambe, doppio conteggio);
      - non leggeva limit_confidence_level: indovinava il CL da labels[0] col
        fallback 0.95, e per i limiti senza err_up usava come sigma il valore
        stesso di x_HI (su Totani 2006: 0.17 invece di 0.3635);
      - non conosceva joint_constraint, quindi di Greig et al. 2022 caricava 5
        punti invece di 4, includendo il vincolo congiunto che E' gia' la
        ricombinazione degli altri quattro;
      - non controllava che la sezione YAML combaciasse con i flag di corecon.

    Mantenere due lettori per gli stessi file significa correggerne uno solo.
    Ora si delega a ReioLike, che quelle cose le fa tutte ed e' coperto da
    Tests/verifica_analisi.py.

    CONVERSIONE.  ReioLike lavora in x_HII, il GP in x_HI = 1 - x_HII.  Oltre
    ai valori si invertono i flag: un limite INFERIORE su x_HII e' un limite
    SUPERIORE su x_HI, e viceversa.

    SIGMA.  Due casi:
      - detection: le barre asimmetriche di ReioLike vengono simmetrizzate,
        0.5*(err_up + err_down).  Il GP vuole un sigma solo per punto;
      - limite:    si usa limit_sigma di ReioLike, cioe' d/Phi^-1(CL) con il CL
        LETTO dal dato.  La trasformazione x_HII -> x_HI ha pendenza -1, quindi
        la larghezza non cambia.

    ATTENZIONE, differenza di forma non risolta qui.  ReioLike appoggia la
    mezza gaussiana al bordo fisico (x_HII = 0 o 1), mentre la verosimiglianza
    del GP la appoggia al limite stesso.  Il sigma che si passa e' quello di
    ReioLike, ma la forma resta quella del GP: sono due convenzioni diverse e
    uniformarle e' una decisione di modello, non di lettura.

    Restituisce la stessa struttura di prima: una lista di dizionari con
    chiavi name, z, xHI, sigma, sigma_z, is_lower, is_upper.
    """
    import contextlib
    import io

    from .reio_like import ReioLike

    like = ReioLike.__new__(ReioLike)
    like.corecon_config_file = corecon_config_file
    like.print_sigma_table = False
    like.debug_points = False
    like.print_dataset_logl = False

    if verbose:
        like.initialize()
    else:
        with contextlib.redirect_stdout(io.StringIO()):
            like.initialize()

    data_list: list[dict] = []
    for a in like.analyses:
        n = len(a["z"])
        lower_xHII = np.asarray(a["lower_lim"], dtype=bool)
        upper_xHII = np.asarray(a["upper_lim"], dtype=bool)
        is_lim = lower_xHII | upper_xHII

        # sigma: dalle barre per le detection, da limit_sigma per i limiti.
        #
        # La media si fa solo sui lati EFFETTIVAMENTE misurati.  Un dato al
        # bordo fisico ha una barra sola — x_HI non puo' superare 1 ne' scendere
        # sotto 0 — e corecon mette 0 dall'altra parte.  Mediare quello zero
        # dimezzerebbe il sigma e farebbe pesare il punto QUATTRO volte troppo.
        #
        # Umeda et al. 2026 ne ha tre, in x_HI:
        #     z= 5.0  x_HI=0.00  barre 0.120 / 0        -> 0.120, non 0.060
        #     z= 8.6  x_HI=1.00  barre 0     / 0.200    -> 0.200, non 0.100
        #     z=10.4  x_HI=1.00  barre 0     / 0.400    -> 0.400, non 0.200
        #
        # E' anche cio' che rende coerenti le due descrizioni dello stesso dato:
        # una detection a x_HI=1 con la sola barra verso il basso E' un limite
        # inferiore, perche' la sigmoide non supera mai 1 e il lato alto non si
        # attiva.  Con la media su entrambi i lati le due letture darebbero
        # risultati diversi, il che non ha senso.
        err_up = np.asarray(a["errors"], dtype=float)
        err_dn = np.asarray(a.get("errors_down", a["errors"]), dtype=float)
        lati = np.vstack([err_up, err_dn])
        buoni = np.isfinite(lati) & (lati > 0)
        n_buoni = buoni.sum(axis=0)
        somma = np.where(buoni, lati, 0.0).sum(axis=0)
        sigma = np.divide(somma, n_buoni, out=np.full(n, np.nan),
                          where=n_buoni > 0)
        lim_sigma = np.asarray(a["limit_sigma"], dtype=float)
        sigma = np.where(is_lim, lim_sigma, sigma)

        # (1) Per i limiti che PUBBLICANO la loro incertezza, si usa quella.
        #
        # limit_sigma vale d/Phi^-1(CL), e quando corecon non dichiara il CL si
        # assume 0.95.  Quella formula pero' lega sigma al VALORE del limite:
        # se il limite e' vicino a zero, sigma va a zero con lui, e il vincolo
        # diventa artificialmente rigidissimo.  Davies et al. 2026 a z=5.481 da'
        # x_HI < 0.030 e dichiara un'incertezza di 0.048 su quel limite; la
        # formula produceva 0.018, cioe' 2.6 volte piu' stretto del pubblicato.
        #
        # Conseguenza misurata: quel soffitto entrava in conflitto con Umeda
        # 2026 (x_HI = 0.25 a z=5.8) e Totani 2014 (0.47), distanti appena
        # Delta log10(1+z) = 0.021, e il GP reagiva accorciando la lunghezza di
        # correlazione fino a r = 0.024 — esattamente quella distanza — per far
        # convivere i due.  La storia ricostruita usciva non monotona.
        # Usando l'errore dichiarato, r torna a 0.082 (il paper da' 0.076).
        #
        # SPENTO di default dal 14 ott 2026 (errore_dichiarato=False): per
        # riprodurre Kageura et al. i limiti prendono sigma dal CL, come nel
        # sampling, e GP e sampling usano le stesse sigma per tutti i punti.
        # Resta attivabile per confronto.
        #
        # Riguarda solo i limiti che hanno una barra.  Dal 19 ott 2026 sia
        # Zhu 2022 sia Davies 2026 sono limiti con errore (integrale, automatico):
        # li' la barra entra come larghezza dell'integrale sul limite, uguale
        # nei due livelli, e questo interruttore non serve piu'.
        if errore_dichiarato:
            barra = np.where(np.isfinite(err_up) & (err_up > 0), err_up, np.nan)
            giu = np.where(np.isfinite(err_dn) & (err_dn > 0), err_dn, np.nan)
            barra = np.where(np.isnan(barra), giu, barra)
            usa = is_lim & np.isfinite(barra)
            # NON per i limiti con errore: li' la barra entra gia' come
            # larghezza dell'integrale, uguale nel GP e nel sampling.
            origine = np.asarray(a.get("limit_sigma_origine", [""] * n), dtype=object)
            usa = usa & (origine != "limite con errore")
            sigma = np.where(usa, barra, sigma)

        # (2) Le detection appoggiate al bordo fisico diventano bound, spostati.
        #
        # Un dato con x_HI = 1 e barra solo verso il basso non e' una misura con
        # incertezza simmetrica: e' l'affermazione "x_HI e' almeno 1 - sigma".
        # Umeda et al. 2026 ne ha tre (z = 5.0, 8.6, 10.4), e nella Fig. 1 del
        # paper sono disegnati con una freccia, non con un punto.
        #
        # Si sposta il bound a 1 - sigma (o a 0 + sigma) invece di lasciarlo sul
        # bordo.  Lasciandolo sul bordo il risultato e' identico alla detection
        # — verificato: stessi s, r ed ELBO a quattro cifre, che e' la prova che
        # le due descrizioni sono coerenti — ma impone al GP un vincolo esatto
        # sul bordo, e questo crea massimi multipli: una partenza su cinque
        # finiva altrove.  Con il bound spostato le cinque partenze concordano.
        if bordi_come_bound:
            xHI = 1.0 - np.asarray(a["values"], dtype=float)
            al_massimo = (~is_lim) & (xHI >= 1.0 - 1e-12)
            al_minimo = (~is_lim) & (xHI <= 1e-12)
            lower_xHII = lower_xHII.copy()
            upper_xHII = upper_xHII.copy()
            # x_HI = 1 -> limite INFERIORE su x_HI, cioe' superiore su x_HII
            upper_xHII[al_massimo] = True
            # x_HI = 0 -> limite SUPERIORE su x_HI, cioe' inferiore su x_HII
            lower_xHII[al_minimo] = True

        # Larghezza del bin in redshift, simmetrizzata.  ReioLike la estrae gia'
        # da err_left/err_right di corecon, gestendo anche i dataset in cui
        # quelle colonne sono 2D (Durovcikova: [errore su z, errore
        # sull'etichetta], e si deve prendere la colonna 0).
        el = a.get("err_z_left")
        er = a.get("err_z_right")
        el = np.zeros(n) if el is None else np.asarray(el, dtype=float)
        er = np.zeros(n) if er is None else np.asarray(er, dtype=float)
        sigma_z = 0.5 * (np.nan_to_num(el) + np.nan_to_num(er))

        # Le DUE code, tenute separate.  'sigma' sopra e' la loro media, che
        # serve a chi vuole un numero solo, ma butta via l'asimmetria, e in
        # qualche punto l'asimmetria e' grossa: Umeda 2026 a z=7 ha
        # x_HII = 0.35 (+0.35, -0.27), cioe' x_HI = 0.65 (+0.27, -0.35).  Con
        # il sigma medio 0.31 la barra diventa [0.34, 0.96] e NON tocca piu'
        # quella di Umeda 2025 [LF] ([0.06, 0.32]); con le code vere la barra
        # e' [0.30, 0.92] e le due si sovrappongono, come nella Fig. 1 del
        # paper.  Chi disegna o costruisce la verosimiglianza usi queste.
        #
        # Attenzione all'inversione x_HI = 1 - x_HII: la coda ALTA su x_HII
        # diventa la coda BASSA su x_HI.
        _su = np.where(np.isfinite(err_dn) & (err_dn > 0), err_dn, np.nan)
        _giu = np.where(np.isfinite(err_up) & (err_up > 0), err_up, np.nan)
        _su = np.where(np.isnan(_su), _giu, _su)      # un lato solo: si specchia
        _giu = np.where(np.isnan(_giu), _su, _giu)
        # per i limiti le due code coincidono: la mezza gaussiana ha una
        # larghezza sola
        sigma_up = np.where(is_lim, sigma, _su)
        sigma_dn = np.where(is_lim, sigma, _giu)
        sigma_up = np.where(np.isfinite(sigma_up), sigma_up, sigma)
        sigma_dn = np.where(np.isfinite(sigma_dn), sigma_dn, sigma)

        # IL BERSAGLIO DELLA VEROSIMIGLIANZA PER I LIMITI E' IL BORDO FISICO.
        #
        # Un limite "x_HI < g al 99.7%" non significa "va bene qualunque valore
        # sotto g": significa che la posteriore e' una mezza gaussiana ANCORATA
        # A ZERO con sigma = g / Phi^-1(CL).  Il livello astrofisico lo fa gia'
        # cosi' (reio_like.py, compute_upper_limit_loglike:
        #     log_l = where(x_th > 0, -0.5 (x_th/sigma)^2, 0)
        # ancorata a 0, e la gemella ancorata a 1 per i limiti inferiori).
        #
        # Il livello GP invece ancorava al VALORE del limite: penalita' nulla
        # fino a g, poi quadratica.  Per Fausey et al., che quota x_HII > 0.43
        # al 99.7% (cioe' x_HI < 0.57), questo lasciava salire x_HI fino a 0.57
        # a costo zero, mentre il contenuto a 1 sigma di quel dato e'
        # x_HI < 0.207.  Lo stesso valeva per tutti gli altri 19 limiti.
        #
        # Si passa quindi alla verosimiglianza il BORDO (0 per un limite
        # superiore su x_HI, 1 per uno inferiore); 'xHI' resta il valore
        # pubblicato, che serve per disegnare e per i controlli.
        _ancora = _valori_xHI(a, sigma, is_lim, bordi_come_bound).copy()
        _ancora = np.where(upper_xHII, 1.0, _ancora)   # lim. INF su x_HI -> 1
        _ancora = np.where(lower_xHII, 0.0, _ancora)   # lim. SUP su x_HI -> 0

        # LIMITE CON ERRORE (automatico se il limite ha err_up/err_down; oggi
        # Zhu 2022 e Davies 2026): qui il
        # bersaglio NON e' il bordo ma il limite pubblicato, perche' la
        # likelihood e' Phi((c - x_HI)/e), con c il limite in x_HI ed e la sua
        # incertezza.  Vedi FSpaceLikelihood (flag 3 e 4).
        _orig = np.asarray(a.get("limit_sigma_origine", [""] * n), dtype=object)
        _con_errore = _orig == "limite con errore"
        _ancora = np.where(_con_errore, 1.0 - np.asarray(a["values"], dtype=float), _ancora)
        # e le sue DUE larghezze (split-normal sul limite vero, Zhu 2022 ha
        # +0.09/-0.10).  Stessa inversione delle detection: la larghezza
        # SOTTO in x_HII (err_down) e' quella SOPRA in x_HI.
        _inc_lo = np.asarray(a.get("limit_err_lo", np.full(n, np.nan)), dtype=float)
        _inc_hi = np.asarray(a.get("limit_err_hi", np.full(n, np.nan)), dtype=float)
        sigma_up = np.where(_con_errore, _inc_lo, sigma_up)
        sigma_dn = np.where(_con_errore, _inc_hi, sigma_dn)

        # Taglio in redshift, se richiesto.  Il paper ricostruisce la storia su
        # z = 5-15 (i 10 nodi di pag. 4 partono da z=5), quindi i punti sotto
        # z=5 stanno fuori dall'intervallo ricostruito: li' la reionizzazione
        # e' finita e il dato non vincola la transizione, ma entra lo stesso
        # nella verosimiglianza e tira la curva.
        _tieni = np.ones(n, dtype=bool) if z_min is None else \
            (np.asarray(a["z"], dtype=float) >= z_min)
        if not _tieni.any():
            if verbose:
                print(f"  {a['name']}: tutti i punti sotto z_min={z_min}, saltato.")
            continue
        if z_min is not None and not _tieni.all():
            print(f"  {a['name']}: {int((~_tieni).sum())} punti sotto z={z_min} "
                  f"scartati.")

        data_list.append({
            "name":     a["name"],
            "sigma_z":  sigma_z[_tieni],
            "xHI_ancora": _ancora[_tieni],
            "sigma_up": sigma_up[_tieni],
            "sigma_dn": sigma_dn[_tieni],
            "z":        np.asarray(a["z"], dtype=float)[_tieni],
            "xHI":      _valori_xHI(a, sigma, is_lim, bordi_come_bound)[_tieni],
            "sigma":    sigma[_tieni],
            # inversione dei flag: lower su x_HII  ->  upper su x_HI
            "is_lower": upper_xHII.copy()[_tieni],
            "is_upper": lower_xHII.copy()[_tieni],
            # True dove il limite va integrato sulla sua incertezza
            "limite_con_errore": _con_errore[_tieni],
        })

    return data_list


# ──────────────────────────────────────────────────────────────────────────────
# Main GP class
# ──────────────────────────────────────────────────────────────────────────────

class GPReio(Likelihood):
    # Cobaya-configurable inputs (can be overridden from YAML).
    corecon_config_file: str = "corecon_config.yaml"
    mean_z0: float = 7.67
    nugget: float = 1e-6
    add_prior_z15: bool = True
    prior_z15_sigma: float = 0.001
    # Taglio in redshift: il paper ricostruisce su z = 5-15 (vedi
    # load_xHI_data).  None = nessun taglio.
    z_min: float | None = 5.0

    # Iperparametri del kernel (Eq. A3).  NON si stimano qui: arrivano dallo
    # stadio 1, cioe' da gp_fit_hyperparams.py, che massimizza l'ELBO.
    # Si impostano da YAML, oppure si caricano con load_hyperparams().
    # I default sono i valori del paper (Kageura et al., arXiv:2601.09644).
    #
    # Prima erano assegnati dentro initialize() come attributi runtime, quindi
    # NON erano configurabili: qualunque valore si mettesse nello YAML veniva
    # sovrascritto da 0.83 e 0.076 a ogni avvio.
    s: float = 0.83
    r: float = 0.076

    # ──────────────────────────────────────────────────────────────
    # Initialization
    # ──────────────────────────────────────────────────────────────

    def initialize(self) -> None:
        # Attributi di sola esecuzione.  s e r NON si toccano qui: sono
        # attributi di classe, quindi configurabili da YAML (vedi sopra).
        self._z = None
        self._xHI = None
        self._sigma = None
        self._is_lower = None
        self._is_upper = None
        self._initialized = False

        datasets = load_xHI_data(self.corecon_config_file, z_min=self.z_min)
        z_list, xHI_list, sigma_list = [], [], []
        su_list, giu_list = [], []
        lower_list, upper_list, inc_list = [], [], []

        # Tre cose che questa classe deve fare ESATTAMENTE come
        # gp_fit_hyperparams.py, altrimenti gli iperparametri stimati la' non
        # valgono qui (erano tutte e tre sbagliate fino all'8 ott 2026):
        #   - bersaglio = xHI_ancora: per i limiti il bordo fisico 0 o 1, non
        #     il valore pubblicato (vedi load_xHI_data);
        #   - due code separate, sigma_up / sigma_dn (split-normal);
        #   - taglio z >= z_min.
        for ds in datasets:
            mask = np.isfinite(ds["xHI_ancora"]) & np.isfinite(ds["sigma"])
            z_list.append(ds["z"][mask])
            xHI_list.append(ds["xHI_ancora"][mask])
            sigma_list.append(ds["sigma"][mask])
            su_list.append(ds["sigma_up"][mask])
            giu_list.append(ds["sigma_dn"][mask])
            lower_list.append(ds["is_lower"][mask])
            upper_list.append(ds["is_upper"][mask])
            inc_list.append(ds.get("limite_con_errore", np.zeros(len(ds["z"]), bool))[mask])

        # Prior a z=15: x_HI > 0.999 (1 sigma), limite inferiore su x_HI.
        # Bersaglio 1.0, cioe' il bordo, come ogni altro limite inferiore.
        if self.add_prior_z15:
            z_list.append(np.array([15.0]))
            xHI_list.append(np.array([1.0]))
            sigma_list.append(np.array([self.prior_z15_sigma]))
            su_list.append(np.array([self.prior_z15_sigma]))
            giu_list.append(np.array([self.prior_z15_sigma]))
            lower_list.append(np.array([True]))
            upper_list.append(np.array([False]))
            inc_list.append(np.array([False]))


        # here we concatenate all the datasets, so that the GP will access to only one array
        self._z        = np.concatenate(z_list)
        self._xHI      = np.concatenate(xHI_list)
        self._sigma    = np.concatenate(sigma_list)
        self._sigma_su = np.maximum(np.concatenate(su_list), 1e-6)
        self._sigma_giu = np.maximum(np.concatenate(giu_list), 1e-6)
        self._is_lower = np.concatenate(lower_list)
        self._is_upper = np.concatenate(upper_list)
        self._con_errore  = np.concatenate(inc_list).astype(bool)

        # Sort by redshift for readability
        idx = np.argsort(self._z)
        self._z        = self._z[idx]
        self._xHI      = self._xHI[idx]
        self._sigma    = self._sigma[idx]
        self._sigma_su = self._sigma_su[idx]
        self._sigma_giu = self._sigma_giu[idx]
        self._is_lower = self._is_lower[idx]
        self._is_upper = self._is_upper[idx]
        self._con_errore  = self._con_errore[idx]

        print(
            f"GPReio: loaded {len(self._z)} data points "
            f"({np.sum(~self._is_lower & ~self._is_upper)} detections, "
            f"{np.sum(self._is_lower)} lower limits, "
            f"{np.sum(self._is_upper)} upper limits)."
        )
        self._initialized = True

    # RIMOSSI: _logsumexp e _get_mc_standard_normals.
    # Servivano solo a _log_marginal_likelihood_mc, che era chiamata ma non e'
    # mai stata scritta (latent_treatment="mc_marginal" dava AttributeError).
    # Era il tentativo di marginalizzare f con Monte Carlo dal prior: in 33
    # dimensioni non puo' funzionare, perche' le f estratte dal prior non
    # cadono mai dove i dati le vogliono e la media resta dominata da
    # pochissimi campioni.  La marginalizzazione la fa l'ELBO, nello stadio 1
    # (gp_fit_hyperparams.py).

    # ──────────────────────────────────────────────────────────────
    # GP prior components
    # ──────────────────────────────────────────────────────────────

    def _mean(self, z: np.ndarray) -> np.ndarray:
        """Mean function: μ(z) = z − z₀. This is a line as this is giving
           a simple monotonic function. This helps and guides the sampler"""
        return np.asarray(z, dtype=float) - self.mean_z0

    def _K(self, z: np.ndarray, s: float | None = None, r: float | None = None) -> np.ndarray:
        """Kernel matrix at training points."""
        s = self.s if s is None else s
        r = self.r if r is None else r
        return _kernel_matrix(z, s, r, self.nugget)

    def _log_gp_prior(
        self,
        f: np.ndarray,
        z: np.ndarray,
        s: float | None = None,
        r: float | None = None,
    ) -> float:
        s = self.s if s is None else s
        r = self.r if r is None else r
        mu = self._mean(z) #mean of the f function
        K  = _kernel_matrix(z, s, r, self.nugget) #building the kernel
        df = f - mu
        try:
            L      = np.linalg.cholesky(K)
            alpha  = np.linalg.solve(L.T, np.linalg.solve(L, df))
            log_det = 2.0 * np.sum(np.log(np.diag(L)))
            return -0.5 * (np.dot(df, alpha) + log_det)
        except np.linalg.LinAlgError:
            return -np.inf

    # ──────────────────────────────────────────────────────────────
    # Data log-likelihood in f-space
    # ──────────────────────────────────────────────────────────────

    def _log_likelihood(self, f: np.ndarray) -> float:
        """
        Log-likelihood of the data given f (Eq. A5, second term).

        For each data point:
            detection   : log N(x_HI(f) | obs, σ²) + log |dx_HI/df(f)|
            lower limit : half-Gaussian penalty if x_HI(f) < limit, else Jacobian only
            upper limit : half-Gaussian penalty if x_HI(f) > limit, else Jacobian only
        """
        return float(np.sum(self._log_likelihood_terms(f)[0]))

    def _active_mask(self, xHI: np.ndarray) -> np.ndarray:
        """
        Dove la penalita' e' accesa, punto per punto.

        detection   : sempre
        lower limit : solo se x_HI(f) < obs  (sta sotto il limite inferiore)
        upper limit : solo se x_HI(f) > obs  (sta sopra il limite superiore)

        Nei rami dei limiti la penalita' si spegne con continuita': in
        x_HI = obs vale zero da entrambe le parti, e anche la sua derivata
        tende a zero.  Serve perche' il gradiente analitico sia continuo.
        """
        active = np.ones_like(xHI, dtype=bool)
        active[self._is_lower] = xHI[self._is_lower] < self._xHI[self._is_lower]
        active[self._is_upper] = xHI[self._is_upper] > self._xHI[self._is_upper]
        return active

    def _log_likelihood_terms(self, f: np.ndarray):
        """
        (termini per punto, x_HI, maschera attiva).  Vettorizzato.

        Prima qui c'era un ciclo Python su tutti i punti, rifatto a ogni
        valutazione: con il gradiente per differenze finite sono N+1
        valutazioni per passo di ottimizzazione, quindi il ciclo veniva
        percorso decine di migliaia di volte per una sola log_prob_sr.

        IL JACOBIANO CI VA: e' nell'Eq. (A5) del paper.  p_i(x_HI) e' la
        DENSITA' di probabilita' di x_HI che viene da ogni analisi, non una
        verosimiglianza, quindi portarla in f richiede |dx/df|.  Vedi il
        commento esteso in Reiolike/gp_fit_hyperparams.py.
        """
        f = np.asarray(f, dtype=float)
        xHI = _sigmoid(f)
        log_jac = np.log(np.maximum(_sigmoid_deriv(f), 1e-300))
        active = self._active_mask(xHI)
        resid = (xHI - self._xHI) / self._sigma_eff(xHI)
        log_p = np.where(active, -0.5 * resid ** 2, 0.0)
        # limite con errore: L = P(x_HI dal lato giusto del limite), integrale
        # sulla split-normal del limite (funzioni di reio_like.py; vedi
        # FSpaceLikelihood in gp_fit_hyperparams.py, flag 3 e 4)
        if self._con_errore.any():
            log_p = np.where(self._con_errore, self._log_con_errore(xHI), log_p)
        return log_p + log_jac, xHI, active

    def _log_con_errore(self, xHI):
        """log L delle righe 'limite con errore' (in x_HI: sotto = sigma_giu,
        sopra = sigma_su).  limite SUP su x_HI: x <= C;  INF: x >= C."""
        a = (xHI, self._xHI, self._sigma_giu, self._sigma_su)
        return np.where(self._is_upper, logL_limite_superiore_con_errore(*a),
                        logL_limite_inferiore_con_errore(*a))

    def _d_log_con_errore(self, xHI):
        """d/dx_HI di _log_con_errore."""
        a = (xHI, self._xHI, self._sigma_giu, self._sigma_su)
        return np.where(self._is_upper, dlogL_limite_superiore_con_errore(*a)[0],
                        dlogL_limite_inferiore_con_errore(*a)[0])

    def _sigma_eff(self, xHI):
        """Split-normal: coda alta se il modello sta sopra il bersaglio."""
        return np.where(xHI > self._xHI, self._sigma_su, self._sigma_giu)

    def _log_posterior_and_grad(self, f: np.ndarray, s=None, r=None):
        """
        (log-posterior, d log-posterior / df).  Il gradiente e' analitico.

        Derivazione, con x = sigma(f) e x' = sigma(f)(1-sigma(f)):

          termine GP      -0.5 (f-mu)^T K^-1 (f-mu)   ->  -K^-1 (f-mu)
          jacobiano       log x'                      ->  1 - 2 sigma(f)
          dato            -0.5 ((x-obs)/sig)^2        ->  -((x-obs)/sig^2) x'
                          (solo dove la penalita' e' accesa)

        Il termine -0.5 log|K| non dipende da f e non entra nel gradiente.

        PERCHE' SERVE.  _find_map chiamava scipy.minimize senza jac, quindi
        scipy stimava il gradiente per differenze finite: N+1 = 34 valutazioni
        per passo.  Con maxiter=2000 sono 68 000 valutazioni, misurate a
        0.895 ms l'una: ~79 s per UNA chiamata di log_prob_sr, e il budget si
        esauriva prima della convergenza ("TOTAL NO. OF F,G EVALUATIONS
        EXCEEDS LIMIT").
        """
        s = self.s if s is None else s
        r = self.r if r is None else r
        f = np.asarray(f, dtype=float)

        mu = self._mean(self._z)
        K = _kernel_matrix(self._z, s, r, self.nugget)
        df = f - mu
        try:
            L = np.linalg.cholesky(K)
            alpha = np.linalg.solve(L.T, np.linalg.solve(L, df))
            log_det = 2.0 * np.sum(np.log(np.diag(L)))
        except np.linalg.LinAlgError:
            return -np.inf, np.zeros_like(f)

        log_prior = -0.5 * (np.dot(df, alpha) + log_det)

        terms, xHI, active = self._log_likelihood_terms(f)
        log_lk = float(np.sum(terms))

        dx = _sigmoid_deriv(f)                      # dx_HI/df
        g_data = np.where(active,
                          -(xHI - self._xHI) / self._sigma_eff(xHI) ** 2 * dx, 0.0)
        if self._con_errore.any():
            g_data = np.where(self._con_errore, self._d_log_con_errore(xHI) * dx, g_data)
        g_jac = 1.0 - 2.0 * xHI
        grad = -alpha + g_jac + g_data

        return log_prior + log_lk, grad

    # ──────────────────────────────────────────────────────────────
    # Log-posterior
    # ──────────────────────────────────────────────────────────────

    def _log_posterior(
        self,
        f: np.ndarray,
        s: float | None = None,
        r: float | None = None,
    ) -> float:
        """Full log-posterior: GP prior + data log-likelihood. """
        return self._log_gp_prior(f, self._z, s, r) + self._log_likelihood(f)

    # ──────────────────────────────────────────────────────────────
    # MAP estimation
    # ──────────────────────────────────────────────────────────────

    # RIMOSSO: _find_map.
    #
    # Cercava il massimo su f (il MAP) e usava quel valore come punteggio della
    # coppia (s, r): e' la "profile likelihood", cioe' sostituire l'integrale
    # di Eq. A4 col suo massimo.  Due problemi, entrambi misurati:
    #
    #  1. METODO.  Il massimo misura l'altezza del picco, l'integrale il suo
    #     volume.  Profilare butta via il fattore di Occam, che su questi dati
    #     vale fino a 8.5 unita' di log-verosimiglianza fra un angolo e l'altro
    #     dello spazio (s, r).  L'integrale lo fa l'ELBO, nello stadio 1.
    #
    #  2. NON FUNZIONAVA.  L'ottimizzazione non convergeva mai: partiva da
    #     f0 = logit(x_HI), che e' una funzione RUVIDA, mentre il kernel con
    #     r ~ 0.076 descrive funzioni lisce e penalizza la ruvidita' in modo
    #     enorme (K ha numero di condizionamento 1.5e7).  Misurato su 33 punti:
    #         log_posterior(logit(x_HI)) = -6.06e+07
    #         log_posterior(mu(z))       =    +84.81
    #     L'ottimizzatore doveva coprire 60 milioni di unita' e non ci
    #     arrivava: riportava 68.49, cioe' PEGGIO del valore che si ottiene
    #     partendo da mu senza nemmeno ottimizzare.  Siccome quel numero doveva
    #     essere un massimo, era semplicemente sbagliato.
    #
    # Quello che resta — _log_posterior_and_grad, con il suo gradiente
    # analitico — e' Eq. A5 e serve ancora: e' esattamente la funzione da dare
    # a emcee per campionare f a (s, r) fissati, se un giorno si vuole
    # verificare l'approssimazione variazionale contro un MCMC vero.

    def log_posterior_f(self, f: np.ndarray,
                        s: float | None = None,
                        r: float | None = None) -> float:
        """
        log p(f | dati, s, r) a meno di una costante — Eq. A5 del paper.

        E' la funzione da campionare con emcee per ottenere la storia di
        reionizzazione a (s, r) fissati.  Il gradiente analitico, se serve un
        campionatore che lo usa, viene da _log_posterior_and_grad.

        Sostituisce log_prob_sr(), che faceva un'altra cosa: cercava di dare un
        punteggio a (s, r) profilando f.  Vedi il commento al posto di
        _find_map per il perche' sia stata tolta.
        """
        self._check_initialized()
        return self._log_posterior(f, s=s, r=r)

    def logp(self, **params_values) -> float:
        """
        Punto d'ingresso di cobaya.  Qui non c'e' niente da campionare.

        Questo modulo e' lo STADIO 2 della pipeline: usa (s, r) gia' trovati,
        non li stima.  Lo stadio 1 e' gp_fit_hyperparams.py, che massimizza
        l'ELBO (Eq. A4).

        Prima logp() valutava la profile likelihood su (s, r) campionati,
        duplicando male il lavoro dello stadio 1.  Finche' non e' deciso cosa
        questo modulo debba fornire a una catena, e' meglio un errore chiaro di
        un numero che sembra buono e non lo e'.
        """
        raise NotImplementedError(
            "GPReio non e' una likelihood campionabile: non stima (s, r). "
            "Gli iperparametri si trovano con Reiolike/gp_fit_hyperparams.py "
            "(ELBO, Eq. A4) e si passano qui da YAML o con load_hyperparams(). "
            "Per la posteriore della storia a (s, r) fissati usa "
            "log_posterior_f()."
        )

    # ──────────────────────────────────────────────────────────────
    # Save / load optimised hyperparameters (interface to Stage 2)
    # ──────────────────────────────────────────────────────────────

    def save_hyperparams(self, path: str) -> None:
        """
        Save the optimised hyperparameters s and r to a .npz file.

        Stage 2 of the pipeline loads these with ``load_hyperparams()``
        to reconstruct the reionization history without re-running the fit.
        """
        np.savez(path, s=np.array(self.s), r=np.array(self.r))
        print(f"GPReio: hyperparameters saved to '{path}' (s={self.s:.4f}, r={self.r:.4f}).")

    @staticmethod
    def load_hyperparams(path: str) -> tuple[float, float]:
        """
        Load hyperparameters saved by ``save_hyperparams``.

        Returns
        -------
        (s, r) : floats
        """
        data = np.load(path)
        return float(data["s"]), float(data["r"])

    # ──────────────────────────────────────────────────────────────
    # Utilities
    # ──────────────────────────────────────────────────────────────

    def _check_initialized(self) -> None:
        if not self._initialized:
            raise RuntimeError(
                "GPReio is not initialised.  Call initialize() first."
            )

    def summary(self) -> str:
        """Return a short summary of the GP configuration for the s,r stage."""
        lines = [
            "GPReio summary",
            f"  config file : {self.corecon_config_file}",
            f"  mean_z0     : {self.mean_z0}",
            f"  nugget      : {self.nugget}",
            f"  s           : {self.s:.4f}   (dallo stadio 1, non stimato qui)",
            f"  r           : {self.r:.4f}   (idem)",
        ]
        if self._initialized:
            n = len(self._z)
            n_det = int(np.sum(~self._is_lower & ~self._is_upper))
            n_lo  = int(np.sum(self._is_lower))
            n_up  = int(np.sum(self._is_upper))
            lines.append(
                f"  data points : {n} total "
                f"({n_det} detections, {n_lo} lower limits, {n_up} upper limits)"
            )
        lines.append("  stadio      : 2 — usa (s, r) dello stadio 1, non li stima")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"GPReio(config='{self.corecon_config_file}', "
            f"s={self.s:.3f}, r={self.r:.3f}, "
            f"initialized={self._initialized})"
        )
