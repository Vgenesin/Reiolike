"""
Frazione di elio primordiale Y_He dalla nucleosintesi (BBN), letta dalla
STESSA tabella e con lo STESSO algoritmo che usa CLASS.

Perche' serve.  La storia di reionizzazione che passiamo a CLASS contiene gli
elettroni dell'elio: il plateau vale 1 + f_He (idrogeno + prima ionizzazione
dell'elio) e, a basso z, 1 + 2 f_He.  f_He = n_He/n_H dipende da Y_He, che a
sua volta dipende da omega_b, un parametro campionato.  CLASS ricava Y_He da
omega_b in thermodynamics_helium_from_bbn(), ma lo fa DOPO che noi abbiamo
gia' costruito la tabella x_e(z): non possiamo chiederglielo.  Quindi lo
calcoliamo noi dalla stessa tabella, e poi lo passiamo a CLASS (parametro
YHe) perche' i due usino lo stesso numero per costruzione.

L'algoritmo di CLASS (packages/code/classy/source/thermodynamics.c,
thermodynamics_helium_from_bbn), replicato qui passo per passo:
  1. il file contiene una griglia (omega_b, DeltaN) -> Y_He, con omega_b che
     varia piu' in fretta;
  2. per ogni omega_b della griglia si fa una spline cubica NATURALE lungo
     DeltaN e la si valuta al DeltaN richiesto;
  3. sui valori ottenuti si fa una spline cubica naturale lungo omega_b e la
     si valuta all'omega_b richiesto;
  4. fuori dalla griglia CLASS si ferma con un errore: lo stesso qui.

DeltaN e' definito come in CLASS:  DeltaN = N_eff(BBN) - 3.046.  L'intestazione
di sBBN_2025.dat parla di 3.044, ma il codice di CLASS sottrae 3.046 (con un
commento che lo giustifica): copiamo il codice, non l'intestazione, perche' lo
scopo e' coincidere con CLASS.
"""
import os

import numpy as np
from scipy.interpolate import CubicSpline

# T_nu / T_gamma nel limite istantaneo, (4/11)^(1/3)
_T_NU_STD = (4.0 / 11.0) ** (1.0 / 3.0)
# Default di CLASS (input.c): N_ur = 3.044 se non dato; T_ncdm = 0.71611
_N_UR_DEFAULT = 3.044 #Numero di neutrini leggeri
_T_NCDM_DEFAULT = 0.71611 #Numero di neutrini massivi
_NEFF_RIF_CLASS = 3.046     # quello che CLASS sottrae per ottenere DeltaN


def file_bbn_di_class(nome="sBBN_2025.dat"):
    """Percorso della tabella BBN della copia di CLASS installata.

    sBBN_2025.dat e' il default di CLASS (include/precisions.h, sBBN_file).
    """
    try:
        import classy
        base = os.path.dirname(os.path.abspath(classy.__file__))
        cand = [os.path.join(base, "external", "bbn", nome)]
    except ImportError:
        cand = []
    radice = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cand.append(os.path.join(radice, "packages", "code", "classy",
                             "external", "bbn", nome))
    for c in cand:
        if os.path.exists(c):
            return c
    raise FileNotFoundError(
        f"Tabella BBN '{nome}' non trovata. Cercata in:\n  " + "\n  ".join(cand))


def _lista(v, n, default):
    """Legge un parametro di CLASS che puo' essere numero, lista o stringa
    '0.06, 0.06' e lo restituisce come lista di n float."""
    if v is None:
        return [default] * n
    if isinstance(v, str):
        v = [float(x) for x in v.replace(",", " ").split()]
    v = np.atleast_1d(np.asarray(v, dtype=float)).tolist()
    if len(v) == 1 and n > 1:
        v = v * n
    return v


def delta_neff_bbn(params):
    """DeltaN alla BBN come lo calcola CLASS, dai parametri dei neutrini.
    CLASS fa  N_eff(BBN) = (rho_r - rho_g) / (7/8 (4/11)^(4/3) rho_g)  a
    T = 0.1 MeV.  A quella temperatura anche i neutrini massivi sono
    ultrarelativistici (m/T ~ 1e-6), quindi ciascuna specie ncdm conta
    deg * (T_ncdm / (4/11)^(1/3))^4 e il conto si chiude analiticamente.
    params : dict con i parametri passati a CLASS (N_ur, N_ncdm, deg_ncdm,
             T_ncdm).  Quelli assenti prendono il default di CLASS.
    """
    if any(k in params for k in ("Omega_ur", "omega_ur")):
        raise NotImplementedError(
            "delta_neff_bbn: Omega_ur/omega_ur non gestiti; usare N_ur oppure "
            "fissare reio_bbn_delta_neff a mano.")
    n_ur = float(params.get("N_ur", _N_UR_DEFAULT))
    n_ncdm = int(params.get("N_ncdm", 0) or 0)
    neff = n_ur
    if n_ncdm > 0:
        deg = _lista(params.get("deg_ncdm"), n_ncdm, 1.0)
        tnc = _lista(params.get("T_ncdm"), n_ncdm, _T_NCDM_DEFAULT)
        neff += sum(d * (t / _T_NU_STD) ** 4 for d, t in zip(deg, tnc))
    return neff - _NEFF_RIF_CLASS


class TabellaBBN:
    """Y_He(omega_b, DeltaN) letta da un file BBN nel formato di CLASS."""

    def __init__(self, percorso=None):
        self.percorso = percorso or file_bbn_di_class()
        righe = []
        dims = None
        with open(self.percorso) as fh:
            for riga in fh:
                s = riga.strip()
                # CLASS tiene le righe il cui primo carattere ha codice > 39
                # (esclude vuote, '#', '%', ...).  Equivalente qui:
                if not s or ord(s[0]) <= 39:
                    continue
                if dims is None:
                    dims = tuple(int(x) for x in s.split()[:2])
                    continue
                righe.append([float(x) for x in s.split()[:3]])
        n_ob, n_dn = dims
        a = np.asarray(righe)
        if a.shape != (n_ob * n_dn, 3):
            raise ValueError(f"{self.percorso}: attese {n_ob * n_dn} righe di dati, "
                             f"trovate {len(a)}")
        # ordine del file: omega_b varia piu' in fretta  (riga = ob + n_ob*dn)
        #
        # ATTENZIONE, dettaglio che conta.  La colonna di omega_b NON e'
        # identica nei 13 blocchi di DeltaN: in sBBN_2025.dat differisce fino a
        # 2e-5 da un blocco all'altro.  CLASS tratta comunque il file come una
        # griglia regolare: legge omegab[riga % n_ob] riga per riga, quindi
        # ogni blocco sovrascrive il precedente e alla fine restano i valori
        # dell'ULTIMO blocco.  Prendendo quelli del primo, Y_He differiva da
        # quello di CLASS di 3e-6 - 8e-6; con quelli dell'ultimo coincide.
        self.omegab = a[-n_ob:, 0].copy()
        self.deltaN = a[::n_ob, 1].copy()
        self.YHe = a[:, 2].reshape(n_dn, n_ob)          # [dn, ob]
        self._dn_corrente = None
        self._spline_ob = None

    def _prepara(self, deltaN):
        """Passo 2 di CLASS: spline lungo DeltaN per ogni omega_b, valutata a
        deltaN.  Il risultato dipende solo da deltaN, quindi si tiene in
        memoria: in una catena DeltaN e' fisso e il passo si fa una volta."""
        if self._dn_corrente == deltaN:
            return #skip the interpolation on this axis
        if not (self.deltaN[0] <= deltaN <= self.deltaN[-1]):
            raise ValueError(
                f"DeltaN = {deltaN:.6g} fuori dalla tabella BBN "
                f"[{self.deltaN[0]:g}, {self.deltaN[-1]:g}] (come in CLASS).")
        y_dn = CubicSpline(self.deltaN, self.YHe, axis=0, bc_type="natural")(deltaN)
        self._spline_ob = CubicSpline(self.omegab, y_dn, bc_type="natural")
        self._dn_corrente = deltaN

    def __call__(self, omega_b, deltaN=0.0):
        """Y_He per un dato omega_b = Omega_b h^2 e DeltaN = N_eff(BBN) - 3.046."""
        omega_b = float(omega_b)
        if not (self.omegab[0] <= omega_b <= self.omegab[-1]):
            raise ValueError(
                f"omega_b = {omega_b:.6g} fuori dalla tabella BBN "
                f"[{self.omegab[0]:g}, {self.omegab[-1]:g}] (come in CLASS).")
        self._prepara(float(deltaN))
        return float(self._spline_ob(omega_b)) #this happens at every MCMC stepvo
