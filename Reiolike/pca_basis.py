"""
Lettura e uso di una base di autofunzioni PCA per xe(z).

Questo modulo e' il ponte fra lo script che GENERA le autofunzioni
(PCA_eigenfunction_reconstruction.py) e il modello di reionizzazione che le
USA (ReioTheory, reio_model='pca').

Il principio e': **il file delle autofunzioni si descrive da solo**.  Chi usa
il pacchetto non deve ripetere nel YAML z_min, z_max, delta_z, xe fiduciale,
numero di bin, frazione di elio: tutto questo e' gia' scritto nel file insieme
alle autofunzioni, e ricopiarlo a mano e' solo un modo per sbagliare.  Il YAML
dice soltanto *quale* file usare e *quanti* modi campionare.

    theory:
      Reiolike.reio_theory.ReioTheory:
        reio_model: pca
        reio_pca_file: eigenfunctions_PCA.npz
        reio_pca_n_modes: 5

Formati letti
-------------
`.npz`  — quello scritto da np.savez nello script generatore (preferito).
`.dat`/`.txt` — formato testo auto-descrittivo: un header di righe
        `# chiave = valore`, poi le colonne  z  S1 S2 ... SN.
        Serve a chi vuole portarsi la base fuori da numpy o riceverla da
        un'altra pipeline.

Riferimenti
-----------
Mortonson & Hu 2008 (arXiv:0705.1132) — costruzione della base, Eq. (5)-(6);
    priori di fisicita', Sect. 3, Eq. (10) e (13).
Ilic et al. 2025 (arXiv:2504.13254) Sect. 2.6 — il modello 'pca-5'.
"""

import os

import numpy as np

from .reio_models import XE_MAX_HI_HEI, f_he, _Y_HE

# Chiavi che una base deve avere per essere utilizzabile.  Le altre
# (tau_fiducial, l_max, cosmologia, smussamento) sono documentazione: utili da
# stampare e da confrontare, non necessarie al calcolo.
_REQUIRED = ("z_centers", "eigenvectors", "eigenvalues",
             "z_min", "z_max", "delta_z", "xe_fiducial")

# Cache: leggere e validare un file a ogni passo dell'MCMC sarebbe assurdo.
# La chiave include mtime e dimensione cosi' rigenerare la base durante una
# sessione interattiva non lascia in memoria quella vecchia.
_CACHE = {}

# Numero minimo di autofunzioni che una base deve contenere, e minimo di modi
# che ha senso campionare.
#
# Mortonson & Hu 2008 (arXiv:0705.1132) mostrano che i primi 5 modi esauriscono
# l'informazione che C_l^EE porta su xe(z) fino al limite della varianza
# cosmica: i modi successivi sono oscillazioni cosi' rapide in redshift da
# essere mediate via nello spettro, e la loro sigma supera l'intervallo
# fisicamente ammesso (cioe' li decide il prior, non il dato).  Ilic et al.
# 2025 adottano la stessa scelta e chiamano il modello 'pca-5':
#
#   "We keep only the first 5 components which according to Mortonson & Hu
#    (2008) are sufficient to describe all the information on xe carried by
#    C_l^EE up to the cosmic variance limit."
#
# Averne di PIU' nel file non e' un problema (se ne usano i primi
# reio_pca_n_modes); averne di MENO significa che la base non puo' descrivere
# la reionizzazione, e va rigenerata.
MIN_MODES = 5


def _as_float(x):
    return float(np.asarray(x).reshape(()))


class PCABasis:
    """
    Una base di autofunzioni PCA, con i suoi metadati e le operazioni che
    servono per usarla come modello di reionizzazione.

    Attributi principali
    --------------------
    z_centers   : (Nz,)      centri dei bin in redshift
    S           : (Nz, Nz)   autofunzioni in colonna, normalizzate MH08
                             (sum_i S_a(z_i)^2 = Nz+1)
    eigenvalues : (Nz,)      autovalori della Fisher, in ordine decrescente
    z_min, z_max, delta_z, xe_fiducial, delta_xe, tau_fiducial, l_max
    YHe, f_He   : bookkeeping dell'elio con cui la base e' stata costruita
    cosmo       : dict o None — cosmologia fiduciale del generatore
    """

    # ------------------------------------------------------------------ init
    def __init__(self, data, path=None):
        self.path = path
        missing = [k for k in _REQUIRED if k not in data]
        if missing:
            raise ValueError(
                f"PCABasis: al file '{path}' mancano le chiavi {missing}. "
                "Rigeneralo con PCA_eigenfunction_reconstruction.py: le "
                "autofunzioni da sole non bastano, servono anche i metadati "
                "(z_min, z_max, delta_z, xe fiduciale) che le definiscono."
            )

        self.z_centers = np.asarray(data["z_centers"], dtype=float).ravel()
        self.S = np.asarray(data["eigenvectors"], dtype=float)
        self.eigenvalues = np.asarray(data["eigenvalues"], dtype=float).ravel()

        self.z_min = _as_float(data["z_min"])
        self.z_max = _as_float(data["z_max"])
        self.delta_z = _as_float(data["delta_z"])
        self.xe_fiducial = _as_float(data["xe_fiducial"])

        self.delta_xe = _as_float(data["delta_xe"]) if "delta_xe" in data else np.nan
        self.tau_fiducial = (_as_float(data["tau_fiducial"])
                             if "tau_fiducial" in data else np.nan)
        self.l_max = int(_as_float(data["l_max"])) if "l_max" in data else -1
        self.YHe = _as_float(data["YHe"]) if "YHe" in data else _Y_HE
        self.f_He = _as_float(data["f_He"]) if "f_He" in data else f_he(self.YHe)

        self.smooth_median_bins = (int(_as_float(data["smooth_median_bins"]))
                                   if "smooth_median_bins" in data else -1)
        self.smooth_sigma_z = (_as_float(data["smooth_sigma_z"])
                               if "smooth_sigma_z" in data else np.nan)

        # Cosmologia fiduciale: serve al peso di tau nel prior di fisicita'.
        # Se manca si ricade sulla dominazione di materia, che nell'intervallo
        # z = 5-30 sbaglia H(z) di meno dello 0.3% — irrilevante per un PRIOR,
        # ma va detto invece che assunto in silenzio.
        self.cosmo = None
        if "cosmo_keys" in data and "cosmo_vals" in data:
            keys = [str(k) for k in np.asarray(data["cosmo_keys"]).ravel()]
            vals = [float(v) for v in np.asarray(data["cosmo_vals"]).ravel()]
            self.cosmo = dict(zip(keys, vals))

        self._validate()
        self._tau_bin_weights = None   # calcolato pigramente

    # -------------------------------------------------------------- validate
    def _validate(self):
        nz = self.z_centers.size
        if self.S.ndim != 2 or self.S.shape[0] != nz:
            raise ValueError(
                f"PCABasis: eigenvectors ha forma {self.S.shape} ma z_centers "
                f"ha {nz} punti. Le autofunzioni vanno in COLONNA: "
                "eigenvectors[:, a] e' il modo a."
            )
        if self.eigenvalues.size != self.S.shape[1]:
            raise ValueError(
                f"PCABasis: {self.eigenvalues.size} autovalori per "
                f"{self.S.shape[1]} autofunzioni."
            )
        if np.any(np.diff(self.z_centers) <= 0):
            raise ValueError("PCABasis: z_centers non e' strettamente crescente.")
        if not (self.z_max > self.z_min >= 0):
            raise ValueError(
                f"PCABasis: intervallo non valido z_min={self.z_min}, "
                f"z_max={self.z_max}."
            )
        if not np.isfinite(self.S).all():
            raise ValueError("PCABasis: le autofunzioni contengono NaN o inf.")

        # Convenzione MH08:  z_i = z_min + i*dz  per i = 1..Nz,  con
        # z_max = z_min + (Nz+1)*dz.  Se il file non la rispetta, la
        # normalizzazione sum S^2 = Nz+1 e i limiti di Eq. (10) non valgono.
        expected = np.arange(1, nz + 1) * self.delta_z + self.z_min
        if not np.allclose(self.z_centers, expected, atol=1e-6):
            raise ValueError(
                "PCABasis: i centri dei bin non seguono la convenzione MH08 "
                f"z_i = z_min + i*delta_z. Attesi {expected[0]:g}..{expected[-1]:g}, "
                f"trovati {self.z_centers[0]:g}..{self.z_centers[-1]:g}. "
                "La normalizzazione delle autofunzioni dipende da questa "
                "convenzione, quindi il file non e' utilizzabile cosi' com'e'."
            )

        if self.S.shape[1] < MIN_MODES:
            raise ValueError(
                f"PCABasis: the basis in '{self.path}' contains only "
                f"{self.S.shape[1]} eigenfunction(s), but at least {MIN_MODES} "
                "are required. Mortonson & Hu (2008) show that the first five "
                "principal components exhaust the information that the E-mode "
                "polarisation spectrum carries about the reionisation history "
                "up to the cosmic-variance limit; with fewer modes the basis "
                "cannot describe reionisation. Ilic et al. (2025) adopt the "
                "same choice and name the model 'pca-5'. Regenerate the basis "
                "with PCA_eigenfunction_reconstruction.py using a redshift "
                "binning that yields at least 5 bins."
            )

        norm = np.sum(self.S ** 2, axis=0)
        if not np.allclose(norm, nz + 1, rtol=0.02):
            raise ValueError(
                "PCABasis: le autofunzioni non sono normalizzate secondo MH08 "
                f"(sum_i S_a(z_i)^2 = Nz+1 = {nz+1}); trovato "
                f"{norm.min():.3f}..{norm.max():.3f}. Senza questa "
                "normalizzazione i limiti di fisicita' Eq. (10) e (13) sono "
                "sbagliati e i coefficienti m_a non hanno la scala giusta."
            )

    # ------------------------------------------------------------- proprieta'
    @property
    def n_bins(self):
        return self.z_centers.size

    @property
    def n_modes_available(self):
        return self.S.shape[1]

    @property
    def xe_max(self):
        """Massimo fisico di xe nel settore descritto dalla base (H + HeI)."""
        return 1.0 + self.f_He

    def sigma(self, a):
        """
        sigma(m_a) per un esperimento limitato dalla varianza cosmica.

        Il fattore (Nz+1) viene dalla normalizzazione MH08: senza, sigma
        dipende dal binning (differenze del 40% fra delta_z 0.25 e 0.5).
        """
        lam = self.eigenvalues[a]
        return 1.0 / np.sqrt((self.n_bins + 1) * lam) if lam > 0 else np.inf

    # ------------------------------------------------- priori di fisicita'
    def mode_bounds(self, n_modes=None, xe_lo=0.0, xe_hi=None):
        """
        Limiti per singolo modo, MH08 Eq. (10).

        Ogni storia con 0 <= xe(z) <= xe_hi ha coefficienti dentro questi
        limiti; il viceversa NON vale (MH08 lo dice esplicitamente: "even if
        all m_mu satisfy equation (10), xe(z) may still be unphysical for some
        z").  Sono quindi la scatola piu' STRETTA che contiene tutto lo spazio
        fisico: la scelta naturale per i prior uniformi del YAML, che Ilic
        et al. chiamano "very large uniform priors".

        Derivazione: m_a = (1/(Nz+1)) sum_i S_a(z_i) [xe(z_i) - xe_fid].
        Per massimizzare m_a si mette xe al suo massimo dove S_a > 0 e al
        minimo dove S_a < 0.  Con c = (xe_hi+xe_lo)/2 e d = (xe_hi-xe_lo)/2:

            m_a^(+-) = (1/(Nz+1)) sum_i [ S_a(z_i)*(c - xe_fid) +- |S_a(z_i)|*d ]

        Con xe_lo=0, xe_hi=1 si riduce esattamente a MH08 Eq. (10).  Qui il
        default e' xe_hi = 1+f_He, cioe' la convenzione di Ilic et al. 2025,
        che include la prima ionizzazione dell'elio.

        Returns
        -------
        (m_minus, m_plus) : due array di lunghezza n_modes
        """
        if xe_hi is None:
            xe_hi = self.xe_max
        n = self._n(n_modes)
        S = self.S[:, :n]
        c, d = 0.5 * (xe_hi + xe_lo), 0.5 * (xe_hi - xe_lo)
        centre = np.sum(S * (c - self.xe_fiducial), axis=0) / (self.n_bins + 1)
        half = np.sum(np.abs(S) * d, axis=0) / (self.n_bins + 1)
        return centre - half, centre + half

    def sum_sq_bound(self, xe_lo=0.0, xe_hi=None):
        """
        Limite sulla somma dei quadrati, MH08 Eq. (13):  sum_a m_a^2 <= f,
        con  f = max[(xe_fid - xe_lo)^2, (xe_hi - xe_fid)^2].

        Viene dall'ortogonalita': integrare [xe - xe_fid]^2 sull'intervallo da'
        (z_max - z_min) * sum_a m_a^2, e quell'integrale non puo' superare
        (z_max - z_min) * f se xe resta nel suo intervallo fisico.

        E' un vincolo GLOBALE sui modi presi insieme, quindi taglia via una
        parte dello spazio che i limiti per singolo modo lascerebbero passare.
        Anche questo e' necessario ma non sufficiente.
        """
        if xe_hi is None:
            xe_hi = self.xe_max
        return max((self.xe_fiducial - xe_lo) ** 2, (xe_hi - self.xe_fiducial) ** 2)

    def tau_bin_weights(self):
        """
        Peso di ogni bin nella profondita' ottica, normalizzato a somma 1.

            tau ~ integral xe(z) (1+z)^2 / E(z) dz,      E = H/H0

        Restituisce w_i tale che  tau_parziale / tau(xe = xe_max) = sum_i w_i
        xe_i / xe_max  sull'intero intervallo della base.

        Usa la cosmologia fiduciale salvata nel file, se c'e'.  Il prior deve
        essere FISSO, non dipendere dai parametri campionati: altrimenti il
        volume del prior cambia da punto a punto e la posterior ne risente.
        """
        if self._tau_bin_weights is not None:
            return self._tau_bin_weights

        if self.cosmo:
            h = self.cosmo.get("h", 0.6731)
            om = (self.cosmo.get("omega_b", 0.02222)
                  + self.cosmo.get("omega_cdm", 0.1197)) / h ** 2
        else:
            om = 0.315
        ol = 1.0 - om

        def integrand(z):
            E = np.sqrt(om * (1.0 + z) ** 3 + ol)
            return (1.0 + z) ** 2 / E

        # Simpson a 5 punti su ogni bin: l'integranda e' liscia, l'errore e'
        # ~1e-8 relativo, e comunque questo e' un prior.
        half = 0.5 * self.delta_z
        offs = np.linspace(-half, half, 5)
        simp = np.array([1.0, 4.0, 2.0, 4.0, 1.0]) / 12.0 * self.delta_z
        w = np.array([np.sum(simp * integrand(zc + offs)) for zc in self.z_centers])

        self._tau_bin_weights = w / w.sum()
        return self._tau_bin_weights

    def tau_fraction(self, xe_bins, xe_hi=None, band=None):
        """
        Frazione di profondita' ottica su una banda di redshift, rispetto a
        quella di una storia completamente ionizzata sulla stessa banda.

        Fisico  <=>  0 <= tau_fraction <= 1.  E' il criterio che MH08 usano al
        posto del taglio punto per punto su xe: "Reionization histories with an
        unphysical optical depth over a large range in redshift are considered
        to be unphysical models since the addition of higher modes can not
        perturb the optical depth enough to give it a physical value."

        Il senso e' che i modi che NON campioniamo (a > 5) possono correggere
        xe punto per punto — sono proprio le oscillazioni veloci in z — ma non
        possono spostare di molto un integrale su un intervallo largo.  Quindi
        un xe negativo qui e la' e' recuperabile e non va scartato; una tau
        negativa su tutta la banda no.

        Parameters
        ----------
        band : (z_lo, z_hi) or None — None usa l'intero intervallo della base.
               Si contano i bin il cui CENTRO cade nella banda.
        """
        if xe_hi is None:
            xe_hi = self.xe_max
        w = self.tau_bin_weights()
        xe = np.asarray(xe_bins, dtype=float)
        if band is not None:
            sel = (self.z_centers >= band[0]) & (self.z_centers <= band[1])
            if not sel.any():
                raise ValueError(
                    f"PCABasis.tau_fraction: la banda {band} non contiene "
                    f"nessun centro di bin (z = {self.z_centers[0]:g}.."
                    f"{self.z_centers[-1]:g}, passo {self.delta_z:g})."
                )
            w, xe = w[sel], xe[sel]
        return float(np.dot(w, xe) / (w.sum() * xe_hi))

    # ------------------------------------------------- il test di fisicita'
    def physicality_report(self, m, xe_lo=0.0, xe_hi=None, bands=None,
                           strict=False):
        """
        Le condizioni che i coefficienti devono soddisfare, scritte una per una.

        Restituisce una lista di dizionari, uno per condizione:

            {'nome', 'formula', 'valore', 'limite', 'ok', 'commento'}

        cosi' il chiamante puo' sia DECIDERE (tutte ok?) sia STAMPARLE, senza
        che la logica sia sepolta dentro un if.  Sono esattamente le condizioni
        di Mortonson & Hu 2008, Sect. 3.

        LE CONDIZIONI, IN CHIARO
        ------------------------
        Sia  xe(z) = xe_fid + sum_{a=1..N} m_a S_a(z),  con  xe fisica in
        [xe_lo, xe_hi]  (di default [0, 1+f_He]).

        (1)  Eq. (10) — un limite per OGNI modo, preso da solo:

                 m_a^(-)  <=  m_a  <=  m_a^(+)

             con  m_a^(+-) = (1/(Nz+1)) sum_i [ S_a(z_i)*(c - xe_fid)
                                                +- |S_a(z_i)|*d ],
             c = (xe_hi+xe_lo)/2,  d = (xe_hi-xe_lo)/2.

             Viene da Eq. (6): per rendere m_a piu' grande possibile si mette
             xe al massimo dove S_a > 0 e al minimo dove S_a < 0.  Ogni storia
             fisica ha coefficienti dentro questi limiti; il viceversa NON
             vale, ed e' per questo che servono anche le altre condizioni.

        (2)  Eq. (13) — un limite sui modi presi INSIEME:

                 sum_a m_a^2  <=  f,   f = max[(xe_fid-xe_lo)^2, (xe_hi-xe_fid)^2]

             Viene dall'ortogonalita': integrare [xe - xe_fid]^2 sull'intervallo
             da' (z_max - z_min) * sum_a m_a^2, e quell'integrale non puo'
             superare (z_max - z_min)*f se xe resta nel suo intervallo fisico.
             Taglia combinazioni che la (1) da sola lascerebbe passare.

        (3)  tau parziale — su ogni banda larga di redshift:

                 0  <=  tau(banda) / tau(xe = xe_hi, stessa banda)  <=  1

             E' il criterio vero di MH08.  NON si taglia punto per punto su xe,
             e lo motivano: i modi che non campioniamo (a > N) sono oscillazioni
             veloci in z, quindi possono correggere xe localmente ma non possono
             spostare un integrale su banda larga.  Una xe negativa qua e la' e'
             recuperabile dai modi alti; una tau negativa su tutta la banda no.

        (4)  solo con strict=True — il taglio punto per punto:

                 xe_lo  <=  xe(z_i)  <=  xe_hi   per ogni bin i

             Piu' restrittivo del paper.  Serve a misurare quanto pesa la
             scelta, non a riprodurlo: lascia passare circa l'1% dello spazio.
        """
        if xe_hi is None:
            xe_hi = self.xe_max
        m = np.asarray(m, dtype=float).ravel()
        n = self._n(m.size)
        xe_bins = self.xe_fiducial + self.S[:, :n] @ m
        out = []

        # (1) Eq. (10), modo per modo
        lo_b, hi_b = self.mode_bounds(n, xe_lo, xe_hi)
        for a in range(n):
            out.append(dict(
                nome=f"MH08 Eq.(10)  modo {a+1}",
                formula=f"{lo_b[a]:+.4f} <= m{a+1} <= {hi_b[a]:+.4f}",
                valore=float(m[a]),
                limite=(float(lo_b[a]), float(hi_b[a])),
                ok=bool(lo_b[a] <= m[a] <= hi_b[a]),
                commento="ogni storia fisica ha m_a dentro questi limiti",
            ))

        # (2) Eq. (13), i modi insieme
        f = self.sum_sq_bound(xe_lo, xe_hi)
        s2 = float(np.sum(m ** 2))
        out.append(dict(
            nome="MH08 Eq.(13)  vincolo globale",
            formula=f"sum_a m_a^2 <= {f:.4f}",
            valore=s2, limite=(0.0, f), ok=bool(s2 <= f),
            commento="taglia combinazioni che i limiti per singolo modo "
                     "lascerebbero passare",
        ))

        # (3) tau parziale sulle bande
        for band in (bands or [(self.z_min, self.z_max)]):
            r = self.tau_fraction(xe_bins, xe_hi=xe_hi, band=band)
            out.append(dict(
                nome=f"tau parziale  z = {band[0]:g}-{band[1]:g}",
                formula="0 <= tau(banda)/tau_completa <= 1",
                valore=r, limite=(0.0, 1.0), ok=bool(0.0 <= r <= 1.0),
                commento="il criterio vero di MH08: i modi alti non possono "
                         "spostare un integrale su banda larga",
            ))

        # (4) opzionale: taglio punto per punto
        if strict:
            worst = int(np.argmax(np.maximum(xe_lo - xe_bins, xe_bins - xe_hi)))
            out.append(dict(
                nome="strict: xe punto per punto",
                formula=f"{xe_lo:g} <= xe(z_i) <= {xe_hi:.4f}  per ogni bin",
                valore=float(xe_bins[worst]), limite=(xe_lo, xe_hi),
                ok=bool(np.all((xe_bins >= xe_lo) & (xe_bins <= xe_hi))),
                commento=f"peggiore a z = {self.z_centers[worst]:g}; NON e' il "
                         "criterio del paper",
            ))
        return out

    @staticmethod
    def format_report(report, indent="    "):
        """Le condizioni in forma leggibile, una per riga."""
        w = max(len(r["nome"]) for r in report)
        out = []
        for r in report:
            out.append(f"{indent}[{'ok ' if r['ok'] else 'NO '}] "
                       f"{r['nome']:<{w}}  {r['formula']:<34} "
                       f"valore = {r['valore']:+.5f}")
        return "\n".join(out)

    # -------------------------------------------------------- ricostruzione
    def _n(self, n_modes):
        n = self.n_modes_available if n_modes is None else int(n_modes)
        if n > self.n_modes_available:
            raise ValueError(
                f"PCABasis: {n} modes requested, but the basis only provides "
                f"{self.n_modes_available}."
            )
        if n < MIN_MODES:
            raise ValueError(
                f"PCABasis: {n} modes requested, but at least {MIN_MODES} are "
                "required. Mortonson & Hu (2008) show that five principal "
                "components are needed to describe all the information on "
                "xe(z) contained in C_l^EE up to the cosmic-variance limit; "
                "Ilic et al. (2025) follow the same prescription ('pca-5'). "
                "Using more than five is allowed — the number of sampled "
                "coefficients reio_pca_m1..reio_pca_mN simply follows "
                "reio_pca_n_modes."
            )
        return n

    def reconstruct(self, m):
        """
        xe nei bin:  xe(z_i) = xe_fid + sum_a m_a S_a(z_i).

        NESSUN taglio: il risultato puo' uscire da [0, 1+f_He], ed e' voluto.
        Il troncamento a pochi modi rompe la completezza della base, quindi
        una storia perfettamente fisica puo' avere una ricostruzione a 5 modi
        che sfora.  Chi decide cosa farne e' il chiamante.
        """
        m = np.asarray(m, dtype=float).ravel()
        n = self._n(m.size)
        return self.xe_fiducial + self.S[:, :n] @ m

    def knots(self, m, xe_complete=None, xe_early=0.0, smooth_sigma_z=0.0,
              z_out_max=None):
        """
        Storia completa come spezzata: (z_knots, xe_knots).

        Questa e' la risposta al problema delle DUE GRIGLIE — quella delle
        autofunzioni e quella che va in CLASS.  I nodi sono:

            z = 0        xe = xe_complete     (reionizzazione conclusa)
            z = z_min    xe = xe_complete
            z = z_1      xe(z_1)     \
            ...                       |  i centri dei bin della base
            z = z_Nz     xe(z_Nz)    /
            z = z_max    xe = xe_early        (prima della reionizzazione)

        Fra un nodo e l'altro si interpola LINEARMENTE.  Il punto chiave:
        anche CLASS, in reio_inter, interpola linearmente.  Se la griglia che
        gli passiamo contiene esattamente questi nodi, CLASS ricostruisce la
        spezzata in modo ESATTO — errore di interpolazione zero, non piccolo.
        Le due griglie smettono di essere due.

        Perche' spezzata e non gradini.  La base e' formalmente una base di
        funzioni costanti a tratti (la Fisher e' stata costruita perturbando
        xe di una costante dentro ogni bin), quindi la rappresentazione
        "ortodossa" sarebbe una scala di Nz gradini.  Ma con soli 5 modi la
        ricostruzione e' a banda limitata: la struttura piu' fine che i 5 modi
        possono contenere ha scala ~ (z_max - z_min)/5, venti volte piu' larga
        di un bin da 0.5.  Rappresentarla a gradini aggiungerebbe struttura ad
        alta frequenza che nella ricostruzione non c'e'.  In piu' non ci
        starebbe: Nz+1 bordi da risolvere sono ~500 nodi, oltre il buffer del
        parser di CLASS.

        SMOOTHING AI BORDI (smooth_sigma_z > 0)
        ---------------------------------------
        Mortonson & Hu 2008, Sect. 5:

            "Since xe(z) during reionization must match onto xe ~ 1 at
             z = zmin and xe ~ 0 at z = zmax, there are often sharp
             transitions at these redshifts since nothing in equation (5)
             forces xe(z) to satisfy these boundary conditions.  To avoid
             problems with the time integration in CAMB, we smooth the
             reionization history by convolving xe(z) with a Gaussian of
             width sigma_z = 0.5."

        Il motivo e' NUMERICO, non fisico: la ricostruzione sum_a m_a S_a(z)
        agli estremi non vale ne' xe_complete ne' xe_early, quindi raccordarla
        al plateau crea un salto, e il salto fa impazzire l'integratore
        temporale del codice di Boltzmann.  Lo stesso problema che abbiamo
        misurato con CLASS: xe che varia troppo in fretta fa esplodere
        dln(kappa)/dz e il solutore raffina all'infinito.

        Con smooth_sigma_z > 0 si convolve la storia con una gaussiana di
        quella larghezza (in unita' di redshift) prima di estrarne i nodi.
        L'estremo del dominio non viene "tirato": mode='nearest' prolunga i
        plateau, quindi la convoluzione arrotonda gli spigoli senza spostare i
        livelli.

        NOTA: con lo smoothing acceso la ricostruzione NON e' piu' esattamente
        una spezzata sui centri dei bin, quindi la proprieta' di riproduzione
        esatta da parte di CLASS vale solo entro l'errore di campionamento
        della curva liscia.  Per questo i nodi vengono infittiti dove la
        gaussiana agisce.

        Parameters
        ----------
        m              : array dei coefficienti
        xe_complete    : xe sotto z_min.  None => 1 + f_He della base.
        xe_early       : xe sopra z_max (default 0).
        smooth_sigma_z : larghezza della gaussiana in redshift.  0 = nessuno
                         smoothing (default; il raccordo lineare su Delta z
                         che facciamo gia' e' di per se' una regolarizzazione).
        z_out_max      : estremo superiore dei nodi restituiti.  Serve solo
                         allo smoothing, per avere margine oltre z_max.
        """
        if xe_complete is None:
            xe_complete = self.xe_max
        xe_bins = self.reconstruct(m)
        z_knots = np.concatenate(([0.0, self.z_min], self.z_centers, [self.z_max]))
        xe_knots = np.concatenate(([xe_complete, xe_complete], xe_bins, [xe_early]))

        if not smooth_sigma_z or smooth_sigma_z <= 0:
            return z_knots, xe_knots

        return self._smooth_knots(z_knots, xe_knots, smooth_sigma_z,
                                  z_out_max or self.z_max)

    def _smooth_knots(self, z_knots, xe_knots, sigma_z, z_out_max):
        """
        Convoluzione con una gaussiana di larghezza sigma_z, come MH08.

        Si valuta la spezzata su una griglia fine uniforme (il passo deve
        essere piccolo rispetto a sigma_z, altrimenti si convolve male), si
        applica il filtro, e si ricampiona a passo sigma_z/2: la curva liscia
        ha curvatura limitata da sigma_z, quindi quel passo basta perche'
        l'interpolazione lineare di CLASS la segua.
        """
        from scipy.ndimage import gaussian_filter1d

        # margine di 4 sigma oltre gli estremi: oltre, la gaussiana pesa < 1e-4
        lo = 0.0
        hi = max(z_out_max, self.z_max) + 4.0 * sigma_z
        dz_fine = min(sigma_z / 10.0, self.delta_z / 4.0)
        z_fine = np.arange(lo, hi + dz_fine, dz_fine)
        xe_fine = np.interp(z_fine, z_knots, xe_knots)

        # mode='nearest': i plateau vengono prolungati, cosi' la convoluzione
        # arrotonda gli spigoli senza abbassare il livello di xe a z piccolo
        xe_sm = gaussian_filter1d(xe_fine, sigma_z / dz_fine, mode='nearest')

        # ricampionamento: fitto dove la gaussiana ha agito, rado sui plateau
        step = min(sigma_z / 2.0, self.delta_z)
        z_new = np.unique(np.concatenate([
            [0.0],
            np.arange(max(0.0, self.z_min - 4 * sigma_z),
                      min(hi, self.z_max + 4 * sigma_z) + step, step),
            self.z_centers,
            [min(hi, z_out_max)],
        ]))
        z_new = z_new[(z_new >= 0.0) & (z_new <= z_out_max)]
        if z_new[-1] != z_out_max:
            z_new = np.append(z_new, z_out_max)
        return z_new, np.interp(z_new, z_fine, xe_sm)

    # ------------------------------------------------------------ diagnostica
    def describe(self):
        """Riassunto leggibile della base, da stampare all'avvio di una catena."""
        lo, hi = self.mode_bounds()
        lines = [
            f"base PCA: {self.path or '(in memoria)'}",
            f"  z = {self.z_min:g} .. {self.z_max:g},  delta_z = {self.delta_z:g},  "
            f"{self.n_bins} bin  (normalizzazione sum S^2 = {self.n_bins + 1})",
            f"  xe fiduciale = {self.xe_fiducial:g}"
            + (f",  tau fiduciale = {self.tau_fiducial:.5f}"
               if np.isfinite(self.tau_fiducial) else ""),
            f"  delta_xe = {self.delta_xe:g},  l_max = {self.l_max}"
            if np.isfinite(self.delta_xe) else "",
            f"  YHe = {self.YHe:g},  f_He = {self.f_He:.6f},  "
            f"xe_max = {self.xe_max:.6f}",
            f"  modi disponibili: {self.n_modes_available}",
            f"  cosmologia fiduciale: "
            + (", ".join(f"{k}={v:g}" for k, v in self.cosmo.items())
               if self.cosmo else "NON salvata nel file (uso Omega_m = 0.315 "
                                  "per il peso di tau)"),
            f"  limite MH08 Eq.(13): sum m_a^2 <= {self.sum_sq_bound():.4f}",
            "",
            f"  {'modo':>4} {'lambda':>12} {'sigma(m_a)':>11} "
            f"{'m_min':>10} {'m_max':>10}",
        ]
        for a in range(min(8, self.n_modes_available)):
            lines.append(f"  {a+1:4d} {self.eigenvalues[a]:12.4f} "
                         f"{self.sigma(a):11.5f} {lo[a]:10.4f} {hi[a]:10.4f}")
        return "\n".join(x for x in lines if x != "")

    def priors_yaml(self, n_modes=5, indent="  "):
        """
        Blocco 'params' pronto da incollare nel YAML di Cobaya.

        I prior sono i limiti MH08 Eq. (10), che e' precisamente il senso di
        "very large uniform priors + physicality prior" in Ilic et al. 2025:
        la scatola e' la piu' piccola che contiene tutto lo spazio fisico, e
        il resto del taglio lo fa il prior di fisicita' dentro calculate().

        ref/proposal sono presi da sigma(m_a) della base: e' la larghezza che
        un esperimento limitato dalla varianza cosmica puo' distinguere, quindi
        la scala giusta per i passi dell'MCMC.  Senza, il campionatore parte
        con passi sbagliati di ordini di grandezza fra il modo 1 e il modo 5.
        """
        n = self._n(n_modes)
        lo, hi = self.mode_bounds(n)
        out = ["params:"]
        for a in range(n):
            s = self.sigma(a)
            out += [
                f"{indent}reio_pca_m{a+1}:",
                f"{indent*2}prior: {{min: {lo[a]:.5f}, max: {hi[a]:.5f}}}",
                f"{indent*2}ref: {{dist: norm, loc: 0.0, scale: {min(s, (hi[a]-lo[a])/6):.5f}}}",
                f"{indent*2}proposal: {min(s, (hi[a]-lo[a])/6):.5f}",
                f"{indent*2}latex: m_{a+1}",
            ]
        return "\n".join(out)


# =========================================================================
# LETTURA DEI FILE
# =========================================================================
def _read_npz(path):
    with np.load(path, allow_pickle=False) as d:
        return {k: d[k] for k in d.files}


def _read_dat(path):
    """
    Formato testo auto-descrittivo:

        # chiave = valore
        # ...
        # z   S1   S2   ...
        6.5  0.12  -0.4 ...

    Le righe di commento nella forma `# chiave = valore` diventano metadati;
    le altre si ignorano.  Le colonne sono z e poi le autofunzioni.
    """
    meta, rows = {}, []
    with open(path) as fh:
        for line in fh:
            s = line.strip()
            if not s:
                continue
            if s.startswith("#"):
                body = s.lstrip("#").strip()
                if "=" in body:
                    k, _, v = body.partition("=")
                    k, v = k.strip(), v.strip()
                    try:
                        meta[k] = float(v)
                    except ValueError:
                        meta[k] = v
                continue
            rows.append([float(x) for x in s.split()])

    if not rows:
        raise ValueError(f"PCABasis: nessuna riga di dati in '{path}'.")
    arr = np.asarray(rows, dtype=float)
    if arr.shape[1] < 2:
        raise ValueError(
            f"PCABasis: '{path}' ha una sola colonna. Servono z e almeno "
            "un'autofunzione."
        )

    data = dict(meta)
    data["z_centers"] = arr[:, 0]
    data["eigenvectors"] = arr[:, 1:]
    if "eigenvalues" in meta and isinstance(meta["eigenvalues"], str):
        data["eigenvalues"] = np.array(
            [float(x) for x in meta["eigenvalues"].replace(",", " ").split()])
    elif "eigenvalues" not in data:
        raise ValueError(
            f"PCABasis: '{path}' non contiene gli autovalori. Aggiungi una "
            "riga di header `# eigenvalues = l1 l2 ...`: senza autovalori non "
            "si possono calcolare le sigma(m_a) ne' le proposal dell'MCMC."
        )

    # cosmologia: righe `# cosmo.h = 0.6731`
    ck = [k for k in meta if k.startswith("cosmo.")]
    if ck:
        data["cosmo_keys"] = np.array([k[len("cosmo."):] for k in ck])
        data["cosmo_vals"] = np.array([float(meta[k]) for k in ck])
    return data


def load_basis(path):
    """
    Carica (con cache) una base PCA da .npz o da .dat/.txt.

    La cache e' indicizzata su (percorso, mtime, dimensione): rigenerare il
    file durante una sessione interattiva non lascia in memoria quello vecchio.
    """
    if not path:
        raise ValueError(
            "PCABasis: nessun file specificato. Imposta reio_pca_file nel "
            "blocco theory del YAML, puntando al .npz prodotto da "
            "PCA_eigenfunction_reconstruction.py."
        )
    path = os.path.abspath(os.path.expanduser(str(path)))
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"PCABasis: '{path}' non esiste. Genera la base con "
            "PCA_eigenfunction_reconstruction.py, oppure correggi "
            "reio_pca_file nel YAML."
        )

    st = os.stat(path)
    key = (path, st.st_mtime_ns, st.st_size)
    if key in _CACHE:
        return _CACHE[key]

    ext = os.path.splitext(path)[1].lower()
    data = _read_npz(path) if ext == ".npz" else _read_dat(path)
    basis = PCABasis(data, path=path)
    _CACHE[key] = basis
    return basis
