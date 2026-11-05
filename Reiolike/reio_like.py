# from cobaya.likelihood import DataSetLikelihood
import re
import numpy as np
import yaml
import corecon as con
from scipy.special import log_ndtr
from scipy.stats import norm
from cobaya.likelihood import Likelihood
from cobaya.log import LoggedError
from .reio_models import _F_HE


# ---------------------------------------------------------------------------
# Dal confidence level alla larghezza della gaussiana
# ---------------------------------------------------------------------------
# Un limite e' una MEZZA GAUSSIANA appoggiata al bordo fisico (0 per un upper
# limit su x_HII, 1 per un lower limit): una distribuzione vera, definita solo
# dal bordo verso l'interno, con densita'
#
#     p(x) = 2 / (sigma sqrt(2 pi)) * exp(-x^2 / (2 sigma^2)),   x >= 0
#
# (x = distanza dal bordo; il 2 la normalizza a 1).  "Limite g al CL" vuol dire
# che l'area di p fra il bordo e g e' CL:
#
#     int_0^g p(x) dx = 2 Phi(g/sigma) - 1 = CL
#     =>  z_CL = g/sigma = Phi^-1((1 + CL)/2)
#
# cioe' 0.9945 per CL=0.68, 1.9600 per CL=0.95, 2.9677 per CL=0.997.  Da qui
#
#     upper limit su x_HII:   sigma = g / z_CL            (bordo in 0)
#     lower limit su x_HII:   sigma = (1 - g) / z_CL      (bordo in 1)
#
# E' l'UNICA lettura del CL nel pacchetto: non c'e' niente da scegliere, ne'
# nello YAML ne' in corecon.  Coincide con la lettura
# "68% = 1 sigma, 95% = 2 sigma" dei paper, perche' 2 Phi(z) - 1 e' anche
# l'area di una gaussiana intera fra -z e +z; ma il conto e' quello sulla
# mezza gaussiana.  Kageura et al. 2026 usano questa lettura: le barre dei
# limiti nella loro Fig. 1 ("error bars indicate 1 sigma uncertainties")
# coincidono al millesimo per Totani 2006, Bolan 2022, Wold 2022 e Mason 2019.
#
# STORIA.  Prima qui c'era 1.645 scritto a mano in quattro punti (ogni limite
# al 95%), poi z = Phi^-1(CL) ("una coda": CL letto come area a sinistra di g
# di una gaussiana INTERA, che comprende meta' massa oltre il bordo, dove non
# esiste niente; sulla mezza gaussiana mette il 36% sotto un limite "al 68%").
# Fino al 21 ott 2026 era un'opzione dello YAML (convenzione_CL), con questa
# lettura chiamata prima 'due_code' e poi 'mezza_gaussiana'; ora e' l'unica e
# la chiave non esiste piu'.  Il backup del 6 ott 2026, fatto con Phi^-1(CL),
# si riproduce con Tests/gp_riproduci_spot_on.py, che la emula da se'.
# L'argomento per Phi^-1(CL) era Totani 2006, che tabula la STESSA misura a
# 0.68 e 0.95:
#     Phi^-1(CL):          0.170/0.4677 = 0.3635   e   0.600/1.6449 = 0.3648
#     Phi^-1((1+CL)/2):    0.170/0.9945 = 0.171    e   0.600/1.9600 = 0.306
# Che con Phi^-1(CL) le due righe tornino NON dimostra che sia giusta:
# vuol dire che la posteriore di Totani non e' una mezza gaussiana (code piu'
# larghe), come quelle di Mason 2019 (rapporto 1.56 fra le due righe) e Bolan
# 2022 (2.00).  La scelta della riga e' una decisione fisica, da fare nello
# YAML con [CL=...].

# Usato quando corecon marca una riga come limite ma non dice a che CL.
# Non e' un default silenzioso: _limit_sigma avvisa.
DEFAULT_LIMIT_CL = 0.95


def z_for_confidence_level(cl):
    """
    Numero di sigma del limite: z_CL = Phi^-1((1 + CL)/2).

    E' la soluzione di  2 Phi(z) - 1 = CL, cioe' dell'area della mezza
    gaussiana fra il bordo e il limite (vedi il commento sopra).
    """
    cl = float(cl)
    if not (0.0 < cl < 1.0): #if the confidence level out of this bound, an error rises up.
        raise ValueError(f"confidence level fuori da (0,1): {cl}")
    #norm.ppf is the inverse of the partition function of the gaussian.
    return float(norm.ppf(0.5 * (1.0 + cl)))


# ---------------------------------------------------------------------------
# Valutazione del modello: interpolazione lineare, integrale in forma chiusa
# ---------------------------------------------------------------------------
# ReioTheory consegna xe(z) a CLASS come tabella con reio_parametrization=
# 'reio_inter', e CLASS la interpola LINEARMENTE (thermodynamics.c:4311-4350).
# Quindi il modello vero, quello da cui esce tau, e' la spezzata sui nodi:
# la likelihood deve valutare la stessa funzione, non un'altra.
#
# Questo risolve da solo il problema delle transizioni ripide.  Con una spline
# cubica serviva una quadratura adattiva per integrarla sulle finestre dei
# dati; su una spezzata la media su una finestra e' invece ESATTA con la regola
# dei trapezi applicata ai nodi del modello, perche' i nodi di un gradino
# finiscono fra i punti di integrazione per costruzione.
#
# Misurato sul modello a bin (transizioni larghe 0.1 in z):
#   - esattezza : scarto a precisione macchina contro quad ad alta precisione,
#                 contro un errore fino a 2.6e-4 della vecchia via;
#   - velocita' : ~58x (29 ms -> 0.5 ms per 40 finestre, per dataset, per passo);
#   - coerenza  : la spline cubica oscillava di ~1e-3 fra i nodi, cioe'
#                 valutava una funzione diversa da quella da cui esce tau;
#   - estrapolazione: fuori griglia la cubica dava xe negativo (-0.04 a z=40),
#                 np.interp estende con il valore all'estremo (0).


def model_at(z_model, model_values, z):
    """
    Modello valutato in z per interpolazione lineare, la stessa che usa CLASS.

    Fuori dalla griglia estende con il valore all'estremo (np.interp) invece di
    estrapolare: per xe(z) e' la scelta fisica giusta (sopra la griglia la
    tabella vale gia' zero) ed evita le escursioni di un'estrapolazione cubica.
    """
    return np.interp(z, z_model, model_values)


def split_normal_chi2(x_th, mu, sigma_lo, sigma_hi):
    """
    chi^2 di una split normal (gaussiana asimmetrica) valutata in x_th.

    Il dato vale mu con barre d'errore asimmetriche: sigma_hi sopra, sigma_lo
    sotto.  La distribuzione e' centrata sul DATO (la barra d'errore e' sua,
    non del modello) e si valuta dove cade il modello:

        x_th >= mu  ->  ((x_th - mu)/sigma_hi)^2
        x_th <  mu  ->  ((x_th - mu)/sigma_lo)^2

    Il valore centrale mu e' la MODA, non la media: la media di una split
    normal e' spostata di sqrt(2/pi)*(sigma_hi - sigma_lo).

    NORMALIZZAZIONE.  La densita' normalizzata porta il fattore
        sqrt(2/pi) / (sigma_lo + sigma_hi)
    che serve perche' le due mezze gaussiane hanno aree diverse.  Qui viene
    OMESSO di proposito: sigma_lo e sigma_hi vengono dai dati e non dipendono
    dai parametri campionati, quindi log C e' una costante additiva identica a
    ogni passo della catena e non tocca il posterior.  Andrebbe invece tenuto
    se un giorno gli errori venissero riscalati da un parametro di nuisance.

    LATO LIBERO.  sigma <= 0 sul lato attivo non e' una misura di larghezza
    nulla: e' un bordo fisico (x_HII non puo' superare 1 ne' scendere sotto 0,
    vedi Umeda et al. 2026).  Su quel lato il dato non porta informazione,
    quindi il contributo e' zero invece di una divisione per zero.  Il punto
    resta comunque nella likelihood e vincola dal lato non degenere.

    CONTINUITA'.  In x_th = mu entrambi i rami danno 0, quindi la logL e'
    continua; la sua DERIVATA no (c'e' un angolo).  Innocuo per Metropolis,
    che e' il campionatore di cobaya; sarebbe un problema per campionatori a
    gradiente tipo HMC.
    """
    x_th     = np.asarray(x_th,     dtype=float)
    mu       = np.asarray(mu,       dtype=float)
    sigma_lo = np.asarray(sigma_lo, dtype=float)
    sigma_hi = np.asarray(sigma_hi, dtype=float)

    sigma = np.where(x_th >= mu, sigma_hi, sigma_lo)
    free  = ~(sigma > 0)                      # copre anche NaN
    safe  = np.where(free, 1.0, sigma)        # evita il warning di divisione
    return np.where(free, 0.0, ((x_th - mu) / safe) ** 2)


# ---------------------------------------------------------------------------
# Limiti pubblicati con un errore sul limite stesso
# ---------------------------------------------------------------------------
# Alcuni limiti vengono con err_up / err_down (oggi, nella selezione di
# analisi: Zhu et al. 2022 e Davies et al. 2026, frazione di pixel/gap scuri).
# Quell'errore e' l'incertezza sul VALORE del limite: il vincolo e' x <= B (o
# x >= B) con B non noto esattamente.  B ha distribuzione split-normal centrata
# sul valore pubblicato g, larga err_down sotto ed err_up sopra (in x_HII), e
# si integra su B:
#
#     x >= B (lower limit su x_HII):  L(x) = P(B <= x) =     F(x)
#     x <= B (upper limit su x_HII):  L(x) = P(B >= x) = 1 - F(x)
#
# con F la funzione di ripartizione della split-normal, che e' esatta:
#
#     x <= g :   F(x) = a_lo Phi((x - g) / s_lo)
#     x >  g :   F(x) = 1 - a_hi Phi((g - x) / s_hi)
#     a_lo = 2 s_lo / (s_lo + s_hi),   a_hi = 2 s_hi / (s_lo + s_hi)
#
# Con s_lo = s_hi = e si ritrova F(x) = Phi((x - g)/e) (Davies).  Zhu 2022 ha
# l'ultima riga asimmetrica (x_HII > 0.71, err_down 0.09, err_up 0.10).
# Tutto in logaritmo, stabile anche quando L e' piccolissima.  Le derivate
# servono al gradiente analitico del GP (gp_likelihood.py usa queste stesse
# funzioni).

def logL_limite_inferiore_con_errore(x, g, s_lo, s_hi):
    """log F(x): vincolo x >= B, con B split-normal (g, s_lo, s_hi)."""
    x, g, s_lo, s_hi = np.broadcast_arrays(*map(np.asarray, (x, g, s_lo, s_hi)))
    a_lo = 2.0 * s_lo / (s_lo + s_hi)
    a_hi = 2.0 * s_hi / (s_lo + s_hi)
    with np.errstate(invalid="ignore", divide="ignore"):   # ramo scartato da where
        sotto = np.log(a_lo) + log_ndtr((x - g) / s_lo)
        sopra = np.log1p(-a_hi * np.exp(log_ndtr((g - x) / s_hi)))
    return np.where(x <= g, sotto, sopra)


def logL_limite_superiore_con_errore(x, g, s_lo, s_hi):
    """log (1 - F(x)): vincolo x <= B, con B split-normal (g, s_lo, s_hi)."""
    x, g, s_lo, s_hi = np.broadcast_arrays(*map(np.asarray, (x, g, s_lo, s_hi)))
    a_lo = 2.0 * s_lo / (s_lo + s_hi)
    a_hi = 2.0 * s_hi / (s_lo + s_hi)
    with np.errstate(invalid="ignore", divide="ignore"):   # ramo scartato da where
        sotto = np.log1p(-a_lo * np.exp(log_ndtr((x - g) / s_lo)))
        sopra = np.log(a_hi) + log_ndtr((g - x) / s_hi)
    return np.where(x <= g, sotto, sopra)


def dlogL_limite_inferiore_con_errore(x, g, s_lo, s_hi):
    """(d/dx, d2/dx2) di logL_limite_inferiore_con_errore."""
    a_hi = 2.0 * s_hi / (s_lo + s_hi)
    # x <= g: log a_lo + log Phi(u), u = (x-g)/s_lo;  r = phi(u)/Phi(u)
    u = (x - g) / s_lo
    r = np.exp(norm.logpdf(u) - log_ndtr(u))
    d1_s, d2_s = r / s_lo, -r * (u + r) / s_lo ** 2
    # x > g: log(1 - a_hi Phi(v)), v = (g-x)/s_hi
    v = (g - x) / s_hi
    with np.errstate(invalid="ignore", divide="ignore"):
        L = -np.expm1(np.log(a_hi) + log_ndtr(v))        # 1 - a_hi Phi(v)
        Fp = a_hi * norm.pdf(v) / s_hi
        Fpp = a_hi * norm.pdf(v) * v / s_hi ** 2
        d1_o = Fp / L
        d2_o = Fpp / L - d1_o ** 2
    return np.where(x <= g, d1_s, d1_o), np.where(x <= g, d2_s, d2_o)


def dlogL_limite_superiore_con_errore(x, g, s_lo, s_hi):
    """(d/dx, d2/dx2) di logL_limite_superiore_con_errore."""
    # simmetria: 1 - F(x; g, s_lo, s_hi) = F(-x; -g, s_hi, s_lo)
    d1, d2 = dlogL_limite_inferiore_con_errore(-np.asarray(x), -np.asarray(g), s_hi, s_lo)
    return -d1, d2


def model_window_mean(z_model, model_values, z_lo, z_hi):
    """
    Media del modello sulla finestra [z_lo, z_hi], calcolata esattamente.

    Il modello e' lineare a tratti con nodi in z_model, quindi l'integrale e'
    esatto sommando i trapezi sui nodi interni alla finestra piu' i due tratti
    parziali agli estremi.  Nessuna dipendenza dalla ripidita' delle
    transizioni: se un gradino cade dentro la finestra, i suoi nodi sono fra i
    punti di integrazione per costruzione.

    Se la finestra e' degenere (z_hi <= z_lo) restituisce il valore puntuale.
    """
    if not (z_hi > z_lo):
        return float(model_at(z_model, model_values, z_lo))

    inner = z_model[(z_model > z_lo) & (z_model < z_hi)]
    zs = np.concatenate(([z_lo], inner, [z_hi]))
    vs = model_at(z_model, model_values, zs)
    integral = 0.5 * np.sum(np.diff(zs) * (vs[:-1] + vs[1:]))
    return float(integral / (z_hi - z_lo))

class ReioLike(Likelihood):
    """
    This likelihood compares the neutral hydrogen fraction
    obtained by a certain reionization history xe(z)
    calculated by the theory (ReioTheory) with astrophysical data.

    It uses the 'corecon' package to compute likelihoods
    for different astrophysical analyses.

    Assumption:
    - The theory (ReioTheory) calculates xe(z) on a redshift grid.
    - The theory provides the arrays 'z' and 'xe'.
    """

    # This is the path to the corecon configuration file,
    # which specifies which analyses to run and with which datasets.
    corecon_config_file: str = "corecon_config.yaml"

    # Diagnostica.  Entrambi stampano, quindi in una catena vera vanno a False.
    #
    # print_sigma_table: una tabella per dataset a initialize(), con il sigma di
    #   ogni punto.  Non dipende dal modello, quindi costa una volta sola ed e'
    #   il posto dove verificare a mano la larghezza delle gaussiane.
    # debug_points: a OGNI chiamata di logp, per ogni punto, x_th e il chi
    #   quadro che ne esce.  Questo serve per i test di controllo, non per le
    #   catene: con 15 dataset produce migliaia di righe al secondo.
    # print_dataset_logl: una riga per dataset a OGNI chiamata di logp.  Era
    #   incondizionato: con 12 dataset e una catena da 100k passi faceva 1.2
    #   milioni di righe su stdout, che rallentano la catena e rendono il log
    #   illeggibile.  Ora e' spento di default; per un controllo interattivo
    #   basta accenderlo, o usare debug_points che da' molto piu' dettaglio.
    print_sigma_table: bool = False
    debug_points: bool = False
    print_dataset_logl: bool = False

    def initialize(self):

        """
        Initializes the Corecon analyses specified in the configuration file.
        """
        print(f"CoCORITA initialized. Reading corecon config from: {self.corecon_config_file}")

        # Read the corecon configuration file to determine which analyses to run
        with open(self.corecon_config_file, "r") as f:
            self.config = yaml.safe_load(f) #we turn the YAML into a python dictionary
            print(f"Loaded corecon config: {self.config}")

        self.analyses = [] #output of initialize, list with dictionaries related to each analysis in the yaml
        self._seen_datasets = set()  #For duplicate detection
        self._seen_corecon_names = {} #This is necessary to check for different models in the same analysis
        # nome corecon -> [(riga dello YAML, 'indipendente'?), ...] per ogni selezione caricata
        self._coverage_warned = set()  # for one-time extrapolation warnings, so that we don't repeat them in the MCMC many times.
        #There are four possibilities: detection, lower bound, upper bound or a mix.
        KNOWN_SECTIONS = {
            "HII_fraction", "upper_limit_HII_fraction",
            "lower_limit_HII_fraction", "mixed_HII_fraction",
        }

        # La lettura del CL e' fissa (mezza gaussiana, vedi z_for_confidence_level):
        # la vecchia chiave convenzione_CL non e' piu' un'opzione.  Si ferma
        # invece di ignorarla, cosi' un YAML vecchio non passa in silenzio.
        if "convenzione_CL" in self.config:
            raise RuntimeError(
                f"convenzione_CL: {self.config['convenzione_CL']!r} nello YAML non e' "
                f"piu' un'opzione: il CL di un limite si legge sempre come area della "
                f"mezza gaussiana fra il bordo e il limite.  Togli la riga.")

        for observable_type, datasets in self.config.items():
            #check the KNOWN_SECTIONS written above
            if observable_type not in KNOWN_SECTIONS: #observable type can be HII_fraction, upper_limit_HII_fraction, lower_limit_HII_fraction, mixed_HII_fraction
                raise RuntimeError(
                    f"Unknown YAML section '{observable_type}'. "
                    f"Valid sections are: {sorted(KNOWN_SECTIONS)}. "
                    f"Check for typos in corecon_config.yaml."
                )
            print(f"Found observable type: {observable_type}")

            if not datasets: #may happen that we leave a section which is empty, for example we don't use lower limit. This is printed however it is an error.
                print(f"  - No datasets listed for '{observable_type}', skipping.")
                continue

            # OGGI tutte e quattro le sezioni descrivono la STESSA osservabile,
            # x_HII.  Quello che cambia fra loro non e' la grandezza misurata ma
            # come va trattata statisticamente: detection (gaussiana), limite
            # superiore, limite inferiore, misto.  Quindi la condizione qui sotto
            # e' sempre vera e corecon_key vale sempre "x_HII"; la prima riga non
            # sopravvive mai.  E' scritta comunque perche' e' il punto in cui
            # andra' innestata una seconda osservabile (tau, flusso Lyman-alpha,
            # temperatura dell'IGM, ...) quando servira'.
            #
            # QUANDO SI AGGIUNGERA' UN'OSSERVABILE, attenzione: l'elenco delle
            # sezioni e' scritto in DUE posti, qui e in KNOWN_SECTIONS sopra.
            # Aggiungerne una sola a KNOWN_SECTIONS la fa accettare, ma poi le
            # assegna corecon_key = nome_della_sezione e con.get() fallisce con
            # un errore che non indica la causa vera.  Conviene allora passare a
            # un dizionario come unica fonte di verita':
            #
            #     SECTION_TO_CORECON = {"HII_fraction": "x_HII", ..., "tau_section": "tau"}
            #     KNOWN_SECTIONS     = set(SECTION_TO_CORECON)
            #     corecon_key        = SECTION_TO_CORECON[observable_type]
            #
            # Nota anche che con.get() sta dentro il loop sulle sezioni, quindi
            # oggi viene chiamata 4 volte con lo stesso argomento "x_HII".  Costa
            # solo tempo all'avvio; con piu' osservabili converra' mettere in
            # cache i dizionari gia' scaricati, es. {chiave: dati}.
            corecon_key = observable_type
            if observable_type in ("HII_fraction", "upper_limit_HII_fraction",
                                     "lower_limit_HII_fraction", "mixed_HII_fraction"):
                corecon_key = "x_HII"

            try:
                corecon_data_dict = con.get(corecon_key) #this actually is necessary to extract the data from the paper
            except Exception as e:
                print(f"  - ERROR loading corecon data for '{corecon_key}': {e}")
                raise RuntimeError(f"CRITICAL ERROR loading corecon data for '{corecon_key}': {e}")

            if isinstance(datasets, list):
                for dataset_spec in datasets:
                    # Support "Dataset Name [LABEL]" notation for multi-realization or multi-type datasets
                    # e.g. "Durovcikova et al. 2024 [ATON]" or "Davies et al. 2026 [threshold]"
                    (corecon_name, label, cl_req, joint_req,
                     indip_req,
                     err95_req) = self._parse_dataset_spec(dataset_spec) #separa il nome dalle parentesi quadrate dove specifichiamo il nome di modelli o del cl
                                                                         # tipo "Bolan et al. 2022 [CL=0.68]" viene separato in "Bolan et al. 2022" e 0.68 sotto cl_req
                                                                         # oppure "Durovcikova et al. 2024 [ATON]" diventa "Durovcikova et al. 2024" e ["ATON"] diventa l'etichetta da leggere per i modelli
                    print(f"  - Initializing analysis for dataset: {dataset_spec}")

                    if corecon_name in corecon_data_dict:
                        data_obj = corecon_data_dict[corecon_name] #data_obj e' l'oggetto corecon con tutte le righe del paper, se il nome non e' in corecon piu' sotto c'e' l'else che blocca tutto con "not found"

                        # Some datasets are intrinsically multi-model and must be disambiguated in YAML.
                        # Example: Totani et al. 2014 has IGM_model choices like [fixed_dz], [variable_dz].
                        self._ensure_required_label(corecon_name, label, data_obj, dataset_spec) #alcuni modelli hanno labels non opzionali

                        if label is not None:
                            # Multi-realization or multi-type dataset: extract only the rows matching label.
                            # For some dataset (like Umeda et a. 2026) data_obj.axes is a flat array and we can read it automatically.
                            # If a dataset has more models, usually it is identified by a key, thus we need to extract the redshift information
                            # and separate it with respect to the label (like Durovcikova et al. 2024[CROC] is a tuple like (z, label), thus you need to
                            # split z from the label)
                            label_str = ", ".join(label) #labels and redshift are splitted by a comma
                            print(f"    - Multi-label dataset, extracting label: '[{label_str}]'")
                            z_arr, val_arr, err_up_arr, err_dn_arr, err_z_left, err_z_right, \
                                lower_lim_arr, upper_lim_arr, cl_arr, \
                                err_up2_arr, err_dn2_arr = \
                                self._extract_realization(data_obj, label)
                        else: #if we don't have a label, we need to transform every piece in corecon in an array
                            # Standard dataset: axes is a plain array of redshifts, in aclcuni casi alcuni campi sono vuoti e qui c'e' scritto come gestirli.
                            z_arr       = np.array(data_obj.axes,   dtype=float)
                            val_arr     = np.array(data_obj.values, dtype=float)
                            err_up_arr  = np.array(data_obj.err_up, dtype=float)
                            # Barra inferiore sul valore; assente -> errore simmetrico. Se corecon ha err_down ed è pieno, usalo; altrimenti copia err_up
                            err_dn_arr  = np.array(data_obj.err_down, dtype=float) \
                                if hasattr(data_obj, "err_down") and data_obj.err_down is not None \
                                else err_up_arr.copy()
                            # Barre al 95% (due code), dove corecon le fornisce.
                            err_up2_arr = self._get_err2_array(data_obj, "err_up2", len(z_arr)) #NaN: "nessuna barra al 95%"
                            err_dn2_arr = self._get_err2_array(data_obj, "err_down2", len(z_arr)) #None: nessuna finestra in z, il dato si valuta nel punto
                            err_z_left  = self._get_error_array(data_obj, "err_left")
                            err_z_right = self._get_error_array(data_obj, "err_right")
                            # reading the arrays and doing basic checks: empty, NaN, inf object
                            # "Se corecon ha lower_lim, usalo; altrimenti tutti False." 
                            # Un dataset senza flag è fatto solo di detection. np.zeros(..., dtype=bool) è un array di False lungo quanto le righe. 
                            # Lo stesso vale per upper_lim.
                            lower_lim_arr = np.array(data_obj.lower_lim, dtype=bool) \
                                if hasattr(data_obj, "lower_lim") and data_obj.lower_lim is not None \
                                else np.zeros(len(z_arr), dtype=bool)
                            
                            upper_lim_arr = np.array(data_obj.upper_lim, dtype=bool) \
                                if hasattr(data_obj, "upper_lim") and data_obj.upper_lim is not None \
                                else np.zeros(len(z_arr), dtype=bool)
                            # Confidence level di ciascun limite.  NaN dove manca
                            # o dove la riga non e' un limite.
                            cl_arr = self._get_cl_array(data_obj, len(z_arr))

                        if len(z_arr) == 0:
                            raise RuntimeError(
                                f"Dataset '{dataset_spec}' in corecon '{corecon_key}' has no data points (axes is empty)."
                            )

                        # Selezione delle righe.  Si fa qui, dopo l'estrazione,
                        # cosi' vale allo stesso modo per i dataset con e senza
                        # label.  Prima il vincolo congiunto, poi il confidence
                        # level: in quest'ordine il guard sul CL lavora su un
                        # insieme di righe gia' coerente.
                        arrays = [z_arr, val_arr, err_up_arr, err_dn_arr,
                                  lower_lim_arr, upper_lim_arr, cl_arr,
                                  err_z_left, err_z_right,
                                  err_up2_arr, err_dn2_arr] #con questo array mettiamo gli array in una lista
                        
                        #la funzione di keep serve per selezionare. 
                        keep = self._select_joint_rows(
                            dataset_spec, joint_req,
                            self._get_joint_array(data_obj, len(z_arr)), z_arr)
                        arrays = self._apply_row_mask(arrays, keep, dataset_spec)
                        (z_arr, val_arr, err_up_arr, err_dn_arr, lower_lim_arr,
                         upper_lim_arr, cl_arr, err_z_left, err_z_right,
                         err_up2_arr, err_dn2_arr) = arrays
                        #se piu' cl sono disponibili, tende a selezionare quello scelto dall'autore.
                        keep = self._select_cl_rows(
                            dataset_spec, cl_req, z_arr, cl_arr,
                            lower_lim_arr, upper_lim_arr)
                        arrays = self._apply_row_mask(arrays, keep, dataset_spec) #questa funzione tiene solo le righe che servono, sfruttando l'informazione del keep
                        (z_arr, val_arr, err_up_arr, err_dn_arr, lower_lim_arr,
                         upper_lim_arr, cl_arr, err_z_left, err_z_right,
                         err_up2_arr, err_dn2_arr) = arrays #dopo il taglio devo tenere a memoria questo nuovo array.
                        #Attenzione all'ordine. Python assegna per posizione, non per nome. Se i nomi fossero scritti in un ordine diverso da quello della lista, per esempio (z, cl, val) = arrays, non ci sarebbe nessun errore, ma cl conterrebbe i valori e val i CL. Per questo nel codice la lista e la riga dei nomi hanno esattamente lo stesso ordine:
                        #lista:  z, values, err_up, err_down, lower_lim, upper_lim, CL, err_left, err_right, err_up2, err_down2
                        #nomi:   z_arr, val_arr, err_up_arr, err_dn_arr, lower_lim_arr, upper_lim_arr, cl_arr, err_z_left, err_z_right, err_up2_arr, err_dn2_arr

                        # Barre al livello di confidenza chiesto: se si prende la
                        # riga-limite al 95%, le detection dello stesso dataset
                        # devono usare le loro barre al 95%, non quelle al 68%.

                        err_up_arr, err_dn_arr = self._select_error_bars(
                            dataset_spec, cl_req, err_up_arr, err_dn_arr,
                            err_up2_arr, err_dn2_arr, lower_lim_arr, upper_lim_arr)
                        #se non ho nemmeno un valore numerico buono letto da corecon allora si ferma tutto,
                        #nel caso di singoli valori che non funzionano allora questo viene gestito dalla likelihood
                        n_valid = np.sum(np.isfinite(val_arr))
                        if n_valid == 0:
                            raise RuntimeError(
                                f"Dataset '{dataset_spec}' in corecon '{corecon_key}' has no finite values "
                                f"(all NaN/Inf). Cannot use it in the likelihood."
                            )

                        # Physical range check: x_HII must be in [0, 1]
                        finite_vals = val_arr[np.isfinite(val_arr)]
                        out_of_range = finite_vals[(finite_vals < 0) | (finite_vals > 1)]
                        if len(out_of_range) > 0:
                            raise RuntimeError(
                                f"Dataset '{dataset_spec}': {len(out_of_range)} value(s) outside physical "
                                f"range [0, 1]: {out_of_range}. Check the corecon data."
                            )

                        # Duplicate detection: same spec in two YAML sections or listed twice
                        if dataset_spec in self._seen_datasets:
                            raise RuntimeError(
                                f"Dataset '{dataset_spec}' appears more than once in corecon_config.yaml. "
                                f"This would double-count its contribution to the log-likelihood."
                            )
                        self._seen_datasets.add(dataset_spec)

                        # Stessa entry di corecon presa con selettori diversi.  Il
                        # controllo sopra confronta la stringa intera, quindi
                        # "[ATON]" e "[CROC]" gli sembrano due dataset distinti,
                        # mentre sono gli stessi 18 QSO di Durovcikova et al. 2024
                        # confrontati con due suite di simulazione.  Lo stesso vale
                        # per [fixed_dz]/[variable_dz] di Totani 2014, i tre metodi
                        # di Davies 2026, i tre modelli IGM di Wold 2022, LF/ACF di
                        # Umeda 2025: in tutti i casi presenti nel dato, due label
                        # dello stesso dataset sono la stessa osservazione
                        # rianalizzata, non due misure indipendenti.
                        #
                        # Regola: due selezioni della stessa entry possono stare
                        # insieme solo se ALMENO UNA delle due dice 'indipendente',
                        # in qualunque ordine siano scritte nello YAML.  Con tre o
                        # piu' selezioni vale per ogni coppia.  (Prima contava solo
                        # la riga letta per seconda: invertire l'ordine delle
                        # sezioni faceva fallire una configurazione corretta.)
                        precedenti = self._seen_corecon_names.get(corecon_name, [])
                        # le selezioni gia' caricate con cui questa NON e' dichiarata
                        # indipendente: ne' lei ne' l'altra dicono 'indipendente'
                        in_conflitto = [spec for spec, indip in precedenti
                                        if not (indip or indip_req)]
                        if in_conflitto:
                            raise RuntimeError(
                                f"Dataset '{dataset_spec}': l'entry corecon '{corecon_name}' e' "
                                f"gia' stata caricata come '{in_conflitto[0]}'.  Sono selezioni "
                                f"diverse della STESSA osservazione, quindi sommarle la "
                                f"conterebbe piu' volte: tienine una sola.  Se invece sai che "
                                f"sono misure DISTINTE, dichiaralo aggiungendo 'indipendente' "
                                f"fra le parentesi quadre di una delle due."
                            ) #"Se questa entry e' gia' stata caricata e nessuna delle due righe dice indipendente, fermati."
                        #
                        if precedenti:
                            gia = ", ".join(f"'{spec}'" for spec, _ in precedenti)
                            print(f"  NOTA: '{dataset_spec}' carica un'altra selezione di "
                                  f"'{corecon_name}', gia' presa come {gia}.  Dichiarate "
                                  f"indipendenti a mano: vengono dallo stesso campione e "
                                  f"restano correlate.") #Hai dichiarato tu che sono misure distinte, quindi il pacchetto ti crede e la carica. Ti ricorda però che vengono dallo stesso campione.
                        # si registrano TUTTE le selezioni, ciascuna con il suo 'indipendente'
                        self._seen_corecon_names.setdefault(corecon_name, []).append(
                            (dataset_spec, bool(indip_req)))

                        # Validate that lower_lim/upper_lim flags in corecon match the declared YAML section
                        # If in your yaml you declare that something is a detection while it is a bound,
                        # This section will raise an error. Cioe' se per lower_lim e upper _limit ho FALSE e FALSE allora e' una detection, se ho TRUE e TRUE ho errore. e se ho un true o false e' ovvio
                        self._validate_corecon_flags(
                            dataset_spec, observable_type, lower_lim_arr, upper_lim_arr
                        ) 
                        # When working with the detections we need to read their error bars in order to build the Gaussian likelihood.
    
                        if observable_type == "HII_fraction":
                            # La likelihood e' una split normal: basta che UNO dei due
                            # lati sia utilizzabile perche' il punto porti informazione.
                            ok_up = np.isfinite(err_up_arr) & (err_up_arr > 0)
                            ok_dn = np.isfinite(err_dn_arr) & (err_dn_arr > 0)
                            n_valid_err = int(np.sum(ok_up | ok_dn)) #Conta le righe con almeno una barra buona (| = "o"). Nell'esempio sono 2: la terza non ha barre. Basta un lato, perché la gaussiana asimmetrica usa la barra alta se il modello sta sopra il dato e quella bassa se sta sotto.
                            if n_valid_err == 0: #Se nessuna riga ha una barra utilizzabile, il dataset non può entrare in una gaussiana, e il pacchetto si ferma.
                                raise RuntimeError(
                                    f"Dataset '{dataset_spec}' in corecon '{corecon_key}' has no finite positive "
                                    f"error bars on either side (err_up, err_down). "
                                    f"Cannot use it in a Gaussian likelihood."
                                )
                            # Punti con un lato degenere: NON vengono scartati, quel lato
                            # resta semplicemente libero (vedi split_normal_chi2).  Si
                            # avvisa perche' e' una scelta, non un dettaglio.
                            #^ vuol dire "uno o l'altro, ma non entrambi": sono le righe con una sola barra (nell'esempio, la seconda). 
                            # Non vengono scartate, ma il lato senza barra resta libero, cioè non penalizza. 
                            # Succede ai dati appoggiati al bordo fisico, come i punti di Umeda 2026 con x_HII = 0: sotto lo zero non si può andare, quindi da quel lato non c'è barra.
                            #  Il pacchetto stampa una NOTE, perché è una scelta.
                            one_sided = np.where(ok_up ^ ok_dn)[0] 
                            if len(one_sided) > 0:
                                detail = ", ".join(
                                    f"z={z_arr[i]:g} value={val_arr[i]:g} "
                                    f"({'err_up' if not ok_up[i] else 'err_down'}=0)"
                                    for i in one_sided)
                                print(f"    - NOTE: '{dataset_spec}': {len(one_sided)} point(s) with a "
                                      f"one-sided error bar; that side is left unconstrained "
                                      f"(physical boundary of x_HII). [{detail}]")


                            #n_asym conta le righe con barre diverse sopra e sotto.
                            n_asym = int(np.sum(~np.isclose(err_up_arr, err_dn_arr)))
                            print(f"    - Dataset '{dataset_spec}': DETECTION "
                                  f"({len(z_arr)} points, {n_valid} finite values, {n_valid_err} usable "
                                  f"error bars, {n_asym} asymmetric -> split normal).")
                        self.analyses.append({
                            "type": observable_type, #mixed_fraction, detection, upper or lower bound
                            "name": dataset_spec, # esempio: Bolan et al. 2022 [CL=0.68]
                            "z": z_arr, #[6.7, 7.6],
                            "values": val_arr, # [0.75, 0.17]
                            "errors": err_up_arr, #valore della barra alta
                            "errors_down": err_dn_arr, #valore della barra bassa
                            "err_z_left": err_z_left,
                            "err_z_right": err_z_right, #danno la finestra in redshift
                            "lower_lim": lower_lim_arr,
                            "upper_lim": upper_lim_arr,
                            "limit_cl": cl_arr,
                            # sigma della mezza gaussiana di ogni limite: dipende
                            # solo dal dato e dal suo CL, non dal modello, quindi
                            # si calcola una volta qui e non a ogni passo.
                            "limit_sigma": self._limit_sigma(
                                dataset_spec, val_arr, lower_lim_arr,
                                upper_lim_arr, cl_arr,
                                err_up=err_up_arr, err_down=err_dn_arr,
                                err_up2=err_up2_arr, err_down2=err_dn2_arr,
                                integral_errxhi_95=err95_req), #la sigma si calcola con questa funzione visto che io immetto il valore del confidence level
                            "limit_sigma_origine": self._ultima_origine,
                            # errore sul limite, sotto/sopra (in x_HII): NaN
                            # dove il limite non ne ha
                            "limit_err_lo": self._ultimo_errore_limite[0],
                            "limit_err_hi": self._ultimo_errore_limite[1],
                        })
                        if observable_type != "HII_fraction":
                            print(self._riassunto_limiti(self.analyses[-1], err95_req))
                            self._controlla_limiti_con_errore_al_bordo(self.analyses[-1])
                        self._print_sigma_table(self.analyses[-1])
                    else:
                        print(f"    - WARNING: Dataset '{corecon_name}' not found in corecon '{corecon_key}'")
                        raise RuntimeError(
                            f"CRITICAL ERROR loading corecon data for '{corecon_key}': "
                            f"Dataset '{corecon_name}' not found"
                        )
            else:
                raise RuntimeError(
                    f"Section '{observable_type}' in {self.corecon_config_file} must be a "
                    f"YAML list, got {type(datasets).__name__}. This is almost always an "
                    f"indentation typo: every dataset name must be on its own line, "
                    f"preceded by '- '. Example:\n"
                    f"  {observable_type}:\n"
                    f"    - Greig et al. 2022\n"
                    f"    - Jung et al. 2020\n"
                    f"Refusing to continue: a malformed section would silently drop every "
                    f"dataset it contains from the likelihood."
                )























    @staticmethod
    def _parse_dataset_spec(dataset_spec):
        """
        Parse a dataset specification string into name, axis labels and CL.

        Dentro le parentesi quadre convivono due selettori diversi, perche' in
        corecon ci sono due modi diversi in cui lo stesso dataset contiene piu'
        righe:

          - una LABEL seleziona su una colonna extra di axes (il modello IGM di
            Totani 2014, l'indice spettrale di Fausey, la realizzazione di
            Durovcikova);
          - CL=x seleziona su limit_confidence_level, che NON sta in axes ma e'
            un campo a parte (le due righe di Totani 2006, Mason 2019, Bolan
            2022: la stessa misura scritta al 68% e al 95%).

        Si distinguono dal prefisso 'CL=', quindi i due si possono combinare e
        l'ordine fra loro non conta.

        Il quarto selettore e' 'indipendente', che non prende un valore.  Di
        norma la stessa entry corecon non si puo' caricare due volte con label
        diverse, perche' quasi sempre sono la stessa osservazione rianalizzata
        e sommarle la conterebbe due volte.  'indipendente' dichiara che in
        QUESTO caso le due label sono misure distinte e disattiva la guardia.
        Serve a Umeda et al. 2025, che in Kageura et al. compare sia fra i
        vincoli da clustering di LAE ([ACF]) sia fra quelli da funzione di
        luminosita' ([LF]): in Fig. 1 sono due categorie di legenda separate,
        2 pentagoni viola e 4 croci ciano.  Resta vero che vengono dallo stesso
        campione e sono correlate: e' una scelta, non un fatto.

        Formati:
        Il terzo selettore e' 'joint', che non prende un valore: sceglie la riga
        di vincolo congiunto (joint_constraint=True) invece delle misure singole.
        Serve a Greig et al. 2022.

        Formati:
          "Nome"                          -> ("Nome", None, None, False)
          "Nome [LAB]"                    -> ("Nome", ["LAB"], None, False)
          "Nome [CL=0.68]"                -> ("Nome", None, 0.68, False)
          "Nome [LAB, CL=0.68]"           -> ("Nome", ["LAB"], 0.68, False)
          "Nome [joint]"                  -> ("Nome", None, None, True)
          "Nome [LAB, indipendente]"      -> ("Nome", ["LAB"], None, False, True)

        Esempi:
          "Durovcikova et al. 2024 [ATON]"      -> (..., ["ATON"], None)
          "Totani et al. 2014 [fixed_dz]"       -> (..., ["fixed_dz"], None)
          "Fausey et al. 2024 [0.60, McQuinn et al. 2008]"
                                                -> (..., ["0.60", "McQuinn et al. 2008"], None)
          "Totani et al. 2006 [CL=0.68]"        -> (..., None, 0.68, False)
          "Wold et al. 2022 [McQuinn et al. (2007), CL=0.68]"
                                                -> (..., ["McQuinn et al. (2007)"], 0.68, False)
          "Greig et al. 2022 [joint]"           -> (..., None, None, True)

        I limiti pubblicati con un errore sul limite (Zhu 2022, Davies 2026)
        NON hanno bisogno di un selettore: l'integrale su err_up / err_down si
        fa sempre, da solo (vedi _limit_sigma).

        Il quinto selettore e' 'integral_errxhi_95': l'integrale si fa sugli
        errori al 95% (err_up2 / err_down2) invece che su quelli a 1 sigma.
        Vale per i limiti che hanno entrambe le versioni (Davies 2026, Tab. 3:
        "+1 sigma (+2 sigma)").  Vedi _limit_sigma.

        Returns (corecon_name, labels|None, cl|None, joint, indipendente,
                 integral_errxhi_95).
        """
        match = re.match(r"^(.+?)\s*\[([^\]]+)\]\s*$", dataset_spec)
        if not match:
            return dataset_spec, None, None, False, False, False

        name = match.group(1).strip()
        labels, cl, joint, indipendente, err95 = [], None, False, False, False
        for token in match.group(2).split(","):
            token = token.strip()
            if token.lower() == "joint":
                joint = True
                continue
            if token.lower() == "indipendente":
                indipendente = True
                continue
            if token.lower() == "limite_1sigma":
                raise RuntimeError(
                    f"Dataset '{dataset_spec}': 'limite_1sigma' non esiste piu'.  I "
                    f"limiti con errore si integrano sempre su err_up / err_down "
                    f"(o su err_up2 / err_down2 con 'integral_errxhi_95').")
            if token.lower() == "integral_errxhi_95":
                err95 = True
                continue
            if token.lower() == "bound_incerto":
                raise RuntimeError(
                    f"Dataset '{dataset_spec}': 'bound_incerto' non esiste piu'.  "
                    f"Per i limiti con errore sul limite l'integrale si fa sempre, "
                    f"togli l'etichetta dallo YAML.")
            m_cl = re.match(r"^CL\s*=\s*(.+)$", token, flags=re.IGNORECASE)
            if m_cl:
                if cl is not None:
                    raise RuntimeError(
                        f"Dataset '{dataset_spec}': CL= indicato piu' di una volta. "
                        f"Si sceglie una riga sola.")
                try:
                    cl = float(m_cl.group(1).strip())
                except ValueError:
                    raise RuntimeError(
                        f"Dataset '{dataset_spec}': CL='{m_cl.group(1).strip()}' non e' "
                        f"un numero.  Atteso un confidence level come CL=0.68.")
                if not (0.0 < cl < 1.0):
                    raise RuntimeError(
                        f"Dataset '{dataset_spec}': CL={cl} fuori da (0,1).  Il confidence "
                        f"level e' una probabilita': 0.68, non 68.")
            else:
                labels.append(token)

        return name, (labels or None), cl, joint, indipendente, err95

    @staticmethod
    def _ensure_required_label(corecon_name, label, data_obj, dataset_spec):
        """
        Enforce explicit label selection for datasets that have multiple physical models.

        Due controlli, uno generico e uno con messaggi su misura:

        1. GENERICO: se axes ha colonne extra oltre al redshift e lo YAML non da'
           nessuna label, si solleva un errore con le combinazioni disponibili.
           Prima questo caso non era coperto per i dataset fuori dalla lista qui
           sotto, e il risultato era
               ValueError: could not convert string to float: 'ATON'
           da np.array(data_obj.axes, dtype=float) nel ramo "standard dataset" —
           un errore che non dice all'utente cosa fare.  Essendo ricavato da
           axes, questo controllo vale per tutti i dataset, anche quelli che non
           abbiamo ancora guardato.

        2. LISTA: per alcuni dataset si pretende un numero PRECISO di label,
           perche' una label sola matcherebbe l'ultima colonna lasciando aperta
           la prima (Fausey ha indice spettrale e modello IGM, e scegliere solo
           il modello IGM selezionerebbe tre righe).
        """
        # --- 1. controllo generico, ricavato dal dato
        axes = getattr(data_obj, "axes", [])
        n_extra = max((len(a) - 1 for a in axes if hasattr(a, "__len__")), default=0)
        if n_extra > 0 and label is None:
            combos = sorted({", ".join(str(a[k]) for k in range(1, len(a)))
                             for a in axes if hasattr(a, "__len__")})
            descr = list(getattr(data_obj, "dimensions_descriptors", []))
            cols = ", ".join(str(d) for d in descr[1:]) or f"{n_extra} colonna/e extra"
            raise RuntimeError(
                f"Dataset '{dataset_spec}': axes non contiene solo il redshift, ha anche "
                f"{cols}.  Va specificato fra parentesi quadre quale riga si vuole, "
                f"altrimenti non e' definito quale dato usare.  Combinazioni disponibili: "
                f"{combos}.  Esempio: '{corecon_name} [{combos[0]}]'."
            )

        # --- 1b. il caso opposto: label data dove axes non ha colonne extra.
        # Senza questo si finisce in _extract_realization, che fa entry[0] su un
        # float e muore con "TypeError: 'float' object is not subscriptable".
        # E' l'errore che si prende scrivendo "Mason et al. 2019 [0.68]", cioe'
        # la sintassi del formato VECCHIO, quando il confidence level era una
        # colonna di axes: su questo branch e' il campo limit_confidence_level e
        # si seleziona con [CL=0.68].  Vale la pena riconoscerlo e dirlo, perche'
        # e' lo sbaglio naturale di chi aggiorna un YAML scritto prima.
        if n_extra == 0 and label is not None:
            hint = ""
            cls_avail = sorted({
                f"{c:g}" for c in np.atleast_1d(np.asarray(
                    getattr(data_obj, "limit_confidence_level", np.nan),
                    dtype=float).ravel()) if np.isfinite(c)})
            for lab in label:
                try:
                    if f"{float(lab):g}" in cls_avail:
                        hint = (f"  '{lab}' sembra un confidence level: in quel caso la "
                                f"sintassi e' '{corecon_name} [CL={lab}]', perche' il CL non "
                                f"sta in axes ma nel campo limit_confidence_level.")
                        break
                except ValueError:
                    pass
            raise RuntimeError(
                f"Dataset '{dataset_spec}': e' stata indicata una label "
                f"{label} ma axes contiene solo il redshift, quindi non c'e' nessuna "
                f"colonna su cui selezionare."
                + (hint or (f"  Confidence level disponibili: {cls_avail}."
                            if cls_avail else ""))
            )

        REQUIRED_LABEL_DATASETS = {
            "Totani et al. 2014": {
                "n_labels": 1,
                "description": "IGM model",
                "example": "Totani et al. 2014 [fixed_dz]",
            },
            "Fausey et al. 2024": {
                "n_labels": 2,
                "description": "spectral index and IGM model",
                "example": "Fausey et al. 2024 [free, McQuinn et al. 2008]",
            },
            "Umeda et al. 2025": {
                "n_labels": 1,
                "description": "method (LF or ACF)",
                "example": "Umeda et al. 2025 [LF]",
            },
            # Mason et al. 2019 stava qui con n_labels=1 e description
            # "confidence level (0.68 or 0.95)", da quando il CL era una colonna
            # di axes e si selezionava come una label.  Ora e' il campo
            # limit_confidence_level, Mason non ha nessuna label, e quella voce
            # rifiutava proprio la sintassi corretta "[CL=0.68]".  La scelta
            # resta obbligatoria, ma la impone _select_cl_rows, che la ricava dal
            # dato invece di averla scritta a mano qui: vale per tutti i dataset
            # con piu CL allo stesso z, non solo per quelli in questa lista.
        }

        if corecon_name not in REQUIRED_LABEL_DATASETS:
            return

        info = REQUIRED_LABEL_DATASETS[corecon_name]
        if label is not None and len(label) == info["n_labels"]:
            return

        # Collect all available label combinations from the axes to help the user
        # understand which explicit labels are required in the YAML for disambiguation.
        available = []
        for entry in getattr(data_obj, "axes", []):
            if hasattr(entry, "__len__") and len(entry) > info["n_labels"]:
                combo = ", ".join(str(entry[k]) for k in range(1, info["n_labels"] + 1))
                available.append(combo)
        available = sorted(set(available))

        raise RuntimeError(
            f"Dataset '{dataset_spec}' is ambiguous. For '{corecon_name}' you must specify "
            f"an explicit {info['description']} label in YAML ({info['n_labels']} label(s) required). "
            f"Available combinations: {available}. "
            f"Example: '{info['example']}'."
        )

    @staticmethod
    def _extract_realization(data_obj, label):
        """
        Extract data for a single realization or type label from a multi-entry corecon dataset.

        In datasets like 'Durovcikova et al. 2024' or 'Davies et al. 2026', each entry in
        data_obj.axes is a tuple (redshift, label) instead of a plain redshift.
        This method filters to the requested label (realization name or type like
        'threshold', 'mixture', 'negative') and returns flat arrays.

        Returns: z_arr, val_arr, err_up_arr, err_down_arr, err_z_left, err_z_right,
                 lower_lim_arr, upper_lim_arr, cl_arr
        """
        z_list, val_list, err_up_list, err_left_list, err_right_list = [], [], [], [], []
        err_down_list = []
        err_up2_list, err_down2_list = [], []
        lower_lim_list, upper_lim_list, cl_list = [], [], []

        # Confidence level riga per riga; NaN dove corecon non lo dichiara.
        cl_full = ReioLike._get_cl_array(data_obj, len(data_obj.axes))
        # Barre al 95% (due code), NaN dove assenti.
        eu2_full = ReioLike._get_err2_array(data_obj, "err_up2", len(data_obj.axes))
        ed2_full = ReioLike._get_err2_array(data_obj, "err_down2", len(data_obj.axes))

        # err_down e' la barra d'errore INFERIORE sul valore (non una larghezza
        # in z).  Se manca si ricade su err_up, cioe' sul caso simmetrico.
        has_err_down  = hasattr(data_obj, "err_down")  and data_obj.err_down  is not None
        has_err_left  = hasattr(data_obj, "err_left")  and data_obj.err_left  is not None
        has_err_right = hasattr(data_obj, "err_right") and data_obj.err_right is not None
        has_lower_lim = hasattr(data_obj, "lower_lim") and data_obj.lower_lim is not None
        has_upper_lim = hasattr(data_obj, "upper_lim") and data_obj.upper_lim is not None

        # labels is a list of strings to match against the LAST n columns (after z)
        # e.g. label=["McQuinn et al. 2008"] matches the last extra column regardless of spectral index
        n_labels = len(label)
        available = set()
        for i, entry in enumerate(data_obj.axes):
            z = entry[0]
            entry_labels = [str(entry[k]) for k in range(1, len(entry))]
            combo = ", ".join(entry_labels)
            available.add(combo)
            # Match labels against the last n_labels extra columns
            if entry_labels[-n_labels:] == label:
                z_list.append(float(z))
                val_list.append(data_obj.values[i])
                err_up_list.append(data_obj.err_up[i])
                err_down_list.append(
                    data_obj.err_down[i] if has_err_down else data_obj.err_up[i])
                if has_err_left:
                    try:
                        el = data_obj.err_left[i]
                        err_left_list.append(el[0] if hasattr(el, "__len__") else float(el))
                    except Exception:
                        err_left_list.append(0.0)
                else:
                    err_left_list.append(0.0)
                if has_err_right:
                    try:
                        er = data_obj.err_right[i]
                        err_right_list.append(er[0] if hasattr(er, "__len__") else float(er))
                    except Exception:
                        err_right_list.append(0.0)
                else:
                    err_right_list.append(0.0)
                lower_lim_list.append(bool(data_obj.lower_lim[i]) if has_lower_lim else False)
                upper_lim_list.append(bool(data_obj.upper_lim[i]) if has_upper_lim else False)
                cl_list.append(float(cl_full[i]))
                err_up2_list.append(float(eu2_full[i]))
                err_down2_list.append(float(ed2_full[i]))

        if not z_list:
            label_str = ", ".join(label)
            raise RuntimeError(
                f"Label '[{label_str}]' not found in dataset. "
                f"Available label combinations: {sorted(available)}"
            )

        return (
            np.array(z_list,       dtype=float),
            np.array(val_list,     dtype=float),
            np.array(err_up_list,   dtype=float),
            np.array(err_down_list, dtype=float),
            np.nan_to_num(np.array(err_left_list,  dtype=float), nan=0.0),
            np.nan_to_num(np.array(err_right_list, dtype=float), nan=0.0),
            np.array(lower_lim_list, dtype=bool),
            np.array(upper_lim_list, dtype=bool),
            np.array(cl_list, dtype=float),
            np.array(err_up2_list, dtype=float),
            np.array(err_down2_list, dtype=float),
        )

    @staticmethod
    def _validate_corecon_flags(dataset_spec, observable_type, lower_lim_arr, upper_lim_arr):
        """
        Validates that the lower_lim / upper_lim flags stored in corecon are consistent
        with the YAML section the user placed the dataset in.

        Rules:
          lower_lim=False, upper_lim=False  ->  detection   (HII_fraction)
          upper_lim=True,  lower_lim=False  ->  upper limit (upper_limit_HII_fraction)
          lower_lim=True,  upper_lim=False  ->  lower limit (lower_limit_HII_fraction)
          lower_lim=True,  upper_lim=True   ->  INCONSISTENT (error in the dataset)

        If ALL points agree with the declared type, print a confirmation.
        If SOME points disagree, print a warning with the counts.
        If ALL points disagree with the declared type, raise a RuntimeError.
        """
        _YAML_TO_TYPE = {
            "HII_fraction":               "detection",
            "upper_limit_HII_fraction":   "upper limit",
            "lower_limit_HII_fraction":   "lower limit",
            "mixed_HII_fraction":         "mixed",
        }
        declared = _YAML_TO_TYPE.get(observable_type, observable_type)

        # For mixed datasets, just verify there really are multiple flag types
        if observable_type == "mixed_HII_fraction":
            n_lower  = int(np.sum(lower_lim_arr & ~upper_lim_arr))
            n_upper  = int(np.sum(upper_lim_arr & ~lower_lim_arr))
            n_detect = int(np.sum(~lower_lim_arr & ~upper_lim_arr))
            n_types  = sum([n_lower > 0, n_upper > 0, n_detect > 0])
            if n_types < 2:
                suggestion = "lower_limit_HII_fraction" if n_lower > 0 else \
                             "upper_limit_HII_fraction" if n_upper > 0 else "HII_fraction"
                print(
                    f"    - WARNING: Dataset '{dataset_spec}' is declared as 'mixed' "
                    f"but all {len(lower_lim_arr)} point(s) are actually '{suggestion}'. "
                    f"Consider moving it to '{suggestion}'."
                )
            else:
                flag_parts = []
                if n_lower  > 0: flag_parts.append(f"{n_lower} lower limit(s)")
                if n_upper  > 0: flag_parts.append(f"{n_upper} upper limit(s)")
                if n_detect > 0: flag_parts.append(f"{n_detect} detection(s)")
                print(f"    - Flag check OK: mixed dataset with {', '.join(flag_parts)} — routing per-point.")
            return

        n = len(lower_lim_arr)
        # Check for internally inconsistent points (both flags True)
        both_true = lower_lim_arr & upper_lim_arr
        if np.any(both_true):
            raise RuntimeError(
                f"Dataset '{dataset_spec}': {np.sum(both_true)} point(s) have BOTH "
                f"lower_lim=True AND upper_lim=True. This is inconsistent in the corecon data."
            )

        # Classify each point according to corecon flags
        n_lower = int(np.sum(lower_lim_arr & ~upper_lim_arr))
        n_upper = int(np.sum(upper_lim_arr & ~lower_lim_arr))
        n_detect = int(np.sum(~lower_lim_arr & ~upper_lim_arr))

        # Determine implied type from flags (majority or unanimous)
        flag_summary = []
        if n_lower  > 0: flag_summary.append(f"{n_lower} lower limit(s)")
        if n_upper  > 0: flag_summary.append(f"{n_upper} upper limit(s)")
        if n_detect > 0: flag_summary.append(f"{n_detect} detection(s)")
        flag_str = ", ".join(flag_summary)

        # Map declared observable_type to expected flag pattern
        if observable_type == "HII_fraction":
            n_matching = n_detect
            expected_flag_str = "lower_lim=False, upper_lim=False (detection)"
        elif observable_type == "upper_limit_HII_fraction":
            n_matching = n_upper
            expected_flag_str = "upper_lim=True, lower_lim=False"
        elif observable_type == "lower_limit_HII_fraction":
            n_matching = n_lower
            expected_flag_str = "lower_lim=True, upper_lim=False"
        else:
            # Unknown type; skip validation
            return

        if n_matching == n:
            print(f"    - Flag check OK: all {n} point(s) match declared type '{declared}'.")
        elif n_matching == 0:
            # Suggest the right section
            if n_lower == n:
                suggestion = "lower_limit_HII_fraction"
            elif n_upper == n:
                suggestion = "upper_limit_HII_fraction"
            elif n_detect == n:
                suggestion = "HII_fraction"
            else:
                suggestion = "(mixed — split by label first)"
            raise RuntimeError(
                f"Dataset '{dataset_spec}' is declared as '{declared}' in the YAML, "
                f"but its corecon flags say: {flag_str}. "
                f"Expected flags: {expected_flag_str}. "
                f"Suggested YAML section: '{suggestion}'."
            )
        else:
            # Flag misti dentro una sezione a tipo unico.  Prima qui c'era solo un
            # WARNING, e la conseguenza non era che il punto discordante venisse
            # trattato male: veniva PERSO.  Umeda et al. 2025 [LF] ha 3 detection
            # piu' un lower limit a z=5.7; un limite non ha barre d'errore
            # (err_up = err_down = 0), quindi la maschera di compute_gaussian_loglike
            # lo scartava e quel vincolo spariva dalla likelihood senza che nessun
            # numero lo dicesse.  Il confronto misurato: logL = -18.65 nella
            # sezione sbagliata contro -26.34 instradando per punto.
            #
            # La sezione giusta esiste gia' ('mixed_HII_fraction', che instrada
            # punto per punto sui flag), quindi qui si solleva un errore invece di
            # tirare avanti.  Non si instrada d'ufficio di proposito: se un domani
            # corecon cambia un flag su una riga, un errore all'avvio lo fa vedere,
            # un instradamento automatico cambierebbe il trattamento in silenzio.
            #
            # Riguarda 7 dataset su 50 in HII_fraction: Bolan 2022, Bosman 2022,
            # Nakane 2024, Schenker 2014, Spina 2024, Umeda 2025, Yang 2020b.
            odd = [i for i in range(n)
                   if (observable_type == "HII_fraction"
                       and (lower_lim_arr[i] or upper_lim_arr[i]))
                   or (observable_type == "upper_limit_HII_fraction"
                       and not upper_lim_arr[i])
                   or (observable_type == "lower_limit_HII_fraction"
                       and not lower_lim_arr[i])]
            raise RuntimeError(
                f"Dataset '{dataset_spec}' e' dichiarato '{declared}' ma i flag di corecon "
                f"dicono: {flag_str} ({n_matching}/{n} righe coincidono).  Le righe "
                f"discordanti sono {odd}.  In una sezione a tipo unico quelle righe non "
                f"verrebbero trattate come limiti: verrebbero SCARTATE, perche' un limite "
                f"non ha barre d'errore e la likelihood gaussiana le maschera.  Metti questo "
                f"dataset in 'mixed_HII_fraction', che instrada ogni punto in base ai suoi "
                f"flag, oppure separa le righe con una label."
            )

    # Nomi degli array, nell'ordine in cui initialize() li passa a
    # _apply_row_mask: servono solo al messaggio d'errore.
    _NOMI_ARRAY = ("z", "values", "err_up", "err_down", "lower_lim", "upper_lim",
                   "limit_confidence_level", "err_left", "err_right",
                   "err_up2", "err_down2")

    @staticmethod
    def _apply_row_mask(arrays, keep, dataset_spec="?"):
        """
        Applica la stessa selezione di righe a tutti gli array di un dataset.

        keep=None vuol dire "niente da filtrare".  Gli elementi None
        (err_left/err_right assenti) restano None.

        Esiste perche' gli array da tagliare sono undici e tagliarne dieci e'
        esattamente il tipo di errore che non da' un'eccezione: produce un
        disallineamento fra z e il resto, cioe' dati attribuiti al redshift
        sbagliato.

        Per lo stesso motivo, PRIMA di tagliare si controlla che tutti gli array
        abbiano una riga per ogni z.  Se no ci si ferma: prima un array troppo
        corto veniva lasciato intero, senza tagliarlo, e i suoi valori finivano
        abbinati alle righe sbagliate senza nessun messaggio.
        """
        n = len(np.asarray(arrays[0]))
        for i, a in enumerate(arrays):
            if a is None:
                continue
            a = np.asarray(a)
            if a.ndim == 0 or len(a) != n:
                nome = (ReioLike._NOMI_ARRAY[i] if i < len(ReioLike._NOMI_ARRAY)
                        else f"array {i}")
                lung = "uno scalare" if a.ndim == 0 else f"{len(a)} valori"
                raise RuntimeError(
                    f"Dataset '{dataset_spec}': il campo '{nome}' ha {lung}, ma le "
                    f"righe (z) sono {n}.  Ogni campo deve avere un valore per riga: "
                    f"controlla il data file di corecon.")
        if keep is None:
            return arrays
        return [None if a is None else np.asarray(a)[keep] for a in arrays]

    @staticmethod
    def _lunghezza_campo(data_obj, campo, k, n):
        """
        Un campo di corecon ha un valore per riga (k = n) o uno solo per tutte
        (k = 1, gestito da chi chiama).  Qualsiasi altra lunghezza e' un data
        file scritto male: ci si ferma.  Prima il campo veniva troncato o
        riempito (NaN, False) in silenzio, e per limit_confidence_level il
        risultato era un "CL non dichiarato" che portava fuori strada.
        """
        if k != n:
            rif = str(getattr(data_obj, "reference", "") or "")[:70]
            raise RuntimeError(
                f"corecon: il campo '{campo}' ha {k} valori ma il dataset ha {n} "
                f"righe (axes).  Deve averne {n}, oppure uno solo per tutte.  "
                f"Dataset: {rif!r}.")

    @staticmethod
    def _get_joint_array(data_obj, n):
        """
        joint_constraint di corecon come array bool, o None se il campo manca.

        Oggi lo dichiara solo Greig et al. 2022, dove segna la riga che e' il
        prodotto delle likelihood dei quattro QSO: non e' una quinta misura, e'
        le altre quattro ricombinate.
        """
        val = getattr(data_obj, "joint_constraint", None)
        if val is None:
            return None
        arr = np.atleast_1d(np.asarray(val).ravel())
        if arr.size == 1:
            return np.full(n, bool(arr[0]))
        ReioLike._lunghezza_campo(data_obj, "joint_constraint", arr.size, n)
        return arr.astype(bool)

    @staticmethod
    def _select_joint_rows(dataset_spec, joint_req, joint_arr, z):
        """
        Sceglie fra il vincolo congiunto e le misure singole.

            "Greig et al. 2022 [joint]"  ->  solo la riga con joint_constraint=True
            "Greig et al. 2022"          ->  solo le righe con joint_constraint=False

        I due insiemi descrivono gli stessi oggetti: in Greig 2022 la riga a
        z=7.29 e' il prodotto delle likelihood dei quattro QSO a z=7.00, 7.09,
        7.51 e 7.54.  Caricarli insieme conta quei quattro QSO due volte, una
        volta singolarmente e una volta dentro il congiunto, e il congiunto ha il
        sigma piu' piccolo di tutti (0.13) proprio perche' li contiene tutti.

        Il default sono le misure singole: sono quattro vincoli a quattro redshift
        diversi, cioe' piu' informazione sulla forma di xe(z) del singolo punto
        congiunto, che sta al redshift medio.  Chi vuole il congiunto lo chiede.

        Sui dataset senza joint_constraint non fa nulla.
        """
        if joint_arr is None:
            if joint_req:
                raise RuntimeError(
                    f"Dataset '{dataset_spec}': richiesto [joint] ma questo dataset non ha "
                    f"il campo joint_constraint in corecon, quindi non ha un vincolo congiunto."
                )
            return None

        if joint_req:
            keep = np.nonzero(joint_arr)[0]
            if len(keep) == 0:
                raise RuntimeError(
                    f"Dataset '{dataset_spec}': richiesto [joint] ma nessuna riga ha "
                    f"joint_constraint=True."
                )
            print(f"    - [joint]: tenuta/e {len(keep)} riga/he di vincolo congiunto "
                  f"(z={np.round(np.asarray(z)[keep], 3).tolist()}), "
                  f"scartate {len(joint_arr) - len(keep)} misure singole.")
            return keep

        keep = np.nonzero(~joint_arr)[0]
        if len(keep) == 0:
            raise RuntimeError(
                f"Dataset '{dataset_spec}': tutte le righe sono joint_constraint=True, "
                f"quindi non ci sono misure singole.  Usa '[joint]' per prendere il congiunto."
            )
        n_joint = int(np.sum(joint_arr))
        if n_joint:
            print(f"    - vincolo congiunto presente a z="
                  f"{np.round(np.asarray(z)[joint_arr], 3).tolist()}, NON caricato "
                  f"(sarebbe le misure singole ricombinate). Per usarlo invece delle "
                  f"singole: '{dataset_spec} [joint]'.")
        return keep

    @staticmethod
    def _select_cl_rows(dataset_spec, cl_req, z, cl, lower_lim, upper_lim):
        """
        Seleziona le righe-limite con il confidence level richiesto nello YAML.

        Il problema che risolve: corecon scrive la STESSA misura su piu' righe,
        una per confidence level.  Totani 2006 ha un solo GRB ma due righe,
        x_HII > 0.83 al 68% e x_HII > 0.40 al 95%; Mason 2019 e Bolan 2022
        uguale.  Caricarle tutte conta la stessa osservazione due volte: su
        Totani 2006 le due righe danno quasi lo stesso chi quadro, quindi il
        contributo raddoppia.  Le label fra parentesi non bastano perche'
        guardano axes, e il confidence level non sta in axes.

        Regole:
          - le righe che NON sono limiti (detection) passano sempre: in un
            dataset misto come Bolan 2022 la detection a z=7.6 non ha CL, e
            filtrare alla cieca la butterebbe via;
          - con CL=x nello YAML si tengono solo le righe-limite a quel CL;
          - senza CL=x, se due righe-limite allo stesso z hanno CL diversi la
            scelta e' ambigua e si solleva un errore invece di sommarle.

        Ritorna gli indici da tenere, o None se non c'e' niente da filtrare.
        """
        n = len(z)
        is_lim = np.asarray(lower_lim, dtype=bool) | np.asarray(upper_lim, dtype=bool) #quali righe sono limiti? vale sia per upper che per lower
        cl = np.asarray(cl, dtype=float)

        if cl_req is not None: #e' specificato  quanto vale il limite? si? allora usa quello.
            # Tiene le detection piu' le righe-limite al CL richiesto.
            match = is_lim & np.isclose(cl, cl_req, rtol=0, atol=1e-9)
            keep = np.nonzero(~is_lim | match)[0] #questo mi serve per tenere le righe giuste
            if not np.any(match): #se il limite richiesto non c'e' allora partono messaggi di errore, con suggerimenti di come correggere l'errore.
                available = sorted({f"{c:g}" for c in cl[is_lim] if np.isfinite(c)}) #scrive quali cl sono disponibili se indicato un cl sbagliato.
                raise RuntimeError(
                    f"Dataset '{dataset_spec}': CL={cl_req:g} non presente fra i limiti. "
                    f"Confidence level disponibili: {available or 'nessuno (il dato non li dichiara)'}."
                )
            n_drop = n - len(keep)
            if n_drop:
                print(f"    - CL={cl_req:g} richiesto: tenute {int(np.sum(match))} riga/he-limite "
                      f"(scartate {n_drop} ad altri CL).")
            return keep

        # Nessun CL richiesto: si controlla che non ci sia ambiguita'.
        for z_val in np.unique(z[is_lim]) if np.any(is_lim) else []:
            same = np.nonzero(is_lim & (z == z_val))[0]
            if len(same) < 2:
                continue
            cls_here = sorted({f"{c:g}" for c in cl[same] if np.isfinite(c)})
            if len(cls_here) > 1:
                raise RuntimeError(
                    f"Dataset '{dataset_spec}': a z={z_val:g} ci sono {len(same)} righe-limite "
                    f"con confidence level diversi ({', '.join(cls_here)}).  E' la stessa misura "
                    f"scritta a piu' CL: sommarle la conterebbe piu' volte.  Scegline uno nello "
                    f"YAML, per esempio \"{dataset_spec} [CL={cls_here[0]}]\"."
                )
            print(f"    - NOTA: a z={z_val:g} ci sono {len(same)} righe-limite con lo stesso "
                  f"CL ({cls_here[0] if cls_here else 'non dichiarato'}): non sono distinguibili "
                  f"per CL, verifica a mano che siano misure indipendenti.")
        return None

    def _limit_sigma(self, dataset_spec, values, lower_lim, upper_lim, cl,
                     err_up=None, err_down=None,
                     err_up2=None, err_down2=None, integral_errxhi_95=False):
        """
        sigma della mezza gaussiana di ogni riga-limite.  NaN sulle detection.

            upper limit su x_HII (bordo in 0):  sigma = g / z_CL
            lower limit su x_HII (bordo in 1):  sigma = (1 - g) / z_CL

        con z_CL = Phi^-1((1 + CL)/2) (z_for_confidence_level): il CL e' l'area
        della mezza gaussiana fra il bordo e il limite, 2 Phi(z_CL) - 1 = CL,
        cioe' 68% = 0.9945 sigma, 95% = 1.96 sigma, 99.7% = 2.97 sigma.  Vedi il
        commento in testa al modulo, anche per la lettura Phi^-1(CL) usata fino
        al 21 ott 2026 e poi abbandonata.

        Dove corecon non dichiara il CL si usa DEFAULT_LIMIT_CL, avvisando: e'
        un'assunzione sul significato statistico del dato, non un dettaglio.

        LIMITE CON ERRORE (nessun selettore: automatico).  Se la riga-limite
        ha err_up / err_down, quello e' l'errore sul valore del limite, e si
        integra su di esso (vedi il blocco "Limiti pubblicati con un errore
        sul limite stesso" in testa al modulo):

            lower limit su x_HII (x >= B):  L = F(x)
            upper limit su x_HII (x <= B):  L = 1 - F(x)

        B split-normal con err_down sotto ed err_up sopra; con errore
        simmetrico e: Phi((x - g)/e) e Phi((g - x)/e).  Il CL della riga in
        questo caso non entra.  Le due larghezze vanno in
        self._ultimo_errore_limite; sigma = lato del vincolo serve solo alle
        stampe.  La forma la applicano le compute_*_loglike guardando
        l'origine 'limite con errore'.

        ERRORI AL 95% (selettore 'integral_errxhi_95').  Stesso integrale, ma
        su err_up2 / err_down2, che sono la semiampiezza dell'intervallo al
        95% sul limite.  La larghezza della gaussiana e' quindi

            sigma = err2 / z_95,   z_95 = Phi^-1((1 + 0.95)/2) = 1.95996

        perche' fra -1.96 sigma e +1.96 sigma c'e' il 95% dell'area.  E'
        un intervallo +- attorno al valore, su una gaussiana intera (l'errore
        sul limite, non la mezza gaussiana del limite).  Esempio, Davies
        [negative] a z = 5.056: err2 = 0.036, sigma = 0.0184 (a 1 sigma il
        dato da' 0.017).

        Registra in self._ultima_origine, riga per riga, da dove viene sigma:
        'CL', 'CL assunto', 'limite con errore', oppure '' per
        le detection.
        """
        n = len(values)
        sigma = np.full(n, np.nan)
        origine = np.array([""] * n, dtype=object)
        # errore sul limite, sotto/sopra, in x_HII (solo limiti che ce l'hanno)
        err_lo = np.full(n, np.nan)
        err_hi = np.full(n, np.nan)
        missing = []

        for i in range(n):
            is_lower = bool(lower_lim[i])
            is_upper = bool(upper_lim[i])
            if not (is_lower or is_upper):
                continue                      # detection: sigma viene da err_up/err_down

            g = float(values[i])
            d = (1.0 - g) if is_lower else g

            # errore pubblicato sul limite: e' lui che decide, non lo YAML
            def _e(arr):
                if arr is None:
                    return np.nan
                v = float(arr[i])
                return v if (np.isfinite(v) and v > 0) else np.nan
            if integral_errxhi_95:
                # intervallo +- al 95% sulla gaussiana del limite: Phi^-1(0.975)
                z95 = float(norm.ppf(0.975))                       # 1.95996
                lo, hi = _e(err_down2) / z95, _e(err_up2) / z95
            else:
                lo, hi = _e(err_down), _e(err_up)
            lo = hi if not np.isfinite(lo) else lo     # un lato solo: si specchia
            hi = lo if not np.isfinite(hi) else hi
            if integral_errxhi_95 and not np.isfinite(lo):
                raise RuntimeError(
                    f"'{dataset_spec}' riga {i}: 'integral_errxhi_95' richiesto ma il "
                    f"limite non ha err_up2 / err_down2 in corecon.")

            if np.isfinite(lo):
                err_lo[i], err_hi[i] = lo, hi
                # sigma "rappresentativa" (stampe e diagnostica): lato del vincolo
                sigma[i] = lo if is_lower else hi
                origine[i] = "limite con errore"
                continue

            cl_i = float(cl[i]) if cl is not None else np.nan
            if not np.isfinite(cl_i):
                cl_i = DEFAULT_LIMIT_CL
                missing.append(i)
                origine[i] = "CL assunto"
            else:
                origine[i] = "CL"

            z_cl = z_for_confidence_level(cl_i)
            # d <= 0 vuol dire un limite al bordo fisico: non vincola nulla, e
            # sigma = 0 farebbe esplodere il chi quadro.  Resta NaN, e le
            # maschere di validita' a valle lo scartano.
            if d > 0:
                sigma[i] = d / z_cl

        if missing:
            print(f"    - WARNING: '{dataset_spec}': {len(missing)} limite/i senza "
                  f"limit_confidence_level in corecon (righe {missing}). "
                  f"Assumo CL={DEFAULT_LIMIT_CL} (z={z_for_confidence_level(DEFAULT_LIMIT_CL):.4f}). "
                  f"Se il paper dice altro, il sigma e' sbagliato: aggiungi il campo "
                  f"al data file di corecon.")
        self._ultima_origine = origine
        self._ultimo_errore_limite = (err_lo, err_hi)
        return sigma

    @staticmethod
    def _riassunto_limiti(analysis, integral_errxhi_95=False):
        """
        Una riga di log per un dataset di limiti (o misto): come viene trattata
        DAVVERO ogni riga, letto da limit_sigma_origine, cioe' da quello che ha
        fatto _limit_sigma.

        Prima la riga si costruiva guardando solo se il dato aveva un CL, e per
        Davies et al. 2026 scriveva "nessun CL nel dato, assunto 0.95": falso,
        perche' Davies ha l'incertezza sul limite (Tab. 3: "the central value of
        the corresponding dark fraction along with its +1 sigma (+2 sigma)
        uncertainty") e si integra su quella, senza nessun CL.  Per Zhu 2022
        scriveva "CL dal dato: 0.68", anche quello mai usato.  I dataset misti
        non avevano nessuna riga.
        """
        tipo = {"upper_limit_HII_fraction": "UPPER LIMIT",
                "lower_limit_HII_fraction": "LOWER LIMIT",
                "mixed_HII_fraction": "MIXED"}.get(analysis["type"], analysis["type"])
        origine = np.asarray(analysis["limit_sigma_origine"], dtype=object)
        cl = np.asarray(analysis["limit_cl"], dtype=float)
        is_lim = np.asarray(analysis["lower_lim"]) | np.asarray(analysis["upper_lim"])

        parti = []
        n_det = int(np.sum(~is_lim))
        if n_det:
            parti.append(f"{n_det} detection (split normal)")
        n_err = int(np.sum(origine == "limite con errore"))
        if n_err:
            quali = ("err_down2/err_up2 al 95%" if integral_errxhi_95
                     else "err_down/err_up a 1 sigma")
            parti.append(f"{n_err} limite/i con incertezza sul limite: integrale su "
                         f"{quali}, il CL non entra")
        # un limite al bordo fisico (g = 0 o 1) ha sigma NaN: non vincola nulla
        bordo = is_lim & ~np.isfinite(np.asarray(analysis["limit_sigma"], dtype=float))
        m = (origine == "CL") & ~bordo
        if m.any():
            cls = ", ".join(sorted({f"{c:g}" for c in cl[m]}))
            parti.append(f"{int(m.sum())} limite/i a mezza gaussiana, CL dal dato {cls}")
        n_ass = int(np.sum((origine == "CL assunto") & ~bordo))
        if n_ass:
            parti.append(f"{n_ass} limite/i a mezza gaussiana, CL ASSUNTO "
                         f"{DEFAULT_LIMIT_CL:g} (assente nel dato)")
        n_tot = len(analysis["z"])
        if bordo.any():
            parti.append(f"{int(bordo.sum())} limite/i al bordo fisico, senza vincolo")
        return (f"    - Dataset '{analysis['name']}': {tipo} ({n_tot} points: "
                + "; ".join(parti) + ").")

    def _debug_points(self, analysis, mask, x_th, sigma, log_l, branch):
        """
        Stampa, a ogni chiamata, cosa e' entrato nel chi quadro punto per punto.

        A differenza di _print_sigma_table questa dipende dal modello, quindi va
        ricalcolata ogni volta: serve per i test di controllo a mano, non per le
        catene.  Si attiva con debug_points=True.

        mask seleziona le righe sopravvissute, e x_th/sigma/log_l sono GIA'
        mascherati: gli indici stampati sono quelli del dataset originale, cosi'
        si confrontano con la tabella dei sigma.
        """
        if not self.debug_points:
            return
        z    = np.asarray(analysis["z"])[mask]
        g    = np.asarray(analysis["values"])[mask]
        idx  = np.nonzero(np.asarray(mask))[0]
        print(f"      [{branch}] '{analysis['name']}':")
        print(f"        {'riga':>4} {'z':>7} {'g':>7} {'x_th':>8} {'sigma':>9} "
              f"{'(x-c)/sigma':>12} {'logL':>11}")
        for k in range(len(idx)):
            # c e' il centro della gaussiana: 1 per un lower limit, 0 per un
            # upper limit, g per una detection.
            c = 1.0 if branch == "lower lim" else (0.0 if branch == "upper lim" else g[k])
            if sigma[k] > 0:
                s_txt, dev_txt = f"{sigma[k]:9.5f}", f"{(x_th[k] - c) / sigma[k]:12.5f}"
            else:
                # Lato libero di una split normal: la barra d'errore da quel lato
                # vale 0 perche' il dato sta al bordo fisico di x_HII (1 a z=5,
                # 0 a z alto), quindi quel lato non vincola.  Stampare "0.00000"
                # e un dev "nan" sembrerebbe un errore di calcolo: non lo e'.
                s_txt, dev_txt = "   libero", "           -"
            print(f"        {idx[k]:>4} {z[k]:7.3f} {g[k]:7.3f} {x_th[k]:8.5f} "
                  f"{s_txt} {dev_txt} {log_l[k]:11.5f}")
        print(f"        somma logL = {float(np.sum(log_l)):.6f}")

    def _print_sigma_table(self, analysis):
        """
        Stampa, per un dataset, cosa entra nella likelihood punto per punto.

        Tutto quello che c'e' in questa tabella dipende solo dai dati, non dal
        modello, quindi e' fissato una volta per sempre a questo punto: e' la
        tabella da guardare per controllare il sigma a mano.  La parte che
        cambia a ogni passo (x_th, chi quadro) la stampa debug_points in logp.
        """
        if not self.print_sigma_table:
            return

        z        = analysis["z"]
        vals     = analysis["values"]
        err_up   = analysis["errors"]
        err_dn   = analysis["errors_down"]
        lower    = analysis["lower_lim"]
        upper    = analysis["upper_lim"]
        cl       = analysis["limit_cl"]
        sig_lim  = analysis["limit_sigma"]

        print(f"    - SIGMA che entrano nella likelihood per '{analysis['name']}':")
        print(f"        {'z':>7} {'ramo':>10} {'g/mu':>7} {'CL':>6} {'z_CL':>7} "
              f"{'sigma':>9}   {'sigma da':<22}")
        for i in range(len(z)):
            if bool(lower[i]):
                branch, cl_i = "lower lim", cl[i]
                src = "(1-g)/z_CL"
            elif bool(upper[i]):
                branch, cl_i = "upper lim", cl[i]
                src = "g/z_CL"
            else:
                branch, cl_i = "detection", np.nan
                src = "err_up / err_down"

            if branch == "detection":
                s_txt = f"{err_up[i]:.4f}/{err_dn[i]:.4f}"
                zcl_txt, cl_txt = "-", "-"
            else:
                cl_used = cl_i if np.isfinite(cl_i) else DEFAULT_LIMIT_CL
                zcl_txt = f"{z_for_confidence_level(cl_used):.4f}"
                cl_txt = f"{cl_used:.3f}" + ("*" if not np.isfinite(cl_i) else "")
                s_txt = f"{sig_lim[i]:.4f}" if np.isfinite(sig_lim[i]) else "nan"

            print(f"        {z[i]:7.3f} {branch:>10} {vals[i]:7.3f} {cl_txt:>6} "
                  f"{zcl_txt:>7} {s_txt:>9}   {src:<22}")
        if np.any(~np.isfinite(cl) & (np.asarray(lower) | np.asarray(upper))):
            print(f"        (* = CL assunto, non presente in corecon)")

    @staticmethod
    def _get_err2_array(data_obj, attr_name, n):
        """
        err_up2 / err_down2 come array float di lunghezza n, NaN dove assenti.

        Sono le barre d'errore al 95% a DUE CODE, cioe' la controparte di
        err_up/err_down.  Come per _get_cl_array non si usa nan_to_num: uno zero
        verrebbe letto come una barra valida di larghezza nulla.

        Popolate in 4 dei 50 dataset di HII_fraction: Bolan 2022 (solo la riga di
        detection a z=7.6), Davies 2026 (tutte e 24, ma la' sono l'incertezza sul
        LIMITE, non su una detection), Spina 2024, Wang 2020.
        """
        val = getattr(data_obj, attr_name, None)
        if val is None:
            return np.full(n, np.nan)
        arr = np.atleast_1d(np.asarray(val, dtype=float).ravel())
        if arr.size == 1:
            return np.full(n, float(arr[0]))
        ReioLike._lunghezza_campo(data_obj, attr_name, arr.size, n)
        return arr

    @staticmethod
    def _select_error_bars(dataset_spec, cl_req, err_up, err_dn,
                           err_up2, err_dn2, lower_lim, upper_lim):
        """
        Sceglie quale coppia di barre d'errore usare sulle DETECTION, in base al
        confidence level chiesto nello YAML.

        Il problema: con [CL=0.95] si sceglie la riga-limite al 95%, ma in un
        dataset misto le righe di detection restano con le barre al 68%, e il
        dataset finisce letto a due livelli di confidenza diversi.  Bolan et al.
        2022 e' il caso: due righe-limite a z=6.7 (CL 0.68 e 0.95) piu' una
        detection a z=7.6 che ha ENTRAMBE le versioni, err_up/err_down al 68% e
        err_up2/err_down2 al 95%.

        ATTENZIONE alla normalizzazione.  err_up2 NON e' un sigma: e' la
        semiampiezza dell'intervallo al 95% a due code, cioe' ~1.97 sigma.
        Metterla direttamente come sigma sottostimerebbe il chi quadro di ~4x.
        Il fattore e'

            Phi^-1((1+0.95)/2) / Phi^-1((1+0.68)/2) = 1.9600 / 0.9945 = 1.9709

        normalizzato in modo che la barra al 68% resti sigma com'e', che e' la
        convenzione con cui la usano tutti gli altri dataset.

        Verifica sul dato (Bolan, riga z=7.6):
            err_up2/err_up     = 1.909   vs 1.971 atteso per una gaussiana -> torna
            err_down2/err_down = 1.375   vs 1.971                          -> non torna
        quindi il lato basso di quella posteriore non e' gaussiano, e passare da
        68% a 95% cambia sigma_down del 30% (0.0800 -> 0.0558) contro il 3% del
        lato alto.  E' un'informazione sul dato, non un errore di conversione.

        Le righe-limite non vengono toccate: il loro sigma esce da _limit_sigma,
        che usa values e il CL, non le barre.

        Ritorna (err_up_eff, err_dn_eff).
        """
        err_up = np.asarray(err_up, dtype=float).copy()
        err_dn = np.asarray(err_dn, dtype=float).copy()
        if cl_req is None or cl_req <= 0.9:
            return err_up, err_dn          # la barra al 68% e' sigma, come sempre

        ratio = norm.ppf((1.0 + 0.95) / 2.0) / norm.ppf((1.0 + 0.68) / 2.0)
        is_det = ~(np.asarray(lower_lim, dtype=bool) | np.asarray(upper_lim, dtype=bool))
        swapped, missing = [], []
        for i in np.nonzero(is_det)[0]:
            e2u, e2d = float(err_up2[i]), float(err_dn2[i])
            if np.isfinite(e2u) or np.isfinite(e2d):
                if np.isfinite(e2u):
                    err_up[i] = e2u / ratio
                if np.isfinite(e2d):
                    err_dn[i] = e2d / ratio
                swapped.append(i)
            else:
                missing.append(i)

        if swapped:
            print(f"    - CL={cl_req:g}: per coerenza le detection usano le barre al 95% "
                  f"(err_up2/err_down2 / {ratio:.4f}) invece di quelle al 68%; "
                  f"righe {swapped}.")
        if missing:
            print(f"    - NOTA: CL={cl_req:g} richiesto ma le righe di detection {missing} "
                  f"non hanno err_up2/err_down2 in corecon: restano con le barre al 68%. "
                  f"Quel dataset viene letto a due livelli di confidenza diversi.")
        return err_up, err_dn

    @staticmethod
    def _get_cl_array(data_obj, n):
        """
        limit_confidence_level di corecon come array float di lunghezza n.

        NaN dove il campo manca: significa "CL non dichiarato", ed e' diverso da
        zero.  Non si usa nan_to_num qui, al contrario di _get_error_array: uno
        zero verrebbe poi letto come un CL valido e Phi^-1(0) = -inf.
        corecon espande gli scalari, ma un campo assente o di lunghezza sbagliata
        va riempito a mano.
        """
        val = getattr(data_obj, "limit_confidence_level", None)
        if val is None:
            return np.full(n, np.nan)
        arr = np.atleast_1d(np.asarray(val, dtype=float).ravel())
        if arr.size == 1:
            return np.full(n, float(arr[0]))
        ReioLike._lunghezza_campo(data_obj, "limit_confidence_level", arr.size, n)
        return arr

    def _get_error_array(self, data_obj, attr_name):
        """
        Larghezze della finestra in z (err_left / err_right), o None se il campo
        manca: in quel caso il dato si valuta nel punto, ed e' legittimo.

        NaN -> 0 e' voluto: una riga senza larghezza (es. i QSO singoli di Greig
        2022) si valuta nel punto.

        Se il campo c'e' ma non si converte in numeri, ci si ferma.  Prima
        l'errore veniva ingoiato e si restituiva None: la finestra spariva e il
        dato veniva confrontato con il valore puntuale del modello invece che
        con la media sul bin (Bolan 2022 a z=7.6: 0.084 contro 0.143).
        """
        val = getattr(data_obj, attr_name, None)
        if val is None:
            return None
        try:
            arr = np.array(val, dtype=float)
        except (ValueError, TypeError) as e:
            rif = str(getattr(data_obj, "reference", "") or "")[:70]
            raise RuntimeError(
                f"corecon: il campo '{attr_name}' (larghezza della finestra in z) "
                f"non si converte in numeri ({e}).  Dataset: {rif!r}.") from e
        return np.nan_to_num(arr, nan=0.0)

    def get_requirements(self):
        """
        Chiede la storia di reionizzazione come prodotto vero.

        Prima qui c'era {"tau_reio": None}: non serviva tau, era un espediente
        per costringere cobaya a eseguire ReioTheory.calculate() prima di
        logp(), e la storia si andava poi a prendere da una globale di modulo.
        Con una dipendenza dichiarata l'ordine lo sa gia' cobaya, e la storia
        arriva per il canale che la sua cache gestisce correttamente.

        Insieme alla richiesta passa a ReioTheory i redshift dei dati
        puntuali (z_nodi): la teoria ci mette un nodo fisso, cosi' il modello
        in quei z e' il valore esatto e non quello di una retta fra due nodi
        che scivolano con i parametri campionati.  Vedi _z_dati_puntuali.
        """
        return {"reio_history": {"z_nodi": self._z_dati_puntuali()}}

    def _z_dati_puntuali(self):
        """
        Redshift distinti dei dati senza larghezza in z, in tutte le analisi.

        Solo quelli: un dato con una finestra (err_z_left + err_z_right > 0)
        e' confrontato con la MEDIA del modello sulla finestra, che
        model_window_mean calcola gia' in forma chiusa sulla spezzata, quindi
        e' esatta comunque e un nodo nel suo z nominale non cambierebbe nulla.
        """
        z = []
        for a in self.analyses:
            zz = np.asarray(a["z"], dtype=float)
            w = np.zeros_like(zz)
            for k in ("err_z_left", "err_z_right"):
                if a[k] is not None:
                    w = w + np.asarray(a[k], dtype=float)
            z.extend(zz[w <= 0.0].tolist())
        return sorted(set(float(v) for v in z))

    @staticmethod
    def _storia_sbagliata(msg):
        """
        Eccezione che ferma la catena.  NON un RuntimeError: cobaya cattura le
        eccezioni ordinarie di una likelihood e, con stop_at_error: False (il
        default), le registra a livello debug e assegna likelihood zero
        (cobaya/theory.py, check_cache_and_compute).  La catena andrebbe
        avanti rifiutando ogni punto, in silenzio.  LoggedError invece sta in
        cobaya.log.always_stop_exceptions: stampa il messaggio come errore e
        interrompe sempre.
        """
        return LoggedError("ReioLike", msg)

    def _reio_history(self):
        """
        La storia del punto corrente.  Se non e' quella giusta, ci si FERMA.

        DENTRO COBAYA (self.provider esiste, lo imposta cobaya quando monta il
        modello) la storia ha UNA sola sorgente: provider.get_result, cioe' lo
        state che ReioTheory.calculate() ha scritto per QUESTO punto.  Se
        quella chiamata fallisce, o restituisce qualcosa di incompleto, si
        solleva un errore e la catena si interrompe.

        Prima, in quel caso, l'errore veniva ignorato (except: pass) e si
        prendeva la copia globale _shared_reio_history.  Quella copia e'
        unica e la scrive l'ultima calculate() eseguita: su un cache hit di
        cobaya puo' essere la storia del punto PRECEDENTE.  La likelihood
        avrebbe confrontato i dati con il modello sbagliato senza dirlo.

        FUORI DA COBAYA (script che costruiscono ReioLike a mano, es.
        Tests/verifica_analisi.py) non c'e' provider e si usa la globale,
        riempita con set_reio_history.  Se e' vuota, errore anche qui.
        """
        from Reiolike.reio_theory import _shared_reio_history

        provider = getattr(self, "provider", None)
        if provider is not None:
            try:
                h = provider.get_result("reio_history")
            except Exception as e:
                raise self._storia_sbagliata(
                    "[ReioLike] cobaya non ha fornito 'reio_history' per il punto "
                    "corrente.  La likelihood si ferma invece di usare una storia "
                    "che potrebbe essere di un altro punto.  Controlla che la "
                    "theory sia Reiolike.reio_theory.ReioTheory e che calculate() "
                    "non sia fallita.  Errore originale: "
                    f"{type(e).__name__}: {e}") from e
            self._controlla_storia(h, "cobaya (provider)", serve_elio=True)
            return h

        if _shared_reio_history.get("z") is None:
            raise self._storia_sbagliata(
                "[ReioLike] nessuna storia di reionizzazione: fuori da cobaya va "
                "iniettata prima di logp() con "
                "Reiolike.reio_theory.set_reio_history(z, xe, includes_HeII=...).")
        self._controlla_storia(_shared_reio_history, "globale (fuori da cobaya)",
                               serve_elio=False)
        return _shared_reio_history

    @staticmethod
    def _controlla_storia(h, sorgente, serve_elio):
        """Controlli minimi di forma: se falliscono, la storia non e' usabile."""
        def errore(msg):
            raise ReioLike._storia_sbagliata(
                f"[ReioLike] storia di reionizzazione da {sorgente} non valida: {msg}")
        if not isinstance(h, dict):
            errore(f"atteso un dizionario, arrivato {type(h).__name__}.")
        for k in ("z", "xe") + (("xe_HeII", "f_He") if serve_elio else ()):
            if h.get(k) is None:
                errore(f"manca la chiave '{k}'.")
        z, xe = np.asarray(h["z"], float), np.asarray(h["xe"], float)
        if z.ndim != 1 or z.shape != xe.shape or len(z) < 2:
            errore(f"z e xe devono essere 1D, della stessa lunghezza, almeno 2 "
                   f"punti: {z.shape} e {xe.shape}.")
        if not (np.all(np.isfinite(z)) and np.all(np.isfinite(xe))):
            errore("z o xe contengono NaN o infiniti.")
        if serve_elio:
            he = np.asarray(h["xe_HeII"], float)
            if he.shape != z.shape or not np.all(np.isfinite(he)):
                errore(f"xe_HeII ha forma {he.shape} (attesa {z.shape}) o "
                       f"contiene NaN.")
            if not np.isfinite(float(h["f_He"])):
                errore("f_He non e' finito.")

    def _model_on_windows(self, analysis_dict, z_model, model_values,
                          integration_width=None):
        """
        Modello valutato in corrispondenza di ciascun dato, mediato sulla sua
        finestra in redshift quando il dato ne ha una.

        Molte misure di x_HII non sono a un singolo z: sono medie su un bin di
        survey (Kageura et al. 2025 a z=8.41 copre 7.51-9.43).  err_left e
        err_right di corecon danno l'estensione del bin, e in quel caso il
        confronto corretto e' fra il dato e la MEDIA del modello sul bin, non
        il suo valore nel z nominale.  Su finestre larghe la differenza e'
        grande: Bolan et al. 2022 a z=7.6 da' 0.090 puntuale contro 0.154
        mediato.

        Se il dato non ha larghezze (err_left/err_right assenti o nulli) la
        finestra e' degenere e si ricade sulla valutazione puntuale, che e' la
        cosa giusta per una misura quotata a un singolo redshift.

        L'integrale e' in forma chiusa sulla spezzata del modello (vedi
        model_window_mean): esatto anche quando dentro la finestra cade una
        transizione ripida, e senza quadratura adattiva da far fallire.

        Questo metodo e' usato da TUTTI i rami della likelihood - rilevazioni,
        limiti superiori, limiti inferiori e dataset misti - perche' la
        finestra e' una proprieta' del DATO, non del modo in cui quel dato
        viene trattato statisticamente.

        Restituisce (model_values_ai_dati, z_model, model_values), con gli
        ultimi due riordinati in z crescente se serviva.
        """
        z_corecon = analysis_dict["z"]
        err_left  = analysis_dict["err_z_left"]
        err_right = analysis_dict["err_z_right"]

        if integration_width is None:
            w_left  = err_left  if err_left  is not None else np.zeros_like(z_corecon)
            w_right = err_right if err_right is not None else np.zeros_like(z_corecon)
            w_left  = np.array(w_left,  dtype=float)
            w_right = np.array(w_right, dtype=float)
            widths_z = w_left + w_right
        else:
            widths_z = np.full_like(z_corecon, float(integration_width))

        if z_model[0] > z_model[-1]:
            z_model      = z_model[::-1]
            model_values = model_values[::-1]

        self._warn_extrapolation(analysis_dict["name"], z_corecon, z_model)

        out = []
        for i, z in enumerate(z_corecon):
            if widths_z[i] > 0:
                if integration_width is None:
                    z_min, z_max = z - w_left[i], z + w_right[i]
                else:
                    z_min, z_max = z - widths_z[i] / 2.0, z + widths_z[i] / 2.0
            else:
                z_min = z_max = z
            out.append(model_window_mean(z_model, model_values, z_min, z_max))

        return np.array(out, dtype=float), z_model, model_values

    def compute_gaussian_loglike(self, analysis_dict, z_model, model_values, integration_width=None):
        """
        Split-normal likelihood: confronta il modello con i dati di corecon
        usando barre d'errore ASIMMETRICHE.

        La distribuzione e' centrata sul valore misurato, con larghezza err_up
        sopra ed err_down sotto.  Si riduce esattamente alla gaussiana normale
        quando err_up == err_down.  Vedi split_normal_chi2 per normalizzazione,
        lati degeneri e continuita'.
        """
        corecon_values = analysis_dict["values"]
        corecon_err_hi = analysis_dict["errors"]
        corecon_err_lo = analysis_dict.get("errors_down")
        if corecon_err_lo is None:                 # storia messa a mano, test
            corecon_err_lo = corecon_err_hi
        corecon_err_hi = np.asarray(corecon_err_hi, dtype=float)
        corecon_err_lo = np.asarray(corecon_err_lo, dtype=float)
        model_at_corecon, z_model, model_values = self._model_on_windows(
            analysis_dict, z_model, model_values, integration_width)

        # Basta un lato utilizzabile: un punto al bordo fisico (err_up = 0 a
        # x_HII = 1, err_down = 0 a x_HII = 0) vincola comunque dall'altro lato.
        ok_hi = np.isfinite(corecon_err_hi) & (corecon_err_hi > 0)
        ok_lo = np.isfinite(corecon_err_lo) & (corecon_err_lo > 0)
        mask_valid = (
            np.isfinite(corecon_values) &
            np.isfinite(model_at_corecon) &
            (ok_hi | ok_lo)
        )

        if np.sum(mask_valid) == 0:
            self._avvisa_una_volta(
                ("rilevazioni", analysis_dict["name"]),
                "nessun punto valido per '%s' (punti %d, valori finiti %d, "
                "barre d'errore usabili %d, modello finito %d): la logL e' -inf "
                "e ogni punto della catena viene rifiutato.",
                analysis_dict["name"], len(corecon_values),
                int(np.sum(np.isfinite(corecon_values))), int(np.sum(ok_hi | ok_lo)),
                int(np.sum(np.isfinite(model_at_corecon))))
            return -np.inf

        chi2_each = split_normal_chi2(
            model_at_corecon[mask_valid],
            corecon_values[mask_valid],
            corecon_err_lo[mask_valid],
            corecon_err_hi[mask_valid],
        )
        if self.debug_points:
            # Su una split normal il sigma ATTIVO dipende da che lato del dato
            # cade il modello, quindi si stampa quello effettivamente usato.
            x_th_m = model_at_corecon[mask_valid]
            mu_m   = corecon_values[mask_valid]
            s_act  = np.where(x_th_m >= mu_m,
                              corecon_err_hi[mask_valid], corecon_err_lo[mask_valid])
            self._debug_points(analysis_dict, mask_valid, x_th_m, s_act,
                               -0.5 * np.asarray(chi2_each), "detection")
        return -0.5 * float(np.sum(chi2_each))

    def _warn_heii_missing(self, z_model):
        """
        Avvisa, una volta sola, che la storia iniettata non dichiara il
        bookkeeping dell'elio, e quantifica l'errore che questo comporta.

        Nel percorso normale non scatta mai: ReioTheory riempie xe_HeII e f_He
        a ogni passo.  Scatta solo quando qualcuno scrive in
        _shared_reio_history a mano, tipicamente in uno script di test.  In
        quel caso si assume che xe NON contenga il gradino HeII->HeIII, che e'
        il caso giusto se l'array viene direttamente da reio_models, ma e'
        un'assunzione e come tale va detta ad alta voce.
        """
        if getattr(self, "_heii_warned", False):
            return
        self._heii_warned = True

        from .reio_models import helium_fullreio_step
        z_data = np.concatenate([a["z"] for a in self.analyses]) \
            if getattr(self, "analyses", None) else np.asarray([])

        print("\n" + "!" * 72)
        print("  ATTENZIONE (una volta sola): storia di reionizzazione iniettata a mano.")
        print("  _shared_reio_history non contiene 'xe_HeII' e/o 'f_He', quindi ReioLike")
        print("  ASSUME che xe NON includa la seconda ionizzazione dell'elio (HeII->HeIII)")
        print(f"  e usa f_He = {_F_HE:.6f}.  La conversione applicata e':")
        print(f"      x_HII = xe / (1 + f_He) = xe / {1.0 + _F_HE:.6f}")
        if len(z_data):
            he_at_data = helium_fullreio_step(z_data)
            i = int(np.argmax(he_at_data))
            err = float(he_at_data[i]) / (1.0 + _F_HE)
            print(f"  Se l'assunzione fosse SBAGLIATA (cioe' se xe includesse gia' HeII),")
            print(f"  l'errore su x_HII sarebbe al massimo {err:.2e}, a z = {z_data[i]:g}")
            print(f"  (su {len(z_data)} punti dati, z = {z_data.min():g}..{z_data.max():g}).")
        print("  Per rendere la scelta esplicita e far sparire questo avviso:")
        print("      from Reiolike.reio_theory import set_reio_history")
        print("      set_reio_history(z, xe, includes_HeII=False)   # o True")
        print("!" * 72 + "\n")

    def _avvisa_una_volta(self, chiave, msg, *args):
        """Avviso nel logger una sola volta per chiave.  Al posto dei print
        "DEBUG" di prima: un caso che rende la logL -inf a OGNI valutazione
        deve vedersi (warning) ma non riempire il log di una catena."""
        visti = self.__dict__.setdefault("_avvisi_dati", set())
        if chiave in visti:
            return
        visti.add(chiave)
        self.log.warning(msg, *args)

    def _warn_extrapolation(self, name, z_data, z_model):
        """Emit a one-time warning if any data points lie outside the model redshift grid."""
        if name in self._coverage_warned:
            return
        z_min = float(np.min(z_model))
        z_max = float(np.max(z_model))
        out = z_data[(z_data < z_min) | (z_data > z_max)]
        if len(out) > 0:
            print(
                f"    WARNING (once): dataset '{name}' has {len(out)} point(s) at "
                f"z={np.round(out, 2).tolist()} outside the model grid "
                f"[{z_min:.2f}, {z_max:.2f}]. Values will be extrapolated."
            )
        self._coverage_warned.add(name)

    def _controlla_limiti_con_errore_al_bordo(self, analysis):
        """
        Ferma l'avvio se un limite CON errore sta sul bordo fisico o oltre.

        Le maschere di compute_lower_limit_loglike, compute_upper_limit_loglike
        e compute_mixed_loglike saltano un limite inferiore con g fuori da
        (0, 1) e un limite superiore con g <= 0.  Per un limite SENZA errore e'
        giusto: la mezza gaussiana ha sigma = (1-g)/z_CL (o g/z_CL), che al
        bordo vale zero o non porta informazione, e _riassunto_limiti lo
        segnala gia' ("al bordo fisico, senza vincolo").

        Per un limite CON errore no: la sua logL e' P(B <= x) con B
        split-normal (logL_limite_inferiore_con_errore), che non usa (1-g) ed e'
        informativa anche con g = 1.  Esempio: g = 1.0 +- 0.05 penalizza un
        modello a x_HII = 0.9 di -3.78, ma la maschera lo scarterebbe e la riga
        non conterebbe nulla, in silenzio.  Nessun dato di corecon_config.yaml
        e' in questa situazione (28 ott 2026, test B5 della tanh); il controllo
        e' per i dataset che si aggiungeranno.  Correzione possibile: applicare
        le condizioni sul bordo solo alle righe senza errore.
        """
        g = np.asarray(analysis["values"], dtype=float)
        con_err = self._righe_con_errore(analysis)
        tipo = analysis["type"]
        n = len(g)
        inf = (np.ones(n, bool) if tipo == "lower_limit_HII_fraction"
               else np.asarray(analysis["lower_lim"], bool) if tipo == "mixed_HII_fraction"
               else np.zeros(n, bool))
        sup = (np.ones(n, bool) if tipo == "upper_limit_HII_fraction"
               else np.asarray(analysis["upper_lim"], bool) if tipo == "mixed_HII_fraction"
               else np.zeros(n, bool))
        persi = con_err & np.isfinite(g) & ((inf & ((g <= 0) | (g >= 1))) | (sup & (g <= 0)))
        if persi.any():
            righe = ", ".join(
                f"z={analysis['z'][i]:g} ({'inferiore' if inf[i] else 'superiore'}, "
                f"g={g[i]:g})" for i in np.flatnonzero(persi))
            raise LoggedError(
                "ReioLike",
                f"'{analysis['name']}': {int(persi.sum())} limite/i CON errore sul "
                f"bordo fisico: {righe}. La likelihood li scarterebbe in silenzio "
                "(le maschere escludono i limiti inferiori con g fuori da (0, 1) e "
                "quelli superiori con g <= 0, regola pensata per i limiti SENZA "
                "errore), ma con l'errore sono informativi: P(B <= x) con B "
                "split-normal non usa (1 - g). Va adattata la maschera per queste "
                "righe (vedi _controlla_limiti_con_errore_al_bordo), oppure il "
                "dataset va escluso esplicitamente dallo YAML.")

    @staticmethod
    def _righe_con_errore(analysis_dict):
        """Maschera delle righe-limite con errore sul limite (integrale)."""
        o = analysis_dict.get("limit_sigma_origine")
        n = len(analysis_dict["values"])
        if o is None:
            return np.zeros(n, dtype=bool)
        return np.asarray(o, dtype=object) == "limite con errore"

    @staticmethod
    def _errori_limite(analysis_dict, mask):
        """Errore sotto/sopra (in x_HII) sul limite, righe mask.
        Dove il limite non ne ha valgono NaN: si riempiono con 1, tanto
        np.where scarta quelle righe."""
        n = len(analysis_dict["values"])
        lo = analysis_dict.get("limit_err_lo", np.full(n, np.nan))
        hi = analysis_dict.get("limit_err_hi", np.full(n, np.nan))
        lo, hi = np.asarray(lo, float)[mask], np.asarray(hi, float)[mask]
        # riempitivo innocuo per le righe non incerte (evita NaN nei rami scartati)
        return np.where(np.isfinite(lo), lo, 1.0), np.where(np.isfinite(hi), hi, 1.0)

    def compute_upper_limit_loglike(self, analysis_dict, z_model, model_values):
        """
        Log-likelihood per non-detection date come upper limit a una coda.

        sigma_i viene da limit_confidence_level riga per riga (vedi _limit_sigma
        e il commento in testa al modulo), NON da un CL fisso:

            sigma_i = g_i / Phi^-1(CL_i)        (bordo in x_HII = 0)
            logL_i  = -0.5 * (x_th_i / sigma_i)^2   se x_th_i > 0
                    =  0                            se x_th_i <= 0
        """
        upper_limits = analysis_dict["values"]
        sigma_all    = analysis_dict["limit_sigma"]

        # Stessa media sulla finestra in redshift usata per le rilevazioni: la
        # larghezza del bin e' una proprieta' del dato, non del suo trattamento
        # statistico.  Senza larghezze si ricade sul valore puntuale.
        x_theory_at_data, z_model, model_values = self._model_on_windows(
            analysis_dict, z_model, model_values)

        # sigma non finito = limite inutilizzabile (CL assurdo, o g al bordo):
        # va escluso qui, altrimenti entra un NaN nel chi quadro.
        mask_valid = (
            np.isfinite(upper_limits) & (upper_limits > 0) &
            np.isfinite(x_theory_at_data) &
            np.isfinite(sigma_all) & (sigma_all > 0)
        )

        if np.sum(mask_valid) == 0:
            self._avvisa_una_volta(
                ("limiti superiori", analysis_dict["name"]),
                "nessun limite superiore valido per '%s': la logL e' -inf e "
                "ogni punto della catena viene rifiutato.", analysis_dict["name"])
            return -np.inf

        x_th  = x_theory_at_data[mask_valid]
        sigma = sigma_all[mask_valid]

        log_l = np.where(x_th > 0, -0.5 * (x_th / sigma) ** 2, 0.0)
        # limite con errore: L = P(B >= x), integrale sul limite
        ce = self._righe_con_errore(analysis_dict)[mask_valid]
        if ce.any():
            g = upper_limits[mask_valid]
            lo, hi = self._errori_limite(analysis_dict, mask_valid)
            log_l = np.where(ce, logL_limite_superiore_con_errore(x_th, g, lo, hi), log_l)
        self._debug_points(analysis_dict, mask_valid, x_th, sigma, log_l, "upper lim")
        return float(np.sum(log_l))

    def compute_lower_limit_loglike(self, analysis_dict, z_model, model_values):
        """
        Log-likelihood per vincoli dati come lower limit a una coda.

        sigma_i viene da limit_confidence_level riga per riga (vedi _limit_sigma):

            sigma_i = (1 - g_i) / Phi^-1(CL_i)      (bordo in x_HII = 1)
            logL_i  = -0.5 * ((x_th_i - 1)/sigma_i)^2   se x_th_i < 1
                    =  0                                se x_th_i >= 1

        Mezza gaussiana appoggiata al bordo fisico x_HII = 1, speculare al ramo
        upper limit che e' appoggiato a x_HII = 0.  Il limite g non e' il punto
        di taglio: entra attraverso sigma, e in x_th = g la penalita' vale
        -0.5 z_CL^2, cioe' "g sta a z_CL sigma dal bordo".
        """
        lower_limits = analysis_dict["values"]
        sigma_all    = analysis_dict["limit_sigma"]

        # Vedi compute_upper_limit_loglike: media sulla finestra anche qui.
        x_theory_at_data, z_model, model_values = self._model_on_windows(
            analysis_dict, z_model, model_values)

        mask_valid = (
            np.isfinite(lower_limits) &
            (lower_limits > 0) &
            (lower_limits < 1) &
            np.isfinite(x_theory_at_data) &
            np.isfinite(sigma_all) & (sigma_all > 0)
        )

        if np.sum(mask_valid) == 0:
            self._avvisa_una_volta(
                ("limiti inferiori", analysis_dict["name"]),
                "nessun limite inferiore valido per '%s': la logL e' -inf e "
                "ogni punto della catena viene rifiutato.", analysis_dict["name"])
            return -np.inf

        g     = lower_limits[mask_valid]
        x_th  = x_theory_at_data[mask_valid]
        sigma = sigma_all[mask_valid]

        # Taglio in 1, non in g: e' la mezza gaussiana appoggiata al bordo
        # fisico x_HII = 1, immagine speculare del ramo upper limit, che taglia
        # in 0 col centro in 0.
        #
        # Prima il taglio era in g mentre il centro restava in 1, quindi logL
        # saltava da 0 a -0.5 z_CL^2 attraversando g: 0.109 a CL=0.68, 1.353 a
        # CL=0.95, 3.775 a CL=0.997.  Misurato su ogni dataset di lower limit da
        # Tests/verifica_analisi.py (test 4).  Non era una mezza gaussiana, era
        # una mezza gaussiana con uno scalino, e lo scalino e' un muro per la
        # catena MCMC.
        #
        # Col taglio in 1 il significato del dato non cambia: g entra comunque,
        # attraverso sigma = (1-g)/z_CL, e in x_th = g la penalita' vale
        # esattamente -0.5 z_CL^2, cioe' "g sta a z_CL sigma dal bordo".  Cambia
        # solo che adesso ci si arriva con continuita'.
        log_l = np.where(x_th < 1.0, -0.5 * ((x_th - 1.0) / sigma) ** 2, 0.0)
        # limite con errore: L = P(B <= x), integrale sul limite
        ce = self._righe_con_errore(analysis_dict)[mask_valid]
        if ce.any():
            lo, hi = self._errori_limite(analysis_dict, mask_valid)
            log_l = np.where(ce, logL_limite_inferiore_con_errore(x_th, g, lo, hi), log_l)
        self._debug_points(analysis_dict, mask_valid, x_th, sigma, log_l, "lower lim")
        return float(np.sum(log_l))

    def compute_mixed_loglike(self, analysis_dict, z_model, model_values):
        """
        Log-likelihood for datasets that contain a mix of detections, upper limits,
        and/or lower limits in the same corecon entry (e.g. Bolan et al. 2022).

        Each point is dispatched individually based on its corecon lower_lim / upper_lim flags:
          lower_lim=True  -> lower limit penalty
          upper_lim=True  -> upper limit penalty
          both False      -> Gaussian (detection)
        """
        # Media sulla finestra anche qui: Bolan et al. 2022 ha larghezze in z
        # (+-0.2 e +-0.6) e i suoi punti finiscono in rami diversi, quindi senza
        # questo la stessa quantita' verrebbe trattata in due modi diversi.
        x_theory_at_data, z_model, model_values = self._model_on_windows(
            analysis_dict, z_model, model_values)
        errors      = analysis_dict["errors"]
        errors_down = analysis_dict.get("errors_down")
        if errors_down is None:
            errors_down = errors
        lower_lim   = analysis_dict["lower_lim"]
        upper_lim   = analysis_dict["upper_lim"]
        sigma_lim   = analysis_dict["limit_sigma"]
        con_errore  = self._righe_con_errore(analysis_dict)

        log_total = 0.0
        # Per debug_points: ogni punto va in un ramo diverso, quindi si tiene
        # traccia di quale, e con che sigma, invece di stampare una tabella sola.
        dbg = []
        for i in range(len(analysis_dict["values"])):
            x_th = x_theory_at_data[i]
            g    = analysis_dict["values"][i]
            if not np.isfinite(x_th) or not np.isfinite(g):
                continue

            if lower_lim[i]:
                # Lower limit: penalise if x_th < g
                if g <= 0 or g >= 1:
                    continue
                # sigma per riga, dal suo limit_confidence_level: in un dataset
                # misto le righe possono avere CL diversi fra loro (Bolan et al.
                # 2022 ha 0.68 e 0.95 nello stesso oggetto corecon).
                sigma = sigma_lim[i]
                if not (np.isfinite(sigma) and sigma > 0):
                    continue
                # Taglio in 1, come in compute_lower_limit_loglike: stessa mezza
                # gaussiana, stessa ragione.  I due rami vanno tenuti allineati.
                if con_errore[i]:
                    contrib = float(logL_limite_inferiore_con_errore(
                        x_th, g, analysis_dict["limit_err_lo"][i],
                        analysis_dict["limit_err_hi"][i]))
                else:
                    contrib = float(np.where(x_th < 1.0, -0.5 * ((x_th - 1.0) / sigma) ** 2, 0.0))
                dbg.append((i, "lower lim", g, x_th, sigma, 1.0, contrib))
            elif upper_lim[i]:
                # Upper limit: penalise if x_th > 0
                if g <= 0:
                    continue
                sigma = sigma_lim[i]
                if not (np.isfinite(sigma) and sigma > 0):
                    continue
                if con_errore[i]:
                    contrib = float(logL_limite_superiore_con_errore(
                        x_th, g, analysis_dict["limit_err_lo"][i],
                        analysis_dict["limit_err_hi"][i]))
                else:
                    contrib = float(np.where(x_th > 0, -0.5 * (x_th / sigma) ** 2, 0.0))
                dbg.append((i, "upper lim", g, x_th, sigma, 0.0, contrib))
            else:
                # Detection: split normal, come in compute_gaussian_loglike.
                s_hi, s_lo = errors[i], errors_down[i]
                ok_hi = np.isfinite(s_hi) and s_hi > 0
                ok_lo = np.isfinite(s_lo) and s_lo > 0
                if not (ok_hi or ok_lo):
                    continue
                contrib = -0.5 * float(split_normal_chi2(x_th, g, s_lo, s_hi))
                dbg.append((i, "detection", g, x_th,
                            (s_hi if x_th >= g else s_lo), g, contrib))
            log_total += contrib

        if self.debug_points and dbg:
            print(f"      [mixed] '{analysis_dict['name']}':")
            print(f"        {'riga':>4} {'ramo':>10} {'g':>7} {'x_th':>8} {'sigma':>9} "
                  f"{'(x-c)/sigma':>12} {'logL':>11}")
            for i, branch, g, x_th, sigma, c, contrib in dbg:
                dev = (x_th - c) / sigma if (np.isfinite(sigma) and sigma > 0) else np.nan
                print(f"        {i:>4} {branch:>10} {g:7.3f} {x_th:8.5f} {sigma:9.5f} "
                      f"{dev:12.5f} {contrib:11.5f}")
            print(f"        somma logL = {log_total:.6f}")

        return log_total

    def logp(self, **params_values):
        """
        Calculate the log-likelihood.
        """
        # Solleva un errore se la storia non c'e' o non e' quella del punto
        # corrente: niente -inf silenzioso (prima la catena restava ferma
        # senza dire perche').  Vedi _reio_history.
        storia = self._reio_history()
        z, xe = storia["z"], storia["xe"]

        he = storia.get("xe_HeII")
        fHe = storia.get("f_He")
        # Solo fuori da cobaya: dentro, _reio_history ha gia' preteso le due
        # chiavi, quindi questo ramo non puo' scattare in una catena.
        if he is None or fHe is None:      # storia messa a mano (script di test)
            self._warn_heii_missing(z)
            he, fHe = 0.0, _F_HE
        x_HII = (xe - he) / (1.0 + fHe)

        log_prob = 0.0

        if hasattr(self, "analyses"):
            for analysis in self.analyses:
                model_to_compare = None
                if analysis["type"] in ("HII_fraction", "upper_limit_HII_fraction",
                                        "lower_limit_HII_fraction", "mixed_HII_fraction"):
                    model_to_compare = x_HII

                if analysis["type"] == "upper_limit_HII_fraction":
                    log_L = self.compute_upper_limit_loglike(
                        analysis_dict=analysis,
                        z_model=z,
                        model_values=model_to_compare,
                    )
                elif analysis["type"] == "lower_limit_HII_fraction":
                    log_L = self.compute_lower_limit_loglike(
                        analysis_dict=analysis,
                        z_model=z,
                        model_values=model_to_compare,
                    )
                elif analysis["type"] == "mixed_HII_fraction":
                    log_L = self.compute_mixed_loglike(
                        analysis_dict=analysis,
                        z_model=z,
                        model_values=model_to_compare,
                    )
                else:
                    log_L = self.compute_gaussian_loglike(
                        analysis_dict=analysis,
                        z_model=z,
                        model_values=model_to_compare,
                        integration_width=None,
                    )

                if self.print_dataset_logl:
                    print(f"  [{analysis['type']}] '{analysis['name']}':  logL = {log_L:.6f}")
                if np.isfinite(log_L):
                    log_prob += log_L
                else:
                    return -np.inf

        return log_prob
