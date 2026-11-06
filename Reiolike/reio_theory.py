from cobaya.theory import Theory
from cobaya.theories.classy import classy
from cobaya.log import LoggedError
import os

import numpy as np
from .reio_models import (tanh_model, tanh_class_model, class_reio_start,
                          tanh_pinned_end_model,
                          tanh_pinned_end_delta_z, gompertz_sr_model,
                          power_law_model, binned_model, make_bin_edges,
                          bin_edges_from_data,
                          pca_model, helium_fullreio_step, f_he, _Y_HE)
from .bbn_helium import TabellaBBN, delta_neff_bbn, file_bbn_di_class
from .pca_basis import load_basis

try:
    from classy import Class as _ClassPy
except ImportError:
    _ClassPy = None

# Deposito condiviso fra ReioTheory e ReioLike.  Oltre alla storia contiene il
# bookkeeping dell'elio effettivamente usato, cosi' la likelihood puo' tornare
# da xe (elettroni per nucleo di H) a x_HII (idrogeno ionizzato, quello che
# misurano i dati) senza riassumere niente sui parametri dell'elio.
_shared_reio_history = {'z': None, 'xe': None, 'xe_HeII': None, 'f_He': None}


def set_reio_history(z, xe, includes_HeII, f_He=None,
                     z_reHeII=3.5, delta_z_HeII=0.5):
    """
    Inietta a mano una storia di reionizzazione in _shared_reio_history.

    Serve agli script di test e ai confronti fuori da cobaya, dove ReioTheory
    non gira e quindi nessuno riempie il deposito condiviso.  Usare questa
    funzione invece di scrivere nel dizionario a mano: cosi' il bookkeeping
    dell'elio e' esplicito e ReioLike non deve indovinare niente.

    PERCHE' SERVE DICHIARARE L'ELIO
    -------------------------------
    ReioLike deve tornare da xe (elettroni per nucleo di idrogeno) a x_HII
    (frazione di idrogeno ionizzato, cio' che misurano i dati di CoReCon):

        x_HII = (xe - xe_HeII) / (1 + f_He)

    Le due correzioni hanno peso molto diverso.  La divisione per (1 + f_He)
    vale circa l'8%% ed e' sempre necessaria, perche' i modelli di reio_models
    saturano a 1 + f_He (idrogeno piu' PRIMA ionizzazione dell'elio) e non a 1.
    La sottrazione di xe_HeII riguarda la SECONDA ionizzazione dell'elio
    (HeII -> HeIII), una tanh centrata a z = 3.5: nell'intervallo dei dati
    (z >~ 5) vale ~3e-4 ed e' trascurabile, ma a z = 3 senza di essa si
    otterrebbe x_HII = 1.067, cioe' idrogeno ionizzato al 106.7%%.

    Non c'e' modo di dedurre dall'array xe se il gradino HeII sia gia' dentro
    oppure no, quindi `includes_HeII` e' un argomento OBBLIGATORIO: chi inietta
    la storia deve dichiararlo.

    Parameters
    ----------
    z : array-like
        Griglia in redshift, crescente.
    xe : array-like
        Elettroni per nucleo di idrogeno, sulla stessa griglia.
    includes_HeII : bool
        True  se `xe` contiene gia' il gradino HeII->HeIII, cioe' se e' stata
              prodotta come fa ReioTheory (modello + helium_fullreio_step).
              Il gradino viene ricostruito e registrato, e ReioLike lo
              sottrarra'.
        False se `xe` e' idrogeno + PRIMA ionizzazione dell'elio soltanto,
              come restituito direttamente da tanh_model, power_law_model,
              pca_model e simili.  Non viene sottratto nulla.
    f_He : float, optional
        n_He/n_H.  Default: il valore del pacchetto, f_he(_Y_HE) = 0.081885.
    z_reHeII, delta_z_HeII : float, optional
        Centro e larghezza del gradino HeII, usati solo se includes_HeII=True.
        I default coincidono con quelli di CLASS.

    Returns
    -------
    dict : il deposito condiviso aggiornato (utile per ispezionarlo nei test).

    Examples
    --------
    >>> from Reiolike.reio_models import tanh_model
    >>> z  = np.linspace(0, 30, 3001)
    >>> xe = tanh_model(z, z_re=7.0, delta_z=0.5)      # H + HeI soltanto
    >>> set_reio_history(z, xe, includes_HeII=False)
    """
    z  = np.asarray(z,  dtype=float)
    xe = np.asarray(xe, dtype=float)
    if z.shape != xe.shape:
        raise ValueError(
            f"set_reio_history: z e xe hanno forme diverse, {z.shape} vs {xe.shape}.")
    if not isinstance(includes_HeII, (bool, np.bool_)):
        raise TypeError(
            "set_reio_history: includes_HeII deve essere True o False, esplicitamente. "
            "True = l'array xe contiene gia' il gradino HeII->HeIII; "
            "False = xe e' idrogeno + prima ionizzazione dell'elio soltanto.")

    f_He_val = f_he(_Y_HE) if f_He is None else float(f_He)
    he = (helium_fullreio_step(z, z_reHeII=z_reHeII, delta_z=delta_z_HeII,
                               Y_He=_Y_HE) if includes_HeII
          else np.zeros_like(z))

    _shared_reio_history['z']       = z
    _shared_reio_history['xe']      = xe
    _shared_reio_history['xe_HeII'] = he
    _shared_reio_history['f_He']    = f_He_val
    return _shared_reio_history

# Lunghezza massima di un valore di parametro accettata dal parser di CLASS
# (_ARGUMENT_LENGTH_MAX_ in include/parser.h; FileArg e' un char[4096] di
# dimensione fissa).  Superarla non da' un errore pulito: la stringa trabocca
# nel buffer adiacente, e CLASS finisce per leggere reio_inter_z e
# reio_inter_xe concatenati.  Con ~9 caratteri per nodo il tetto pratico e'
# intorno ai 450 nodi.
# CHE COS'E' QUESTO LIMITE
# ------------------------
# CLASS legge i parametri in una struttura `file_content` fatta di array C di
# dimensione FISSA (include/parser.h):
#
#     #define _ARGUMENT_LENGTH_MAX_ 1024
#     typedef char FileArg[_ARGUMENT_LENGTH_MAX_];
#
# Ogni NOME e ogni VALORE di parametro viene copiato in un FileArg.  Non c'e'
# allocazione dinamica e non c'e' controllo di lunghezza: passare un valore piu'
# lungo scrive oltre la fine dell'array.  Per noi il valore critico e'
# reio_inter_z / reio_inter_xe, che sono liste di centinaia di numeri separati
# da virgola: a ~10 caratteri per nodo, 1024 caratteri sono ~100 nodi.
#
# `_LINE_LENGTH_MAX_`, sempre in parser.h, e' invece la lunghezza massima di una
# RIGA letta da un file .ini.  Passando da python (classy riempie la struttura
# direttamente) non entra in gioco, ma lo leggiamo lo stesso per avvertire chi
# lancia CLASS anche da riga di comando con un .ini.
#
# IL VALORE NON E' UNIVERSALE.  CLASS v3.3.4 di serie usa 1024; chi ha bisogno
# di tabelle lunghe alza quel numero e ricompila.  Su questa macchina:
#     class_public/include/parser.h                     -> 1024  (originale)
#     cobaya_corecon/packages/code/classy/include/...   -> 4096  (modificato)
# Quindi il valore va LETTO dalla CLASS effettivamente importata, non assunto.
#
# Se non si riesce a leggerlo si assume 1024, che e' il default di CLASS ed e'
# sempre sicuro: al massimo si usano meno nodi del necessario, e quanto costa
# e' misurato in Tests/test_limite_parser_class.py (meno di 0.03 sigma su tau).
_CLASS_ARG_MAX_FALLBACK = 1024

# Caratteri per nodo, stimati: "13.755102," e' 10.  Serve solo a dimensionare
# il budget di nodi PRIMA di formattare; il taglio esatto lo fa poi
# _fit_class_strings() misurando le stringhe vere.
_CHARS_PER_NODE = 10


def _class_header_candidates():
    """Posti dove puo' stare il parser.h della CLASS importata, in ordine di
    fiducia decrescente."""
    import os
    paths = []
    try:
        import classy
        d = os.path.dirname(os.path.abspath(classy.__file__))
        for base in (d, os.path.dirname(d), os.path.dirname(os.path.dirname(d))):
            paths.append(os.path.join(base, "include", "parser.h"))
    except Exception:
        pass
    for var in ("CLASS_DIR", "CLASSDIR", "CLASS_PATH"):
        v = os.environ.get(var)
        if v:
            paths.append(os.path.join(v, "include", "parser.h"))
    seen, out = set(), []
    for p in paths:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


def _eval_define(expr, known):
    """
    Valuta il corpo di una #define: un intero, oppure una piccola espressione
    come `2*1024` o `_LINE_LENGTH_MAX_`.  Chi modifica CLASS a mano non scrive
    per forza un numero nudo, e una regex che accetta solo cifre fallirebbe
    proprio su quelle installazioni.
    """
    import re
    expr = expr.split("/*")[0].split("//")[0].strip()
    expr = expr.rstrip("uUlL")
    if not expr or not re.fullmatch(r"[0-9_A-Za-z\s()+\-*]+", expr):
        return None
    for name, val in known.items():
        expr = re.sub(rf"\b{re.escape(name)}\b", str(val), expr)
    if not re.fullmatch(r"[0-9\s()+\-*]+", expr):
        return None
    try:
        val = eval(expr, {"__builtins__": {}}, {})     # solo cifre e + - * ( )
    except Exception:
        return None
    return int(val) if isinstance(val, int) and 0 < val < 10 ** 7 else None


def _detect_class_arg_max(default=_CLASS_ARG_MAX_FALLBACK):
    """
    Legge _ARGUMENT_LENGTH_MAX_ dal parser.h della CLASS importata.

    Returns
    -------
    (arg_max, line_max, sorgente) — sorgente e' il percorso del file letto,
    oppure None se non si e' trovato nulla e si sta usando il default.
    """
    import re
    for path in _class_header_candidates():
        try:
            with open(path) as fh:
                text = fh.read()
        except Exception:
            continue
        vals = {}
        for name in ("_LINE_LENGTH_MAX_", "_ARGUMENT_LENGTH_MAX_"):
            m = re.search(rf"^\s*#\s*define\s+{name}\s+(.+)$", text, re.M)
            if m:
                v = _eval_define(m.group(1), vals)
                if v is not None:
                    vals[name] = v
        if "_ARGUMENT_LENGTH_MAX_" in vals:
            return (vals["_ARGUMENT_LENGTH_MAX_"],
                    vals.get("_LINE_LENGTH_MAX_"), path)
    return default, None, None

# np.trapezoid in numpy >= 2, np.trapz prima. Con l'alias si evita sia il
# DeprecationWarning sia l'AttributeError, a seconda della versione installata.
_trapz = getattr(np, "trapezoid", None) or np.trapz

class ReioTheory(classy):

    def _to_native(self, val):
        """Recursively convert numpy types to Python native types."""
        if hasattr(val, "item"):
            return val.item()
        if isinstance(val, dict):
            return {k: self._to_native(v) for k, v in val.items()}
        if isinstance(val, (list, tuple)):
            return type(val)(self._to_native(v) for v in val)
        return val

    # Un errore dentro calculate() deve fermare la catena, non diventare un
    # punto a -inf.  Il default di cobaya e' False (theory.py:55), cioe' ogni
    # eccezione viene convertita in logp = -inf: con un refuso nel nome di un
    # parametro la catena gira per ore senza muoversi e senza un messaggio.
    # Misurato: scrivendo 'z_re' invece di 'reio_z_re' nel blocco params,
    # calculate() solleva ValueError e cobaya restituisce -inf a ogni punto.
    # Chi vuole il comportamento permissivo lo rimette a False nel suo YAML.
    stop_at_error: bool = True

    reio_model: str = "tanh"
    # "tanh"       : tanh lineare in z, x_e = xe_max/2 [1 + tanh((z_re - z)/dz)]
    # "tanh_class" : la tanh di CLASS reio_camb (e di CAMB/Planck), in
    #                y = (1+z)^1.5, con x_noreio(z): stessi parametri reio_z_re
    #                e reio_delta_z, ma e' un altro modello (vedi
    #                tanh_class_model).  Riproduce reio_camb nativo a 2e-6 su
    #                tau (Tests/test_tanh_class.py).  Esempio:
    #                    theory:
    #                      Reiolike.ReioTheory:
    #                        reio_model: tanh_class
    #                    params:
    #                      reio_z_re:    {prior: {min: 5, max: 12}}
    #                      reio_delta_z: 0.5        # larghezza di Planck
    # Estremo inferiore della griglia.  CLASS *impone* reio_inter_z[0] == 0
    # (thermodynamics.c:1447), quindi l'unico valore ammesso e' 0: l'opzione
    # resta accettata per compatibilita' con i YAML esistenti.
    z_min:    float = 0.0
    z_max:    float = 30.0  # estremo superiore della griglia passata a CLASS
    z_points: int   = 100   # punti nella finestra della transizione principale

    # --- Parametri della griglia adattiva ----------------------------------
    # La finestra di raffinamento va da centro-z_window_widths*delta_z a
    # centro+z_window_widths*delta_z: a 6 larghezze la tanh e' entro 1e-5 del
    # plateau, quindi oltre non serve risoluzione.
    z_window_widths: float = 6.0
    # Punti per delta_z dentro la finestra.  L'errore dell'interpolazione
    # lineare di una tanh vale ~0.048*A*(h/w)^2, quindi 8 punti per larghezza
    # (h = w/8) danno ~0.08% di errore su xe.
    z_pts_per_width: int   = 8
    # Griglia di fondo su [0, z_max], dove xe e' piatta (plateau o ~0).
    z_bg_points:     int   = 25
    # Distanza minima fra due nodi consecutivi.  Serve solo a evitare che due
    # nodi collassino nello stesso valore una volta formattati a 8 cifre
    # significative (a z~25 la risoluzione della stringa e' ~1e-6), quindi 1e-5
    # lascia un fattore 10 di margine.  Non va alzato: per power_law con
    # alpha<1 la griglia deve potersi addensare fortemente vicino a z_beg.
    z_min_spacing:   float = 1.0e-5
    # Tetto SUPERIORE al numero di nodi.  Il tetto che conta davvero non e'
    # questo: e' quello imposto dalla lunghezza della stringa che CLASS
    # accetta, che dipende dalla versione di CLASS installata e viene ricavato
    # a runtime (vedi _class_arg_limit / _node_budget).  Questo resta come
    # limite di buon senso indipendente dal parser.
    z_max_nodes:     int   = 400

    # Lunghezza massima di reio_inter_z / reio_inter_xe.  0 = leggila dal
    # parser.h di CLASS (raccomandato).  Metterla a mano serve solo se CLASS
    # e' installato in un modo che non espone gli header:
    #   grep _ARGUMENT_LENGTH_MAX_ <cartella_class>/include/parser.h
    class_arg_max:   int   = 0

    # --- Seconda reionizzazione dell'elio (HeII -> HeIII) -------------------
    # ATTENZIONE: CLASS *non* aggiunge alcun termine di elio quando
    # reio_parametrization='reio_inter'.  I parametri index_re_helium_fullreio_*
    # sono allocati solo per reio_camb / reio_half_tanh (thermodynamics.c
    # 1033-1044) e il ramo reio_inter di thermodynamics_reionization_function()
    # restituisce la tabella interpolata cosi' com'e' (thermodynamics.c
    # 4311-4350).  Verificato: una tabella che satura a 1+fHe da' xe(0)=1.08000
    # esatto, contro 1.16366 di reio_camb.
    #
    # Con reio_include_HeII=True lo step viene aggiunto qui, in modo uniforme
    # per TUTTI i modelli di reio_model, cosi' che xe(z->0) -> 1 + 2*fHe.
    # Mettere a False solo per riprodurre catene vecchie.
    reio_include_HeII: bool  = True
    reio_z_HeII:       float = 3.5    # = helium_fullreio_redshift di CLASS
    reio_delta_z_HeII: float = 0.5    # = helium_fullreio_width di CLASS
    # Frazione di elio Y_He.  -1 (default) => dalla BBN, punto per punto, in
    # funzione di omega_b: si legge la STESSA tabella di CLASS con lo stesso
    # algoritmo (Reiolike/bbn_helium.py) e il valore si passa anche a CLASS
    # come parametro YHe, cosi' storia di reionizzazione e CLASS usano lo
    # stesso numero per costruzione.  Un valore > 0 fissa Y_He a mano (per
    # riprodurre catene vecchie: era 0.2454 fisso fino al 14 ott 2026).
    # Se YHe e' gia' dato a CLASS come numero (params o extra_args), vince
    # quello.
    reio_YHe:          float = -1.0
    # Tabella BBN.  "" => quella di CLASS (sBBN_file degli extra_args se
    # c'e', altrimenti il default sBBN_2025.dat).
    reio_bbn_file:     str   = ""
    # DeltaN_eff alla BBN, definito come CLASS: N_eff(BBN) - 3.046.  None =>
    # calcolato da N_ur / N_ncdm / deg_ncdm / T_ncdm (bbn_helium.delta_neff_bbn).
    reio_bbn_delta_neff: float | None = None

    # PAVIMENTO DELLA TABELLA = FONDO DI RICOMBINAZIONE DI CLASS x 1.05
    # (dal 29 ott 2026; prima era il numero fisso reio_xe_floor = 1e-3).
    #
    # Dopo la ricombinazione resta un fondo di elettroni liberi, x_e ~ 2.2e-4
    # per atomo di H (cosmologia Planck), che CLASS calcola da se' fino
    # all'inizio della nostra tabella (z_max) e chiama xe_before.  La tabella
    # pero' SOSTITUISCE x_e per z < z_max, e i modelli prima della
    # reionizzazione vanno a zero: serve un valore minimo, per due motivi.
    #
    # 1. CLASS non accetta xe = 0 nei nodi intermedi: divide per dkappa e va
    #    in "variation rate diverges" (thermodynamics.c:3493).
    #
    # 2. CLASS definisce tau_reio come la profondita' ottica fino al MINIMO
    #    GLOBALE di x_e (thermodynamics.c:3062-3082).  Se un tratto della
    #    tabella scende sotto xe_before, il minimo si sposta li' e tau_reio
    #    viene troncata, in silenzio.  Succede ai modelli che possono avere
    #    x_e ~ 0 IN MEZZO alla storia (bin, PCA).  Misurato (29 ott 2026) su
    #    una storia a bin con due buchi:
    #        pavimento 0.99 xe_before -> tau_reio 0.04775   TRONCATA
    #        pavimento 1.00 xe_before -> tau_reio 0.05733
    #        pavimento 1.01 xe_before -> tau_reio 0.05733   (integrale: 0.05733)
    #
    # Il numero fisso 1e-3 soddisfaceva entrambi i vincoli con largo margine,
    # ma e' ~5 volte il fondo vero: CLASS vedeva elettroni che non ci sono, e
    # tau usciva piu' alta (+1.8e-4 per la tanh al massimo della posteriore,
    # 0.03 sigma di Planck; EE a l = 2-30 spostato dell'1.3%).
    #
    # Ora il pavimento e' xe_before x _MARGINE_SUL_FONDO (arrotondato per
    # eccesso a 2 cifre significative, vedi _arrotonda_per_eccesso), con xe_before
    # calcolato da CLASS (solo termodinamica, reio_none) PER LA COSMOLOGIA DEL
    # PUNTO: il minimo globale resta sull'ultimo nodo per qualunque modello, e
    # il bias residuo e' il 5% del fondo (~1e-5 in x_e, qualche 1e-6 su tau).
    # Il fondo dipende solo dalla cosmologia, non dalla reionizzazione: si
    # tiene in memoria e si ricalcola solo quando la cosmologia cambia
    # (_pavimento_del_punto).  Dopo la run vera si verifica che CLASS abbia
    # usato proprio quel fondo (calculate).  Non e' un'opzione dello YAML.
    _MARGINE_SUL_FONDO = 1.05

    # Parametri da NON passare al calcolo del fondo: quelli dei modelli di
    # reionizzazione (tutti con prefisso reio_, piu' zeta_uv di Gompertz) e
    # quelli dello spettro primordiale, che non toccano la ricombinazione
    # (sigma8 come input farebbe partire il calcolo di P(k)).
    _NON_PER_IL_FONDO = frozenset({
        "zeta_uv", "tau_reio", "z_reio", "sigma8", "A_s", "logA",
        "ln10^{10}A_s", "ln_A_s_1e10"})

    # Modelli che contengono gia' il fondo vero x_noreio(z), come reio_camb di
    # CLASS: niente pavimento x 1.05, e il calcolo del fondo tabula x_noreio(z)
    # su [0, z_max] (_calcola_fondo).  Per questi il minimo globale di x_e, e
    # quindi l'estremo dell'integrale di tau_reio, cade dove lo mette CLASS con
    # reio_camb, non sull'ultimo nodo.
    _MODELLI_CON_FONDO = frozenset({"tanh_class"})
    # Passo della tabella di x_noreio(z): la ricombinazione residua varia del
    # 10% fra z = 0 e 30, l'interpolazione lineare a passo 0.05 sbaglia ~1e-10.
    _PASSO_TABELLA_FONDO = 0.05

    # --- Gompertz+SR options ------------------------------------------------
    # How to obtain sigma8 for the SR formula:
    #   "sample"    : sigma8 is a direct MCMC parameter (replaces logA/A_s).
    #                 CLASS normalises A_s internally to match the sampled sigma8.
    #   "class_mPk" : A_s is sampled normally; a quick CLASS run with
    #                 output='mPk' (no CMB Cl) derives sigma8 before the full
    #                 run.  ~10-50x overhead vs a full CLASS call but avoids
    #                 changing the standard CMB parameterisation.
    gompertz_sigma8_source: str = "sample"

    # --- tanh_pinned_end options --------------------------------------------
    # These are FIXED configuration values chosen by the user in the YAML.
    # They are NOT sampled by MCMC — only reio_z_re is sampled.
    #
    # NOTA: prefissate 'reio_pin_' per non collidere con reio_z_end, che nel
    # modello power_law e' invece un parametro CAMPIONATO (blocco params).  Un
    # nome usato sia come opzione della theory sia come parametro di Cobaya e'
    # ambiguo e va evitato.
    #
    # reio_pin_z_end      : redshift at which xe reaches the pinning value.
    #                       Physically: when does reionization "end"?
    # reio_pin_xe_end     : absolute xe value at z_end.  If set, overrides
    #                       reio_pin_xe_end_frac.  E.g. 1.0 means H fully ionised.
    # reio_pin_xe_end_frac: fraction of xe_max (1.0819) at z_end.  Default 0.95,
    #                       i.e. ~5% neutral (in H+HeI combined) at z_end.
    #
    # Example YAML:
    #   theory:
    #     Reiolike.reio_theory.ReioTheory:
    #       reio_model: tanh_pinned_end
    #       reio_pin_z_end: 5.6
    #       reio_pin_xe_end_frac: 0.95   # or reio_pin_xe_end: 1.026
    #   params:
    #     reio_z_re:
    #       prior: {min: 6, max: 15}
    reio_pin_z_end:       float = 5.6
    reio_pin_xe_end:      float = -1.0   # -1 => usa reio_pin_xe_end_frac
    reio_pin_xe_end_frac: float = 0.95

    # --- power_law options ---------------------------------------------------
    # Modello di Douspis et al. 2015 / Planck Int. XLVII 2016, nella forma di
    # Ilic et al. 2025 (arXiv:2504.13254) Eq. (3).  Parametri CAMPIONATI, da
    # mettere nel blocco 'params' del YAML:
    #
    #   reio_z_end : redshift di fine reionizzazione   (paper: U(5, 20) o U(5, 25))
    #   reio_alpha : esponente della power law         (paper: U(0, 20))
    #   reio_z_beg : redshift di inizio                (solo variante 3-pow)
    #
    # Le due varianti del paper:
    #   2-pow : reio_z_beg FISSO a 20 -> lasciarlo fuori da 'params' e usare
    #           l'opzione reio_z_beg qui sotto;
    #   3-pow : reio_z_beg campionato -> metterlo in 'params', e questa
    #           opzione viene ignorata.
    #
    # Example YAML (2-pow):
    #   theory:
    #     Reiolike.reio_theory.ReioTheory:
    #       reio_model: power_law
    #       reio_z_beg: 20.0
    #   params:
    #     reio_z_end: {prior: {min: 5, max: 20}}
    #     reio_alpha: {prior: {min: 0, max: 20}}
    reio_z_beg:       float = 20.0

    # --- binned options ------------------------------------------------------
    # Modello a bin di Ilic et al. 2025 (arXiv:2504.13254) Sect. 2.5.
    #
    # CONVENZIONE: si definiscono i BORDI (edge) dei bin, non i centri.  Con
    # n_bins bin ci sono n_bins+1 bordi; il bin i copre [edge_{i-1}, edge_i] ed
    # e' numerato dal BASSO redshift.  Le transizioni tanh sono centrate sui
    # bordi.  (CLASS, nella sua reio_bins_tanh nativa, vuole invece i CENTRI e
    # mette i salti ai punti medi: stesso modello, input diverso.)
    #
    # Parametri CAMPIONATI, da mettere in 'params':  reio_xHII_bin1 ... binN.
    # Sono la FRAZIONE IONIZZATA x_HII del bin, prior U(0, 1); il pacchetto
    # la moltiplica per 1 + f_He del punto (Y_He dalla BBN, che varia con
    # omega_b e N_eff).  Cosi' x_e del bin ~ U(0, 1 + f_He) per QUALSIASI
    # cosmologia, che e' la definizione del paper.  Fino al 28 ott 2026 si
    # chiamavano reio_xe_binN ed erano x_e, con un tetto fisso 1.0819 giusto
    # solo per la Y_He di Planck (con Delta N_eff = +1: x_HII al massimo
    # 0.995).  Il vecchio nome, o un prior fuori da [0, 1], fermano l'avvio
    # con un messaggio (_controlla_prior_bin).
    #
    # Lo schema si sceglie in uno di due modi:
    #   a) esplicito — reio_bin_edges: [5, 6, 7, 8.5, 10, 13, 17, 25]
    #      Ha la precedenza su tutto il resto.  Serve a chi vuole allineare i
    #      bin ai propri dati invece di usare uno schema regolare.
    #   b) generato — da reio_bin_z_min/z_max piu' reio_n_bins oppure
    #      reio_bin_width, con reio_bin_spacing.
    #
    # Il default riproduce il modello '10-bins' del paper: 10 bin uniformi di
    # larghezza 2 fra z=5 e z=25.
    #
    # reio_bin_fixed_scheme decide fra i due approcci:
    #   True  (default) — schema fisso, a) o b) qui sopra.
    #   False — bordi dai dati: messi nei vuoti fra i redshift dei dati
    #           puntuali, ad almeno reio_bin_data_distance * reio_bin_sharpness
    #           da ognuno, piu' reio_bin_tail_edges sopra l'ultimo dato
    #           (vedi bin_edges_from_data).  Nessun dato puntuale cade su un
    #           gradino, quindi ognuno vincola un bin solo.  Il numero di bin
    #           dipende dal catalogo: viene stampato all'avvio insieme ai
    #           parametri reio_xHII_bin* da dichiarare.  Con reio_bin_edges
    #           da' errore; le opzioni di b) sono ignorate.
    reio_bin_fixed_scheme:  bool  = True
    reio_bin_data_distance: float = 1.5
    reio_bin_tail_edges:    list  = (12.0, 16.0, 25.0)
    reio_bin_edges:     list  = None
    reio_bin_z_min:     float = 5.0
    reio_bin_z_max:     float = 25.0
    reio_bin_width:     float = 2.0
    reio_n_bins:        int   = 0        # se > 0 ha precedenza su reio_bin_width
    reio_bin_spacing:   str   = "uniform"    # "uniform" | "log1p"
    reio_bin_sharpness: float = 0.1      # Delta z delle transizioni (paper: 0.1)
    # Se True, una storia non monotona (xe che cresce al crescere di z) fa
    # scartare il punto: calculate() ritorna False e cobaya gli assegna
    # likelihood nulla.  Di default False, come nel paper, che vuole poter
    # esplorare anche storie non monotone.
    reio_bin_monotonic: bool  = False
    # Nodi di griglia per ogni transizione.  Con molti bordi il budget totale
    # di nodi si esaurisce in fretta (vedi z_max_nodes), quindi qui si usa una
    # finestra +/- 4 larghezze (la tanh e' entro 3.4e-4 dall'asintoto) invece
    # delle 6 usate altrove.
    z_bin_pts_per_edge: int   = 24

    # Limiti di sanita' sullo schema di binning.  Non sono capricci: vedere
    # _binned_edges() per il perche' di ciascuno.
    reio_bin_zmin_warn:  float = 5.0    # sotto: contraddice la foresta Ly-alpha
    reio_bin_zmax_warn:  float = 30.0   # sopra: reionizzazione quasi certo non iniziata
    reio_bin_zmax_hard:  float = 50.0   # sopra: CLASS rifiuta (reionization_z_start_max)
    reio_bin_nmax_warn:  int   = 12     # sopra: il CMB non regge tanti modi
    reio_bin_nmax_hard:  int   = 25     # sopra: MCMC non praticabile

    # --- pca options ---------------------------------------------------------
    # Modello 'pca-5' di Ilic et al. 2025 (arXiv:2504.13254) Sect. 2.6, che a
    # sua volta segue Mortonson & Hu 2008 (arXiv:0705.1132).
    #
    #     xe(z) = xe_fid + sum_{a=1..N} m_a S_a(z)
    #
    # Le autofunzioni S_a e TUTTO il contorno che le definisce (z_min, z_max,
    # delta_z, xe fiduciale, delta_xe, l_max, elio, cosmologia) stanno nel file
    # prodotto da PCA_eigenfunction_reconstruction.py.  Il YAML non li ripete:
    # dice solo quale file e quanti modi.  Vedere Reiolike/pca_basis.py.
    #
    # Parametri CAMPIONATI, da mettere in 'params':  reio_pca_m1 ... mN.
    # I prior giusti si ottengono da:
    #     python -c "from Reiolike.pca_basis import load_basis; \
    #                print(load_basis('eigenfunctions_PCA.npz').priors_yaml(5))"
    #
    # Example YAML:
    #   theory:
    #     Reiolike.reio_theory.ReioTheory:
    #       reio_model: pca
    #       reio_pca_file: eigenfunctions_PCA.npz
    #       reio_pca_n_modes: 5
    reio_pca_file:     str   = ""
    reio_pca_n_modes:  int   = 5

    # Come trattare le ricostruzioni non fisiche.  Con un numero finito di modi
    # la base non e' completa, quindi xe(z) puo' uscire da [0, 1+f_He] anche
    # per storie perfettamente fisiche: Ilic et al. lo dicono esplicitamente
    # ("by construction, ... the pca-5 model allows the ionisation fraction to
    # reach values beyond the physical range").
    #
    #   "mh08"   — (default, = il paper) priori di fisicita' di MH08 Sect. 3:
    #              limiti per singolo modo Eq. (10) [che sono anche i prior del
    #              YAML], vincolo globale sum m_a^2 <= f Eq. (13), e tau
    #              parziale nell'intervallo fisico su una banda larga.  NON si
    #              taglia punto per punto: MH08 spiegano perche' non si deve
    #              ("the uncertainty in modes higher than the first few
    #              prevents us from simply excluding all models where xe < 0 or
    #              xe > 1 at any redshift").
    #   "strict" — in piu' scarta se xe esce da [0, 1+f_He] in un solo bin.
    #              Piu' restrittivo del paper; utile per capire quanto pesa la
    #              scelta, non per riprodurlo.
    #   "none"   — nessun taglio.  Resta comunque il floor numerico di CLASS.
    #
    # DIFFERENZA DICHIARATA RISPETTO AL PAPER.  Ilic et al. usano CAMB per il
    # solo pca-5, "as it allows for the computation of CMB power spectra even
    # in cases where the ionisation fraction temporarily goes negative".  CLASS
    # in reio_inter non lo permette: divide per dkappa e va in "variation rate
    # diverges".  Qui gli xe negativi vengono quindi portati al pavimento
    # (fondo di ricombinazione x 1.05, vedi _MARGINE_SUL_FONDO).
    # L'effetto e' misurato e riportato: vedere il warning emesso quando
    # l'area troncata supera reio_pca_floor_warn.
    reio_pca_physicality: str = "mh08"

    # Bande su cui si controlla la tau parziale, come lista di coppie:
    #   reio_pca_tau_bands: [[6, 30], [15, 30]]
    # None => una sola banda, l'intero intervallo della base.  MH08 usano
    # proprio 15 < z < 30 come esempio di "sufficiently large range".
    reio_pca_tau_bands: list = None

    # xe imposto sotto z_min della base ("reionizzazione conclusa").
    # -1 => 1 + f_He della base.
    reio_pca_xe_complete: float = -1.0

    # Se l'area di xe portata al floor supera questa frazione dell'area totale
    # della storia, si avverte: vuol dire che la catena sta esplorando una
    # regione dove la differenza fra noi e il paper (CLASS vs CAMB) conta.
    reio_pca_floor_warn: float = 0.02

    # Soglia oltre la quale il punto viene SCARTATO invece che solo segnalato.
    # 1.0 (default) = mai, cioe' il comportamento del paper: si accetta tutto
    # quello che i prior di fisicita' lasciano passare, e il floor si subisce.
    #
    # Metterla a un valore piccolo (es. 0.05) confina la catena dove CLASS
    # riproduce fedelmente la storia, ma e' un prior IN PIU' rispetto al paper,
    # dipendente dal codice invece che dalla fisica.  Va usata consapevolmente,
    # per esempio per misurare quanto pesa il floor confrontando due catene.
    reio_pca_max_floor_frac: float = 1.0

    # Smoothing gaussiano della storia ricostruita, come Mortonson & Hu 2008
    # Sect. 5: "there are often sharp transitions at these redshifts since
    # nothing in equation (5) forces xe(z) to satisfy these boundary
    # conditions.  To avoid problems with the time integration in CAMB, we
    # smooth the reionization history by convolving xe(z) with a Gaussian of
    # width sigma_z = 0.5."
    #
    # Il motivo e' NUMERICO: sum_a m_a S_a(z) agli estremi dell'intervallo non
    # vale ne' 1+f_He ne' 0, quindi raccordarla ai plateau crea un salto, e il
    # salto fa impazzire l'integratore temporale del codice di Boltzmann.
    #
    # 0.0 = spento (default).  Il raccordo lineare su Delta z che facciamo gia'
    # in PCABasis.knots() e' di per se' una regolarizzazione, quindi lo
    # smoothing non e' sempre necessario; 0.5 riproduce esattamente MH08.
    reio_pca_smooth_sigma_z: float = 0.0

    # -----------------------------------------------------------------------
    def _compute_sigma8_from_As(self, omega_b, omega_cdm, H0, n_s,
                                 logA=None, A_s=None):
        """
        Derive sigma8 from A_s using a lightweight CLASS call.

        Only requests output='mPk' (matter power spectrum, no CMB Cl).
        sigma8 is independent of the reionisation history, so a dummy
        tau_reio is used; this introduces no error.

        Parameters
        ----------
        omega_b, omega_cdm : float  — Omega_b*h², Omega_cdm*h²
        H0                 : float  — Hubble constant [km/s/Mpc]
        n_s                : float  — spectral index
        logA               : float, optional  — ln(10^10 A_s)
        A_s                : float, optional  — A_s directly
        (at least one of logA / A_s must be provided)

        Returns
        -------
        sigma8 : float
        """
        if _ClassPy is None:
            raise ImportError(
                "Il modulo 'classy' non è disponibile. "
                "Usa gompertz_sigma8_source='sample' oppure installa classy."
            )
        if logA is not None:
            A_s_val = float(np.exp(logA) / 1.0e10)
        elif A_s is not None:
            A_s_val = float(A_s)
        else:
            raise ValueError(
                "_compute_sigma8_from_As: fornire 'logA' oppure 'A_s'."
            )

        cosmo = _ClassPy()
        try:
            cosmo.set({
                'output'         : 'mPk',
                'P_k_max_h/Mpc'  : 1.0,
                'omega_b'        : float(omega_b),
                'omega_cdm'      : float(omega_cdm),
                'H0'             : float(H0),
                'n_s'            : float(n_s),
                'A_s'            : A_s_val,
                'reio_parametrization': 'reio_camb',
                'tau_reio'       : 0.055,
            })
            cosmo.compute()
            s8 = cosmo.sigma8()
        finally:
            cosmo.struct_cleanup()
            cosmo.empty()
        return float(s8)

    # -----------------------------------------------------------------------
    def initialize(self):
        super().initialize()
        if self.z_min != 0.0:
            self.log.warning(
                "z_min=%g ignorato: CLASS richiede che il primo nodo di "
                "reio_inter sia z=0. La griglia parte comunque da 0.", self.z_min,
            )
        # Era un print(): finiva sempre su stdout, anche dentro una catena.
        # Attraverso il logger si vede solo con debug attivo.
        self.log.debug("z_max=%g, z_points=%d, reio_include_HeII=%s",
                       self.z_max, self.z_points, self.reio_include_HeII)
        if self.reio_model == "gompertz_sr" and self.z_max > self._GOMPERTZ_Z_MAX:
            raise ValueError(
                f"ReioTheory (gompertz_sr): z_max={self.z_max:g} supera "
                f"{self._GOMPERTZ_Z_MAX:g}. Il polinomio P5 dell'Eq. 2.3 di "
                "Montero-Camacho et al. 2024 ha l'ultimo coefficiente negativo "
                "(c5 = -6.518e-5): a redshift abbastanza alto P5 torna a "
                "crescere e x_HI = exp(-exp(P5)) riscende verso 0, cioe' "
                "compare una seconda ionizzazione FINTA, fuori dall'intervallo "
                "in cui il polinomio e' stato fittato sulle simulazioni. Dentro "
                "l'intervallo di validita' del paper (Eq. 2.1) comincia sopra "
                "z ~ 32 (con la cosmologia di Planck sopra z ~ 40): una tabella "
                "oltre z = 30 rischia di passarla a CLASS, che la conterebbe in "
                f"tau. Usa z_max <= {self._GOMPERTZ_Z_MAX:g}.")

    def initialize_with_provider(self, provider):
        super().initialize_with_provider(provider)
        if self.reio_model == "tanh_pinned_end":
            self._controlla_prior_pinned(getattr(provider, "model", None))
        elif self.reio_model == "gompertz_sr":
            self._controlla_prior_gompertz(getattr(provider, "model", None))
        elif self.reio_model == "binned":
            self._controlla_prior_bin(getattr(provider, "model", None))

    def _controlla_prior_bin(self, model):
        """
        I reio_xHII_binN sono frazioni x_HII in [0, 1] (dal 28 ott 2026): un
        prior che esce da [0, 1] ferma l'avvio.  Il caso tipico e' uno YAML
        scritto per la convenzione precedente, prior U(0, 1.0819) su x_e: oggi
        darebbe bin fino a x_HII = 1.08, che calculate() scarterebbe in
        silenzio, troncando il prior all'8% in meno.  Prior illimitati
        (gaussiani): avviso, le code fuori da [0, 1] vengono scartate.
        """
        if model is None:
            return
        # Il nome di prima: senza questo controllo i parametri finirebbero a
        # CLASS, che si fermerebbe con "did not read input parameter(s)".
        vecchi = sorted(p for p in model.parameterization.input_params()
                        if p.startswith("reio_xe_bin"))
        if vecchi:
            raise ValueError(
                f"ReioTheory (binned): {', '.join(vecchi)}: dal 28 ott 2026 i "
                "parametri dei bin si chiamano reio_xHII_binN e sono la "
                "frazione ionizzata x_HII in [0, 1], non piu' x_e in [0, 1.0819]. "
                "Rinominali e metti il prior U(0, 1): il pacchetto moltiplica "
                "per 1 + f_He del punto, cosi' x_e ~ U(0, 1 + f_He) per "
                "qualsiasi Y_He, come nel paper.")
        nomi = list(model.prior.params)
        bin_nomi = [p for p in nomi if p.startswith("reio_xHII_bin")]
        costanti = {k: v for k, v in model.parameterization.constant_params().items()
                    if k.startswith("reio_xHII_bin")}
        fuori = [f"{k} = {float(v):g}" for k, v in costanti.items()
                 if not 0.0 <= float(v) <= 1.0]
        illimitati = []
        if bin_nomi:
            b = model.prior.bounds(confidence_for_unbounded=1.0)
            for k in bin_nomi:
                lo, hi = (float(x) for x in b[nomi.index(k)])
                if not (np.isfinite(lo) and np.isfinite(hi)):
                    illimitati.append(k)
                elif lo < -1e-12 or hi > 1.0 + 1e-12:
                    fuori.append(f"{k} in [{lo:g}, {hi:g}]")
        if fuori:
            raise ValueError(
                "ReioTheory (binned): i parametri reio_xHII_binN sono la frazione "
                "ionizzata x_HII del bin, in [0, 1]; il pacchetto la moltiplica "
                "per 1 + f_He del punto, cosi' x_e ~ U(0, 1 + f_He) per qualsiasi "
                "Y_He (definizione di Ilic et al. 2025). Fuori da [0, 1]: "
                + "; ".join(fuori) + ". Se lo YAML usa il vecchio prior "
                "U(0, 1.0819) su x_e, il nuovo e' U(0, 1).")
        if illimitati:
            self.log.warning(
                "binned: prior illimitato su %s: i punti con x_HII fuori da "
                "[0, 1] vengono scartati, il prior effettivo e' troncato li'.",
                ", ".join(illimitati))

    # Eq. 2.1 di Montero-Camacho et al. 2024: l'intervallo dei parametri su cui
    # sono state fittate le formule SR (Eqs. 3.1-3.2) e la forma P5 (Eq. 2.3).
    _GOMPERTZ_BOX = {"sigma8": (0.74, 0.90), "ns": (0.92, 1.00), "h": (0.61, 0.73),
                     "Omega_b": (0.04, 0.06), "Omega_m": (0.24, 0.40),
                     "zeta_uv": (20.0, 35.0)}

    def _controlla_prior_gompertz(self, model):
        """
        Avvisa se i prior del run escono dall'intervallo di validita' del
        paper (Eq. 2.1).  Fuori da li' le formule SR sono estrapolazioni senza
        garanzie, e il polinomio P5 puo' produrre una seconda ionizzazione
        finta: misurato il 28 ott 2026 (Tests/test_affidabilita_gompertz.py),
        con zeta_uv fino a 100 succede sotto z = 30 nel 14% dei punti e falsa
        tau in mediana di 0.08.  Dentro l'Eq. 2.1 non succede mai sotto z = 32.

        Un avviso e non un errore: allargare il prior puo' essere una scelta
        voluta (test, confronti).  Gli intervalli vengono dal prior dei
        parametri campionati (al 99.99% se illimitato) o dal valore di quelli
        fissati; Omega_b = omega_b/h^2 e Omega_m = (omega_b + omega_cdm)/h^2
        si prendono agli estremi della combinazione dei tre prior.
        """
        if model is None:
            return
        costanti = model.parameterization.constant_params()
        nomi = list(model.prior.params)
        bounds = model.prior.bounds(confidence_for_unbounded=0.9999) if nomi else []

        def intervallo(p):
            if p in costanti:
                v = float(costanti[p])
                return v, v
            if p in nomi:
                lo, hi = bounds[nomi.index(p)]
                return float(lo), float(hi)
            return None

        r = {"zeta_uv": intervallo("zeta_uv"), "ns": intervallo("n_s")}
        if self.gompertz_sigma8_source == "sample":
            r["sigma8"] = intervallo("sigma8")
        H0, ob, oc = intervallo("H0"), intervallo("omega_b"), intervallo("omega_cdm")
        if H0 is not None:
            r["h"] = (H0[0] / 100.0, H0[1] / 100.0)
            if ob is not None:
                r["Omega_b"] = (ob[0] / r["h"][1] ** 2, ob[1] / r["h"][0] ** 2)
                if oc is not None:
                    r["Omega_m"] = ((ob[0] + oc[0]) / r["h"][1] ** 2,
                                    (ob[1] + oc[1]) / r["h"][0] ** 2)

        fuori = [f"{k} in [{v[0]:.4g}, {v[1]:.4g}] (paper: [{self._GOMPERTZ_BOX[k][0]:g}, "
                 f"{self._GOMPERTZ_BOX[k][1]:g}])"
                 for k, v in r.items()
                 if v is not None and (v[0] < self._GOMPERTZ_BOX[k][0] - 1e-12
                                       or v[1] > self._GOMPERTZ_BOX[k][1] + 1e-12)]
        non_controllati = [k for k in self._GOMPERTZ_BOX if r.get(k) is None]
        if fuori:
            self.log.warning(
                "gompertz_sr: i prior escono dall'intervallo di validita' del "
                "paper (Montero-Camacho et al. 2024, Eq. 2.1): %s. Fuori da "
                "li' le formule SR sono estrapolazioni senza garanzie, e il "
                "polinomio P5 puo' generare una seconda ionizzazione finta ad "
                "alto z che falsa tau (con zeta_uv fino a 100: tau +0.08 in "
                "mediana). Se non e' voluto, restringi i prior.",
                "; ".join(fuori))
        if non_controllati:
            self.log.debug("gompertz_sr: non controllati contro l'Eq. 2.1 "
                           "(non campionati ne' fissati con questo nome): %s",
                           ", ".join(non_controllati))

    def _controlla_prior_pinned(self, model):
        """
        Il pinned richiede z_re > reio_pin_z_end: sotto, il midpoint starebbe
        dopo il punto fisso e la storia non esiste.  calculate() scarta quei
        punti, ma se e' il PRIOR a includerli il problema e' di configurazione
        (tipicamente: il prior U(5, 12) della run tanh riusato col punto fisso
        a 5.6), e conviene saperlo all'avvio, non dopo ore di cluster con una
        fetta di prior rifiutata in silenzio.

        - prior con estremo inferiore finito sotto z_end -> errore;
        - prior illimitato (gaussiano...) -> avviso: la coda sotto z_end viene
          scartata da calculate();
        - z_re fissato a un valore <= z_end -> errore: ogni punto sarebbe nullo.
        """
        if model is None:
            return
        # Chi passa dalla tanh al pinned e lascia reio_delta_z nel blocco
        # params: senza questo controllo arrivava a CLASS, che si fermava con
        # "did not read input parameter(s): reio_delta_z" senza dire perche'.
        if ("reio_delta_z" in model.parameterization.sampled_params()
                or "reio_delta_z" in model.parameterization.constant_params()):
            raise ValueError(
                "ReioTheory (tanh_pinned_end): reio_delta_z e' nel blocco "
                "'params', ma col modello pinned la larghezza NON si campiona: "
                "e' ricavata da z_re e dal punto fisso (reio_pin_z_end, "
                "reio_pin_xe_end_frac). Togli reio_delta_z dal blocco params "
                "(o usa reio_model: tanh se vuoi la larghezza libera).")
        z_end = float(self.reio_pin_z_end)
        costanti = model.parameterization.constant_params()
        if "reio_z_re" in costanti:
            if float(costanti["reio_z_re"]) <= z_end:
                raise ValueError(
                    f"ReioTheory (tanh_pinned_end): reio_z_re e' fissato a "
                    f"{costanti['reio_z_re']:g}, non sopra reio_pin_z_end="
                    f"{z_end:g}: il midpoint starebbe dopo il punto fisso, "
                    "nessuna storia e' possibile.")
            return
        nomi = list(model.prior.params)
        if "reio_z_re" not in nomi:
            return
        lo = float(model.prior.bounds(confidence_for_unbounded=1.0)[nomi.index("reio_z_re")][0])
        if not np.isfinite(lo):
            self.log.warning(
                "tanh_pinned_end: il prior di reio_z_re non ha estremo inferiore; "
                "i punti con z_re <= reio_pin_z_end=%g vengono scartati "
                "(likelihood nulla), quindi il prior effettivo e' troncato li'.",
                z_end)
        elif lo < z_end:
            raise ValueError(
                f"ReioTheory (tanh_pinned_end): il prior di reio_z_re parte da "
                f"{lo:g}, sotto reio_pin_z_end={z_end:g}. Fra {lo:g} e {z_end:g} "
                "il midpoint starebbe dopo il punto fisso e la storia non "
                "esiste: quei punti verrebbero tutti scartati. Metti il minimo "
                f"del prior ad almeno {z_end:g} (o abbassa reio_pin_z_end).")

    # -----------------------------------------------------------------------
    # Costruzione della griglia
    # -----------------------------------------------------------------------
    @staticmethod
    def estimate_transition(z, xe):
        """
        Stima (z_mid, delta_z) di una transizione tipo tanh a partire da una
        storia xe(z) tabulata, senza sapere che modello l'ha generata.

        Per una tanh, xe normalizzata vale 0.75 / 0.25 in
            z = z_mid -/+ 0.5493 * delta_z
        da cui  delta_z = (z_25 - z_75) / 1.0986.

        Parameters
        ----------
        z  : array crescente
        xe : array, decrescente in z (la reionizzazione avviene a z bassi)

        Returns
        -------
        (z_mid, delta_z) : tuple of float.  Se xe e' piatta restituisce
        (mediana della griglia, span/10) come fallback innocuo.
        """
        z  = np.asarray(z,  dtype=float)
        xe = np.asarray(xe, dtype=float)

        lo, hi = float(np.min(xe)), float(np.max(xe))
        span = hi - lo
        if not np.isfinite(span) or span <= 0.0:
            return float(np.median(z)), max((z[-1] - z[0]) / 10.0, 0.1)

        xn = (xe - lo) / span
        # np.interp vuole ascisse crescenti: xn cresce al diminuire di z,
        # quindi si interpola su z invertito.
        zr, xr = z[::-1], xn[::-1]
        if not np.all(np.diff(xr) >= 0):
            order = np.argsort(xr, kind="stable")
            zr, xr = zr[order], xr[order]

        z_mid = float(np.interp(0.50, xr, zr))
        z_75  = float(np.interp(0.75, xr, zr))
        z_25  = float(np.interp(0.25, xr, zr))

        delta_z = (z_25 - z_75) / 1.0986
        if not np.isfinite(delta_z) or delta_z <= 0.0:
            delta_z = max((z[-1] - z[0]) / 10.0, 0.1)
        return z_mid, delta_z

    # -----------------------------------------------------------------------
    # Budget imposto dal parser di CLASS
    # -----------------------------------------------------------------------
    def _class_arg_limit(self):
        """Lunghezza massima ammessa per reio_inter_z / reio_inter_xe."""
        lim = getattr(self, "_class_arg_limit_cached", None)
        if lim is not None:
            return lim

        if self.class_arg_max > 0:
            lim = int(self.class_arg_max)
            self.log.info(
                "limite del parser di CLASS: %d caratteri (imposto a mano con "
                "class_arg_max) -> al massimo ~%d nodi.",
                lim, (lim - 1) // _CHARS_PER_NODE)
        else:
            lim, line_max, src = _detect_class_arg_max()
            if src is None:
                self.log.warning(
                    "non sono riuscito a leggere _ARGUMENT_LENGTH_MAX_ da "
                    "nessun include/parser.h della CLASS importata: assumo il "
                    "default di CLASS, %d caratteri (~%d nodi). Se la tua "
                    "CLASS e' stata ricompilata con un valore piu' alto, "
                    "impostalo con class_arg_max nel blocco theory del YAML "
                    "per non sprecare risoluzione. Per leggerlo:  grep "
                    "_ARGUMENT_LENGTH_MAX_ <cartella_class>/include/parser.h",
                    lim, (lim - 1) // _CHARS_PER_NODE)
            else:
                self.log.info(
                    "limite del parser di CLASS: %d caratteri (letto da %s) "
                    "-> al massimo ~%d nodi nella tabella reio_inter.",
                    lim, src, (lim - 1) // _CHARS_PER_NODE)
                if line_max is not None and line_max < lim:
                    self.log.warning(
                        "in %s _LINE_LENGTH_MAX_ (%d) e' piu' piccolo di "
                        "_ARGUMENT_LENGTH_MAX_ (%d). Da python non e' un "
                        "problema (classy riempie la struttura direttamente), "
                        "ma lanciando CLASS da riga di comando con un .ini le "
                        "righe lunghe verrebbero troncate.", src, line_max, lim)

        self._class_arg_limit_cached = lim
        return lim

    def _node_budget(self):
        """
        Numero massimo di nodi utilizzabile, dal piu' stretto fra il tetto di
        buon senso (z_max_nodes) e quello imposto dal parser di CLASS.

        Con CLASS a 4096 caratteri si arriva a ~400 nodi; con CLASS a 1024
        (versioni piu' vecchie e diversi fork) si scende a ~100.  Il vecchio
        codice committato usava z_points=100, e infatti stava sotto 1024
        senza saperlo: e' la ragione per cui non aveva mai dato problemi.
        """
        return max(min(int(self.z_max_nodes),
                       (self._class_arg_limit() - 1) // _CHARS_PER_NODE), 12)

    def _fit_class_strings(self, z_grid, xe, protect=None):
        """
        Formatta la tabella per CLASS, decimandola quanto basta perche' le due
        stringhe stiano nel buffer.

        _node_budget() e' una stima (10 caratteri per nodo); qui invece si
        MISURA la stringa vera e si decima finche' non entra.  E' l'ultima
        garanzia: superare il buffer non da' un errore pulito, la stringa
        scrive oltre la fine di FileArg e CLASS legge spazzatura o aborta.

        I nodi in `protect` (kink della power law, bordi dei bin, centri dei
        bin del PCA) sopravvivono sempre alla decimazione: sono i punti in cui
        la spezzata deve avere un vertice perche' il modello sia quello giusto.
        """
        z = np.asarray(z_grid, dtype=float)
        x = np.asarray(xe, dtype=float)
        prot = np.zeros(len(z), dtype=bool)
        prot[0] = prot[-1] = True
        if protect is not None and len(np.atleast_1d(protect)):
            prot |= np.isin(z, np.asarray(protect, dtype=float))

        limit = self._class_arg_limit()
        n0 = len(z)
        while True:
            sz = ','.join(f'{v:.8g}' for v in z)
            sx = ','.join(f'{v:.6g}' for v in x)
            longest = max(len(sz), len(sx))
            if longest < limit:
                if len(z) < n0:
                    self.log.warning(
                        "tabella ridotta da %d a %d nodi per stare nei %d "
                        "caratteri del parser di CLASS. Per piu' risoluzione "
                        "bisogna alzare _ARGUMENT_LENGTH_MAX_ in "
                        "include/parser.h e ricompilare (di serie e' 1024).",
                        n0, len(z), limit)
                return z, x, sz, sx

            per_node = longest / len(z)
            target = int((limit - 1) / per_node)
            if target >= len(z):
                target = len(z) - 1
            if target < int(prot.sum()):
                raise ValueError(
                    f"ReioTheory: la tabella non sta nei {limit} caratteri del "
                    f"parser di CLASS nemmeno tenendo solo i {int(prot.sum())} "
                    "nodi indispensabili (kink, bordi dei bin o centri dei bin "
                    "del PCA). Le opzioni sono: usare uno schema con meno bin, "
                    "una base PCA con delta_z piu' largo, oppure una CLASS "
                    "compilata con _ARGUMENT_LENGTH_MAX_ piu' grande "
                    f"(di serie e' 1024, qui ne risultano {limit})."
                )
            keep = np.unique(np.concatenate([
                np.linspace(0, len(z) - 1, max(target, 2)).round().astype(int),
                np.flatnonzero(prot),
            ]))
            if len(keep) >= len(z):        # non si riesce a ridurre oltre
                keep = keep[:-1] if not prot[keep[-1]] else keep
                if len(keep) >= len(z):
                    raise ValueError(
                        "ReioTheory: impossibile ridurre ulteriormente la "
                        f"tabella sotto i {limit} caratteri del parser di CLASS."
                    )
            z, x, prot = z[keep], x[keep], prot[keep]

    def _refine(self, center, width, n_points):
        """Blocco di nodi attorno a `center`, largo +/- z_window_widths*width."""
        half = self.z_window_widths * width
        lo = max(0.0, center - half)
        hi = min(self.z_max, center + half)
        if hi <= lo:
            return np.array([])
        return np.linspace(lo, hi, max(int(n_points), 2))

    def _build_grid(self, z_re=None, delta_z=None, extra_windows=(),
                    extra_nodes=None, protect_nodes=None):
        """
        Griglia z adattiva: unione di una griglia di fondo su [0, z_max] e di
        una o piu' "finestre di raffinamento" centrate dove xe(z) varia.

        Il passo dentro una finestra e' legato alla larghezza della transizione,
        non fissato a priori: l'errore dell'interpolazione lineare di una tanh
        di larghezza w campionata con passo h vale ~ 0.048*A*(h/w)^2, quindi
        h ~ w/z_pts_per_width tiene l'errore sotto ~0.1% con
        z_pts_per_width = 8.

        Finestre create automaticamente:
          - la transizione principale, centrata su z_re con larghezza delta_z;
          - lo step HeII (z=3.5, larghezza 0.5) se reio_include_HeII e' True:
            senza questa, per z_re alto la zona attorno a z~3.5 restava
            campionata a passo ~1.2 e lo step dell'elio veniva spianato.

        Parameters
        ----------
        z_re    : float or None — centro della transizione principale.  None
                  disattiva la finestra principale: lo usa power_law, che
                  copre gia' tutto [z_end, z_beg] via extra_nodes e altrimenti
                  spenderebbe ~100 nodi in doppioni.
        delta_z : float or None — larghezza; se None si usa una larghezza
                  equivalente alla vecchia finestra fissa z_re +/- 5
        extra_windows : iterable di (centro, larghezza) — finestre aggiuntive
        extra_nodes   : array-like or None — nodi espliciti da includere.
                  Serve ai modelli la cui transizione non e' una tanh
                  localizzata: power_law li usa per campionare uniformemente
                  in xe e per mettere un nodo esattamente sui due kink.
        protect_nodes : array-like or None — nodi che devono sopravvivere sia
                  al filtro di spaziatura minima sia alla decimazione finale.
                  Vanno messi qui i punti dove xe ha una discontinuita' nella
                  derivata: se il nodo cade esattamente sul kink,
                  l'interpolazione lineare di CLASS lo attraversa senza errore.

        Returns
        -------
        z_grid : ndarray strettamente crescente, z[0]=0, z[-1]=z_max
        """
        if delta_z is None or not np.isfinite(delta_z) or delta_z <= 0.0:
            # equivalente alla vecchia finestra fissa z_re +/- 5
            delta_z = 5.0 / self.z_window_widths

        # punti necessari per l'accuratezza richiesta sulla finestra
        n_acc = int(2 * self.z_window_widths * self.z_pts_per_width)

        blocks = [np.linspace(0.0, self.z_max, max(int(self.z_bg_points), 2))]
        # finestra principale: si rispetta z_points del YAML se piu' fitto
        if z_re is not None:
            blocks.append(self._refine(z_re, delta_z,
                                       max(int(self.z_points), n_acc)))

        if self.reio_include_HeII:
            blocks.append(self._refine(self.reio_z_HeII,
                                       self.reio_delta_z_HeII, n_acc))
        for centre, width in extra_windows:
            blocks.append(self._refine(centre, width, n_acc))

        if extra_nodes is not None:
            en = np.asarray(extra_nodes, dtype=float).ravel()
            blocks.append(en[(en >= 0.0) & (en <= self.z_max)])

        # nodi fissi nei z dei dati puntuali, protetti come i kink
        z_dati = self._nodi_dati(protect_nodes)
        blocks.append(z_dati)

        z_grid = np.concatenate([b for b in blocks if b.size])
        z_grid = np.unique(z_grid)

        # Estremi imposti dalla convenzione reio_inter di CLASS
        z_grid = np.clip(z_grid, 0.0, self.z_max)
        z_grid[0], z_grid[-1] = 0.0, self.z_max

        # Scarta nodi troppo vicini: dopo la formattazione a 8 cifre
        # significative due z quasi uguali diventerebbero identici e CLASS
        # rifiuterebbe la tabella ("expected in growing order").
        #
        # I nodi "protetti" (estremi + quelli in protect_nodes, tipicamente i
        # kink) sopravvivono sempre: sono i vicini a essere scartati.  Senza
        # questo, con power_law e alpha piccolo il campionamento si addensa a
        # meno di z_min_spacing da z_beg, il filtro tiene il primo nodo che
        # incontra salendo in z e butta via z_beg stesso — trasformando uno
        # stacco quasi verticale in una rampa lunga e sbagliata.
        prot = np.zeros(len(z_grid), dtype=bool)
        prot[0] = prot[-1] = True
        if protect_nodes is not None:
            prot |= np.isin(z_grid, np.asarray(protect_nodes, dtype=float))
        prot |= np.isin(z_grid, z_dati)

        keep = [0]
        for i in range(1, len(z_grid)):
            close = z_grid[i] - z_grid[keep[-1]] < self.z_min_spacing
            if not close:
                keep.append(i)
            elif prot[i]:
                # sacrifica i vicini non protetti pur di tenere questo nodo
                while (len(keep) > 1 and not prot[keep[-1]]
                       and z_grid[i] - z_grid[keep[-2]] < self.z_min_spacing):
                    keep.pop()
                if not prot[keep[-1]]:
                    keep[-1] = i
                elif z_grid[i] - z_grid[keep[-1]] >= self.z_min_spacing:
                    keep.append(i)
        z_grid = z_grid[keep]

        # Tetto al numero di nodi.  Il valore viene da _node_budget(), cioe'
        # dal piu' stretto fra z_max_nodes e quello che il parser della CLASS
        # installata puo' ricevere: con CLASS a 4096 caratteri sono ~400 nodi,
        # con CLASS a 1024 sono ~100.  Il sottocampionamento e' uniforme negli
        # indici, quindi conserva la densita' relativa fra finestre fitte e
        # griglia di fondo.
        budget = self._node_budget()
        if len(z_grid) > budget:
            prot = prot[keep]
            idx = np.unique(np.concatenate([
                np.linspace(0, len(z_grid) - 1, budget).round().astype(int),
                np.flatnonzero(prot),          # i kink non si decimano mai
            ]))
            if not getattr(self, "_budget_warned", False):
                self._budget_warned = True
                self.log.warning(
                    "griglia da %d nodi ridotta a %d: e' il massimo che entra "
                    "nei %d caratteri del parser della CLASS installata. Per "
                    "avere piu' risoluzione bisogna alzare "
                    "_ARGUMENT_LENGTH_MAX_ in include/parser.h e ricompilare "
                    "CLASS (di serie e' 1024). Il costo di lavorare cosi' e' "
                    "misurato in Tests/test_limite_parser_class.py: meno di "
                    "0.06 sigma su tau, quindi di norma non serve.",
                    len(z_grid), len(idx), self._class_arg_limit(),
                )
            z_grid = z_grid[idx]

        # l'ultimo nodo deve restare esattamente z_max
        if z_grid[-1] != self.z_max:
            z_grid[-1] = self.z_max

        # Memorizza i nodi da proteggere anche nella decimazione finale fatta
        # da _fit_class_strings(), che agisce dopo il calcolo di xe.
        self._last_protect = np.concatenate([
            np.asarray(protect_nodes, dtype=float).ravel()
            if protect_nodes is not None else np.array([]),
            z_dati])
        return z_grid

    def _nodi_dati(self, protect_nodes=None):
        """
        I redshift dei dati puntuali (da must_provide) da usare come nodi.

        Un dato quotato a un solo z si confronta con il modello in quel z.  Se
        li' non c'e' un nodo, la likelihood legge la retta fra i due nodi
        vicini, e siccome i nodi delle finestre di raffinamento si spostano
        con i parametri campionati (z_re, delta_z, z_end, alpha...), l'errore
        della retta sul dato sale e scende a ogni passo fra nodi: un dente di
        sega sulla logL.  Con un nodo fisso nel z del dato il valore e' quello
        esatto del modello per ogni punto della catena.

        Si scartano i z fuori da (0, z_max), dove c'e' gia' un estremo, e
        quelli a meno di z_min_spacing da un nodo di protect_nodes ma diversi
        da esso: due nodi protetti cosi' vicini non sopravvivono entrambi al
        filtro di spaziatura, e il kink del modello conta piu' del nodo del
        dato (a 1e-5 di distanza l'errore sul dato e' ~1e-5 * pendenza).
        """
        z = np.asarray(getattr(self, "_z_nodi_dati", ()), dtype=float)
        z = z[(z > 0.0) & (z < self.z_max)]
        if z.size and protect_nodes is not None:
            p = np.asarray(protect_nodes, dtype=float).ravel()
            if p.size:
                d = np.abs(z[:, None] - p[None, :])
                z = z[~np.any((d > 0.0) & (d < self.z_min_spacing), axis=1)]
        if z.size > 1:
            # dati quasi coincidenti: ne basta uno
            z = z[np.r_[True, np.diff(z) >= self.z_min_spacing]]
        return z

    def _power_law_grid(self, z_end, alpha, z_beg):
        """
        Griglia z per il modello power_law.

        La transizione non e' localizzata come una tanh: occupa tutto
        [z_end, z_beg] e ha due kink (derivata discontinua) agli estremi.
        Servono tre cose:

          1. nodi ESATTAMENTE su z_end e z_beg (protect_nodes), cosi'
             l'interpolazione lineare di CLASS attraversa i kink senza errore;
          2. campionamento uniforme in z sull'intervallo;
          3. campionamento uniforme in xe, cioe' u = t^(1/alpha).  Questo e' il
             punto delicato: per alpha<1 la curvatura diverge a z->z_beg, per
             alpha>>1 si concentra a z->z_end.  Equispaziare in xe mette i nodi
             dove servono in entrambi i regimi, cosa che una griglia uniforme
             in z non fa.

        Metodo separato da calculate() cosi' che gli script di test possano
        ispezionare la stessa griglia che finisce in CLASS, senza duplicarne
        la ricetta.
        """
        n_each = max(int(self.z_points),
                     int(2 * self.z_window_widths * self.z_pts_per_width))
        t = np.linspace(0.0, 1.0, n_each)
        u = t if alpha <= 0 else np.unique(
            np.concatenate([t, t ** (1.0 / alpha)]))
        pl_nodes = np.concatenate([z_beg - (z_beg - z_end) * u,
                                   [z_end, z_beg]])

        # z_re=None: niente finestra principale automatica, pl_nodes copre gia'
        # tutto [z_end, z_beg] e i nodi risparmiati servono all'addensamento
        # vicino a z_beg quando alpha < 1.
        return self._build_grid(z_re=None, extra_nodes=pl_nodes,
                                protect_nodes=[z_end, z_beg])

    def _binned_edges(self):
        """
        Costruisce, valida e mette in cache i bordi dei bin.

        Lo schema e' configurazione FISSA (non e' campionato), quindi si calcola
        una volta sola.  La cache e' pigra cosi' funziona anche se l'oggetto non
        e' passato per initialize() (es. negli script di test).

        I limiti applicati, e il perche':

        - z_min sotto ~5 -> warning.  La foresta Ly-alpha mostra che l'IGM e'
          gia' completamente ionizzato a z < 5.3 (Bosman et al. 2022), quindi
          sotto quel redshift non c'e' niente da vincolare: il modello fissa
          xe = 1+f_He e basta.  Mettere bordi piu' in basso spreca parametri.

        - z_max sopra ~30 -> warning; sopra 50 -> errore.  A z > 30 non ci sono
          ancora sorgenti ionizzanti in quantita' apprezzabile, quindi un bin
          li' e' solo prior: non lo vincola nessun dato e allarga la posterior
          di tau senza informazione.  Il 50 e' il limite duro di CLASS
          (reionization_z_start_max), oltre il quale rifiuta comunque.

        - numero di bin sopra ~12 -> warning; sopra 25 -> errore.  Il CMB da
          solo porta circa 5 modi indipendenti sulla storia di reionizzazione
          (Mortonson & Hu 2008), quindi oltre una decina di bin si sta
          campionando quasi solo il prior; e ogni bin in piu' e' una dimensione
          in piu' per l'MCMC, che con i dati astrofisici a basso z puo' avere
          senso fino a un certo punto ma non oltre.

        - sharpness: deve stare sotto meta' del bin piu' stretto, altrimenti
          transizioni adiacenti si fondono e i "bin" non sono piu' plateau;
          e sopra 0.01, sotto cui servirebbero troppi nodi di griglia.
        """
        cached = getattr(self, "_bin_edges_cache", None)
        if cached is not None:
            return cached

        if not self.reio_bin_fixed_scheme:
            if self.reio_bin_edges:
                raise ValueError(
                    "ReioTheory (binned): reio_bin_fixed_scheme: False mette i "
                    "bordi nei vuoti fra i dati, ma c'e' anche reio_bin_edges "
                    f"({list(self.reio_bin_edges)}). Scegli uno dei due: "
                    "togli reio_bin_edges, oppure metti reio_bin_fixed_scheme: True.")
            z_dati = getattr(self, "_z_nodi_dati", np.array([]))
            if not len(z_dati):
                raise ValueError(
                    "ReioTheory (binned): reio_bin_fixed_scheme: False vuole i "
                    "redshift dei dati puntuali, che arrivano da ReioLike "
                    "dentro un modello cobaya; qui non ce ne sono (nessuna "
                    "ReioLike, o nessun dato senza finestra in z). Usa "
                    "reio_bin_fixed_scheme: True con reio_bin_edges.")
            edges = bin_edges_from_data(
                z_dati, delta_z=self.reio_bin_sharpness,
                min_distance=self.reio_bin_data_distance,
                tail_edges=self.reio_bin_tail_edges)
            source = (f"dai dati: nei vuoti fra {len(z_dati)} redshift di dati "
                      f"puntuali, a >= {self.reio_bin_data_distance:g} x "
                      f"sharpness da ognuno")
        elif self.reio_bin_edges:
            edges = np.asarray(self.reio_bin_edges, dtype=float).ravel()
            if edges.size < 2:
                raise ValueError(
                    "ReioTheory (binned): reio_bin_edges deve avere almeno 2 "
                    f"bordi (= 1 bin), ne ha {edges.size}."
                )
            if np.any(np.diff(edges) <= 0):
                raise ValueError(
                    "ReioTheory (binned): reio_bin_edges deve essere "
                    f"strettamente crescente, ricevuto {list(edges)}."
                )
            source = "espliciti da reio_bin_edges"
        else:
            edges = make_bin_edges(
                z_min=self.reio_bin_z_min, z_max=self.reio_bin_z_max,
                n_bins=(self.reio_n_bins if self.reio_n_bins > 0 else None),
                width=self.reio_bin_width, spacing=self.reio_bin_spacing,
            )
            source = (f"generati: {self.reio_bin_spacing}, "
                      f"z in [{self.reio_bin_z_min}, {self.reio_bin_z_max}]")

        n_bins = edges.size - 1
        widths = np.diff(edges)

        # --- limiti duri ---------------------------------------------------
        if edges[0] < 0.0:
            raise ValueError(
                f"ReioTheory (binned): il primo bordo e' {edges[0]}, "
                "non puo' essere negativo."
            )
        if edges[-1] > self.reio_bin_zmax_hard:
            raise ValueError(
                f"ReioTheory (binned): l'ultimo bordo e' z={edges[-1]:g}, oltre "
                f"il limite di {self.reio_bin_zmax_hard:g} "
                "(reionization_z_start_max di CLASS). A quei redshift la "
                "reionizzazione non e' iniziata: nessun dato vincola quei bin."
            )
        if edges[-1] > self.z_max:
            raise ValueError(
                f"ReioTheory (binned): l'ultimo bordo e' z={edges[-1]:g} ma la "
                f"griglia arriva solo a z_max={self.z_max:g}: la storia "
                "verrebbe troncata. Alza z_max."
            )
        if n_bins > self.reio_bin_nmax_hard:
            raise ValueError(
                f"ReioTheory (binned): {n_bins} bin sono {n_bins} parametri "
                f"liberi in piu'; oltre {self.reio_bin_nmax_hard} l'inferenza "
                "MCMC non e' praticabile. Riduci il numero di bin, oppure usa "
                "il modello pca se ti serve piu' liberta' a parita' di "
                "dimensioni."
            )
        if self.reio_bin_sharpness < 0.01:
            raise ValueError(
                f"ReioTheory (binned): reio_bin_sharpness="
                f"{self.reio_bin_sharpness:g} e' troppo piccola; sotto 0.01 "
                "servirebbero troppi nodi di griglia per risolverla."
            )
        if self.reio_bin_sharpness > widths.min() / 2.0:
            raise ValueError(
                f"ReioTheory (binned): reio_bin_sharpness="
                f"{self.reio_bin_sharpness:g} e' oltre meta' del bin piu' "
                f"stretto ({widths.min():g}): le transizioni adiacenti si "
                "fondono e i bin non sono piu' plateau distinti."
            )

        # --- avvisi ---------------------------------------------------------
        if edges[0] < self.reio_bin_zmin_warn:
            self.log.warning(
                "binned: primo bordo a z=%g, sotto %g. La foresta Ly-alpha "
                "indica che l'IGM e' gia' completamente ionizzato a z<5.3, "
                "quindi i bin sotto quel redshift non sono vincolati da nulla.",
                edges[0], self.reio_bin_zmin_warn,
            )
        if edges[-1] > self.reio_bin_zmax_warn:
            self.log.warning(
                "binned: ultimo bordo a z=%g, sopra %g. A quei redshift e' "
                "molto probabile che la reionizzazione non sia iniziata: quei "
                "bin aggiungono prior, non informazione.",
                edges[-1], self.reio_bin_zmax_warn,
            )
        if n_bins > self.reio_bin_nmax_warn:
            self.log.warning(
                "binned: %d bin. Il CMB da solo porta circa 5 modi "
                "indipendenti sulla storia di reionizzazione (Mortonson & Hu "
                "2008): oltre ~%d bin la posterior e' in buona parte prior. "
                "Ha senso solo se i dati astrofisici a basso z reggono la "
                "risoluzione in piu'.",
                n_bins, self.reio_bin_nmax_warn,
            )
        if self.reio_bin_sharpness > widths.min() / 4.0:
            self.log.warning(
                "binned: sharpness %g contro bin piu' stretto %g: le "
                "transizioni occupano piu' di un quarto del bin, i plateau "
                "sono appena accennati.",
                self.reio_bin_sharpness, widths.min(),
            )

        n_pts = self._binned_pts_per_edge(edges.size)
        self.log.info(
            "binned: %d bin (%s), bordi z=[%s], larghezze %g-%g, "
            "sharpness %g, %d nodi per transizione. Parametri attesi: "
            "reio_xHII_bin1..reio_xHII_bin%d, frazioni x_HII in [0, 1] (bin 1 = il piu' a basso z).",
            n_bins, source,
            ", ".join(f"{e:g}" for e in edges),
            widths.min(), widths.max(), self.reio_bin_sharpness, n_pts, n_bins,
        )
        if n_pts < self.z_bin_pts_per_edge:
            self.log.info(
                "binned: nodi per transizione ridotti da %d a %d per stare nel "
                "budget di %d nodi (limite del parser di CLASS). L'errore su "
                "tau resta sotto ~1e-6 fino a %d nodi.",
                self.z_bin_pts_per_edge, n_pts, self.z_max_nodes,
                self._BIN_PTS_PER_EDGE_MIN,
            )

        self._bin_edges_cache = edges
        return edges

    # Sotto questa soglia l'errore di griglia su tau supera ~1e-6; misurato.
    _BIN_PTS_PER_EDGE_MIN = 10

    # z_max massimo per gompertz_sr: oltre, l'artefatto ad alto z del
    # polinomio P5 entrerebbe nella tabella (vedi initialize).  Misurato il
    # 28 ott 2026 (beta corretta), Tests/test_affidabilita_gompertz.py, B0:
    # dentro l'Eq. 2.1 l'artefatto comincia sopra z = 32.  FUORI dall'Eq. 2.1
    # (zeta_uv > 35 soprattutto) puo' cominciare molto piu' in basso, anche a
    # z < 10: lo z_max non basta, servono prior dentro l'Eq. 2.1.
    _GOMPERTZ_Z_MAX = 30.0

    # Durata minima z_beg - z_end della power law: sotto si scarta il punto.
    # Una reionizzazione piu' rapida di dz = 1e-3 non ha senso fisico, e la
    # soglia sta 100 volte sopra z_min_spacing, cosi' i due kink protetti non
    # possono mai cadere abbastanza vicini da far perdere z_beg nel filtro di
    # spaziatura di _build_grid (prima: tau sbagliato di ~0.01, in silenzio).
    _PL_DURATA_MIN = 1e-3

    def _binned_pts_per_edge(self, n_edges):
        """
        Nodi da spendere su ogni transizione, adattati al budget totale.

        Il numero di nodi cresce col numero di bordi, e oltre il budget la
        stringa passata a CLASS sfonda il buffer del parser: con un valore
        fisso, da ~12 bin in su il tetto scattava a OGNI passo dell'MCMC,
        decimando la griglia e riempiendo il log di warning.  Qui si divide il
        budget disponibile fra i bordi, cosi' il tetto non viene mai raggiunto
        e la degradazione e' graduale e dichiarata.

        Il budget viene da _node_budget(), che dipende dalla CLASS installata:
        ~400 nodi con il parser a 4096 caratteri, ~100 con quello a 1024.

        Misurato (10 bin, sharpness 0.1): 24 nodi/bordo danno un errore su tau
        di 5e-9, 12 nodi 2e-7, 6 nodi 1e-6.  Il minimo di 10 e' quindi gia'
        ampiamente sufficiente; su una CLASS a 1024 caratteri si scende anche
        sotto, ma l'errore resta comunque molto piccolo.
        """
        reserve = int(self.z_bg_points) + len(self._nodi_dati())
        if self.reio_include_HeII:
            reserve += int(2 * self.z_window_widths * self.z_pts_per_width)
        budget = max(self._node_budget() - reserve, 0)
        per_edge = budget // max(n_edges, 1)
        n = min(int(self.z_bin_pts_per_edge), per_edge)
        # Non si scende sotto il minimo utile, salvo che il budget non lo
        # consenta proprio: con CLASS a 1024 caratteri e molti bin puo'
        # succedere, e allora e' meglio pochi nodi per bordo che una griglia
        # gonfiata che poi viene decimata alla cieca.
        return max(n, min(self._BIN_PTS_PER_EDGE_MIN, max(per_edge, 4)))

    def _binned_grid(self, edges):
        """
        Griglia z per il modello a bin: una finestra fitta su ogni bordo (dove
        sta la transizione) piu' la griglia di fondo sui plateau, che essendo
        piatti non hanno bisogno di risoluzione.

        Finestra +/- 4 sharpness invece delle 6 usate altrove: con molti bordi
        il budget di nodi e' stretto, e a 4 larghezze la tanh e' gia' entro
        3.4e-4 dall'asintoto.
        """
        d = self.reio_bin_sharpness
        n = self._binned_pts_per_edge(len(edges))
        nodes = np.concatenate([np.linspace(e - 4.0 * d, e + 4.0 * d, n)
                                for e in edges])
        return self._build_grid(z_re=None, extra_nodes=nodes,
                                protect_nodes=edges)

    # -----------------------------------------------------------------------
    # Modello PCA
    # -----------------------------------------------------------------------
    def _pca_basis(self):
        """
        Carica (una volta) la base di autofunzioni e ne verifica la
        compatibilita' con la griglia di questa theory.

        load_basis ha la sua cache su (percorso, mtime, dimensione), quindi
        chiamarla a ogni passo dell'MCMC costa un os.stat.
        """
        basis = load_basis(self.reio_pca_file)

        if basis.z_max > self.z_max + 1e-9:
            raise ValueError(
                f"ReioTheory (pca): la base arriva a z = {basis.z_max:g} ma la "
                f"griglia passata a CLASS si ferma a z_max = {self.z_max:g}. "
                "Cosi' l'ultima parte della storia verrebbe tagliata via in "
                f"silenzio: alza z_max ad almeno {basis.z_max:g}."
            )
        n = int(self.reio_pca_n_modes)
        if not (1 <= n <= basis.n_modes_available):
            raise ValueError(
                f"ReioTheory (pca): reio_pca_n_modes={n} ma la base "
                f"'{basis.path}' ha {basis.n_modes_available} modi."
            )
        if not getattr(self, "_pca_announced", False):
            self.log.info("modello pca:\n%s", basis.describe())
            self._pca_announced = True
        return basis

    def _pca_bands(self, basis):
        """Bande in z per il controllo sulla tau parziale."""
        if not self.reio_pca_tau_bands:
            return [(basis.z_min, basis.z_max)]
        bands = []
        for b in self.reio_pca_tau_bands:
            if len(b) != 2 or b[1] <= b[0]:
                raise ValueError(
                    f"ReioTheory (pca): banda {b} non valida in "
                    "reio_pca_tau_bands; servono coppie [z_lo, z_hi] crescenti."
                )
            bands.append((float(b[0]), float(b[1])))
        return bands

    def _pca_physicality(self, basis, m, xe_bins):
        """
        Priori di fisicita' di Mortonson & Hu 2008 Sect. 3, nella forma citata
        da Ilic et al. 2025 ("priors to the amplitude of each mode following
        the prescription of Mortonson & Hu (2008), ensuring that reionisation
        histories with unphysical optical depths over a broad redshift range
        are treated as unphysical models").

        Restituisce (ok, motivo).  ok=False fa scartare il punto: calculate()
        ritorna False e cobaya gli assegna likelihood nulla.

        I tre livelli, in ordine di costo crescente e di forza crescente:

          1. Eq. (10) — ogni m_a dentro il suo intervallo.  Sono anche i prior
             uniformi che il YAML dovrebbe avere (priors_yaml li stampa), ma si
             ricontrollano qui: se qualcuno allarga i prior a mano, il taglio
             deve restare.
          2. Eq. (13) — sum_a m_a^2 <= f.  Vincolo GLOBALE: taglia combinazioni
             che i limiti per singolo modo lascerebbero passare.
          3. tau parziale nelle bande — il criterio vero e proprio di MH08.
             NON si taglia punto per punto su xe, di proposito: i modi non
             campionati possono correggere xe localmente (sono oscillazioni
             veloci in z) ma non possono spostare un integrale su banda larga.
        """
        mode = self.reio_pca_physicality
        if mode == "none":
            return True, ""
        if mode not in ("mh08", "strict"):
            raise ValueError(
                f"reio_pca_physicality='{mode}' non riconosciuto. "
                "Usa 'mh08' (come il paper), 'strict' o 'none'."
            )

        report = basis.physicality_report(
            m, bands=self._pca_bands(basis), strict=(mode == "strict"))

        # Alla prima chiamata si stampano TUTTE le condizioni, con la formula
        # accanto: cosi' sono scritte nero su bianco nel log della catena e non
        # bisogna andarle a cercare nel codice.
        if not getattr(self, "_pca_conditions_shown", False):
            self._pca_conditions_shown = True
            self.log.info(
                "pca: condizioni di fisicita' (%s), valutate sul primo punto:\n%s",
                mode, basis.format_report(report))

        for r in report:
            if not r["ok"]:
                return False, f"{r['nome']}: {r['formula']}, ma vale {r['valore']:+.5f}"
        return True, ""

    def _pca_grid(self, z_knots):
        """
        Griglia z per il modello PCA.

        Non servono finestre di raffinamento: la ricostruzione e' una spezzata
        con vertici nei nodi, e CLASS interpola linearmente.  Basta che i nodi
        ci siano (protect_nodes) perche' CLASS riproduca il modello in modo
        esatto ovunque.  Il resto della griglia serve solo allo step HeII e a
        dare a CLASS qualche punto sui plateau.
        """
        return self._build_grid(z_re=None, extra_nodes=z_knots,
                                protect_nodes=z_knots)

    def must_provide(self, **requirements):
        """
        Intercetta 'reio_history' prima che arrivi al wrapper di classy.

        Serve perche' classy, davanti a una richiesta con valore None che non
        riconosce, la aggiunge a derived_extra e la gira a CLASS come parametro
        derivato (classy.py:518-522).  CLASS ovviamente risponde
        "reio_history was not recognized as a derived parameter".

        La storia la produce gia' calculate() a ogni punto, quindi non c'e'
        niente da predisporre: basta toglierla dalle richieste che scendono.

        Le opzioni della richiesta possono contenere 'z_nodi', i redshift dei
        dati puntuali di chi la chiede (ReioLike._z_dati_puntuali): diventano
        nodi fissi della griglia, vedi _nodi_dati.  cobaya chiama must_provide
        una volta per ogni componente che chiede la storia, quindi si fa
        l'unione invece di sovrascrivere.
        """
        req = requirements.pop("reio_history", None)
        if isinstance(req, dict) and req.get("z_nodi") is not None:
            z = np.asarray(req["z_nodi"], dtype=float).ravel()
            z = np.union1d(getattr(self, "_z_nodi_dati", np.array([])),
                           z[np.isfinite(z)])
            self._z_nodi_dati = z
            self.log.info("nodi fissi in %d redshift di dati puntuali: %s",
                          len(z), np.array2string(z, precision=4))
        return super().must_provide(**requirements)

    def get_can_provide(self):
        """
        'reio_history' e' un prodotto vero, chiesto da ReioLike via provider.

        Prima qui c'era solo super().get_can_provide(), cioe' una lista vuota:
        la storia viaggiava SOLO nella globale _shared_reio_history, scritta
        dentro calculate().  Siccome cobaya non chiama calculate() quando il
        punto e' in cache, quella globale poteva restare indietro e la
        likelihood valutava un punto contro la storia di un altro (riprodotto
        da Tests/bug_storia_stantia.py: 8.53 di errore in logL).

        Dichiarandola come prodotto, la storia finisce dentro lo state di
        cobaya, che tiene una copia per punto e serve sempre quella giusta.
        """
        return ["reio_history"] + list(super().get_can_provide())

    def get_reio_history(self):
        """
        La storia del punto CORRENTE: z, xe, il gradino HeII e f_He.

        Legge da current_state, cioe' dallo stato che cobaya ha selezionato per
        il punto in corso.  Su un cache hit current_state e' quello del punto
        giusto, mentre la globale no: e' tutta qui la differenza.
        """
        return self.current_state["reio_history"]

    def _yhe_del_punto(self, native_params):
        """
        Y_He per il punto corrente, e lo imposta come parametro di CLASS.

        Ordine di priorita':
          1. YHe gia' dato a CLASS come numero (params o extra_args): si usa
             quello, cosi' la storia resta coerente con cio' che CLASS vede;
          2. reio_YHe > 0: valore fisso;
          3. altrimenti BBN: Y_He(omega_b, DeltaN) dalla tabella di CLASS.

        Va chiamato PRIMA di costruire la tabella x_e(z): il plateau 1 + f_He
        dei modelli dipende da questo numero.  E' esattamente il motivo per
        cui non si puo' chiederlo a CLASS: CLASS lo calcola dopo.
        """
        dato = native_params.get("YHe", self.extra_args.get("YHe"))
        if dato is not None and not isinstance(dato, str):
            Y = float(dato)
        elif self.reio_YHe > 0:
            Y = float(self.reio_YHe)
        else:
            ob = native_params.get("omega_b")
            if ob is None and "Omega_b" in native_params:
                h = native_params.get("h")
                if h is None and native_params.get("H0") is not None:
                    h = native_params["H0"] / 100.0
                if h is not None:
                    ob = native_params["Omega_b"] * h ** 2
            if ob is None:
                raise ValueError(
                    "ReioTheory: per ricavare Y_He dalla BBN serve omega_b (o "
                    "Omega_b con h/H0) fra i parametri passati a CLASS.  In "
                    "alternativa fissare reio_YHe > 0.")
            if getattr(self, "_bbn", None) is None:
                f = self.reio_bbn_file
                if not f and self.extra_args.get("sBBN_file"):
                    f = file_bbn_di_class(os.path.basename(self.extra_args["sBBN_file"]))
                self._bbn = TabellaBBN(f or None)
                self.log.info("Y_He dalla BBN: tabella %s", self._bbn.percorso)
            dn = self.reio_bbn_delta_neff
            if dn is None:
                dn = delta_neff_bbn({**self.extra_args, **native_params})
            Y = self._bbn(float(ob), float(dn))
        # Lo stesso numero va a CLASS: senza questo CLASS ricalcolerebbe Y_He
        # per conto suo e i due potrebbero divergere (es. sBBN_file diverso).
        native_params["YHe"] = Y
        self._last_YHe = Y
        self._last_f_He = f_he(Y)
        return Y

    def _finalize_xe(self, z_grid, xe, params):
        """
        Passaggi comuni a TUTTI i modelli, applicati subito prima di passare
        la tabella a CLASS via reio_inter:

          1. aggiunge la seconda reionizzazione dell'elio (HeII -> HeIII),
             se reio_include_HeII e' True.  I modelli in reio_models.py
             producono solo il settore idrogeno + prima ionizzazione
             dell'elio (plateau 1+fHe); questo porta il plateau a 1+2*fHe;
          2. impone un floor pari alla ionizzazione residua: CLASS non
             accetta xe=0 nei nodi intermedi (divisione per dkappa).  Non
             per i modelli di _MODELLI_CON_FONDO (tanh_class), che hanno gia'
             x_e >= x_noreio(z) per costruzione;
          3. azzera l'ultimo nodo, come richiede la convenzione reio_inter
             (CLASS lo sostituisce internamente con xe_before, cioe' il
             valore da ricombinazione: thermodynamics.c:4327).

        Restituisce sempre un array nuovo.
        """
        xe = np.array(xe, dtype=float, copy=True)

        # Pavimento del punto: fondo di ricombinazione di CLASS x 1.05, calcolato
        # in calculate() per la cosmologia corrente (_pavimento_del_punto).  Se
        # manca ci si ferma: nessun ritorno silenzioso a un numero fisso.
        pav = getattr(self, "_pavimento", None)
        if pav is None or not np.isfinite(pav) or pav <= 0.0:
            raise RuntimeError(
                "ReioTheory._finalize_xe: pavimento non calcolato.  Dentro "
                "cobaya lo calcola calculate(); fuori da cobaya va impostato con "
                "imposta_pavimento(cosmologia) prima di costruire la tabella.")

        # Y_He del punto, gia' calcolato da _yhe_del_punto() all'inizio di
        # calculate(): lo stesso usato per il plateau dei modelli.
        Y_He = self._last_YHe
        self._last_HeII = np.zeros_like(xe)
        if self.reio_include_HeII:
            self._last_HeII = helium_fullreio_step(
                z_grid,
                z_reHeII=self.reio_z_HeII,
                delta_z=self.reio_delta_z_HeII,
                Y_He=float(Y_He),
            )
            xe = xe + self._last_HeII

        # Il pavimento non si applica ai modelli che contengono gia' il fondo
        # vero, x_noreio(z): li' x_e >= x_noreio per costruzione, e alzare la
        # coda a 1.05 x x_e(z_max) cambierebbe tau di ~7e-5 rispetto a CLASS
        # (misurato 6 nov 2026, plateau contato fino al minimo globale di x_e).
        if self.reio_model not in self._MODELLI_CON_FONDO:
            xe = np.maximum(xe, pav)

        # L'azzeramento dell'ultimo nodo e' legittimo solo se il modello e' gia'
        # tornato alla ionizzazione residua a z_max; altrimenti si sta tagliando
        # via un pezzo di reionizzazione e serve alzare z_max.
        if xe[-1] > 10.0 * pav:
            self.log.warning(
                "xe(z_max=%.1f) = %.3g, ben sopra la ionizzazione residua "
                "(%.1g): la storia viene troncata bruscamente. Aumenta z_max.",
                self.z_max, xe[-1], pav,
            )

        xe[-1] = 0.0
        return xe

    def calculate(self, state, want_derived=True, **params_values_dict):
        native_params = self._to_native(params_values_dict)

        # Elio del punto, PRIMA di costruire la storia: il plateau di tutti i
        # modelli e' 1 + f_He(omega_b).
        self._yhe_del_punto(native_params)
        xe_max = 1.0 + self._last_f_He

        # Pavimento della tabella = fondo di ricombinazione di CLASS x 1.05 per
        # QUESTA cosmologia (Y_He compreso, quindi dopo _yhe_del_punto).
        self._pavimento_del_punto(native_params)

        if self.reio_model == "tanh":
            z_re    = native_params.pop("reio_z_re",    None)
            delta_z = native_params.pop("reio_delta_z", None)
            if z_re is None or delta_z is None:
                raise ValueError(
                    "ReioTheory (tanh): 'reio_z_re' e 'reio_delta_z' "
                    "devono essere nel blocco 'params'."
                )

            # Costruisce griglia adattiva centrata su z_re
            # la finestra fitta viene dimensionata sul delta_z campionato
            z_grid   = self._build_grid(z_re, delta_z=delta_z)
            xe_class = tanh_model(z_grid, z_re, delta_z, xe_max=xe_max)

        elif self.reio_model == "tanh_class":
            # ------------------------------------------------------------------
            # La tanh di CLASS (reio_camb), con la stessa formula, gli stessi
            # parametri e la stessa x_noreio(z): vedi tanh_class_model.  Stessi
            # nomi della tanh lineare (reio_z_re, reio_delta_z), ma e' un altro
            # modello: i due coincidono solo vicino a z_re.
            # ------------------------------------------------------------------
            z_re    = native_params.pop("reio_z_re",    None)
            delta_z = native_params.pop("reio_delta_z", None)
            if z_re is None or delta_z is None:
                raise ValueError(
                    "ReioTheory (tanh_class): 'reio_z_re' e 'reio_delta_z' "
                    "devono essere nel blocco 'params'."
                )
            z_grid  = self._build_grid(z_re, delta_z=delta_z)
            z_start = class_reio_start(z_re, delta_z, self.reio_z_HeII,
                                       self.reio_delta_z_HeII)
            xe_class = tanh_class_model(z_grid, z_re, delta_z,
                                        x_noreio=self._x_noreio(z_grid),
                                        xe_max=xe_max, z_start=z_start)

        elif self.reio_model == "tanh_pinned_end":
            # ------------------------------------------------------------------
            # Tanh model pinned to a user-chosen (z_end, xe_end) point.
            #
            # Sampled MCMC parameter:
            #   reio_z_re  — midpoint of reionisation (xe = xe_max/2)
            #
            # Fixed configuration (set in the theory block of the YAML):
            #   reio_pin_z_end       — redshift of the pinning point
            #   reio_pin_xe_end      — absolute xe value at z_end  (or)
            #   reio_pin_xe_end_frac — fraction of xe_max at z_end
            # ------------------------------------------------------------------
            z_re = native_params.pop("reio_z_re", None)
            if z_re is None:
                raise ValueError(
                    "ReioTheory (tanh_pinned_end): 'reio_z_re' "
                    "deve essere nel blocco 'params' del YAML."
                )

            # Resolve xe_end: explicit value overrides fraction
            xe_end_cfg = None if self.reio_pin_xe_end < 0 else self.reio_pin_xe_end

            if z_re <= self.reio_pin_z_end:
                # Il midpoint sotto il punto fisso non e' una storia
                # (tanh_pinned_end_delta_z solleva).  Un prior di z_re che
                # scende sotto reio_pin_z_end lo ferma gia' l'avvio
                # (_controlla_prior_pinned); qui resta la rete di sicurezza per
                # prior illimitati o valutazioni dirette, che non devono
                # fermare la catena.
                self.log.debug("tanh_pinned_end: z_re=%g <= z_end=%g, punto "
                               "scartato", z_re, self.reio_pin_z_end)
                return False

            # delta_z qui e' derivato dal vincolo di pinning, non campionato:
            # lo si ricava a parte per dimensionare la finestra della griglia
            delta_z_eff = tanh_pinned_end_delta_z(
                z_re, z_end=self.reio_pin_z_end, xe_end=xe_end_cfg,
                xe_end_frac=self.reio_pin_xe_end_frac,
                xe_max=xe_max,
            )
            z_grid   = self._build_grid(z_re, delta_z=delta_z_eff)
            xe_class = tanh_pinned_end_model(
                z_grid,
                z_re=z_re,
                z_end=self.reio_pin_z_end,
                xe_end=xe_end_cfg,               # None if using fraction
                xe_end_frac=self.reio_pin_xe_end_frac,
                xe_max=xe_max,
            )

        elif self.reio_model == "power_law":
            # ------------------------------------------------------------------
            # Power law di Douspis et al. 2015 / Planck Int. XLVII 2016, nella
            # forma di Ilic et al. 2025 (arXiv:2504.13254) Eq. (3).
            #
            # Parametri campionati (blocco 'params'):
            #   reio_z_end — fine reionizzazione
            #   reio_alpha — esponente
            #   reio_z_beg — inizio; se assente si usa l'opzione self.reio_z_beg
            #                (variante 2-pow del paper, z_beg fissato a 20)
            # ------------------------------------------------------------------
            z_end = native_params.pop("reio_z_end", None)
            alpha = native_params.pop("reio_alpha", None)
            if z_end is None or alpha is None:
                raise ValueError(
                    "ReioTheory (power_law): 'reio_z_end' e 'reio_alpha' "
                    "devono essere nel blocco 'params' del YAML."
                )
            # 3-pow se z_beg e' campionato, altrimenti 2-pow con z_beg fisso
            z_beg = native_params.pop("reio_z_beg", None)
            if z_beg is None:
                z_beg = self.reio_z_beg

            if z_beg > self.z_max:
                # NON si solleva un'eccezione. z_beg puo' essere un parametro
                # CAMPIONATO (variante 3-pow del paper, prior U(5, 25)), quindi
                # un'eccezione qui ucciderebbe l'intera catena al primo punto
                # proposto oltre z_max — magari dopo ore di run. Si scarta il
                # punto e si avverte UNA volta, cosi' il run sopravvive e
                # l'utente ha modo di correggere z_max.
                #
                # Attenzione pero': scartare una fetta di prior deforma la
                # posterior. Il warning va preso sul serio, non ignorato.
                n = getattr(self, "_pl_zbeg_rejected", 0) + 1
                self._pl_zbeg_rejected = n
                if not getattr(self, "_pl_zbeg_warned", False):
                    self._pl_zbeg_warned = True
                    self.log.warning(
                        "power_law: z_beg=%g supera z_max=%g, il punto viene "
                        "SCARTATO. Se z_beg e' campionato, il prior sta "
                        "sconfinando oltre la griglia e la posterior ne "
                        "risulta deformata: alza z_max ad almeno l'estremo "
                        "superiore del prior su reio_z_beg.",
                        z_beg, self.z_max)
                return False

            if z_beg - z_end < self._PL_DURATA_MIN:
                # Storia non fisica: la reionizzazione comincerebbe dopo
                # essere finita, o durerebbe meno di _PL_DURATA_MIN.  Nel 3-pow
                # con i prior del paper (z_end, z_beg ~ U(5, 25), indipendenti)
                # e' META' del volume: un'eccezione qui (power_law_model la
                # solleva) fermerebbe la catena alla prima proposta, con
                # stop_at_error=True.  Si scarta il punto: il prior effettivo
                # diventa uniforme sul triangolo z_beg > z_end.  Nessun
                # avviso, e' una regione attesa, non un errore di configurazione.
                self.log.debug("power_law: z_beg=%g - z_end=%g < %g, punto scartato",
                               z_beg, z_end, self._PL_DURATA_MIN)
                return False

            z_grid   = self._power_law_grid(z_end, alpha, z_beg)
            xe_class = power_law_model(z_grid, z_end=z_end, alpha=alpha,
                                       z_beg=z_beg, xe_max=xe_max)

        elif self.reio_model == "binned":
            # ------------------------------------------------------------------
            # Modello a bin di Ilic et al. 2025 (arXiv:2504.13254) Sect. 2.5.
            #
            # Parametri campionati: reio_xHII_bin1 ... reio_xHII_binN, uno per bin,
            # numerati dal BASSO redshift.  Lo schema dei bordi e' fisso e
            # arriva da _binned_edges().
            # ------------------------------------------------------------------
            edges = self._binned_edges()
            n_bins = len(edges) - 1

            vals, missing = [], []
            for i in range(1, n_bins + 1):
                v = native_params.pop(f"reio_xHII_bin{i}", None)
                if v is None:
                    missing.append(f"reio_xHII_bin{i}")
                vals.append(v)
            if missing:
                raise ValueError(
                    f"ReioTheory (binned): mancano {len(missing)} parametri nel "
                    f"blocco 'params': {', '.join(missing)}. Lo schema ha "
                    f"{n_bins} bin (bordi z=[{', '.join(f'{e:g}' for e in edges)}]), "
                    f"quindi servono reio_xHII_bin1..reio_xHII_bin{n_bins}, "
                    "con bin 1 = il piu' a basso redshift."
                )
            vals = np.asarray(vals, dtype=float)
            # x_HII del bin: fuori da [0, 1] non e' una frazione.  Con un prior
            # dentro [0, 1] (controllato all'avvio) non succede; resta per
            # prior illimitati e valutazioni dirette.
            if np.any((vals < 0.0) | (vals > 1.0)):
                self.log.debug("binned: x_HII dei bin fuori da [0, 1], punto "
                               "scartato: %s", np.array2string(vals, precision=3))
                return False

            # Monotonia opzionale: xe non deve crescere al crescere di z.
            # Il punto non viene "corretto", viene scartato — restituire False
            # da calculate() fa assegnare a cobaya likelihood nulla.
            if self.reio_bin_monotonic and np.any(np.diff(vals) > 0):
                self.log.debug(
                    "binned: storia non monotona scartata (reio_bin_monotonic), "
                    "xe = %s", np.array2string(vals, precision=3))
                return False

            z_grid = self._binned_grid(edges)
            # x_HII -> x_e con l'f_He DEL PUNTO: la likelihood fa l'inverso,
            # x_HII = (x_e - x_e,HeII)/(1 + f_He) con lo stesso f_He, quindi sul
            # plateau del bin rivede esattamente il valore campionato.
            xe_class = binned_model(z_grid, edges, vals * xe_max,
                                    delta_z=self.reio_bin_sharpness,
                                    xe_below=xe_max)

        elif self.reio_model == "pca":
            # ------------------------------------------------------------------
            # Modello 'pca-5' di Ilic et al. 2025 Sect. 2.6 / Mortonson & Hu 2008.
            #
            #     xe(z) = xe_fid + sum_{a=1..N} m_a S_a(z)
            #
            # Parametri campionati: reio_pca_m1 ... reio_pca_mN.  Tutto il
            # resto (base, intervallo, binning, fiduciale) viene dal file.
            # ------------------------------------------------------------------
            basis = self._pca_basis()
            n_modes = int(self.reio_pca_n_modes)

            m, missing = [], []
            for a in range(1, n_modes + 1):
                v = native_params.pop(f"reio_pca_m{a}", None)
                if v is None:
                    missing.append(f"reio_pca_m{a}")
                m.append(v)
            if missing:
                raise ValueError(
                    f"ReioTheory (pca): mancano {len(missing)} parametri nel "
                    f"blocco 'params': {', '.join(missing)}. Servono "
                    f"reio_pca_m1..reio_pca_m{n_modes}. I prior corretti si "
                    "stampano con:\n"
                    "  python -c \"from Reiolike.pca_basis import load_basis; "
                    f"print(load_basis('{self.reio_pca_file}')"
                    f".priors_yaml({n_modes}))\""
                )
            m = np.asarray(m, dtype=float)

            xe_bins = basis.reconstruct(m)
            ok, why = self._pca_physicality(basis, m, xe_bins)
            self._pca_n_total = getattr(self, "_pca_n_total", 0) + 1
            if not ok:
                self._pca_n_rejected = getattr(self, "_pca_n_rejected", 0) + 1
                self.log.debug("pca: punto scartato — %s", why)
                return False

            xe_complete = (self.reio_pca_xe_complete
                           if self.reio_pca_xe_complete > 0 else xe_max)
            z_knots, xe_knots = basis.knots(
                m, xe_complete=xe_complete,
                smooth_sigma_z=self.reio_pca_smooth_sigma_z,
                z_out_max=self.z_max)
            z_grid = self._pca_grid(z_knots)
            xe_class = pca_model(z_grid, z_knots, xe_knots)

            # Quanto ci allontaniamo dal paper.  Ilic et al. usano CAMB per il
            # solo pca-5 perche' accetta xe temporaneamente negativo; CLASS in
            # reio_inter no, quindi _finalize_xe alzera' quei tratti a
            # pavimento.  Qui si misura l'area cosi' aggiunta, in modo che
            # la differenza sia dichiarata e non silenziosa.
            below = np.clip(self._pavimento - xe_class, 0.0, None)
            if np.any(below > 0):
                area_lost = _trapz(below, z_grid)
                area_tot = _trapz(np.abs(xe_class), z_grid)
                frac = area_lost / max(area_tot, 1e-30)
                if frac > self.reio_pca_max_floor_frac:
                    self._pca_n_rejected = getattr(self, "_pca_n_rejected", 0) + 1
                    self.log.debug(
                        "pca: punto scartato — area al floor %.1f%% > "
                        "reio_pca_max_floor_frac", 100 * frac)
                    return False
                if frac > self.reio_pca_floor_warn:
                    # Un warning per volta, non uno per passo dell'MCMC: con
                    # xe_fid = 0.15 questa condizione e' la NORMA, non
                    # l'eccezione, e riempirebbe il log rendendolo inutile.
                    prev = getattr(self, "_pca_floor_max", 0.0)
                    if frac > prev * 1.5:
                        self._pca_floor_max = frac
                        self.log.warning(
                            "pca: xe negativo su un tratto non trascurabile "
                            "(area riportata al floor = %.1f%% del totale, "
                            "xe_min = %.4f). CLASS non accetta xe<=0 in "
                            "reio_inter; Ilic et al. per questo modello usano "
                            "CAMB, che invece lo accetta. La nostra tau e' "
                            "quindi leggermente PIU' ALTA della loro su questi "
                            "punti. Vedere Tests/test_pca_sampling.py per la "
                            "misura dell'effetto.",
                            100 * frac, float(np.min(xe_class)))

        elif self.reio_model == "gompertz_sr":
            # ------------------------------------------------------------------
            # Gompertz + Symbolic Regression (Montero-Camacho et al. 2024)
            # arXiv:2405.13680,  Eqs. 2.2–2.4 and 3.1–3.2
            #
            # zeta_uv must always be present in the YAML params.
            # How sigma8 is obtained depends on gompertz_sigma8_source:
            #
            #   "sample"    — sigma8 is a direct MCMC parameter.  A_s/logA
            #                 must NOT be in params; CLASS derives A_s from σ₈.
            #
            #   "class_mPk" — logA (or A_s) is sampled normally.  A quick
            #                 CLASS call (output='mPk' only, ~10-50x faster
            #                 than the full CMB run) derives sigma8 before
            #                 passing the reionisation history to the full run.
            #                 sigma8 does not depend on tau_reio, so no
            #                 circularity is introduced.
            # ------------------------------------------------------------------
            zeta_uv = native_params.pop("zeta_uv", None)
            if zeta_uv is None:
                raise ValueError(
                    "ReioTheory (gompertz_sr): 'zeta_uv' "
                    "deve essere nel blocco 'params' del YAML di Cobaya."
                )

            # --- shared: recover h, Omega_b, Omega_m, ns --------------------
            H0 = native_params.get("H0", None)
            if H0 is None:
                raise ValueError(
                    "ReioTheory (gompertz_sr): 'H0' deve essere disponibile "
                    "nei parametri. Assicurati che CLASS possa fornirlo."
                )
            h = H0 / 100.0

            ombh2 = native_params.get("omega_b",   None)
            omch2 = native_params.get("omega_cdm", None)
            if ombh2 is None or omch2 is None:
                raise ValueError(
                    "ReioTheory (gompertz_sr): 'omega_b' e 'omega_cdm' "
                    "devono essere disponibili nei parametri."
                )
            Omega_b = ombh2 / h**2
            Omega_m = (ombh2 + omch2) / h**2
            ns = native_params.get("n_s", None)
            if ns is None:
                raise ValueError(
                    "ReioTheory (gompertz_sr): 'n_s' deve essere disponibile "
                    "nei parametri."
                )

            # --- obtain sigma8 according to the chosen source ----------------
            if self.gompertz_sigma8_source == "sample":
                # sigma8 is a direct MCMC parameter
                sigma8 = native_params.pop("sigma8", None)
                if sigma8 is None:
                    raise ValueError(
                        "ReioTheory (gompertz_sr, sigma8_source='sample'): "
                        "'sigma8' deve essere nel blocco 'params'."
                    )
                # Pass sigma8 to CLASS so it computes A_s internally;
                # remove any A_s/logA to avoid conflicts.
                native_params.pop("A_s",  None)
                native_params.pop("logA", None)
                native_params.pop("ln10^{10}A_s", None)
                native_params["sigma8"] = sigma8

            elif self.gompertz_sigma8_source == "class_mPk":
                # A_s (or logA) is sampled; derive sigma8 via a quick CLASS run
                logA_val = native_params.get("logA", None)
                As_val   = native_params.get("A_s",  None)
                sigma8 = self._compute_sigma8_from_As(
                    omega_b=ombh2, omega_cdm=omch2, H0=H0, n_s=ns,
                    logA=logA_val, A_s=As_val,
                )
                self.log.debug("gompertz_sr: sigma8=%.4f (da run CLASS mPk)", sigma8)
                # A_s / logA remain in native_params for the full CLASS run

            else:
                raise ValueError(
                    f"gompertz_sigma8_source='{self.gompertz_sigma8_source}' "
                    "non riconosciuto. Usa 'sample' o 'class_mPk'."
                )

            # --- build the grid and compute xe(z) ----------------------------
            # Il modello Gompertz+SR non ha un z_re esplicito: centro e
            # larghezza della transizione dipendono da (sigma8, ns, h,
            # Omega_b, Omega_m, zeta_uv) attraverso le formule SR.  Usare un
            # centro fisso lascerebbe la finestra fitta nel posto sbagliato
            # quando i parametri spostano la reionizzazione.  Quindi: una
            # prima passata su griglia uniforme fitta per misurare dove sta
            # davvero la transizione, poi la griglia definitiva.
            # include_HeII=False in entrambe: lo step HeII lo aggiunge
            # _finalize_xe() come per tutti gli altri modelli.
            kw = dict(sigma8=sigma8, ns=ns, h=h, Omega_b=Omega_b,
                      Omega_m=Omega_m, zeta_uv=zeta_uv, include_HeII=False,
                      Y_He=self._last_YHe)

            z_probe  = np.linspace(0.0, self.z_max, 600)
            xe_probe = gompertz_sr_model(z_probe, **kw)
            z_mid, delta_z_eff = self.estimate_transition(z_probe, xe_probe)
            self.log.debug("gompertz_sr: transizione stimata z_mid=%.2f, "
                           "delta_z=%.2f", z_mid, delta_z_eff)

            z_grid   = self._build_grid(z_mid, delta_z=delta_z_eff)
            xe_class = gompertz_sr_model(z_grid, **kw)

        else:
            raise NotImplementedError(f"reio_model='{self.reio_model}' non implementato.")

        # Trattamento comune a tutti i modelli: step HeII + floor + ultimo nodo
        xe_class = self._finalize_xe(z_grid, xe_class, native_params)

        # Incastro nel buffer del parser di CLASS.  Va fatto QUI, dopo il
        # calcolo di xe e prima di consegnare la storia a chiunque, cosi' la
        # tabella che vede la likelihood astrofisica e quella che vede CLASS
        # sono la stessa.  Se decimassimo dopo, i due userebbero due modelli
        # leggermente diversi.
        z_grid, xe_class, str_z, str_xe = self._fit_class_strings(
            z_grid, xe_class, protect=getattr(self, "_last_protect", None))
        if self.reio_include_HeII:
            self._last_HeII = helium_fullreio_step(
                z_grid, z_reHeII=self.reio_z_HeII,
                delta_z=self.reio_delta_z_HeII,
                Y_He=self._last_YHe)
        else:
            self._last_HeII = np.zeros_like(xe_class)

        # Canale UFFICIALE: la storia dentro lo state di cobaya.  Lo state e'
        # per-punto e la cache ne tiene una copia ciascuno, quindi su un cache
        # hit la likelihood riceve comunque la storia del punto giusto.
        # Letto da get_reio_history(), che ReioLike chiede via provider.
        # La chiave deve chiamarsi esattamente 'reio_history': cobaya risolve
        # provider.get_result('X') in state['X'] (theory.py:160-170).
        state['reio_history'] = {
            'z':       z_grid,
            'xe':      xe_class,
            'xe_HeII': self._last_HeII,
            'f_He':    self._last_f_He,
        }

        self._current_reio_history = (z_grid, xe_class)

        # Canale di COMPATIBILITA', solo per chi usa il pacchetto fuori da
        # cobaya: script che costruiscono ReioLike a mano e iniettano una
        # storia (Tests/verifica_analisi.py fa cosi').  Dentro cobaya NON si
        # usa piu', perche' e' una copia unica che resta indietro quando
        # calculate() non viene chiamata.
        _shared_reio_history['z']       = z_grid
        _shared_reio_history['xe']      = xe_class
        _shared_reio_history['xe_HeII'] = self._last_HeII
        _shared_reio_history['f_He']    = self._last_f_He

        # str_z / str_xe arrivano gia' da _fit_class_strings(), che ha
        # garantito che stiano nel buffer.  Su z servono 8 cifre significative:
        # a 6, due nodi separati da ~1e-3 attorno a z~25 collasserebbero nello
        # stesso valore e CLASS rifiuterebbe la tabella come non crescente.
        self.log.debug("griglia: %d nodi, z=[%.2f..%.2f], passo min %.4g, "
                       "stringhe %d/%d caratteri (limite %d)",
                       len(z_grid), z_grid[0], z_grid[-1],
                       float(np.min(np.diff(z_grid))), len(str_z), len(str_xe),
                       self._class_arg_limit())

        native_params['reio_parametrization'] = 'reio_inter'
        native_params['reio_inter_num']        = len(z_grid)
        native_params['reio_inter_z']          = str_z
        native_params['reio_inter_xe']         = str_xe

        esito = super().calculate(state, want_derived, **native_params)
        if esito is not False and getattr(self, "_fondo_da_verificare", False):
            self._verifica_fondo_di_class()
        return esito

    # -----------------------------------------------------------------------
    # Pavimento della tabella: fondo di ricombinazione di CLASS
    # -----------------------------------------------------------------------
    def _argomenti_fondo(self, native_params):
        """
        Gli argomenti di CLASS per il calcolo del fondo: la STESSA cosmologia
        della run vera (extra_args + parametri del punto), senza i parametri
        della reionizzazione e dello spettro primordiale, con reio_none.
        """
        # Stessa ricetta di cobaya (theories/classy/classy.py, set()): i
        # parametri del punto con i nomi di CLASS, POI extra_args sopra, e senza
        # output si tolgono non_linear e hmcode_version (CLASS li rifiuta).
        tradurre = getattr(self, "translate_param", lambda q: q)
        args = {tradurre(k): v for k, v in native_params.items()}
        args.update(getattr(self, "extra_args", None) or {})
        if not args.get("output"):
            for k in ("non_linear", "hmcode_version"):
                args.pop(k, None)
        args = {k: v for k, v in args.items()
                if not k.startswith("reio_") and k not in self._NON_PER_IL_FONDO}
        args["reio_parametrization"] = "reio_none"
        return args

    def _calcola_fondo(self, args):
        """
        (x_e a z_max, x_e a z_max + 1) di CLASS senza reionizzazione.  Il primo
        e' xe_before, il fondo; il secondo serve a _verifica_fondo_di_class.
        """
        if _ClassPy is None:
            raise ImportError("ReioTheory: serve classy per calcolare il fondo "
                              "di ricombinazione (pavimento della tabella).")
        c = _ClassPy()
        try:
            c.set(args)
            try:
                c.compute(level=["thermodynamics"])
            except Exception as e:
                # Un parametro non letto vuol dire che il filtro di
                # _argomenti_fondo e' sbagliato: errore di configurazione, che
                # rifiuterebbe TUTTI i punti in silenzio.  Si ferma la catena.
                msg = str(e)
                if "read" in msg.lower() and "param" in msg.lower():
                    raise LoggedError(
                        self.log, "ReioTheory: il calcolo del fondo di ricombinazione "
                        "non accetta questi parametri: %s" % msg) from e
                raise
            fondo = (float(c.ionization_fraction(float(self.z_max))),
                     float(c.ionization_fraction(float(self.z_max) + 1.0)))
            if self.reio_model not in self._MODELLI_CON_FONDO:
                return fondo
            n = int(np.ceil(float(self.z_max) / self._PASSO_TABELLA_FONDO)) + 1
            z_tab = np.linspace(0.0, float(self.z_max), n)
            x_tab = np.array([c.ionization_fraction(float(q)) for q in z_tab])
            return fondo + ((z_tab, x_tab),)
        finally:
            c.struct_cleanup()
            c.empty()

    def _pavimento_del_punto(self, native_params):
        """
        Imposta self._fondo (xe_before di CLASS) e self._pavimento (x 1.05)
        per la cosmologia del punto.  In memoria per cosmologia: con la
        cosmologia fissa si calcola una volta sola.
        """
        args = self._argomenti_fondo(native_params)
        chiave = tuple(sorted((k, repr(v)) for k, v in args.items()))
        cache = self.__dict__.setdefault("_cache_fondo", {})
        if chiave not in cache:
            if len(cache) >= 64:
                cache.clear()
            cache[chiave] = self._calcola_fondo(args)
            self._fondo_da_verificare = True
        valori = cache[chiave]
        self._fondo, self._fondo_controllo = valori[0], valori[1]
        self._tabella_fondo = valori[2] if len(valori) > 2 else None
        self._pavimento = self._arrotonda_per_eccesso(
            self._MARGINE_SUL_FONDO * self._fondo)

    def _x_noreio(self, z):
        """
        x_noreio(z) di CLASS (sola ricombinazione) per la cosmologia del punto,
        interpolata dalla tabella del calcolo del fondo.  Solo per i modelli di
        _MODELLI_CON_FONDO; altrimenti la tabella non c'e' e ci si ferma.
        """
        tab = getattr(self, "_tabella_fondo", None)
        if tab is None:
            raise RuntimeError(
                "ReioTheory._x_noreio: tabella del fondo non calcolata (modello "
                "%r).  Dentro cobaya la prepara calculate(); fuori da cobaya "
                "serve imposta_pavimento(cosmologia)." % self.reio_model)
        z = np.asarray(z, dtype=float)
        if np.any(z < 0.0) or np.any(z > tab[0][-1] * (1 + 1e-12)):
            raise ValueError("ReioTheory._x_noreio: z fuori da [0, z_max].")
        return np.interp(z, tab[0], tab[1])

    @staticmethod
    def _arrotonda_per_eccesso(v, cifre=2):
        """
        v arrotondato PER ECCESSO a `cifre` cifre significative (2.327e-4 ->
        2.4e-4).  Per eccesso: il margine sul fondo resta >= 5% (al massimo
        ~10% in piu', cioe' ~1e-5 su tau).  Serve alla lunghezza della
        tabella: il pavimento compare in molti nodi (bin a zero), e
        '0.000232734' (11 caratteri) contro '0.00024' (7) bastava a far
        superare il buffer del parser di CLASS a una tabella a bin da 396
        nodi, che veniva decimata (misurato 29 ott 2026).
        """
        e = np.floor(np.log10(v)) - (cifre - 1)
        q = 10.0 ** e
        return float(np.ceil(v / q * (1 - 1e-12)) * q)

    def imposta_pavimento(self, cosmologia):
        """
        Per chi usa ReioTheory fuori da cobaya (script di test): calcola il
        fondo per il dizionario di parametri CLASS `cosmologia` e imposta il
        pavimento, come farebbe calculate().
        """
        self.extra_args = getattr(self, "extra_args", None) or {}
        self._pavimento_del_punto(dict(cosmologia))
        self._fondo_da_verificare = False
        return self._pavimento

    def _verifica_fondo_di_class(self):
        """
        Dopo la run vera, alla prima volta di ogni cosmologia: la ricombinazione
        che CLASS ha calcolato deve essere la stessa del calcolo del fondo.  Se
        no, il calcolo del fondo non ha visto la cosmologia vera e il pavimento
        potrebbe stare sotto il fondo (tau_reio troncata): ci si ferma.

        Si confronta x_e a z_max + 1, NON a z_max.  A z_max CLASS interpola fra
        punti a cavallo del bordo della tabella, e la lettura e' contaminata
        dai nodi sotto z_max (misurato, 29 ott 2026: +8% col vecchio pavimento
        1e-3, +0.12% col pavimento attuale).  Sopra z_max c'e' solo la
        ricombinazione, e le due run coincidono a ~1e-9.
        """
        self._fondo_da_verificare = False
        z_c = float(self.z_max) + 1.0
        x_class = float(self.classy.ionization_fraction(z_c))
        if not np.isclose(x_class, self._fondo_controllo, rtol=1e-6, atol=0.0):
            raise LoggedError(
                self.log, "ReioTheory: la ricombinazione calcolata da CLASS "
                "(x_e(z=%.1f) = %.9e) non coincide con quella del calcolo del fondo "
                "(%.9e): il calcolo del fondo non ha visto la stessa cosmologia "
                "della run, vedi _argomenti_fondo." % (z_c, x_class, self._fondo_controllo))

    # get_reio_history() sta piu' su, accanto a get_can_provide().  Qui ce
    # n'era una seconda copia che restituiva self._current_reio_history, cioe'
    # un attributo d'istanza riscritto solo dentro calculate(): aveva lo stesso
    # difetto della globale, restava indietro su un cache hit.  Chi chiamava
    # questo metodo per fare un grafico otteneva la curva del punto sbagliato
    # (vedi Tests/bug_storia_stantia.py).