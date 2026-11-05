import numpy as np

# ---------------------------------------------------------------------------
# Helium bookkeeping — kept consistent with CLASS
# ---------------------------------------------------------------------------
# CLASS defines (thermodynamics.c:352)
#     fHe = YHe / (_not4_ * (1 - YHe)),   _not4_ = 3.9715  (m_He/m_H)
# i.e. the He/H *number* density ratio.  The electron budget per hydrogen
# nucleus is therefore
#     1 + fHe   -> H fully ionised + HeI->HeII   ("first helium reionisation")
#     1 + 2*fHe -> H fully ionised + HeII->HeIII ("second helium reionisation")
_NOT4 = 3.9715          # helium-to-hydrogen mass ratio, as in CLASS
# Valore di RISERVA, usato solo da chi chiama queste funzioni fuori da
# ReioTheory (script di test).  Dentro ReioTheory Y_He si ricava punto per
# punto dalla BBN in funzione di omega_b (Reiolike/bbn_helium.py) e si passa
# esplicitamente a ogni modello come xe_max = 1 + f_He.
_Y_HE = 0.2454

# Residual ionisation fraction left over from recombination
_XE_REC = 1.0e-4


def f_he(Y_He=_Y_HE):
    """n_He/n_H for a given helium mass fraction, CLASS convention."""
    return Y_He / (_NOT4 * (1.0 - Y_He))


_F_HE = f_he()                     # 0.081880
XE_MAX_HI_HEI = 1.0 + _F_HE        # 1.081880 — H + first He ionisation
XE_MAX_HI_HEII = 1.0 + 2.0 * _F_HE # 1.163760 — H + second He ionisation


def helium_fullreio_step(z, z_reHeII=3.5, delta_z=0.5, Y_He=_Y_HE):
    """
    Second helium reionisation (HeII -> HeIII) as a tanh step.

    Reproduces the helium term that CLASS applies internally for
    reio_parametrization='reio_camb' (thermodynamics.c:4188-4193):

        Delta xe(z) = fHe * (tanh((z_reHeII - z)/delta_z) + 1) / 2

    which adds fHe electrons per hydrogen nucleus once z << z_reHeII.

    IMPORTANT
    ---------
    CLASS does NOT apply this term when reio_parametrization='reio_inter':
    the index_re_helium_fullreio_* parameters are only allocated for
    reio_camb / reio_half_tanh (thermodynamics.c:1033-1044), and the
    reio_inter branch of thermodynamics_reionization_function() returns the
    interpolated table verbatim (thermodynamics.c:4311-4350).  Verified
    empirically: a table saturating at 1 + fHe gives xe(0) = 1.08000 exactly.
    So when feeding CLASS through reio_inter this step must be added here.

    Parameters
    ----------
    z        : array-like — redshift
    z_reHeII : float — centre of the step (CLASS default helium_fullreio_redshift = 3.5)
    delta_z  : float — width of the step (CLASS default helium_fullreio_width = 0.5)
    Y_He     : float — helium mass fraction

    Returns
    -------
    Delta xe : ndarray — electrons per H nucleus to add, in [0, fHe]
    """
    z = np.asarray(z, dtype=float)
    return f_he(Y_He) * (np.tanh((z_reHeII - z) / delta_z) + 1.0) / 2.0


def tanh_model(z, z_re, delta_z, xe_max=XE_MAX_HI_HEI):
    """
    Standard tanh reionization history model (hydrogen sector only).

        xe(z) = xe_max/2 * (1 + tanh((z_re - z)/delta_z))

    Note this uses a transition linear in z, unlike CLASS's reio_camb which
    uses y = (1+z)^1.5; for the same z_re the two give slightly different
    histories.

    SCOPE: this returns H reionisation + the FIRST helium ionisation
    (HeI -> HeII), which is why it saturates at xe_max = 1 + fHe = 1.0819.
    The SECOND helium ionisation (HeII -> HeIII, around z ~ 3.5) is NOT
    included here and is NOT added by CLASS under reio_inter — see
    helium_fullreio_step().  ReioTheory adds it centrally via its
    reio_include_HeII flag, so that all models are treated identically.

    Parameters:
    -----------
    z : array-like
        Redshift array
    z_re : float
        Redshift at which xe = xe_max/2 (midpoint of reionization)
    delta_z : float
        Width of the transition in redshift space
    xe_max : float
        Fully-ionised H + HeI electron fraction (default 1 + fHe)

    Returns:
    --------
    xe : array-like
        Ionization fraction xe(z)
    """
    z = np.asarray(z, dtype=float)
    xe = xe_max * np.tanh((z_re - z) / delta_z) / 2.0 + xe_max / 2.0
    # Ensure physical bounds
    xe = np.clip(xe, 0.0, xe_max)

    return xe


def tanh_pinned_end_delta_z(z_re, z_end=5.6, xe_end=None, xe_end_frac=0.95,
                            xe_max=XE_MAX_HI_HEI):
    """
    Width delta_z implied by pinning the tanh to (z_end, xe_end).

        xe(z_end) = xe_max/2 * (tanh((z_re - z_end)/delta_z) + 1) = xe_end
        => delta_z = (z_re - z_end) / arctanh(2*xe_end/xe_max - 1)

    Split out of tanh_pinned_end_model() so that callers (e.g. ReioTheory,
    to size the refinement window of the z grid) can obtain the width
    without evaluating the model.  All the validation of the pinning
    constraint lives here.

    Returns
    -------
    delta_z : float, strictly positive
    """
    # Resolve the target xe at z_end
    if xe_end is not None:
        target = float(xe_end)
    else:
        target = xe_end_frac * xe_max

    if not (0.0 < target < xe_max):
        raise ValueError(
            f"Target xe at z_end must be strictly between 0 and xe_max={xe_max}; "
            f"got {target:.4f}. Adjust xe_end or xe_end_frac."
        )
    if z_re <= z_end:
        raise ValueError(
            f"z_re={z_re} must be greater than z_end={z_end} "
            "(midpoint must be at higher redshift than the pinning point)."
        )
    if target <= xe_max / 2.0:
        raise ValueError(
            f"With z_end={z_end} < z_re={z_re}, xe_end={target:.4f} must be "
            f"> xe_max/2 = {xe_max/2:.4f}: the pinning point must lie past the "
            "midpoint (lower z means higher ionisation)."
        )

    t = 2.0 * target / xe_max - 1.0     # required tanh value at z_end
    if not (-1.0 < t < 1.0):
        raise ValueError(
            f"Pinning condition gives |t|={abs(t):.4f} >= 1; "
            "check that 0 < xe_end < xe_max."
        )
    return (z_re - z_end) / np.arctanh(t)


def tanh_pinned_end_model(z, z_re, z_end=5.6, xe_end=None, xe_end_frac=0.95,
                          xe_max=XE_MAX_HI_HEI):
    """
    Tanh reionization history parametrised by midpoint z_re and a
    user-chosen pinning point (z_end, xe_end) instead of the width delta_z.

    The idea is: "I want the tanh to reach xe_end at z_end."
    Given this constraint, delta_z is derived automatically:

        xe(z_end) = xe_max/2 * (tanh((z_re - z_end) / delta_z) + 1) = xe_end
        => tanh((z_re - z_end) / delta_z) = 2*xe_end/xe_max - 1  ≡ t
        => delta_z = (z_re - z_end) / arctanh(t)

    The pinning point can be specified in two ways (mutually exclusive):
      - xe_end      : absolute value of xe at z_end  (e.g. xe_end=1.0)
      - xe_end_frac : fraction of xe_max at z_end    (e.g. xe_end_frac=0.95)
    If both are given, xe_end takes precedence.
    Default: xe_end_frac=0.95 → xe(z_end) = 0.95 * 1.0819 ≈ 1.028, i.e. the
    universe is ~95% ionized (in H+HeI combined) at z_end.

    SCOPE: as for tanh_model, this covers H reionisation + the first helium
    ionisation only.  The HeII -> HeIII step is added centrally by ReioTheory
    (reio_include_HeII), because CLASS does not add it under reio_inter.

    Parameters
    ----------
    z          : array-like — redshift array
    z_re       : float — midpoint: xe(z_re) = xe_max / 2. Must be > z_end.
    z_end      : float — pinning redshift (default 5.6, from Ly-α forest)
    xe_end     : float or None — absolute xe value at z_end; if None, use
                 xe_end_frac * xe_max
    xe_end_frac: float — fraction of xe_max at z_end (default 0.95 → 5%
                 neutral in H+HeI combined); ignored if xe_end is given
    xe_max     : float — fully-ionised H+HeI electron fraction (default 1+fHe).

    Returns
    -------
    xe : array-like — ionisation fraction xe(z), clipped to [0, xe_max]

    Examples
    --------
    # Default: 95% ionised at z=5.6
    xe = tanh_pinned_end_model(z, z_re=7.5)

    # Pin xe=1.0 at z=6.0.  NB: xe counts electrons per H nucleus from H AND
    # the first He ionisation, which proceed together; xe_end=1.0 is NOT
    # "hydrogen fully ionised": it gives x_HII(z_end) = 1/(1+f_He) = 0.924.
    # For x_HII(z_end) = 1 use xe_end = 1 + f_He (= xe_max).
    xe = tanh_pinned_end_model(z, z_re=8.0, z_end=6.0, xe_end=1.0)

    # Pin 90% ionised at z=6.5
    xe = tanh_pinned_end_model(z, z_re=8.0, z_end=6.5, xe_end_frac=0.90)
    """
    z = np.asarray(z, dtype=float)

    delta_z = tanh_pinned_end_delta_z(z_re, z_end=z_end, xe_end=xe_end,
                                      xe_end_frac=xe_end_frac, xe_max=xe_max)

    xe = xe_max * np.tanh((z_re - z) / delta_z) / 2.0 + xe_max / 2.0
    xe = np.clip(xe, 0.0, xe_max)

    return xe


def make_bin_edges(z_min=5.0, z_max=25.0, n_bins=None, width=None,
                   spacing="uniform"):
    """
    Costruisce gli EDGE (bordi) dei bin in redshift.

    Convenzione — leggere con attenzione, e' la fonte di errore piu' comune:
    questa funzione restituisce n_bins+1 BORDI, non i centri.  Il bin i-esimo
    (i = 1..n_bins, numerato dal basso redshift) copre [edges[i-1], edges[i]]
    e ha un solo valore di xe.  Le transizioni tanh sono centrate sui BORDI.

        edges:   z0 -------- z1 -------- z2 -------- z3
        bins:        bin 1       bin 2       bin 3
        xe:          xe_1        xe_2        xe_3

    Per confronto: la parametrizzazione nativa di CLASS reio_bins_tanh vuole
    in ingresso i CENTRI e mette i salti ai punti medi fra centri consecutivi
    (thermodynamics.c:4256), quindi i suoi salti cadono sui nostri bordi: e'
    lo stesso modello, espresso con un'altra convenzione di input.  Centri e
    bordi si convertono con centers = (edges[:-1] + edges[1:]) / 2.

    Parameters
    ----------
    z_min, z_max : float — estremi dello schema.  Sotto z_min la storia e'
        fissata a "reionizzazione completa", sopra z_max a xe=0; non sono
        parametri liberi, sono le assunzioni del modello.
    n_bins  : int or None — numero di bin.  Ha precedenza su `width`.
    width   : float or None — larghezza dei bin (solo per spacing='uniform');
        usato se n_bins e' None.  n_bins = round((z_max - z_min)/width).
    spacing : {'uniform', 'log1p'}
        'uniform' — bin di larghezza costante in z (schema del paper).
        'log1p'   — bin equispaziati in log(1+z), quindi PIU' STRETTI a basso
                    redshift.  Utile perche' i dati astrofisici (Ly-alpha,
                    LAE, QSO damping wings) si concentrano a z ~ 5-8 e li'
                    sono molto piu' informativi: conviene spendere risoluzione
                    dove i dati la possono sostenere, e non a z alto dove
                    vincola solo il CMB integrato.

    Returns
    -------
    edges : ndarray di n_bins+1 valori, strettamente crescenti
    """
    if z_max <= z_min:
        raise ValueError(f"make_bin_edges: z_max={z_max} deve essere > z_min={z_min}.")

    if n_bins is None:
        if width is None:
            raise ValueError("make_bin_edges: fornire n_bins oppure width.")
        if width <= 0:
            raise ValueError(f"make_bin_edges: width={width} deve essere > 0.")
        n_bins = int(round((z_max - z_min) / width))
        if n_bins < 1:
            raise ValueError(
                f"make_bin_edges: width={width} e' piu' larga dell'intervallo "
                f"[{z_min}, {z_max}]: verrebbero {n_bins} bin."
            )
    n_bins = int(n_bins)
    if n_bins < 1:
        raise ValueError(f"make_bin_edges: n_bins={n_bins} deve essere >= 1.")

    if spacing == "uniform":
        return np.linspace(z_min, z_max, n_bins + 1)
    if spacing == "log1p":
        return np.geomspace(1.0 + z_min, 1.0 + z_max, n_bins + 1) - 1.0
    raise ValueError(
        f"make_bin_edges: spacing='{spacing}' non riconosciuto "
        "(usare 'uniform' o 'log1p')."
    )


def bin_edges_from_data(z_points, delta_z=0.1, min_distance=1.5,
                        tail_edges=(12.0, 16.0, 25.0)):
    """
    Bordi dei bin messi nei VUOTI fra i redshift dei dati puntuali.

    Il gradino fra due bin e' una tanh larga delta_z centrata sul bordo: un
    dato a distanza d dal bordo vede il bin accanto con peso
    (1 - tanh(d/delta_z))/2, cioe' 50% sul bordo, 12% a 1 delta_z, 4.7% a
    1.5, 1.8% a 2.  Un dato sul bordo misura la media di due bin, e il
    risultato dipende da dove cade il bordo (schema del paper, bordo a z=7
    con tre dati a z=7.0: spostandolo di 0.05 il bin 2 cambia di ~0.8 sigma).

    La regola: fra due dati consecutivi z_a < z_b si mette un bordo, al centro
    (z_a + z_b)/2, solo se il vuoto e' largo almeno 2*min_distance*delta_z;
    cosi' ogni bordo dista almeno min_distance*delta_z da ogni dato.  Dati piu'
    vicini di cosi' finiscono nello stesso bin.

    Sopra l'ultimo dato puntuale non ci sono vuoti da riempire: i bordi li'
    sono tail_edges (quelli oltre l'ultimo dato + min_distance*delta_z), bin
    larghi dove i dati vincolano poco.  Se nessuno e' oltre, l'ultimo bordo
    della regola chiude lo schema e l'ultimo dato resta sopra, dove il modello
    fissa xe = 0: e' un errore, perche' quel dato non vincolerebbe nulla.

    I dati con una finestra in z NON entrano: misurano gia' una media su un
    intervallo, che la likelihood calcola esatta anche con un bordo dentro.

    Parameters
    ----------
    z_points     : redshift dei dati senza larghezza in z (anche ripetuti)
    delta_z      : larghezza delle transizioni (reio_bin_sharpness)
    min_distance : distanza minima bordo-dato in unita' di delta_z
    tail_edges   : bordi sopra l'ultimo dato puntuale

    Returns
    -------
    edges : ndarray strettamente crescente
    """
    z = np.unique(np.asarray(z_points, dtype=float).ravel())
    z = z[np.isfinite(z)]
    if z.size == 0:
        raise ValueError(
            "bin_edges_from_data: nessun dato puntuale, i bordi non si possono "
            "mettere nei vuoti fra i dati.")
    if delta_z <= 0 or min_distance <= 0:
        raise ValueError(
            f"bin_edges_from_data: delta_z={delta_z} e min_distance="
            f"{min_distance} devono essere > 0.")
    d = min_distance * delta_z
    gap = np.diff(z)
    # 1e-9: vuoti di 0.3 calcolati in floating point (7.3 - 7.0) non devono
    # cadere sotto 2*d = 0.3 per un ulp
    ok = gap >= 2.0 * d - 1e-9
    edges = list(np.round(0.5 * (z[:-1] + z[1:])[ok], 10))
    tail = sorted(float(t) for t in (tail_edges or ()) if t >= z[-1] + d - 1e-9)
    edges += tail
    if not tail:
        raise ValueError(
            f"bin_edges_from_data: nessun bordo sopra l'ultimo dato puntuale "
            f"(z={z[-1]:g}): servono tail_edges oltre z={z[-1] + d:g}, "
            "altrimenti quel dato cade nella zona fissata a xe = 0.")
    edges = np.asarray(edges, dtype=float)
    if edges.size < 2:
        raise ValueError(
            "bin_edges_from_data: i dati puntuali non lasciano vuoti abbastanza "
            f"larghi (>= {2 * d:g}) e c'e' un solo bordo: nessun bin.")
    return edges


def binned_model(z, bin_edges, bin_xe, delta_z=0.1,
                 xe_below=None, xe_above=0.0):
    """
    Storia di reionizzazione a bin (Ilic et al. 2025, arXiv:2504.13254,
    Sect. 2.5; il loro modello '10-bins').

    xe e' costante a tratti su una successione di intervalli di redshift, con
    transizioni tanh strette (Delta z = 0.1 nel paper) fra un plateau e il
    successivo.  E' l'approccio piu' vicino a "model-independent" fra quelli
    del paper: non impone ne' una forma funzionale ne' la monotonia, quindi
    ammette anche storie non monotone (periodi di ri-neutralizzazione).

    Formula — sovrapposizione di gradini centrati sui BORDI:

        xe(z) = v_0 + sum_{k=0}^{n} (v_{k+1} - v_k) *
                      [1 + tanh((z - edges_k)/delta_z)] / 2

    con v = [xe_below, xe_1, ..., xe_n, xe_above].  A z molto basso tutte le
    tanh valgono -1 e resta xe_below; a z molto alto valgono +1 e la somma
    telescopica da' xe_above.

    CONVENZIONE: `bin_edges` sono i BORDI (n+1 valori), `bin_xe` i valori di
    plateau (n valori).  Il bin i copre [edges[i-1], edges[i]], numerato dal
    basso redshift.  Vedere make_bin_edges() per il diagramma.

    Parameters
    ----------
    z         : array-like — redshift
    bin_edges : array-like di n+1 valori strettamente crescenti
    bin_xe    : array-like di n valori di plateau
    delta_z   : float — larghezza delle transizioni tanh (paper: 0.1).
        Non e' fisica, e' un regolarizzatore: serve solo a rendere il gradino
        derivabile.  Deve restare piccola rispetto al bin piu' stretto,
        altrimenti transizioni adiacenti si fondono e i bin smettono di
        essere bin.
    xe_below  : float or None — valore imposto sotto edges[0].  None => 1+f_He,
        cioe' "reionizzazione conclusa" (idrogeno + primo elio ionizzati).
        E' l'assunzione che rende sicuro fissare edges[0] ~ 5: la foresta
        Ly-alpha dice che a z < 5.3 l'IGM e' gia' completamente ionizzato.
    xe_above  : float — valore imposto sopra edges[-1] (default 0).

    Returns
    -------
    xe : ndarray della stessa forma di z

    Notes
    -----
    SCOPE: come per gli altri modelli qui, questo e' il settore idrogeno piu'
    la PRIMA ionizzazione dell'elio.  Il gradino HeII -> HeIII lo aggiunge
    ReioTheory tramite reio_include_HeII, che e' anche la convenzione del
    paper (Sect. 2: l'elio secondario e' incluso in tutti i loro modelli come
    tanh a z=3.5 con larghezza 0.5).
    """
    edges = np.asarray(bin_edges, dtype=float)
    vals = np.asarray(bin_xe, dtype=float)

    if edges.ndim != 1 or edges.size < 2:
        raise ValueError(
            f"binned_model: bin_edges deve avere almeno 2 bordi, ne ha {edges.size}."
        )
    if vals.size != edges.size - 1:
        raise ValueError(
            f"binned_model: con {edges.size} bordi servono {edges.size - 1} "
            f"valori di xe, ne sono stati passati {vals.size}.  Ricorda che "
            "bin_edges sono i BORDI, non i centri."
        )
    if np.any(np.diff(edges) <= 0):
        raise ValueError(
            f"binned_model: bin_edges deve essere strettamente crescente, "
            f"ricevuto {edges}."
        )
    if delta_z <= 0:
        raise ValueError(f"binned_model: delta_z={delta_z} deve essere > 0.")

    if xe_below is None:
        xe_below = XE_MAX_HI_HEI

    v = np.concatenate([[xe_below], vals, [xe_above]])
    dv = np.diff(v)                                    # n+1 gradini

    zz = np.asarray(z, dtype=float)
    steps = (1.0 + np.tanh((zz[..., None] - edges) / delta_z)) / 2.0
    return v[0] + (steps * dv).sum(axis=-1)


def power_law_model(z, z_end, alpha, z_beg=20.0, xe_max=XE_MAX_HI_HEI):
    """
    Power-law reionisation history (Douspis et al. 2015; Planck Int. XLVII
    2016), in the form used by Ilic et al. 2025, "Reconstructing the epoch of
    reionisation with Planck PR4" (arXiv:2504.13254), Eq. (3):

        xe(z) = (1 + f_He)                                     for z <= z_end
        xe(z) = (1 + f_He) * ((z_beg - z)/(z_beg - z_end))^a    for z >  z_end

    i.e. the IGM is fully ionised (H + first He) below z_end, and above it the
    ionised fraction falls off as a power law, reaching exactly zero at z_beg.

    The point of this model versus a tanh is ASYMMETRY.  Hydrodynamical
    simulations of the EoR find reionisation starts slowly, as the first stars
    form, and finishes abruptly - a shape a symmetric tanh cannot produce.  The
    exponent alpha controls that asymmetry.  Halfway between z_end and z_beg
    xe/xe_max = 0.5^alpha, so (verified in Tests/test_affidabilita_power_law.py,
    test A3):

        alpha < 1  -> steep start at z_beg (infinite slope there), then a slow
                      tail towards full ionisation at z_end
        alpha = 1  -> linear ramp between z_beg and z_end
        alpha > 1  -> gradual start (zero slope at z_beg): the IGM stays nearly
                      neutral for a long stretch and most of the ionisation
                      happens close to z_end - late, abrupt completion, the
                      shape simulations prefer
        alpha ->    large: reionisation crowds towards z_END (late, sudden)

    (Until 28 Oct 2026 this list said the opposite for alpha > 1 and large
    alpha; the code was always right, only the description was inverted.)

    The paper considers two variants (their Table 1):

        2-pow : z_beg fixed to 20      z_end ~ U(5, 20),  alpha ~ U(0, 20)
        3-pow : z_beg free             z_end ~ U(5, 25),  alpha ~ U(0, 20),
                                       z_beg ~ U(5, 25)

    Parameters
    ----------
    z      : array-like — redshift
    z_end  : float — redshift at which reionisation completes (xe = xe_max
             below it).  Must satisfy z_end < z_beg.
    alpha  : float — power-law exponent, >= 0.  Controls the asymmetry.
    z_beg  : float — redshift at which reionisation starts (xe = 0 above it).
    xe_max : float — fully-ionised H + HeI electron fraction (default 1 + fHe)

    Returns
    -------
    xe : ndarray — ionisation fraction, exactly 0 for z >= z_beg and exactly
         xe_max for z <= z_end.

    Notes
    -----
    SCOPE: as for the other models here, this is the hydrogen sector plus the
    FIRST helium ionisation only.  The HeII -> HeIII step is added centrally
    by ReioTheory via reio_include_HeII - which matches this paper's own
    convention (Sect. 2: "we include in all models a second reionisation of
    helium ... using a simple tanh function of width 0.5 and central redshift
    3.5").

    The function has genuine kinks (discontinuous first derivative) at z_end
    and at z_beg; ReioTheory places grid nodes exactly there so that the
    linear interpolation CLASS performs on the table stays exact across them.
    """
    z = np.asarray(z, dtype=float)

    if z_beg <= z_end:
        raise ValueError(
            f"power_law_model: z_beg={z_beg} must be > z_end={z_end} "
            "(reionisation starts at higher redshift than it ends)."
        )
    if alpha < 0.0:
        raise ValueError(f"power_law_model: alpha={alpha} must be >= 0.")

    # u = 1 at z_end, 0 at z_beg; clipping handles both plateaux at once
    u = np.clip((z_beg - z) / (z_beg - z_end), 0.0, 1.0)

    # np.where guards the alpha=0 corner: 0**0 is 1 in numpy, but the model
    # must give xe=0 at and above z_beg for every alpha.
    return np.where(u > 0.0, xe_max * np.power(u, alpha), 0.0)


def double_tanh_model(z, z_re1, z_re2, xe1, delta_z, xe_before=0.0,
                      xe_final=XE_MAX_HI_HEI):
    """
    Double tanh reionization history model.

    Mirrors exactly CLASS's reio_many_tanh parametrization with many_tanh_num=2,
    so results are consistent when passing these parameters to CLASS natively via:
        reio_parametrization  = reio_many_tanh
        many_tanh_num         = 2
        many_tanh_z           = z_re2, z_re1        (ascending order required by CLASS)
        many_tanh_xe          = xe_final, xe1        (xe at each center)
        many_tanh_width       = delta_z

    The formula (derived from thermodynamics.c) is a superposition of two tanh steps:

        xe(z) = xe_before
                + (xe1     - xe_before) * (1 + tanh((z_re1 - z) / delta_z)) / 2   [first, high-z step]
                + (xe_final - xe1     ) * (1 + tanh((z_re2 - z) / delta_z)) / 2   [second, low-z step]

    Parameters:
    -----------
    z : array-like
        Redshift array.
    z_re1 : float
        Center of the first (high-z) reionization step — xe reaches xe1 here.
        Must satisfy z_re1 > z_re2.
    z_re2 : float
        Center of the second (low-z) reionization step — xe reaches xe_final here.
    xe1 : float
        Target xe at the midpoint of the first step (partial ionization level).
        Must satisfy xe_before < xe1 < xe_final.
    delta_z : float
        Common transition width for both steps (many_tanh_width in CLASS).
        NOTE: CLASS uses a single shared width for all jumps; independent widths
        per step are not supported natively. Use reio_inter if you need that.
    xe_before : float, optional
        Pre-reionization ionization fraction (default 0.0).
    xe_final : float, optional
        Fully-reionized ionization fraction (default 1 + fHe = H + first He).
        The HeII -> HeIII step is added centrally by ReioTheory.

    Returns:
    --------
    xe : array-like
        Ionization fraction xe(z).
    """
    z = np.asarray(z)
    if z_re1 <= z_re2:
        raise ValueError(f"z_re1={z_re1} must be > z_re2={z_re2} (first event at higher z).")
    if not (xe_before <= xe1 <= xe_final):
        raise ValueError(f"xe1={xe1} must be between xe_before={xe_before} and xe_final={xe_final}.")
    if delta_z <= 0:
        raise ValueError(f"delta_z={delta_z} must be strictly positive.")

    xe = (xe_before
          + (xe1 - xe_before) * (1.0 + np.tanh((z_re1 - z) / delta_z)) / 2.0
          + (xe_final - xe1)  * (1.0 + np.tanh((z_re2 - z) / delta_z)) / 2.0)
    xe = np.clip(xe, 0.0, xe_final)
    return xe


# ---------------------------------------------------------------------------
# Gompertz + Symbolic Regression model  (Montero-Camacho, Li, Cranmer 2024)
# arXiv:2405.13680
# ---------------------------------------------------------------------------

# Fixed polynomial coefficients for the universal shape F  (Eq. 2.3)
# c_0=0 and c_1=1 are absorbed into the (alpha, beta) transformation, so
# the paper lists them as fixed to 0 and 1 respectively.
_GOMP_C = np.array([0.0, 1.0, 0.1130, 0.02600, 0.0005491, -0.00006518])


def _gompertz_xhi(z, ln_alpha, beta):
    """
    Neutral hydrogen fraction x_HI(z) from the Gompertz model (Eqs. 2.2–2.4).

    x_HI(ã) = exp(-exp(P5(ln ã)))
    P5(u)   = sum_{m=0}^{5} c_m * u^m    with u = ln(ã)
    ã       = (a / alpha)^beta            with a = 1/(1+z)

    Parameters
    ----------
    z        : array-like  — redshift
    ln_alpha : float       — ln(alpha) from Eq. 3.1
    beta     : float       — beta from Eq. 3.2;  must be > 0

    Returns
    -------
    x_HI : ndarray, clipped to [0, 1]
    """
    z   = np.asarray(z, dtype=float)
    a   = 1.0 / (1.0 + z)
    # ln(ã) = beta * (ln a - ln alpha)
    ln_atilde = beta * (np.log(a) - ln_alpha)
    # Polynomial P5 evaluated at u = ln(ã)
    p5 = np.polyval(_GOMP_C[::-1], ln_atilde)  # polyval wants highest-degree first
    # Gompertz function:  x_HI = exp(-exp(P5))
    x_hi = np.exp(-np.exp(p5))
    return np.clip(x_hi, 0.0, 1.0)


def _sr_ln_alpha(sigma8, ns, h, Omega_b, Omega_m, zeta_uv):
    """
    SR mapping for ln(alpha)  (Eq. 3.1, SRFull).

    ln alpha = (Omega_b/Omega_m)^h
               - (sigma8 - 0.04835) * (ns + 0.3558 * ln(0.1123 * zeta_uv))
               - Omega_m - ns
    """
    term1 = (Omega_b / Omega_m) ** h
    term2 = (sigma8 - 0.04835) * (ns + 0.3558 * np.log(0.1123 * zeta_uv))
    return term1 - term2 - Omega_m - ns


def _sr_beta(sigma8, ns, h, Omega_b, Omega_m, zeta_uv):
    """
    SR mapping for beta  (Eq. 3.2, SRFull).

    beta = [ 0.005660^{Omega_m} / 0.6015
             - ln(zeta_uv - (Omega_m + ns*h)^{15.05})
             + h ] * ln(Omega_b)
           + h / sigma8

    ATTENZIONE al primo termine: e' 0.005660 ELEVATO a Omega_m, diviso
    0.6015 (Eq. 3.2 di arXiv:2405.13680 v3, pag. 12, letta sull'immagine:
    l'estrazione di testo dal PDF la rende ambigua).  Fino al 28 ott 2026 qui
    c'era 0.005660 / Omega_m^0.6015: beta sbagliata fino a 1.5 nel box
    dell'Eq. 2.1, e con i parametri della Tab. 1 del paper tau = 0.0514
    invece di 0.0526.  Con questa forma il pacchetto riproduce la Tab. 1
    (tau 0.0526, z_re 6.96 contro 6.98(17): il resto e' Omega_m senza
    neutrini, scelta voluta).  Test: Tests/test_affidabilita_gompertz.py, A2.
    """
    inner = zeta_uv - (Omega_m + ns * h) ** 15.05
    if inner <= 0.0:
        raise ValueError(
            f"SR beta: argument of log is non-positive ({inner:.4g}). "
            "Parameters are likely outside the valid training range."
        )
    term1 = 0.005660 ** Omega_m / 0.6015 - np.log(inner) + h
    return term1 * np.log(Omega_b) + h / sigma8


def gompertz_sr_model(z, sigma8, ns, h, Omega_b, Omega_m, zeta_uv,
                      include_HeII=True, Y_He=_Y_HE):
    """
    Ionisation fraction xe(z) from the Gompertz + SRFull model
    (Montero-Camacho, Li & Cranmer 2024, arXiv:2405.13680).

    The model proceeds in three steps:

    1. Compute ln(alpha) and beta from the SR formulae (Eqs. 3.1–3.2).
    2. Compute x_HI(z) via the Gompertz universal shape (Eqs. 2.2–2.4).
    3. Convert x_HI → xe including first He reionisation (Eq. 2.5) and,
       optionally, second He reionisation via a tanh step (Eq. 2.6).

    Parameters
    ----------
    z        : array-like  — redshift
    sigma8   : float       — σ₈ (present-day rms matter fluctuation, 8 h⁻¹ Mpc)
    ns       : float       — scalar spectral index
    h        : float       — dimensionless Hubble constant H0/100
    Omega_b  : float       — baryon density fraction (NOT Omega_b h²)
    Omega_m  : float       — total matter density fraction (NOT Omega_m h²)
    zeta_uv  : float       — ionisation efficiency (astrophysical nuisance)
    include_HeII : bool    — if True, add the HeII reionisation tanh step
                            (Eq. 2.6; z_reHeII=3.5, Delta_z=0.5).  Set False
                            only if the caller adds the step itself (as
                            ReioTheory does via reio_include_HeII).  Note that
                            CLASS does NOT add it under reio_inter, so it must
                            be applied on one side or the other.
    Y_He     : float       — helium mass fraction (default: CLASS value 0.2454)

    Returns
    -------
    xe : ndarray  — free-electron fraction xe(z)

    Notes
    -----
    Valid training range (Eq. 2.1):
        sigma8  in (0.74, 0.90),  ns     in (0.92, 1.00),
        h       in (0.61, 0.73),  Omega_b in (0.04, 0.06),
        Omega_m in (0.24, 0.40),  zeta_uv in (20, 35).
    Calling this function outside these ranges may give unphysical results.

    Omega_b and Omega_m here are the *density fractions* (not Omega*h²).
    If Cobaya/CLASS uses Omega_b h² and Omega_m h², divide by h².
    """
    z = np.asarray(z, dtype=float)

    # --- Step 1: SR mappings -------------------------------------------------
    ln_alpha = _sr_ln_alpha(sigma8, ns, h, Omega_b, Omega_m, zeta_uv)
    beta     = _sr_beta(sigma8, ns, h, Omega_b, Omega_m, zeta_uv)

    if beta <= 0.0:
        raise ValueError(
            f"SR model: beta={beta:.4g} <= 0. Reionisation history would be "
            "inverted. Parameters are outside the valid range."
        )

    # --- Step 2: neutral hydrogen fraction -----------------------------------
    x_hi = _gompertz_xhi(z, ln_alpha, beta)

    # x_e contribution from HI+HeI reionisation (Eq. 2.5)
    # xe_gomp = (1 - x_HI)  is the ionised H fraction
    # NOTE: the paper writes n_He/n_H = Y_He/(4(1-Y_He)); we use CLASS's
    # _not4_ = 3.9715 instead of 4 so that the electron budget matches the
    # fHe CLASS itself uses (0.7% difference on the helium term).
    nhe_nh = f_he(Y_He)
    xe_gomp = 1.0 - x_hi
    xe = (1.0 + nhe_nh - _XE_REC) * xe_gomp + _XE_REC

    # --- Step 3 (optional): HeII reionisation tanh step (Eq. 2.6) -----------
    # Eq. 2.6 is Y_He/(8(1-Y_He)) * (tanh + 1), i.e. exactly fHe/2 * (tanh + 1)
    # = the same step CLASS applies for reio_camb.  Left optional so that
    # ReioTheory can add it centrally for every model via reio_include_HeII.
    if include_HeII:
        xe = xe + helium_fullreio_step(z, Y_He=Y_He)

    return xe



def pca_model(z, z_knots, xe_knots):
    """
    Storia di reionizzazione ricostruita da una base PCA (Mortonson & Hu 2008;
    Ilic et al. 2025, arXiv:2504.13254, Sect. 2.6, il loro modello 'pca-5').

        xe(z) = xe_fid(z) + sum_a m_a S_a(z)

    Questa funzione riceve la somma gia' fatta, sotto forma di nodi
    (z_knots, xe_knots) prodotti da PCABasis.knots(), e la valuta su `z`
    interpolando linearmente.

    Perche' la somma non si fa qui: le autofunzioni S_a esistono solo nei
    centri dei bin con cui e' stata costruita la matrice di Fisher, e insieme a
    esse serve tutto il contorno (z_min, z_max, delta_z, xe fiduciale, elio).
    Tutto questo sta nel file della base e lo gestisce Reiolike.pca_basis; qui
    resta solo la valutazione, come per gli altri modelli di questo modulo.

    LA COSA IMPORTANTE: fuori dai nodi np.interp e' lineare, e anche CLASS in
    reio_inter interpola linearmente.  Se la griglia passata a CLASS contiene
    i nodi, la tabella che CLASS ricostruisce coincide ESATTAMENTE con questa
    funzione — non "entro l'errore di griglia": esattamente.  E' il motivo per
    cui il modello pca non ha bisogno di finestre di raffinamento.

    Oltre gli estremi np.interp tiene il valore del nodo piu' vicino: sotto
    z_min resta "reionizzazione conclusa", sopra z_max resta xe_early.  E'
    esattamente il comportamento voluto.

    SCOPE: come per gli altri modelli qui, questo e' il settore idrogeno piu'
    la PRIMA ionizzazione dell'elio.  Il gradino HeII -> HeIII lo aggiunge
    ReioTheory tramite reio_include_HeII.

    Parameters
    ----------
    z        : array-like — redshift
    z_knots  : array crescente — nodi in redshift (da PCABasis.knots)
    xe_knots : array della stessa lunghezza — xe nei nodi

    Returns
    -------
    xe : ndarray della stessa forma di z
    """
    z_knots = np.asarray(z_knots, dtype=float)
    xe_knots = np.asarray(xe_knots, dtype=float)
    if z_knots.shape != xe_knots.shape:
        raise ValueError(
            f"pca_model: z_knots ha {z_knots.size} elementi e xe_knots "
            f"{xe_knots.size}."
        )
    if np.any(np.diff(z_knots) <= 0):
        raise ValueError("pca_model: z_knots deve essere strettamente crescente.")
    return np.interp(np.asarray(z, dtype=float), z_knots, xe_knots)
