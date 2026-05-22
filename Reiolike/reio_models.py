import numpy as np

def tanh_model(z, z_re, delta_z):
    """
    Standard tanh reionization history model.
    xe(z) = f_He/2 * (1 + tanh((y(z_re)-y(z))/delta_y))
    This is a simplified version:
    xe(z) = (1+f_He)/2 * (1 + tanh((z_re - z)/delta_z))
    
    Parameters:
    -----------
    z : array-like
        Redshift array
    z_re : float
        Redshift at which xe = 0.5 * (1+f_He) (midpoint of reionization)
    delta_z : float
        Width of the transition in redshift space
        
    Returns:
    --------
    xe : array-like
        Ionization fraction xe(z)
    """
    # Helium fraction (consistent with BBN/CLASS standard)
    # Note: CLASS handles helium reionization separately usually (at z~3.5)
    # Here we model Hydrogen reionization + first Helium ionization.

    
    xe_max = 1.08 
    xe_min = 2e-4 # Residual ionization from recombination
    
    # Tanh formula
    
    xe = 1.08 * np.tanh((z_re-z) / delta_z) / 2.0 + 1.08 / 2.0
    # Ensure physical bounds
    xe = np.clip(xe, 0, xe_max)
    
    return xe


def tanh_pinned_end_model(z, z_re):
    """
    Stretched tanh reionization history model with the end of reionization
    pinned to z_end = 5.6 (5% neutral fraction), as suggested by Lyman-alpha
    forest data.

    For a given midpoint z_re, the width delta_z is automatically derived by
    requiring xe(z_end) = 0.95 * xe_max, i.e.:

        tanh((z_re - z_end) / delta_z) = 0.9
        =>  delta_z = (z_re - z_end) / arctanh(0.9)

    so that the low-z end of reionization is always anchored at z = 5.6,
    regardless of z_re.

    Parameters:
    -----------
    z : array-like
        Redshift array
    z_re : float
        Redshift at which xe = 0.5 * xe_max (midpoint of reionization).
        Must satisfy z_re > 5.6.

    Returns:
    --------
    xe : array-like
        Ionization fraction xe(z)
    """
    xe_max = 1.00
    z_end = 5.6  # pinned: 5% neutral (xe = 0.95 * xe_max) at this redshift

    if z_re <= z_end:
        raise ValueError(f"z_re={z_re} must be greater than z_end={z_end}")

    # Derive delta_z so that xe(z_end) = 0.95 * xe_max
    delta_z = (z_re - z_end) / np.arctanh(0.9)

    xe = xe_max * np.tanh((z_re - z) / delta_z) / 2.0 + xe_max / 2.0
    xe = np.clip(xe, 0, xe_max)

    return xe


def double_tanh_model(z, z_re1, z_re2, xe1, delta_z, xe_before=0.0, xe_final=1.08):
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
        Fully-reionized ionization fraction (default 1.08 = H + first He).

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

