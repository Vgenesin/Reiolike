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

