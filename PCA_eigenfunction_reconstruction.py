#!/usr/bin/env python
"""
Fisher Matrix analysis for xe(z) in reionization
96 bins from z=6 to z=30
"""

import numpy as np
import matplotlib.pyplot as plt
from classy import Class
import time

# ==========================================
# SETUP: bins from z=6 to z=30, Δz=0.25
# Seguendo la convenzione del paper:
#   z_i = z_min + i*Δz, i=1..Nz
#   z_1 = z_min + Δz = 6.25  (primo bin libero)
#   z_Nz = z_max - Δz = 29.75 (ultimo bin libero)
#   Nz + 1 = (z_max - z_min)/Δz  →  Nz = 95
# ==========================================

Z_MIN = 5.0
Z_MAX = 25.0
DELTA_Z = 0.25

# Bin centers: z_min+Δz, z_min+2Δz, ..., z_max-Δz
# np.arange garantisce step ESATTAMENTE pari a DELTA_Z (evita errori floating point
# di np.linspace che causano spike nel calcolo delle derivate con CLASS)
z_centers = np.arange(Z_MIN + DELTA_Z, Z_MAX, DELTA_Z)  # [6.25, 6.50, ..., 29.75]
Nz = len(z_centers)   # = 95
N_BINS = Nz

print(f"Binning setup (paper convention):")
print(f"  Nz = {Nz} free bins")
print(f"  z_1  = {z_centers[0]:.3f}  (= z_min + Δz)")
print(f"  z_Nz = {z_centers[-1]:.3f} (= z_max - Δz)")
print(f"  Δz   = {DELTA_Z}")

# Tutti i bin sono liberi: i boundary conditions sono gestiti
# esplicitamente in generate_class_arrays
free_bin_indices = list(range(Nz))
N_FREE = Nz
z_centers_free = z_centers
print(f"  Free bins (PCA): {N_FREE} (tutti i bin, da z={z_centers_free[0]:.3f} a z={z_centers_free[-1]:.3f})")


def generate_class_arrays(xe_values):
    """
    Boundary conditions:
      z=0          : xe = 1.08  (H + He)
      z=3          : xe = 1.08
      z=Z_MIN      : xe = 1.0   (solo H, ancora bassa-z)
      z=Z_MIN+Δz   : xe = xe_values[0]   (primo bin libero)
      ...
      z=Z_MAX-Δz   : xe = xe_values[-1]  (ultimo bin libero)
      z=Z_MAX      : xe = 0.0            (ancora alta-z)
    """
    z_array  = np.array([0.0, 3.0, Z_MIN] + list(z_centers) + [Z_MAX])
    xe_array = np.array([1.08, 1.08, 1.0] + list(xe_values) + [0.0])
    return z_array, xe_array

# ==========================================
# FIDUCIAL MODEL: definito dai paper
# ==========================================

xe_fiducial = np.ones(N_BINS) * 0.15  # paper: xe_fid = 0.15 → τ_fid ≈ 0.066

# Generate CLASS arrays
#gli passo i valori di xe_fiducial, questo definisce  il valore per z_class e xe_class quindi sistema i bins del redshift e 
#sistema il valore di xe per ogni bin
z_class, xe_class = generate_class_arrays(xe_fiducial)

print(f"\nCLASS arrays:")
print(f"  reio_inter_num = {len(z_class)}")
print(f"  reio_inter_z: {z_class}")
print(f"  reio_inter_xe: {xe_class}")

# ==========================================
# RUN FIDUCIAL MODEL
# ==========================================

print("\n" + "="*60)
print("RUNNING FIDUCIAL MODEL")
print("="*60)

M = Class()

params = {
    'output': 'pCl',
    'lensing': 'no',
    'l_max_scalars': 100,
    'h': 0.6732,
    'omega_b': 0.02237,
    'omega_cdm': 0.1201,
    'A_s': 2.120e-9,
    'n_s': 0.9651,
    'reio_parametrization': 'reio_inter',
    'reio_inter_num': len(z_class),
    # Usa formato compatto con meno decimali per ridurre lunghezza stringa
    'reio_inter_z': ','.join([f'{z:.4g}' for z in z_class]),
    'reio_inter_xe': ','.join([f'{xe:.4g}' for xe in xe_class]),
}

start = time.time()
M.set(params)
M.compute()
elapsed = time.time() - start

print(f"✓ Computation completed in {elapsed:.2f} seconds")
derived = M.get_current_derived_parameters(['tau_reio', 'conformal_age'])
print(f"  τ_reio = {derived['tau_reio']:.5f}")

cl_fiducial = M.raw_cl(50)  # unlensed 
print(f"  Available spectra: {list(cl_fiducial.keys())}")

# Get thermodynamics
thermo = M.get_thermodynamics()
z_thermo = thermo['z']
xe_thermo = thermo['x_e']

M.struct_cleanup()

# ==========================================
# PLOT IONIZATION HISTORY
# ==========================================

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

# Plot xe(z) with bins
ax1.semilogx(z_thermo, xe_thermo, 'b-', linewidth=2, label='CLASS output')

# Mark bin centers (ogni 10 per non affollare)
for i, z in enumerate(z_centers):
    if i % 10 == 0:
        ax1.axvline(z, color='green', alpha=0.5, linestyle=':', linewidth=1.5, label='bin centers' if i == 0 else '')
ax1.axvline(Z_MIN, color='red', linewidth=2, label=f'z_min = {Z_MIN}')
ax1.axvline(Z_MAX, color='red', linewidth=2, linestyle='--', label=f'z_max = {Z_MAX}')
ax1.set_xlabel('Redshift z', fontsize=12)
ax1.set_ylabel('Free electron fraction $x_e$', fontsize=12)
ax1.set_title(f'Ionization History ({N_BINS} bins)', fontsize=14)
ax1.set_xlim([1, 50])
ax1.set_ylim([-0.1, 1.3])
ax1.grid(True, alpha=0.3)
ax1.legend()

# Zoom on reionization range
ax2.plot(z_thermo, xe_thermo, 'b-', linewidth=2, label='CLASS output')
ax2.axvline(Z_MIN, color='red', linewidth=2, alpha=0.7, label=f'z_min = {Z_MIN}')
ax2.axvline(Z_MAX, color='red', linewidth=2, linestyle='--', alpha=0.7, label=f'z_max = {Z_MAX}')

# Show bin centers (ogni 4)
for i, z in enumerate(z_centers):
    if i % 4 == 0:
        ax2.axvline(z, color='green', alpha=0.6, linestyle=':', linewidth=1.5, label='bin centers' if i == 0 else '')
        ax2.text(z, 1.15, f'{z:.0f}', ha='center', fontsize=9, color='green')
ax2.set_xlabel('Redshift z', fontsize=12)
ax2.set_ylabel('Free electron fraction $x_e$', fontsize=12)
ax2.set_title(f'Zoom on reionization range', fontsize=14)
ax2.set_xlim([Z_MIN - 2, Z_MAX + 2])
ax2.set_ylim([-0.1, 1.3])
ax2.grid(True, alpha=0.3)

plt.tight_layout()

plt.show()

# ==========================================
# PLOT POWER SPECTRA
# ==========================================


ell = cl_fiducial['ell'][2:30]
T_CMB = 2.7255e6  # microkelvin

# Conversione Cl (adimensionale) -> D_ell (microkelvin^2)
cl_ee = cl_fiducial['ee'][2:30]  # Cl adimensionali
dl_factor = ell * (ell + 1) / (2 * np.pi)
d_ell_ee = dl_factor * cl_ee * T_CMB**2  # D_ell in microkelvin^2

plt.plot(ell, d_ell_ee, 'b-', linewidth=2)
plt.xlabel(r'$\ell$')
plt.ylabel(r'$D_\ell^{EE}$ [$\mu K^2$]')
plt.title('EE Power Spectrum (Fiducial)')
plt.grid(True, alpha=0.3)
plt.show()

plt.plot(ell, cl_fiducial['ee'][2:30] , 'b-', linewidth=2)
plt.xlabel(r'$\ell$')
plt.ylabel(r'$C_\ell^{EE}$ [dimensionless]')
plt.title('EE Power Spectrum (Fiducial)')
plt.grid(True, alpha=0.3)
plt.show()  





# =======================================
# Calcolo della derivata logaritmica
# =======================================

delta_xe = 0.05 # xe_minus = 0.15 - 0.05 = 0.10, lontano da 0 → nessun boundary issue

def compute_derivative_for_bin(bin_index, delta_xe=0.001):

    print(f"\n  Computing derivative for bin {bin_index} (z={z_centers[bin_index]:.2f})...")
    print(f"    Bin {bin_index} corresponds to redshift z={z_centers[bin_index]:.4f}")
    print(f"    This is the {bin_index+1}-th bin out of {N_BINS} total bins")
    
    xe_plus = xe_fiducial.copy()
    print(f"\n    POSITIVE PERTURBATION:")
    print(f"      xe_fiducial[{bin_index}] = {xe_fiducial[bin_index]:.4f}")
    xe_plus[bin_index] += delta_xe
    print(f"      xe_plus[{bin_index}] = {xe_plus[bin_index]:.4f} (added δxe={delta_xe:.4f})")

    
    z_plus, xe_class_plus = generate_class_arrays(xe_plus)
    print(f"\n      Generated CLASS arrays (length={len(z_plus)}):")
    print(f"        First 3 z: {z_plus[:3]}")
    print(f"        First 3 xe: {xe_class_plus[:3]}")
    print(f"        Around perturbed bin (indices {bin_index} to {bin_index+5}):")
    for idx in range(max(0, bin_index), min(bin_index+5, len(xe_plus))):
        marker = " <-- PERTURBED" if idx == bin_index else ""
        print(f"          z[{idx}]={z_centers[idx]:.3f}, xe[{idx}]={xe_plus[idx]:.4f}{marker}")
    print(f"        Last 3 z: {z_plus[-3:]}")
    print(f"        Last 3 xe: {xe_class_plus[-3:]}")
     

    xe_minus = xe_fiducial.copy()
    print(f"\n    NEGATIVE PERTURBATION:")
    xe_minus[bin_index] -= delta_xe
    print(f"      xe_minus[{bin_index}] = {xe_minus[bin_index]:.4f} (subtracted δxe={delta_xe:.4f})")
    
    z_minus, xe_class_minus = generate_class_arrays(xe_minus)
    
    # Run CLASS per +δxe
    M_plus = Class()
    params_plus = params.copy()
    params_plus['reio_inter_xe'] = ','.join([f'{xe:.4g}' for xe in xe_class_plus])
    M_plus.set(params_plus)
    M_plus.compute()
    cl_plus = M_plus.raw_cl(50)
    M_plus.struct_cleanup()
    
    M_minus = Class()
    params_minus = params.copy()
    params_minus['reio_inter_xe'] = ','.join([f'{xe:.4g}' for xe in xe_class_minus])
    M_minus.set(params_minus)
    M_minus.compute()
    cl_minus = M_minus.raw_cl(50)
    M_minus.struct_cleanup()
    
    # Calcola derivata solo per E-mode polarization
    # Rimuovi ℓ=0 e ℓ=1 (monopolo e dipolo) che sono sempre 0
    deriv = {}
    deriv['ell'] = cl_plus['ell'][2:]  # Salva i multipoli da ℓ=2 in poi
    
    with np.errstate(divide='ignore', invalid='ignore'):
        ln_cl_plus = np.log(cl_plus['ee'][2:])   # Solo da ℓ=2
        ln_cl_minus = np.log(cl_minus['ee'][2:]) # Solo da ℓ=2
        deriv['ee'] = (ln_cl_plus - ln_cl_minus) / (2 * delta_xe)
        deriv['ee'] = np.where(np.isfinite(deriv['ee']), deriv['ee'], 0.0)
    
    return deriv



# ==========================================
# COMPUTE DERIVATIVES FOR ALL BINS
# ==========================================
print("\n" + "="*60)
print("COMPUTING DERIVATIVES FOR ALL BINS")
print("="*60)

derivatives = []  # Lista per salvare tutte le derivate

for k, i in enumerate(free_bin_indices):
    print(f"\n[{k+1}/{N_FREE}] Processing bin {i} at z={z_centers[i]:.1f}...")
    deriv = compute_derivative_for_bin(i, delta_xe=delta_xe)
    derivatives.append(deriv)

print("\n✓ All derivatives computed!")
print(f"Total free bins processed: {len(derivatives)} (skipped first and last bin)")

# Estrai solo le derivate EE per costruire la matrice
derivatives_matrix = np.array([d['ee'] for d in derivatives])
ell_values = derivatives[0]['ell']  # I multipoli sono uguali per tutti i bins

print(f"\nStruttura dei dati:")
print(f"  derivatives_matrix.shape = {derivatives_matrix.shape}")
print(f"    -> {derivatives_matrix.shape[0]} free bins × {derivatives_matrix.shape[1]} multipoli")
print(f"  Range multipoli: ℓ = {int(ell_values[0])} to {int(ell_values[-1])}")






# ==========================================
# Fisher Matrix SCRIPT
# ==========================================
print(f"\nStep 2: Building Fisher matrix...")


Fij = np.zeros((N_FREE, N_FREE))

for i in range(N_FREE):
    for j in range(i, N_FREE):  # Solo triangolo superiore (simmetria)
        fisher_element = 0.0
        
        # ell_idx parte da 0 che corrisponde a l=2
        for ell_idx in range(len(ell_values)):
            ell = ell_values[ell_idx]
            # Fisher matrix element: (ℓ + 1/2) * dCl_i * dCl_j
            fisher_element += (ell + 0.5) * derivatives_matrix[i, ell_idx] * derivatives_matrix[j, ell_idx]
        
        Fij[i, j] = fisher_element
        Fij[j, i] = fisher_element  # Simmetria
    
    if (i + 1) % 10 == 0:
        print(f"    Progress: {i+1}/{N_FREE} rows")


print(f"\n  DEBUG Fisher matrix:")
print(f"    Shape: {Fij.shape}")
print(f"    Min element: {np.min(Fij):.3e}")
print(f"    Max element: {np.max(Fij):.3e}")
print(f"    Diagonal min/max: {np.min(np.diag(Fij)):.3e} / {np.max(np.diag(Fij)):.3e}")
print(f"    Condition number: {np.linalg.cond(Fij):.3e}")
print(f"    Is symmetric: {np.allclose(Fij, Fij.T)}")
print(f"    Any NaN: {np.any(np.isnan(Fij))}")
print(f"    Any Inf: {np.any(np.isinf(Fij))}")

# DEBUG: Verifica range multipoli usati
print(f"\n  DEBUG Multipoles:")
print(f"    ell_values range: {int(ell_values[0])} to {int(ell_values[-1])}")
print(f"    Number of multipoles: {len(ell_values)}")
print(f"    derivatives_matrix shape: {derivatives_matrix.shape}")
print(f"    Verifica: derivatives_matrix[0, 0] usa ℓ={int(ell_values[0])}")
print(f"    Verifica: derivatives_matrix[0, -1] usa ℓ={int(ell_values[-1])}")



# ==========================================
# PCA: Eigendecomposition
# ==========================================

print("\n" + "="*60)
print("PCA EIGENDECOMPOSITION")
print("="*60)

# Calcola autovalori e autovettori
eigenvalues, eigenvectors = np.linalg.eigh(Fij)

# DEBUG: controlla autovalori
print(f"\n  DEBUG Eigenvalues:")
print(f"    Min eigenvalue: {np.min(eigenvalues):.3e}")
print(f"    Max eigenvalue: {np.max(eigenvalues):.3e}")
print(f"    Num negative: {np.sum(eigenvalues < 0)}")
print(f"    Num near-zero (<1e-10): {np.sum(np.abs(eigenvalues) < 1e-10)}")

# Ordina in ordine decrescente (modo più vincolato = eigenvalue più grande)
idx = np.argsort(eigenvalues)[::-1]
eigenvalues_sorted = eigenvalues[idx]
eigenvectors_sorted = eigenvectors[:, idx]

# Normalizzazione del paper (Mortonson & Hu 2008), eq. (3)-(4):
# sum_i S_mu(z_i) * S_nu(z_i) = (N_z+1) * delta_mu_nu   [ortogonalità discreta eq.3]
# sum_mu S_mu(z_i) * S_mu(z_j) = (N_z+1) * delta_ij     [completezza eq.4]
# np.linalg.eigh dà sum(e^2)=1 → moltiplico per sqrt(N_z+1)
# eigenvectors_sorted = eigenvectors_sorted * np.sqrt(Nz + 1)


for k in range(eigenvectors_sorted.shape[1]):
    if eigenvectors_sorted[-1, k] < 0:
        eigenvectors_sorted[:, k] *= -1


# # Salva risultati
# np.save('fisher_matrix.npy', Fij)
# np.save('eigenvalues.npy', eigenvalues_sorted)
# np.save('eigenvectors.npy', eigenvectors_sorted)
# print(f"\n✓ Salvati: fisher_matrix.npy, eigenvalues.npy, eigenvectors.npy")

# ==========================================
# PLOT: Autofunzioni (primi 5 modi PCA)
# ==========================================

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

XE_FIDUCIAL = 0.15  # Valore fiduciale di xe(z)
colors = ['blue', 'orange', 'green', 'red', 'purple']  # Colori per ciascun modo

for i in range(5):
    ax = axes[i]
    
    # Autofunzione i-esima normalizzata: S_i(z) con int S_i^2 dz = 1
    Si = eigenvectors_sorted[:, i]
    
    # xe(z) = xe_fid + S_i(z)  (ampiezza unitaria, come nel paper)
    xe_model = XE_FIDUCIAL + Si
    
    # Le autofunzioni sono definite solo sui bin liberi: non aggiungere bordi fissi
    # per evitare discontinuità artificiali nel plot
    z_plot  = z_centers_free
    xe_plot = xe_model
    
    # Linea tratteggiata = storia completa xe_fid(z) da CLASS (come nel paper)
    ax.plot(z_thermo, xe_thermo, 'k--', linewidth=1.5, alpha=0.7, label='xe_fid(z)')
    
    # Plot autofunzione perturbata
    ax.plot(z_plot, xe_plot, 'o-', color=colors[i], linewidth=2, markersize=4, label=f'xe_fid + S_{i}')
    
    ax.set_xlabel('Redshift z', fontsize=11)
    ax.set_ylabel(r'$x_e(z)$', fontsize=11)
    ax.set_xlim([0, Z_MAX + 1])
    ax.set_ylim([-0.15, 1.25])   # stessa scala del paper
    ax.legend(fontsize=9)
    
    # Constraint su questo modo
    sigma_i = 1.0 / np.sqrt(eigenvalues_sorted[i]) if eigenvalues_sorted[i] > 0 else np.inf
    ax.set_title(f'Modo PCA {i}: σ(a_{i}) = {sigma_i:.5f}', fontsize=12)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
# plt.savefig('/Users/valentinagenesini/Desktop/pca_eigenfunctions.jpg', dpi=150, bbox_inches='tight')
# print(f"✓ Plot salvato: pca_eigenfunctions.pdf")
plt.show()

# ==========================================
# LIKELIHOOD vs TAU