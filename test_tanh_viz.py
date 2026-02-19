
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Assicuriamoci di poter importare il modulo Reiolike
current_dir = os.getcwd()
if current_dir not in sys.path:
    sys.path.append(current_dir)

try:
    from Reiolike.reio_models import tanh_model
    print("Modulo 'reio_models' importato con successo.")
except ImportError as e:
    print(f"ERRORE: Impossibile importare 'reio_models'. Assicurati di eseguire lo script dalla root del progetto.\nDettagli: {e}")
    sys.exit(1)

def run_test():
    print("\n=== TEST VISUALIZZAZIONE MODELLO TANH ===")

    # --- 1. CONFIGURAZIONE ---
    # Definiamo l'intervallo di redshift
    z_min = 0.0
    z_max = 20.0
    num_points = 100
    
    # Creiamo l'array di redshift
    z_values = np.linspace(z_min, z_max, num_points)

    # Definiamo LISTE di parametri da testare
    # Esempio: Variamo z_re mantenendo delta_z fisso, e viceversa
    
    scenarios = [
        {"z_re": 6.0, "delta_z": 1.0, "label": "Early End"},
        {"z_re": 7.5, "delta_z": 1.0, "label": "Fiducial"},
        {"z_re": 9.0, "delta_z": 1.0, "label": "Late start"},
        {"z_re": 7.5, "delta_z": 0.5, "label": "Sharp transition"},
        {"z_re": 7.5, "delta_z": 2.5, "label": "Smooth transition"},
    ]
    
    print(f"Testiamo {len(scenarios)} scenari diversi su intervallo z=[{z_min}, {z_max}].")

    try:
        plt.figure(figsize=(10, 6))
        
        # Linee di riferimento (fisse)
        plt.axhline(y=1.08, color='green', linestyle=':', alpha=0.4, label='Max Ionization (H+He)')
        plt.axhline(y=0.0, color='black', linewidth=0.5)

        # CICLO SUI DIVERSI PARAMETRI
        for i, sc in enumerate(scenarios):
            z_mid = sc["z_re"]
            dz = sc["delta_z"]
            lbl = sc["label"]
            
            # Calcolo xe(z)
            xe_values = tanh_model(z_values, z_re=z_mid, delta_z=dz)
            
            # Plot
            plt.plot(z_values, xe_values, linewidth=2, alpha=0.8,
                     label=f'{lbl}: $z_{{re}}={z_mid}, \Delta z={dz}$')
            
            print(f"Plotting scenario {i+1}: {lbl} (z_re={z_mid}, dz={dz})")

        # Configurazione grafico
        plt.xlabel('Redshift (z)', fontsize=12)
        plt.ylabel('Ionization Fraction $x_e(z)$', fontsize=12)
        plt.title('Confronto Scenari di Reionizzazione (Tanh Model)', fontsize=14)
        plt.legend(loc='upper right', fontsize=9)
        plt.grid(True, alpha=0.3)
        plt.xlim(z_min, z_max)
        plt.ylim(-0.1, 1.2)
        
        output_filename = "tanh_comparison_plot.png"
        plt.savefig(output_filename, dpi=150)
        print(f"\nGrafico salvato correttamente: {output_filename}")
        
    except Exception as plotting_error:
        print(f"Errore durante il plot: {plotting_error}")

if __name__ == "__main__":
    run_test()
