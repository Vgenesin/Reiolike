
import numpy as np
import matplotlib.pyplot as plt
import yaml
import sys
import os

# Assicuriamoci di poter importare il modulo Reiolike
current_dir = os.getcwd()
if current_dir not in sys.path:
    sys.path.append(current_dir)

try:
    from Reiolike.reio_models import tanh_model
except ImportError:
    print("Warning: Reiolike module not found in current path. Ensure your PYTHONPATH is correct.")
    # Fallback o exit, per ora exit se manca il modello
    sys.exit(1)

def main(yaml_file="test_run.yaml"):
    print(f"Reading configuration from {yaml_file}...")
    
    if not os.path.exists(yaml_file):
        print(f"Error: {yaml_file} not found.")
        return

    with open(yaml_file, 'r') as f:
        config = yaml.safe_load(f)

    # 1. Parsing Parameters
    params = config.get('params', {})
    
    # Helper to clean Cobaya-style parameters
    def get_param(name, default=None):
        val = params.get(name)
        if val is None:
            return default
        if isinstance(val, dict):
            # Cobaya param definition can be {value: X, prior: ...}
            if 'value' in val:
                return float(val['value'])
            # If no 'value' key, it might be sampled (no fixed value)
            # For this script we need a concrete value. Use 'ref' if available?
            if 'ref' in val:
                val_ref = val['ref']
                if isinstance(val_ref, (int, float)):
                    return float(val_ref)
                # If ref is a list/dict (prior range), take center?
                if isinstance(val_ref, list):
                     return np.mean(val_ref)
                if isinstance(val_ref, dict) and 'min' in val_ref and 'max' in val_ref:
                     return (val_ref['min'] + val_ref['max']) / 2.0
            return default # Cannot determine value
        return float(val)

    # Allow custom parameter names if defined in yaml, otherwise defaults
    z_re = get_param('reio_z_re')
    delta_z = get_param('reio_delta_z')
    
    # If not found in params, check if they are nested in theory logic (unlikely for cobaya input but possible)
    # Check theory configuration for grid settings
    theory_conf = config.get('theory', {})
    
    # Defaults
    z_min = 0.0
    z_max = 30.0
    z_points = 500
    
    # Try to find customization in 'theory' block
    for key, val in theory_conf.items():
        if 'ReioTheory' in key:
            z_min = float(val.get('z_min', z_min))
            z_max = float(val.get('z_max', z_max))
            z_points = int(val.get('z_points', z_points))
            # Also check if fixed params are defined here instead of top-level params (less standard)
            if z_re is None: z_re = val.get('reio_z_re')
            if delta_z is None: delta_z = val.get('reio_delta_z')

    # Fallback defaults if still None
    if z_re is None:
        print("Warning: 'reio_z_re' not found in yaml. Using default 7.0")
        z_re = 7.0
    if delta_z is None:
        print("Warning: 'reio_delta_z' not found in yaml. Using default 0.5")
        delta_z = 0.5

    print(f"Parameters: z_re={z_re}, delta_z={delta_z}")
    print(f"Grid: z=[{z_min}, {z_max}], points={z_points}")

    # 2. Computation
    z_array = np.linspace(z_min, z_max, z_points)
    xe = tanh_model(z_array, z_re=z_re, delta_z=delta_z)

    # 3. Output/Plot
    output_png = "reio_history_test_run.png"
    
    plt.figure(figsize=(8, 5))
    plt.plot(z_array, xe, label=f'Model from {yaml_file}\nz_re={z_re}, dz={delta_z}')
    plt.xlabel('Redshift z')
    plt.ylabel('Ionization Fraction $x_e$')
    plt.title('Reionization History')
    plt.axvline(x=z_re, linestyle='--', color='gray', alpha=0.5, label='z_re')
    plt.axhline(y=1.0, linestyle=':', color='gray', alpha=0.5)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(output_png)
    print(f"Plot saved to {output_png}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        main()
