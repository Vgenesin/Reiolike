import os
import sys
import traceback

import numpy as np
import yaml

# Assicuriamoci che python trovi i moduli
if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())

# Prova import robusto per ReioLike
try:
    from Reiolike import ReioLike

    print("✓ Successfully imported ReioLike")
except ImportError as e:
    print(f"CRITICAL ERROR: Could not import ReioLike: {e}")
    sys.exit(1)

# Importa il modulo reio_models (contiene i modelli teorici come tanh, erf, ecc.)
try:
    from Reiolike import reio_models

    print("✓ Successfully imported reio_models")
except ImportError as e:
    print(f"WARNING: Could not import reio_models: {e}")
    print("  Proceeding anyway, but theory models may not be available.")
    reio_models = None
# ==============================================================================
# 0. CONFIGURAZIONE DEI DUE FILE YAML
# ==============================================================================
REIO_MODEL_YAML = "reio_model_params.yaml"  # Per la teoria (z_re, delta_z)
CORECON_CONFIG_YAML = "corecon_config.yaml"  # Per la likelihood (dati astrofisici)


# Crea i file YAML se non esistono per il test
def create_test_yamls():
    if not os.path.exists(REIO_MODEL_YAML):
        print(f"-> Creating mock {REIO_MODEL_YAML}...")
        with open(REIO_MODEL_YAML, "w") as f:
            yaml.dump(
                {
                    "reio_model": {
                        "theory_model": "tanh_model",  # Specifica quale modello usare
                        "z_min": 0.0,
                        "z_max": 30.0,
                        "z_points": 80,
                        "params": {"z_re": 8.0, "delta_z": 0.8},
                    }
                },
                f,
            )

    if not os.path.exists(CORECON_CONFIG_YAML):
        print(f"-> Creating mock {CORECON_CONFIG_YAML}...")
        with open(CORECON_CONFIG_YAML, "w") as f:
            yaml.dump({"HII_fraction": ["Fan et al. 2006", "Tilvi et al. 2014"]}, f)


# ==============================================================================
# 1. CLASSE MOCK PER IL PROVIDER (SIMULA COBAYA)
# ==============================================================================
class MockProvider:
    def __init__(self):
        self.results = {}

    def get_result(self, key):
        if key in self.results:
            return self.results[key]
        else:
            raise KeyError(f"MockProvider: Key '{key}' not found in results.")

    def set_result(self, key, value):
        self.results[key] = value


# ==============================================================================
# 2. FUNZIONE PER CARICARE IL MODELLO DA reio_models
# ==============================================================================


def load_xe_model_from_reio_models(model_name):
    """
    Carica il modello di ionizzazione dal modulo reio_models.py.

    Args:
        model_name (str): Nome della funzione modello in reio_models (es. 'tanh', 'erf')

    Returns:
        callable: Funzione che calcola xe(z) dato (z, z_re, delta_z)

    Raises:
        ValueError: Se il modello non esiste in reio_models
    """
    if reio_models is None:
        raise RuntimeError(
            "reio_models module not available. Cannot load theory models."
        )

    if hasattr(reio_models, model_name):
        model_func = getattr(reio_models, model_name)
        print(f"   ✓ Successfully loaded '{model_name}' from reio_models")
        return model_func
    else:
        available = [attr for attr in dir(reio_models) if not attr.startswith("_")]
        raise ValueError(
            f"Model '{model_name}' not found in reio_models. "
            f"Available models: {available}"
        )


# ==============================================================================
# 3. MAIN TEST SCRIPT
# ==============================================================================


def run_test():
    print("\n" + "=" * 70)
    print("TESTING REIO_LIKE WITH THEORY MODELS FROM reio_models.py")
    print("=" * 70)

    # Setup iniziale file dummy
    create_test_yamls()

    # --------------------------------------------------------------------------
    # STEP A: LEGGI I PARAMETRI DEL MODELLO DAL YAML
    # --------------------------------------------------------------------------
    print(f"\n[A] Reading reionization model parameters from: {REIO_MODEL_YAML}")

    with open(REIO_MODEL_YAML, "r") as f:
        model_config = yaml.safe_load(f)

    # Estrazione parametri sicuro con default
    reio_conf = model_config.get("reio_model", {})

    # Leggi quale modello usare dal YAML
    model_type = reio_conf.get("theory_model", "tanh_model")
    print(f"    Theory model specified in YAML: '{model_type}'")

    params = reio_conf.get("params", {"z_re": 8.0, "delta_z": 0.8})

    z_min = float(reio_conf.get("z_min", 0.0))
    z_max = float(reio_conf.get("z_max", 30.0))
    z_points = int(reio_conf.get("z_points", 100))

    z_re = float(params.get("z_re"))
    delta_z = float(params.get("delta_z"))

    print(f"    Loaded params: z_re={z_re}, delta_z={delta_z}")
    print(f"    Grid settings: z=[{z_min}, {z_max}] with {z_points} points")

    # --------------------------------------------------------------------------
    # STEP B: CARICA IL MODELLO DA reio_models E CALCOLA xe(z)
    # --------------------------------------------------------------------------
    print(f"\n[B] Loading theory model '{model_type}' from reio_models...")

    try:
        # Carica la funzione modello da reio_models
        xe_model_func = load_xe_model_from_reio_models(model_type)
    except (ValueError, RuntimeError) as e:
        print(f"    ERROR: {e}")
        return

    # 1. Crea griglia redshift
    z_array = np.linspace(z_min, z_max, z_points)

    # 2. Calcola xe(z) teorico usando il modello caricato da reio_models
    print(f"    Computing xe(z) via {model_type}(z, z_re={z_re}, delta_z={delta_z})...")
    try:
        xe_array = xe_model_func(z_array, z_re, delta_z)
    except TypeError as e:
        print(f"    ERROR calling {model_type}: {e}")
        print(
            f"    Make sure {model_type}(z, z_re, delta_z) has the correct signature"
        )
        return

    print(f"    Generated xe(z) array with shape {xe_array.shape}")
    print(f"    xe(z=0) = {xe_array[0]:.4f}")
    print(f"    xe(z={z_re}) = {np.interp(z_re, z_array, xe_array):.4f}")

    # --------------------------------------------------------------------------
    # STEP C: CONFIGURA E INIZIALIZZA IL LIKELIHOOD (ReioLike)
    # --------------------------------------------------------------------------
    print("\n[C] Initializing ReioLike likelihood...")

    try:
        # Istanzia
        like = ReioLike()

        # Override del file di configurazione con quello locale di test
        like.corecon_config_file = CORECON_CONFIG_YAML
        print(f"    Using config file: {like.corecon_config_file}")

        # Inizializza (Carica i dati Corecon)
        like.initialize()

    except Exception as e:
        print(f"    ERROR during ReioLike initialization: {e}")
        print("    -> Is 'corecon' installed? Is the YAML valid?")
        traceback.print_exc()
        return

    # --------------------------------------------------------------------------
    # STEP D: CALCOLA IL LOG-LIKELIHOOD
    # --------------------------------------------------------------------------
    print("\n[D] Calculating Log-Likelihood...")

    # Configura il Provider finto con i dati della Teoria (Step B)
    provider = MockProvider()
    provider.set_result("reio_history_z", z_array)
    provider.set_result("reio_history_xe", xe_array)

    # Collega il provider alla likelihood
    like.provider = provider

    try:
        # Esegui il calcolo del likelihood
        print("    Invoking like.logp()... checking data compatibility:")
        log_prob = like.logp()

        print("\n" + "-" * 70)
        print(f"RESULT: Log-Likelihood = {log_prob}")
        print("-" * 70)

        # Debug: Check how many analyses and data points were used
        if hasattr(like, "analyses"):
            print("\nDebug Info:")
            print(f"  Number of analyses: {len(like.analyses)}")
            for i, analysis in enumerate(like.analyses):
                n_points = len(analysis.get("values", []))
                print(f"  Analysis {i} ({analysis['type']}): {n_points} data points")

        if log_prob == 0.0:
            print(
                " WARNING: Log-Likelihood is exactly 0.0."
            )
        elif log_prob == -np.inf:
            print("  WARNING: Log-Likelihood is -inf.")
            print("   Possible reasons:")
            print("   1. Model values are NaN or Inf")
            print("   2. Corecon data has no valid points (errors <= 0 or NaN)")
            print("   3. Model interpolation failed (z_array range mismatch)")
        else:
            print(f"✓ Likelihood computed successfully with log_prob = {log_prob:.6f}")

    except Exception as e:
        print(f"ERROR during logp calculation: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    run_test()
