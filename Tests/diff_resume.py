"""Why does a `cobaya-run -r` refuse to resume ("Old and new run information
not compatible")?

Rebuilds the run information the way cobaya-run does from the YAML in the
repository (all likelihood and theory defaults filled in) and compares it,
key by key, with the `updated.yaml` saved in the chain folder.  Prints only
the keys that differ.

Usage (from the Reiolike folder, with the environment used for the runs):
    python Tests/diff_resume.py run_ilic/power_law_2pow_ilic.yaml \
        -p /leonardo_work/INF26_indark/vgenesin/Reiolike/packages
"""
import argparse
import os
import sys

sys.path.insert(0, os.getcwd())

import numpy as np
from cobaya.input import load_input, update_info
from cobaya.yaml import yaml_load_file


def _confronta(vecchio, nuovo, percorso=""):
    if isinstance(vecchio, dict) and isinstance(nuovo, dict):
        for k in sorted(set(vecchio) | set(nuovo), key=str):
            p = f"{percorso}.{k}" if percorso else str(k)
            if k not in nuovo:
                print(f"SOLO NELLA CATENA   {p}: {vecchio[k]!r}")
            elif k not in vecchio:
                print(f"SOLO NEL NUOVO      {p}: {nuovo[k]!r}")
            else:
                _confronta(vecchio[k], nuovo[k], p)
        return
    if isinstance(vecchio, (list, tuple)) and isinstance(nuovo, (list, tuple)):
        if len(vecchio) == len(nuovo):
            for i, (a, b) in enumerate(zip(vecchio, nuovo)):
                _confronta(a, b, f"{percorso}[{i}]")
            return
    try:
        uguali = vecchio == nuovo
        if isinstance(uguali, np.ndarray):
            uguali = uguali.all()
    except Exception:
        uguali = str(vecchio) == str(nuovo)
    if not uguali and str(vecchio) != str(nuovo):
        print(f"DIVERSO             {percorso}\n    catena: {vecchio!r}\n    nuovo:  {nuovo!r}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("yaml")
    ap.add_argument("-p", "--packages-path", default=None)
    args = ap.parse_args()

    info = load_input(args.yaml)
    if args.packages_path:
        info["packages_path"] = args.packages_path
    salvato = info["output"] + ".updated.yaml"
    if not os.path.exists(salvato):
        raise SystemExit(f"non trovo {salvato}")
    vecchio = yaml_load_file(salvato)
    nuovo = update_info(info)

    for blocco in ("theory", "likelihood", "params", "prior", "sampler"):
        print(f"===== {blocco}")
        _confronta(vecchio.get(blocco), nuovo.get(blocco), blocco)


if __name__ == "__main__":
    main()
