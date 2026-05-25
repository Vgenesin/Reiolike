# from cobaya.likelihood import DataSetLikelihood
import re
import numpy as np
import yaml
from scipy.interpolate import interp1d
from scipy.integrate import quad
import corecon as con
from cobaya.likelihood import Likelihood

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

    def initialize(self):

        """
        Initializes the Corecon analyses specified in the configuration file.
        """
        print(f"ReioLike initialized. Reading corecon config from: {self.corecon_config_file}")

        # Read the corecon configuration file to determine which analyses to run
        with open(self.corecon_config_file, "r") as f:
            self.config = yaml.safe_load(f)
            print(f"Loaded corecon config: {self.config}")

        self.analyses = []
        self._seen_datasets = set()  # for duplicate detection
        self._coverage_warned = set()  # for one-time extrapolation warnings

        KNOWN_SECTIONS = {
            "HII_fraction", "upper_limit_HII_fraction",
            "lower_limit_HII_fraction", "mixed_HII_fraction",
        }

        for observable_type, datasets in self.config.items():
            if observable_type not in KNOWN_SECTIONS:
                raise RuntimeError(
                    f"Unknown YAML section '{observable_type}'. "
                    f"Valid sections are: {sorted(KNOWN_SECTIONS)}. "
                    f"Check for typos in corecon_config.yaml."
                )
            print(f"Found observable type: {observable_type}")

            if not datasets:
                print(f"  - No datasets listed for '{observable_type}', skipping.")
                continue

            corecon_key = observable_type
            if observable_type in ("HII_fraction", "upper_limit_HII_fraction",
                                     "lower_limit_HII_fraction", "mixed_HII_fraction"):
                corecon_key = "x_HII"

            try:
                corecon_data_dict = con.get(corecon_key)
            except Exception as e:
                print(f"  - ERROR loading corecon data for '{corecon_key}': {e}")
                raise RuntimeError(f"CRITICAL ERROR loading corecon data for '{corecon_key}': {e}")

            if isinstance(datasets, list):
                for dataset_spec in datasets:
                    # Support "Dataset Name [LABEL]" notation for multi-realization or multi-type datasets
                    # e.g. "Durovcikova et al. 2024 [ATON]" or "Davies et al. 2026 [threshold]"
                    corecon_name, label = self._parse_dataset_spec(dataset_spec)
                    print(f"  - Initializing analysis for dataset: {dataset_spec}")

                    if corecon_name in corecon_data_dict:
                        data_obj = corecon_data_dict[corecon_name]

                        # Some datasets are intrinsically multi-model and must be disambiguated in YAML.
                        # Example: Totani et al. 2014 has IGM_model choices like [fixed_dz], [variable_dz].
                        self._ensure_required_label(corecon_name, label, data_obj, dataset_spec)

                        if label is not None:
                            # Multi-realization or multi-type dataset: extract only the rows matching label
                            label_str = ", ".join(label)
                            print(f"    - Multi-label dataset, extracting label: '[{label_str}]'")
                            z_arr, val_arr, err_up_arr, err_z_left, err_z_right, lower_lim_arr, upper_lim_arr = \
                                self._extract_realization(data_obj, label)
                        else:
                            # Standard dataset: axes is a plain array of redshifts
                            z_arr       = np.array(data_obj.axes,   dtype=float)
                            val_arr     = np.array(data_obj.values, dtype=float)
                            err_up_arr  = np.array(data_obj.err_up, dtype=float)
                            err_z_left  = self._get_error_array(data_obj, "err_left")
                            err_z_right = self._get_error_array(data_obj, "err_right")
                            lower_lim_arr = np.array(data_obj.lower_lim, dtype=bool) \
                                if hasattr(data_obj, "lower_lim") and data_obj.lower_lim is not None \
                                else np.zeros(len(z_arr), dtype=bool)
                            upper_lim_arr = np.array(data_obj.upper_lim, dtype=bool) \
                                if hasattr(data_obj, "upper_lim") and data_obj.upper_lim is not None \
                                else np.zeros(len(z_arr), dtype=bool)

                        if len(z_arr) == 0:
                            raise RuntimeError(
                                f"Dataset '{dataset_spec}' in corecon '{corecon_key}' has no data points (axes is empty)."
                            )

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

                        # Validate that lower_lim/upper_lim flags in corecon match the declared YAML section
                        self._validate_corecon_flags(
                            dataset_spec, observable_type, lower_lim_arr, upper_lim_arr
                        )

                        if observable_type == "HII_fraction":
                            n_valid_err = np.sum(np.isfinite(err_up_arr) & (err_up_arr > 0))
                            if n_valid_err == 0:
                                raise RuntimeError(
                                    f"Dataset '{dataset_spec}' in corecon '{corecon_key}' has no finite positive "
                                    f"error bars (err_up). Cannot use it in a Gaussian likelihood."
                                )
                            print(f"    - Dataset '{dataset_spec}': DETECTION "
                                  f"({len(z_arr)} points, {n_valid} finite values, {n_valid_err} finite positive errors).")
                        elif observable_type == "upper_limit_HII_fraction":
                            print(f"    - Dataset '{dataset_spec}': UPPER LIMIT "
                                  f"({len(z_arr)} points, values interpreted as 95% CL upper bounds).")
                        elif observable_type == "lower_limit_HII_fraction":
                            print(f"    - Dataset '{dataset_spec}': LOWER LIMIT "
                                  f"({len(z_arr)} points, values interpreted as 95% CL lower bounds).")

                        self.analyses.append({
                            "type": observable_type,
                            "name": dataset_spec,
                            "z": z_arr,
                            "values": val_arr,
                            "errors": err_up_arr,
                            "err_z_left": err_z_left,
                            "err_z_right": err_z_right,
                            "lower_lim": lower_lim_arr,
                            "upper_lim": upper_lim_arr,
                        })
                    else:
                        print(f"    - WARNING: Dataset '{corecon_name}' not found in corecon '{corecon_key}'")
                        raise RuntimeError(
                            f"CRITICAL ERROR loading corecon data for '{corecon_key}': "
                            f"Dataset '{corecon_name}' not found"
                        )
            else:
                print(f"  - WARNING: Expected list for {observable_type}, got {type(datasets)}")

    @staticmethod
    def _parse_dataset_spec(dataset_spec):
        """
        Parse a dataset specification string, optionally extracting a label.

        Supported formats:
          "Dataset Name"               -> (corecon_name="Dataset Name", label=None)
          "Dataset Name [LABEL]"       -> (corecon_name="Dataset Name", label="LABEL")

        Examples:
          "Durovcikova et al. 2024 [ATON]"         -> ("Durovcikova et al. 2024", ["ATON"])
          "Davies et al. 2026 [threshold]"           -> ("Davies et al. 2026", ["threshold"])
          "Totani et al. 2014 [fixed_dz]"            -> ("Totani et al. 2014", ["fixed_dz"])
          "Fausey et al. 2024 [1.5, Inoue14]"        -> ("Fausey et al. 2024", ["1.5", "Inoue14"])

        Returns a list of labels (one per extra axis column), or None if no brackets.
        """
        match = re.match(r"^(.+?)\s*\[([^\]]+)\]\s*$", dataset_spec)
        if match:
            labels = [l.strip() for l in match.group(2).split(",")]
            return match.group(1).strip(), labels
        return dataset_spec, None

    @staticmethod
    def _ensure_required_label(corecon_name, label, data_obj, dataset_spec):
        """
        Enforce explicit label selection for datasets that have multiple physical models.

        Currently enforced:
          - Totani et al. 2014: requires explicit IGM model label, e.g. [fixed_dz].
        """
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
            "Mason et al. 2019": {
                "n_labels": 1,
                "description": "confidence level (0.68 or 0.95)",
                "example": "Mason et al. 2019 [0.68]",
            },
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

        Returns: z_arr, val_arr, err_up_arr, err_z_left, err_z_right, lower_lim_arr, upper_lim_arr
        """
        z_list, val_list, err_up_list, err_left_list, err_right_list = [], [], [], [], []
        lower_lim_list, upper_lim_list = [], []

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

        if not z_list:
            label_str = ", ".join(label)
            raise RuntimeError(
                f"Label '[{label_str}]' not found in dataset. "
                f"Available label combinations: {sorted(available)}"
            )

        return (
            np.array(z_list,       dtype=float),
            np.array(val_list,     dtype=float),
            np.array(err_up_list,  dtype=float),
            np.nan_to_num(np.array(err_left_list,  dtype=float), nan=0.0),
            np.nan_to_num(np.array(err_right_list, dtype=float), nan=0.0),
            np.array(lower_lim_list, dtype=bool),
            np.array(upper_lim_list, dtype=bool),
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
            print(
                f"    - WARNING: Dataset '{dataset_spec}' declared as '{declared}' "
                f"but has mixed corecon flags: {flag_str}. "
                f"Only {n_matching}/{n} point(s) match. Check your dataset or split by label."
            )

    def _get_error_array(self, data_obj, attr_name):
        """Helper to safely extract error arrays from corecon objects"""
        try:
            val = getattr(data_obj, attr_name, None)
            if val is not None:
                arr = np.array(val, dtype=float)
                return np.nan_to_num(arr, nan=0.0)
            return None
        except (ValueError, TypeError):
            return None

    def get_requirements(self):
        """
        Request tau_reio to ensure ReioTheory.calculate() runs before logp().
        The reio history is read from the shared module-level variable in reio_theory.
        """
        return {"tau_reio": None}

    def compute_gaussian_loglike(self, analysis_dict, z_model, model_values, integration_width=None):
        """
        Gaussian likelihood comparing model values with observed values and errors from corecon.
        """
        corecon_values = analysis_dict["values"]
        corecon_errors = analysis_dict["errors"]
        z_corecon      = analysis_dict["z"]
        err_left       = analysis_dict["err_z_left"]
        err_right      = analysis_dict["err_z_right"]

        if integration_width is None:
            w_left  = err_left  if err_left  is not None else np.zeros_like(z_corecon)
            w_right = err_right if err_right is not None else np.zeros_like(z_corecon)
            w_left  = np.array(w_left,  dtype=float)
            w_right = np.array(w_right, dtype=float)
            widths_z = w_left + w_right
        else:
            widths_z = np.full_like(z_corecon, integration_width)

        if z_model[0] > z_model[-1]:
            z_model      = z_model[::-1]
            model_values = model_values[::-1]

        self._warn_extrapolation(analysis_dict["name"], z_corecon, z_model)
        f_interp = interp1d(z_model, model_values, kind="cubic", fill_value="extrapolate")

        model_at_corecon = []
        for i, z in enumerate(z_corecon):
            w = widths_z[i]
            used_integration = False
            if w > 0:
                if integration_width is None:
                    left_val  = w_left[i]  if w_left  is not None else 0.0
                    right_val = w_right[i] if w_right is not None else 0.0
                    z_min = z - left_val
                    z_max = z + right_val
                else:
                    z_min = z - w / 2.0
                    z_max = z + w / 2.0
                if z_max > z_min:
                    try:
                        val_integral, _ = quad(f_interp, z_min, z_max)
                        model_at_corecon.append(val_integral / (z_max - z_min))
                        used_integration = True
                    except Exception as e:
                        print(f"Integration failed at z={z}: {e}")
            if not used_integration:
                val = f_interp(z)
                print(f"Pointwise interpolation at z={z}: {val}")
                if np.ndim(val) > 0:
                    val = val.item()
                model_at_corecon.append(val)

        model_at_corecon = np.array(model_at_corecon)

        mask_valid = (
            np.isfinite(corecon_values) &
            np.isfinite(corecon_errors) &
            np.isfinite(model_at_corecon) &
            (corecon_errors > 0)
        )

        if np.sum(mask_valid) == 0:
            print(f"DEBUG: No valid data points found for Analysis.")
            print(f"  Total points: {len(corecon_values)}")
            print(f"  Valid values: {np.sum(np.isfinite(corecon_values))}")
            print(f"  Valid errors > 0: {np.sum(np.isfinite(corecon_errors) & (corecon_errors > 0))}")
            print(f"  Valid model values: {np.sum(np.isfinite(model_at_corecon))}")
            return -np.inf

        chi_squared = np.sum(
            ((model_at_corecon[mask_valid] - corecon_values[mask_valid]) / corecon_errors[mask_valid]) ** 2
        )
        return -0.5 * chi_squared

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

    def compute_upper_limit_loglike(self, analysis_dict, z_model, model_values):
        """
        Log-likelihood for non-detections quoted as one-sided 95% CL upper limits.

        sigma_i = g_i / 1.645
        logL_i  = -0.5 * (x_th_i / sigma_i)^2   if x_th_i > 0
                =  0                              if x_th_i <= 0
        """
        upper_limits = analysis_dict["values"]
        z_data       = analysis_dict["z"]

        if z_model[0] > z_model[-1]:
            z_model      = z_model[::-1]
            model_values = model_values[::-1]

        self._warn_extrapolation(analysis_dict["name"], z_data, z_model)
        f_interp = interp1d(z_model, model_values, kind="cubic", fill_value="extrapolate")
        x_theory_at_data = np.asarray(f_interp(z_data), dtype=float)

        mask_valid = np.isfinite(upper_limits) & (upper_limits > 0) & np.isfinite(x_theory_at_data)

        if np.sum(mask_valid) == 0:
            print(f"DEBUG: No valid upper-limit points found for analysis '{analysis_dict['name']}'.")
            return -np.inf

        g     = upper_limits[mask_valid]
        x_th  = x_theory_at_data[mask_valid]
        sigma = g / 1.645

        log_l = np.where(x_th > 0, -0.5 * (x_th / sigma) ** 2, 0.0)
        return float(np.sum(log_l))

    def compute_lower_limit_loglike(self, analysis_dict, z_model, model_values):
        """
        Log-likelihood for detections quoted as one-sided 95% CL lower limits.

        sigma = (1 - g) / 1.645
        logL  = -0.5 * ((x_th - 1) / sigma)^2   if x_th < g
              =  0                                if x_th >= g
        """
        lower_limits = analysis_dict["values"]
        z_data       = analysis_dict["z"]

        if z_model[0] > z_model[-1]:
            z_model      = z_model[::-1]
            model_values = model_values[::-1]

        self._warn_extrapolation(analysis_dict["name"], z_data, z_model)
        f_interp = interp1d(z_model, model_values, kind="cubic", fill_value="extrapolate")
        x_theory_at_data = np.asarray(f_interp(z_data), dtype=float)

        mask_valid = (
            np.isfinite(lower_limits) &
            (lower_limits > 0) &
            (lower_limits < 1) &
            np.isfinite(x_theory_at_data)
        )

        if np.sum(mask_valid) == 0:
            print(f"DEBUG: No valid lower-limit points found for analysis '{analysis_dict['name']}'.")
            return -np.inf

        g     = lower_limits[mask_valid]
        x_th  = x_theory_at_data[mask_valid]
        sigma = (1.0 - g) / 1.645

        log_l = np.where(x_th < g, -0.5 * ((x_th - 1.0) / sigma) ** 2, 0.0)
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
        if z_model[0] > z_model[-1]:
            z_model      = z_model[::-1]
            model_values = model_values[::-1]

        self._warn_extrapolation(analysis_dict["name"], analysis_dict["z"], z_model)
        f_interp = interp1d(z_model, model_values, kind="cubic", fill_value="extrapolate")
        x_theory_at_data = np.asarray(f_interp(analysis_dict["z"]), dtype=float)
        errors      = analysis_dict["errors"]
        lower_lim   = analysis_dict["lower_lim"]
        upper_lim   = analysis_dict["upper_lim"]

        log_total = 0.0
        for i in range(len(analysis_dict["values"])):
            x_th = x_theory_at_data[i]
            g    = analysis_dict["values"][i]
            if not np.isfinite(x_th) or not np.isfinite(g):
                continue

            if lower_lim[i]:
                # Lower limit: penalise if x_th < g
                if g <= 0 or g >= 1:
                    continue
                sigma = (1.0 - g) / 1.645
                log_total += float(np.where(x_th < g, -0.5 * ((x_th - 1.0) / sigma) ** 2, 0.0))
            elif upper_lim[i]:
                # Upper limit: penalise if x_th > 0
                if g <= 0:
                    continue
                sigma = g / 1.645
                log_total += float(np.where(x_th > 0, -0.5 * (x_th / sigma) ** 2, 0.0))
            else:
                # Detection: Gaussian
                err = errors[i]
                if not np.isfinite(err) or err <= 0:
                    continue
                log_total += -0.5 * ((x_th - g) / err) ** 2

        return log_total

    def logp(self, **params_values):
        """
        Calculate the log-likelihood.
        """
        from Reiolike.reio_theory import _shared_reio_history
        z  = _shared_reio_history["z"]
        xe = _shared_reio_history["xe"]
        if z is None or xe is None:
            print("[ReioLike] ERROR: reio_history not available - ReioTheory.calculate() may not have run.")
            return -np.inf

        Y_He = 0.24
        fHe  = Y_He / (1 - Y_He) * (1.008 / 4.003)  # approximately 0.079
        x_HII = xe / (1 + fHe)

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

                print(f"  [{analysis['type']}] '{analysis['name']}':  logL = {log_L:.6f}")
                if np.isfinite(log_L):
                    log_prob += log_L
                else:
                    return -np.inf

        return log_prob
