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
        with open(self.corecon_config_file, 'r') as f:
            self.config = yaml.safe_load(f)
            print(f"Loaded corecon config: {self.config}") # Debug print to check config loading and seeing what is in the config file.

        self.analyses = [] 
        # Mapping between YAML key (observable) and Corecon module/class (to be defined)
        # For now, we assume that 'HII_fraction' maps to something that handles x_HI
        
        for observable_type, datasets in self.config.items():
            print(f"Found observable type: {observable_type}") #checking which observable types we have in the config file, for example 'HII_fraction'.

            # --- Point 2: skip entirely if the dataset list is empty ---
            # This avoids an unnecessary call to con.get() when the user has left
            # a section in the YAML but without any dataset names (e.g. upper_limit_HII_fraction: []).
            if not datasets:
                print(f"  - No datasets listed for '{observable_type}', skipping.")
                continue

            # Mapping YAML observable types to corecon keys.
            corecon_key = observable_type
            if observable_type in ('HII_fraction', 'upper_limit_HII_fraction', 'lower_limit_HII_fraction'):
                corecon_key = 'x_HII'

            #At this level we want everything we could get from corecon with this key.
            try:
                corecon_data_dict = con.get(corecon_key)
            except Exception as e:
                print(f"  - ERROR loading corecon data for '{corecon_key}': {e}")
                raise RuntimeError(f"CRITICAL ERROR loading corecon data for '{corecon_key}': {e}")

            if isinstance(datasets, list):
                for dataset_spec in datasets:
                    # Support "Dataset Name [REALIZATION]" notation for multi-realization datasets
                    # (e.g. "Durovcikova et al. 2024 [ATON]").
                    corecon_name, realization = self._parse_dataset_spec(dataset_spec)
                    print(f"  - Initializing analysis for dataset: {dataset_spec}")

                    if corecon_name in corecon_data_dict:
                        data_obj = corecon_data_dict[corecon_name]

                        if realization is not None:
                            # Multi-realization dataset: extract only the requested realization
                            print(f"    - Multi-realization dataset, extracting realization: '{realization}'")
                            z_arr, val_arr, err_up_arr, err_z_left, err_z_right = \
                                self._extract_realization(data_obj, realization)
                        else:
                            # Standard dataset: axes is a plain array of redshifts
                            z_arr      = np.array(data_obj.axes,   dtype=float)
                            val_arr    = np.array(data_obj.values, dtype=float)
                            err_up_arr = np.array(data_obj.err_up, dtype=float)
                            err_z_left  = self._get_error_array(data_obj, 'err_left')
                            err_z_right = self._get_error_array(data_obj, 'err_right')

                        # --- Sanity checks ---
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

                        if observable_type == 'HII_fraction':
                            n_valid_err = np.sum(np.isfinite(err_up_arr) & (err_up_arr > 0))
                            if n_valid_err == 0:
                                raise RuntimeError(
                                    f"Dataset '{dataset_spec}' in corecon '{corecon_key}' has no finite positive "
                                    f"error bars (err_up). Cannot use it in a Gaussian likelihood."
                                )
                            print(f"    - Dataset '{dataset_spec}': DETECTION "
                                  f"({len(z_arr)} points, {n_valid} finite values, {n_valid_err} finite positive errors).")
                        elif observable_type == 'upper_limit_HII_fraction':
                            print(f"    - Dataset '{dataset_spec}': UPPER LIMIT "
                                  f"({len(z_arr)} points, values interpreted as 95% CL upper bounds).")
                        elif observable_type == 'lower_limit_HII_fraction':
                            print(f"    - Dataset '{dataset_spec}': LOWER LIMIT "
                                  f"({len(z_arr)} points, values interpreted as 95% CL lower bounds).")

                        self.analyses.append({
                            'type': observable_type,
                            'name': dataset_spec,  # use full spec (with [REALIZATION]) as label
                            'z': z_arr,
                            'values': val_arr,
                            'errors': err_up_arr,
                            'err_z_left': err_z_left,
                            'err_z_right': err_z_right,
                        })
                    else:
                        print(f"    - WARNING: Dataset '{corecon_name}' not found in corecon '{corecon_key}'")
                        raise RuntimeError(f"CRITICAL ERROR loading corecon data for '{corecon_key}': Dataset '{corecon_name}' not found")

            else:
                 print(f"  - WARNING: Expected list for {observable_type}, got {type(datasets)}")
                    
    @staticmethod
    def _parse_dataset_spec(dataset_spec):
        """
        Parse a dataset specification string, optionally extracting a realization.

        Supported formats:
          "Dataset Name"               -> (corecon_name="Dataset Name", realization=None)
          "Dataset Name [REALIZATION]" -> (corecon_name="Dataset Name", realization="REALIZATION")

        Example:
          "Durovcikova et al. 2024 [ATON]" -> ("Durovcikova et al. 2024", "ATON")
        """
        match = re.match(r'^(.+?)\s*\[([^\]]+)\]\s*$', dataset_spec)
        if match:
            return match.group(1).strip(), match.group(2).strip()
        return dataset_spec, None

    @staticmethod
    def _extract_realization(data_obj, realization):
        """
        Extract data for a single realization from a multi-realization corecon dataset.

        In datasets like 'Durovcikova et al. 2024', each entry in data_obj.axes
        is a tuple (redshift, realization_name) instead of a plain redshift.
        This method filters to the requested realization and returns flat arrays.

        Returns: z_arr, val_arr, err_up_arr, err_z_left, err_z_right
        """
        z_list, val_list, err_up_list, err_left_list, err_right_list = [], [], [], [], []

        # Check whether err_left/err_right exist and are indexable
        has_err_left  = hasattr(data_obj, 'err_left')  and data_obj.err_left  is not None
        has_err_right = hasattr(data_obj, 'err_right') and data_obj.err_right is not None

        available = set()
        for i, entry in enumerate(data_obj.axes):
            z, real_name = entry[0], entry[1]
            available.add(real_name)
            if real_name == realization:
                z_list.append(float(z))
                val_list.append(data_obj.values[i])
                err_up_list.append(data_obj.err_up[i])
                if has_err_left:
                    try:
                        el = data_obj.err_left[i]
                        err_left_list.append(el[0] if hasattr(el, '__len__') else float(el))
                    except Exception:
                        err_left_list.append(0.0)
                else:
                    err_left_list.append(0.0)
                if has_err_right:
                    try:
                        er = data_obj.err_right[i]
                        err_right_list.append(er[0] if hasattr(er, '__len__') else float(er))
                    except Exception:
                        err_right_list.append(0.0)
                else:
                    err_right_list.append(0.0)

        if not z_list:
            raise RuntimeError(
                f"Realization '{realization}' not found in dataset. "
                f"Available realizations: {sorted(available)}"
            )

        return (
            np.array(z_list,      dtype=float),
            np.array(val_list,    dtype=float),
            np.array(err_up_list, dtype=float),
            np.nan_to_num(np.array(err_left_list,  dtype=float), nan=0.0),
            np.nan_to_num(np.array(err_right_list, dtype=float), nan=0.0),
        )

    def _get_error_array(self, data_obj, attr_name):
        """Helper to safely extract error arrays from corecon objects"""
        try:
            val = getattr(data_obj, attr_name, None) # Lets assume we have different analyses, and some report a symmetric error (err_up)    
                                                     # while others report asymmetric errors (err_left, err_right). 
                                                     # This helper function tries to extract the error array for the given attribute name,
                                                     # and if it is not available or not valid, it returns None. 
                                                     # This allows us to handle different types of datasets without crashing.
                                                    
                                                     # If the attribute does not exist, we keep it as None.
                                                     # If it is nan, it is placed to 0

            if val is not None:
                arr = np.array(val, dtype=float) # If it is not None, we want to make sure it is a numpy array of floats. 
                                                 # If this fails, we catch the exception and return None.
                return np.nan_to_num(arr, nan=0.0) # Convert NaN values in array to 0.0
            return None
        except (ValueError, TypeError):
            return None

    def get_requirements(self):
        """
        Request tau_reio to ensure ReioTheory.calculate() runs before logp().
        The reio history is read from the shared module-level variable in reio_theory.
        """
        return {'tau_reio': None}

    def compute_gaussian_loglike(self, analysis_dict, z_model, model_values, integration_width=None): #I can call thisfunction and set the integration_width at the beginning.
        """
        This likelihood is a gaussian likelihood comparing the model values (interpolated/integrated at the redshifts of the dataset)
        with the observed values and errors from corecon. 
        """
        # Extract observational data saved during initialization
        corecon_values = analysis_dict['values'] # one of the dictionary entries we saved during initialization, which contains the values of the observable (for example, x_HII) at the redshifts given by 'z'.
        corecon_errors = analysis_dict['errors'] # error bars for the observable, we take err_up as a proxy for symmetric errors, but this can be improved later to handle asymmetric errors if needed.
        z_corecon = analysis_dict['z'] # redshift of the points in the dataset
        err_left = analysis_dict['err_z_left'] # error bar on the left in redshift, if available, otherwise None
        err_right = analysis_dict['err_z_right'] # error bar on the right in redshift, if available, otherwise None
        
        # Handle bin widths in z
        if integration_width is None:
            # Use dataset errors if available, otherwise 0
            # If err_left/right are None, be careful
            w_left = err_left if err_left is not None else np.zeros_like(z_corecon)
            w_right = err_right if err_right is not None else np.zeros_like(z_corecon)
            
            # Safe conversion
            w_left = np.array(w_left, dtype=float)
            w_right = np.array(w_right, dtype=float)

            widths_z = w_left + w_right # necessary for the normalization of the integral.
        else:
            widths_z = np.full_like(z_corecon, integration_width) # If a fixed integration width is provided, we use it for all points.

        #widths_z is the total width in redshift for each data point, which we will use for integration if it is greater than 0. If it is 0, we will just take the pointwise value at z. 
        #widths_z is calculated based on the errors on redshift from the dataset if integration_width is not provided, otherwise it is set to a fixed value for all points.

        # Interpolate the theoretical model
        # z_model must be increasing for interp1d, check
        if z_model[0] > z_model[-1]: # The first redshift has to be smaller than the last one for interp1d to work properly. If this is not the case, we reverse the arrays.
            z_model = z_model[::-1]
            model_values = model_values[::-1]
            
        f_interp = interp1d(z_model, model_values, kind='cubic', fill_value='extrapolate')
        
        model_at_corecon = []
        
        # Loop over each data point
        for i, z in enumerate(z_corecon): # We want to loop over the redshifts of the dataset, and for each redshift, we want to compare the model value with the observed value.
            w = widths_z[i] # This is the total width in redshift for this data point, which we will use for integration if it is greater than 0. If it is 0, we will just take the pointwise value at z.   
            used_integration = False
            
            if w > 0: #meaning that we have a finite width in redshift for this data point, either from the dataset errors or from the specified integration width, we want to perform an integration of the model over this redshift range to compare with the data, instead of just taking the pointwise value at z.
                # Determine integration limits
                if integration_width is None: # Meaning the integration width is not specified as a fixed value, but we want to use the errors on redshift from the dataset to determine the integration limits. 
                                              # In this case, we check if we have err_left and err_right available, and if so, we use them to set the integration limits. If they are not available, we fall back to a symmetric integration around z with width w/2.
                     # Extract scalar values from numpy arrays if necessary
                     left_val = w_left[i] if w_left is not None else 0.0
                     right_val = w_right[i] if w_right is not None else 0.0
                     z_min = z - left_val
                     z_max = z + right_val
                else: #if we have a fixed integration width specified, we simply integrate symmetrically around z with width w/2.
                     z_min = z - w/2.0
                     z_max = z + w/2.0
                
                # Check for numerical validity
                if z_max > z_min: 
                    try:
                        # Integration
                        val_integral, _ = quad(f_interp, z_min, z_max)
                        avg_value = val_integral / (z_max - z_min)
                        model_at_corecon.append(avg_value)
                        used_integration = True # It is important to set this flag to True only if the integration was successful,
                                                # so that we know whether we can trust the integrated value or if we need to fall back to pointwise interpolation.
                    except Exception as e:
                        print(f"Integration failed at z={z}: {e}")
                        used_integration = False
            #  What happens if w=0 and thus we do not enter the integration block?
            #  In this case, we will just take the pointwise value at z.
            if not used_integration: # if we did not use integration, either because the width was 0 or because the integration failed,
                                     # we fall back to pointwise interpolation at z.
                # Pointwise interpolation
                val = f_interp(z)
                print(f"Pointwise interpolation at z={z}: {val}") # Debug print to check the interpolated value at this redshift.
                # Ensure it's scalar
                if np.ndim(val) > 0: val = val.item()
                model_at_corecon.append(val)
                
        model_at_corecon = np.array(model_at_corecon)

        # Filter valid data and calculate Chi2
        mask_valid = np.isfinite(corecon_values) & np.isfinite(corecon_errors) & np.isfinite(model_at_corecon) & (corecon_errors > 0)
        
        if np.sum(mask_valid) == 0:
            # Debug info in case of failure
            print(f"DEBUG: No valid data points found for Analysis.")
            print(f"  Total points: {len(corecon_values)}")
            print(f"  Valid values: {np.sum(np.isfinite(corecon_values))}")
            print(f"  Valid errors > 0: {np.sum(np.isfinite(corecon_errors) & (corecon_errors > 0))}")
            print(f"  Valid model values: {np.sum(np.isfinite(model_at_corecon))}")
            if len(model_at_corecon) > 0:
                print(f"  Sample model values: {model_at_corecon[:5]}")
            return -np.inf # No valid data or infinite error

        chi_squared = np.sum(((model_at_corecon[mask_valid] - corecon_values[mask_valid]) / corecon_errors[mask_valid]) ** 2)
        return -0.5 * chi_squared

    def compute_upper_limit_loglike(self, analysis_dict, z_model, model_values):
        """
        Log-likelihood for non-detections quoted as one-sided 95% CL upper limits.

        The dataset 'values' are interpreted as 95% CL upper bounds g_i on the
        observable.  We model the measurement noise as a half-Gaussian centred at
        zero:

            sigma_i = g_i / 1.645          (one-sided 95% CL)
            logL_i  = -0.5 * (x_th_i / sigma_i)^2   if x_th_i > 0
                    =  0                              if x_th_i <= 0  (model is
                                                      already inside the limit)

        Args:
            analysis_dict : dict built in initialize() for an
                            'upper_limit_HII_fraction' analysis.
            z_model       : 1-D redshift array from the theory provider.
            model_values  : 1-D array of the corresponding theoretical
                            observable (e.g. x_HII) on z_model.

        Returns:
            float : total log-likelihood summed over valid data points.
        """
        upper_limits = analysis_dict['values']   # 95% CL upper bounds from corecon
        z_data       = analysis_dict['z']        # redshifts of the upper-limit points

        # Make sure z_model is increasing for interp1d
        if z_model[0] > z_model[-1]:
            z_model      = z_model[::-1]
            model_values = model_values[::-1]

        f_interp = interp1d(z_model, model_values, kind='cubic', fill_value='extrapolate')
        x_theory_at_data = np.asarray(f_interp(z_data), dtype=float)

        # Only use points where the upper limit itself is finite and positive
        mask_valid = np.isfinite(upper_limits) & (upper_limits > 0) & np.isfinite(x_theory_at_data)

        if np.sum(mask_valid) == 0:
            print(f"DEBUG: No valid upper-limit points found for analysis '{analysis_dict['name']}'.")
            return -np.inf

        g      = upper_limits[mask_valid]
        x_th   = x_theory_at_data[mask_valid]
        sigma  = g / 1.645  # convert 95% CL upper limit to Gaussian sigma

        # Half-Gaussian: penalise only when the theory exceeds the limit
        log_l = np.where(x_th > 0, -0.5 * (x_th / sigma) ** 2, 0.0)

        return float(np.sum(log_l))

    def compute_lower_limit_loglike(self, analysis_dict, z_model, model_values):
        """
        Log-likelihood for detections quoted as one-sided 95% CL lower limits.

        Specular analogy to compute_upper_limit_loglike:

          Upper limit g (95% CL):
              Gaussian centred at 0, g sits at 1.645*sigma from the centre.
              sigma = g / 1.645
              logL  = -0.5 * (x_th / sigma)^2       if x_th > 0
                    =  0                             if x_th <= 0

          Lower limit g (95% CL):
              Gaussian centred at the physical maximum (x_HII = 1), g sits at
              1.645*sigma from the centre on the left tail.
              sigma = (1 - g) / 1.645
              logL  = -0.5 * ((x_th - 1) / sigma)^2   if x_th < 1  (always true)
                    =  0                                if x_th >= 1

          In practice the half-Gaussian penalises only when the theory falls
          below the lower limit:
              logL  = -0.5 * ((x_th - 1) / sigma)^2   if x_th < g
                    =  0                                if x_th >= g

        Args:
            analysis_dict : dict built in initialize() for a
                            'lower_limit_HII_fraction' analysis.
            z_model       : 1-D redshift array from the theory provider.
            model_values  : 1-D array of the corresponding theoretical
                            observable (e.g. x_HII) on z_model.

        Returns:
            float : total log-likelihood summed over valid data points.
        """
        lower_limits = analysis_dict['values']   # 95% CL lower bounds from corecon
        z_data       = analysis_dict['z']        # redshifts of the lower-limit points

        # Make sure z_model is increasing for interp1d
        if z_model[0] > z_model[-1]:
            z_model      = z_model[::-1]
            model_values = model_values[::-1]

        f_interp = interp1d(z_model, model_values, kind='cubic', fill_value='extrapolate')
        x_theory_at_data = np.asarray(f_interp(z_data), dtype=float)

        # Only use points where the lower limit itself is finite and in (0, 1)
        mask_valid = np.isfinite(lower_limits) & (lower_limits > 0) & (lower_limits < 1) & np.isfinite(x_theory_at_data)

        if np.sum(mask_valid) == 0:
            print(f"DEBUG: No valid lower-limit points found for analysis '{analysis_dict['name']}'.")
            return -np.inf

        g     = lower_limits[mask_valid]
        x_th  = x_theory_at_data[mask_valid]
        sigma = (1.0 - g) / 1.645  # distance from lower limit to physical maximum (x_HII=1)

        # Half-Gaussian: penalise only when the theory falls below the limit
        log_l = np.where(x_th < g, -0.5 * ((x_th - 1.0) / sigma) ** 2, 0.0)

        return float(np.sum(log_l))

    def logp(self, **params_values):
        """
        Calculate the log-likelihood.
        """
        # Access the reionization history via the shared module-level variable.
        # cobaya guarantees ReioTheory.calculate() runs before logp(), so
        # _shared_reio_history is always populated when we get here.
        from Reiolike.reio_theory import _shared_reio_history
        z  = _shared_reio_history['z']
        xe = _shared_reio_history['xe']
        if z is None or xe is None:
            print("[ReioLike] ERROR: reio_history not available — ReioTheory.calculate() may not have run.")
            return -np.inf
        
        # Calculate hydrogen fraction (x_HI)
        # Assume standard Y_He 
        Y_He = 0.24 
        # fHe = n_He/n_H
        fHe = Y_He / (1 - Y_He) * (1.008 / 4.003)  # ≈ 0.079

        # Assuming hydrogen and singly ionized helium reionization proceed together:

        x_HII = xe / (1 + fHe)
        # x_H_neutral = 1 - x_HII # Neutral hydrogen fraction.

        log_prob = 0.0
        
        # Iterate over the configured analyses
        if hasattr(self, 'analyses'):
             for analysis in self.analyses:
                # Each analysis is a dictionary with the data (saved in initialize)
                
                model_to_compare = None
                if analysis['type'] in ('HII_fraction', 'upper_limit_HII_fraction', 'lower_limit_HII_fraction'):
                    model_to_compare = x_HII

                if analysis['type'] == 'upper_limit_HII_fraction':
                    log_L = self.compute_upper_limit_loglike(
                        analysis_dict=analysis,
                        z_model=z,
                        model_values=model_to_compare,
                    )
                elif analysis['type'] == 'lower_limit_HII_fraction':
                    log_L = self.compute_lower_limit_loglike(
                        analysis_dict=analysis,
                        z_model=z,
                        model_values=model_to_compare,
                    )
                else:
                    log_L = self.compute_gaussian_loglike(
                        analysis_dict=analysis,
                        z_model=z,
                        model_values=model_to_compare,
                        integration_width=None  # Use errors on z from the dataset
                    )
                
                print(f"  [{analysis['type']}] '{analysis['name']}':  logL = {log_L:.6f}")
                if np.isfinite(log_L):
                    log_prob += log_L
                else:
                    # Numerical error handling 
                    return -np.inf
            
        return log_prob

