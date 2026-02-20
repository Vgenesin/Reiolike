# from cobaya.likelihood import DataSetLikelihood
import numpy as np
import yaml
from scipy.interpolate import interp1d
from scipy.integrate import quad
import corecon as con
from cobaya.likelihood import Likelihood

class ReioLike(Likelihood):
    """
    This likelihood compares the neutral hydrogen fraction obtained by a certain reionization history xe(z) 
    calculated by the theory (ReioTheory) with astrophysical data.
    
    It uses the 'corecon' package to compute likelihoods for different astrophysical analyses.
    
    Assumption:
    - The theory (ReioTheory) calculates xe(z) on a redshift grid and saves it in the state as 'reio_history_z' and 'reio_history_xe'.
    - The theory provides the arrays 'z' and 'xe'.
    """
    
    # This is the path to the corecon configuration file, which specifies which analyses to run and with which datasets.
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
            
            # Mapping YAML observable types to corecon keys.
            corecon_key = observable_type
            if observable_type == 'HII_fraction': 
                corecon_key = 'x_HII'  # This is the key we expect to find in corecon for this type of analysis.
                                       # We assume that corecon has a dataset for 'x_HII' which contains the relevant analyses for the ionized fraction. 
                                       # If we had other types of observables, we would map them to their respective corecon keys here.

            #At this level we want everything we could get from corecon with this key.
            try:
                corecon_data_dict = con.get(corecon_key)
            except Exception as e:
                print(f"  - ERROR loading corecon data for '{corecon_key}': {e}")
                raise RuntimeError(f"CRITICAL ERROR loading corecon data for '{corecon_key}': {e}")

            if isinstance(datasets, list):
                for dataset_name in datasets: # We want to loop over the datasets specified in the config file for this observable type. 
                                              # For example, if we have 'HII_fraction' with a list of datasets, 
                                              # we want to loop over them and extract the data from corecon for each dataset.
                    print(f"  - Initializing analysis for dataset: {dataset_name}")
                    
                    if dataset_name in corecon_data_dict: # dataset_name is the key to access the dataset in corecon.
                        data_obj = corecon_data_dict[dataset_name] # estracting the dataset object from corecon, which should contain axes (z), values (x_HII), and errors.
                        
                        self.analyses.append({
                            'type': observable_type, # what kind of astrohysical constraint we are asking from corecon (for example, HII_fraction, x_HI, etc.)
                            'name': dataset_name, # the name of the analysis/dataset we are using (for example, "Umeda et al. 2023 (subm.)")
                            'z': np.array(data_obj.axes, dtype=float), # redshift of the points in the dataset
                            'values': np.array(data_obj.values, dtype=float), # values for the astrophysical observable (for example, x_HII) at the redshifts given by 'z'
                            'errors': np.array(data_obj.err_up, dtype=float), # error bars for the observable, we take err_up as a proxy for symmetric errors, but this can be improved later to handle asymmetric errors if needed.
                            # Safe handling of err_left and err_right for integration
                            'err_z_left': self._get_error_array(data_obj, 'err_left'), # error bar on the left in redshift, if available, otherwise None
                            'err_z_right': self._get_error_array(data_obj, 'err_right') # error bar on the right in redshift, if available, otherwise None
                        })
                    else:
                        print(f"    - WARNING: Dataset '{dataset_name}' not found in corecon '{corecon_key}'")
                        raise RuntimeError(f"CRITICAL ERROR loading corecon data for '{corecon_key}': Dataset '{dataset_name}' not found")

            else:
                 print(f"  - WARNING: Expected list for {observable_type}, got {type(datasets)}")
                    
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
        Defines the quantities that must be calculated by the theory.
        In this case we explicitly ask for 'reio_xe' and 'reio_z'.
        So the common underlying history of reionization is common among cmb power spectrum and astrophysical constraints.
        """
        return {'reio_history_z': None, 'reio_history_xe': None}

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
                        used_integration = True # It is important to set this flag to True only if the integration was successful, so that we know whether we can trust the integrated value or if we need to fall back to pointwise interpolation.
                    except Exception as e:
                        print(f"Integration failed at z={z}: {e}")
                        used_integration = False
            
            #what happens if w=0 and thus we do not enter the integration block? In this case, we will just take the pointwise value at z, which is what we want.
            if not used_integration: #if we did not use integration, either because the width was 0 or because the integration failed, we fall back to pointwise interpolation at z.
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

    def logp(self, **params_values):
        """
        Calculate the log-likelihood.
        """
        # 1. Retrieve the reionization history calculated by the theory (ReioTheory)
        #    This is the starting point: xe(z) and z to "work" with.
        xe = self.provider.get_result('reio_history_xe')
        z = self.provider.get_result('reio_history_z')
        
        # Calculate neutral hydrogen fraction (x_HI)
        # Assume standard Y_He if not passed otherwise
        Y_He = 0.24 
        # fHe = n_He/n_H
        fHe = Y_He / (1 - Y_He) * (1.008 / 4.003)  # ≈ 0.079
        
        # xe_theory = x_HII + fHe*x_HeII + ... 
        # Assuming hydrogen and singly ionized helium reionization proceed together:
        # xe ≈ x_HII * (1 + fHe)
        # Therefore x_HII = xe / (1+fHe)
        x_HII = xe / (1 + fHe)
        # x_H_neutral = 1 - x_HII
        
        # Ensure it's between 0 and 1 (for numerical stability)
        # x_H_neutral = np.clip(x_H_neutral, 0, 1)

        # 2. Here we "work" on xe(z) or x_H_neutral.
        
        # 3. Comparison with Corecon analyses
        log_prob = 0.0
        
        # For now: iterate over the configured analyses
        if hasattr(self, 'analyses'):
             for analysis in self.analyses:
                # Each analysis is a dictionary with the data (saved in initialize)
                # We calculate the likelihood by comparing the theory's x_H_neutral with the data
                
                # If the analysis is of type 'HII_fraction' (x_HII), we need to compare
                # with the IONIZED fraction (x_HII = 1 - x_HI).
                # If it's x_HI, we use x_H_neutral.
                
                model_to_compare = None
                if analysis['type'] == 'HII_fraction': # So we are comparing with x_HII, which is the ionized fraction, so we use x_HII from the theory.
                    model_to_compare = x_HII
                # else: 
                #     # Default or other types, assume comparison with x_HI for now
                #     model_to_compare = x_H_neutral

                log_L = self.compute_gaussian_loglike(
                    analysis_dict=analysis, 
                    z_model=z, 
                    model_values=model_to_compare,
                    integration_width=None # Use errors on z from the dataset
                )
                
                if np.isfinite(log_L):
                    log_prob += log_L
                else:
                    # Numerical error handling (optional: strong penalty)
                    return -np.inf
            
        return log_prob
