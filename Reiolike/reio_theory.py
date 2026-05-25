from cobaya.theory import Theory # Base class for theories in Cobaya
from cobaya.theories.classy import classy # We will extend this to create our ReioTheory exploiting CLASS
import numpy as np
from .reio_models import tanh_model #this is the function that computes xe(z) for the tanh model, we will use it in ReioTheory

# Module-level shared storage: updated by ReioTheory.calculate(), read by ReioLike.logp().
# cobaya guarantees calculate() runs before logp() for each sample, so this is always current.
_shared_reio_history = {'z': None, 'xe': None}

class ReioTheory(classy):
    """
    Una classe Theory che estende CLASS per gestire storie di reionizzazione custom.
    
    Prende in input parametri generici (a, b...) e:
    1. Calcola xe(z)
    2. Passa xe(z) a CLASS
    3. Rende disponibile xe(z) per altre likelihood
    """
    # This class can accept a parameters that is the reionization history written in the .yaml file
    reio_model: str = "tanh" # Name of the history of reionization, or identifier to select it from a library of histories. 
                             # At this stage, let's assume it is the tanh and the parameters sampled by cobaya are 'z_re' and 'delta_z'.
    z_min: float = 0.0  # Minimum redshift for the reionization history
    z_max: float = 30.0 # Maximum redshift for the reionization history
    z_points: int = 100 # Number of points in the redshift grid for
    
    def initialize(self):
        """Inizializzazione della classe base (CLASS)"""
        super().initialize()
        #Here we could place the additional stuff: for example the external PCA eigenfunctions.
        #Here we could also place fixed constraints/requests on the reionization history if we think it is necessary (for example, if we want to force xe(z=0)=1 or similar).
        
        # Grid construction strategy: 
        # We divide the redshift range into 3 parts: [z_min, 6], [6, 15], [15, z_max]
        # Most of the points are allocated in the central part [6, 15] where reionization happens.
        
        z_low_lim = 6.0
        z_high_lim = 25.0
        
        n_low = 5
        n_high = 8
        # The remaining points go to the central part
        n_mid = max(self.z_points - n_low - n_high, 20) 

        # 1. Low redshift range [z_min, z_low_lim)
        # We use endpoint=False to avoid duplicate point at z_low_lim
        z_grid_1 = np.linspace(self.z_min, z_low_lim, n_low, endpoint=False)
        
        # 2. Mid redshift range [z_low_lim, z_high_lim)
        # We use endpoint=False to avoid duplicate point at z_high_lim
        z_grid_2 = np.linspace(z_low_lim, z_high_lim, n_mid, endpoint=False)
        
        # 3. High redshift range [z_high_lim, z_max]
        # Here we include the endpoint
        z_grid_3 = np.linspace(z_high_lim, self.z_max, n_high)

        self.z_class = np.concatenate((z_grid_1, z_grid_2, z_grid_3))


    def get_can_provide(self):
        """ Tells Cobaya what this theory can provide.
            We will provide the reionization history as a function of redshift,
            which CLASS can use to compute derived parameters like tau_reio."""
        return super().get_can_provide()

    def calculate(self, state, want_derived=True, **params_values_dict):
        """
        This is the main method where we compute xe(z) and pass it to CLASS.
        """
        if self.reio_model == "tanh":
            # Extract the parameters for the tanh model from params_values_dict
            z_re = params_values_dict.get("reio_z_re")
            delta_z = params_values_dict.get("reio_delta_z")

            # Check if parameters are provided
            if z_re is None or delta_z is None:
                raise ValueError("ReioTheory (tanh): Parameters 'reio_z_re' and 'reio_delta_z' must be provided in the 'params' block.")
                
            xe_class = tanh_model(self.z_class, z_re, delta_z)
            
            # CLEANUP: Remove long tail of zeros to avoid numerical issues in CLASS splines
            # We keep points where xe > 1e-9, plus a couple of buffer points to anchor 0
            # If we pass 100 points of exactly 0 at high z, CLASS thermodynamics derivatives can diverge 
            mask_significant = xe_class > 1e-9
            
            # Find the last point where xe is significant
            if np.any(mask_significant):
                # Get the index of the last True value
                last_sig_idx = np.where(mask_significant)[0][-1]
                
                # Keep up to 2 points after the last significant one (which will be ~0)
                # to anchor the tail to 0 without providing a super long flat array
                cut_idx = min(len(self.z_class), last_sig_idx + 3)
                
                self.z_class = self.z_class[:cut_idx]
                xe_class = xe_class[:cut_idx]
                
        # Enforce boundary condition: last point is exactly 0
        xe_class[-1] = 0.0

        # Store history for direct access by ReioLike
        self._current_reio_history = (self.z_class, xe_class)
        # Update module-level shared storage so ReioLike.logp() can read it
        _shared_reio_history['z'] = self.z_class
        _shared_reio_history['xe'] = xe_class

        # Construct strings for CLASS
        # We upped _ARGUMENT_LENGTH_MAX_ to 4096 in parser.h, so we can use sufficient precision
        # to ensure z is strictly increasing, avoiding "10.2, 10.2" duplicates.
        str_z = ','.join([f'{z:.8g}' for z in self.z_class])
        str_xe = ','.join([f'{xe:.8g}' for xe in xe_class])


        # Pass to CLASS via params_values_dict
        params_values_dict['reio_parametrization'] = 'reio_inter'
        params_values_dict['reio_inter_num'] = len(self.z_class)
        params_values_dict['reio_inter_z'] = str_z
        params_values_dict['reio_inter_xe'] = str_xe
        print(f" ReioTheory: Configured CLASS with reio_parametrization='reio_inter' and {len(self.z_class)} points for xe(z)")
        print(f" Example: z[0]={self.z_class[0]:.2f}, xe[0]={xe_class[0]:.4f}; z[-1]={self.z_class[-1]:.2f}, xe[-1]={xe_class[-1]:.4f}")
        print(f" (Full arrays passed to CLASS: reio_inter_z and reio_inter_xe with {len(self.z_class)} points)")
        # CLEANUP: Remove custom parameters that CLASS does not recognize
        if 'reio_z_re' in params_values_dict:
            del params_values_dict['reio_z_re']
        if 'reio_delta_z' in params_values_dict:
            del params_values_dict['reio_delta_z']

        # Convert numpy scalar types (even in lists/tuples) to Python native types for CLASS compatibility (robust recursive)
        def _to_native(val):
            if hasattr(val, "item"):
                return val.item()
            if isinstance(val, (list, tuple)):
                return type(val)(_to_native(v) for v in val)
            return val

        for k in list(params_values_dict.keys()):
            params_values_dict[k] = _to_native(params_values_dict[k])

        return super().calculate(state, want_derived, **params_values_dict)
           

    def get_reio_history(self):
        """Helper method for external likelihoods"""
        return self._current_reio_history