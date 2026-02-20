from cobaya.theory import Theory
from cobaya.theories.classy import classy
import numpy as np
from .reio_models import tanh_model

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
    z_max: float = 20.0 # Maximum redshift for the reionization history
    z_points: int = 100 # Number of points in the redshift grid for
    
    def initialize(self):
        """Inizializzazione della classe base (CLASS)"""
        super().initialize()
        #Here we could place the additional stuff: for example the external PCA eigenfunctions.
        #Here we could also place fixed constraints/requests on the reionization history if we think it is necessary (for example, if we want to force xe(z=0)=1 or similar).


    def calculate(self, state, want_derived=True, **params_values_dict):
        """
        Il cuore della modifica.
        intercetta i parametri, calcola xe(z), configura CLASS.
        """
        z_class= np.linspace(self.z_min, self.z_max, self.z_points) # example of z array, we can make it more flexible later
        if self.reio_model == "tanh":
            # Here we take the sampled parameters 'z_re' and 'delta_z' and we calculate xe(z) using the tanh parametrization.
            # In tanh model: z_re -> midpoint, delta_z -> width
            # Nota: Cobaya passerà questi parametri solo se sono definiti nel blocco 'params' dello YAML
            z_re = params_values_dict.get("reio_z_re")
            delta_z = params_values_dict.get("reio_delta_z")
            
            # Check if parameters are provided, otherwise raise an error (since they are essential for the tanh model)
            if z_re is None or delta_z is None:
                raise ValueError("ReioTheory (tanh): Parameters 'reio_z_re' and 'reio_delta_z' must be provided in the 'params' block.")
                
            xe_class = tanh_model(z_class, z_re, delta_z)
           
        # Save history in state for likelihoods to access
        state['reio_history_z'] = z_class
        state['reio_history_xe'] = xe_class

        str_z = ','.join([f'{z:.4g}' for z in z_class])
        str_xe = ','.join([f'{xe:.4g}' for xe in xe_class])

        params_values_dict['reio_parametrization'] = 'reio_inter'
        params_values_dict['reio_inter_num'] = len(z_class)
        params_values_dict['reio_inter_z'] = str_z
        params_values_dict['reio_inter_xe'] = str_xe

        return super().calculate(state, want_derived, **params_values_dict)

    def get_reio_history(self):
        """Metodo helper per le likelihood esterne"""
        return self.current_state.get('reio_history_z'), self.current_state.get('reio_history_xe')
