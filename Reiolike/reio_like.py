from cobaya.likelihood import Likelihood
import numpy as np
import sys
import os
import importlib

class CoreconLike(Likelihood):
    # Parametri configurabili dallo yaml
    corecon_path: str = ""     # Path assoluto alla cartella di Corecon (opzionale se installato)
    corecon_module: str = "corecon.likelihood" # Modulo da importare
    # IMPORTANTE: questa è la classe ESTERNA che fa i conti veri, non questo wrapper
    corecon_worker_class: str = "CoreconCalculator"  # Rinominata per evitare confusione
    
    # Nuovi parametri configurabili dall'utente
    data_file: str = ""        # Path al file dei dati
    reio_model: str = ""       # Nome o identificativo della storia di reionizzazione
    extra_args: dict = {}      # Dizionario per passare altri argomenti opzionali

    def initialize(self):
        """
        Carica dinamicamente il modulo Corecon.
        """
        # 1. Aggiungi il path se specificato e non c'è già
        if self.corecon_path and self.corecon_path not in sys.path:
            sys.path.append(self.corecon_path)
            print(f"Added {self.corecon_path} to sys.path")
            
        try:
            # 2. Importa il modulo
            module = importlib.import_module(self.corecon_module)
            
            # 3. Prendi la classe
            # Qui cerchiamo la classe esterna specificata nello YAML (default: CoreconCalculator)
            LikelihoodClass = getattr(module, self.corecon_worker_class)
            
            # 4. Inizializza la classe passando i parametri configurati
            print(f"Initializing {self.corecon_worker_class} with data={self.data_file} and model={self.reio_model}")
            
            self.corecon_worker = LikelihoodClass(
                data_file=self.data_file, 
                reio_model=self.reio_model,
                **self.extra_args
            )
            print(f"Successfully initialized {self.corecon_worker_class}")
            
        except TypeError as e:
            # Fallback
            try:
                print(f"Direct init failed ({e}), trying setup attributes manually...")
                self.corecon_worker = LikelihoodClass()
                if self.data_file: self.corecon_worker.data_file = self.data_file
                if self.reio_model: self.corecon_worker.reio_model = self.reio_model
                for k, v in self.extra_args.items():
                    setattr(self.corecon_worker, k, v)
                print(f"Successfully initialized {self.corecon_worker_class} (manual setup)")
            except Exception as e2:
                 raise ImportError(f"Errore inizializzazione {self.corecon_worker_class}: {e}. Fallback failed: {e2}")

        except ImportError as e:
            raise ImportError(f"Non riesco a importare {self.corecon_module}. Assicurati che sia installato o specifica 'corecon_path'. Errore: {e}")
        except AttributeError as e:
            raise AttributeError(f"Il modulo {self.corecon_module} non ha la classe {self.corecon_worker_class}. Errore: {e}")

    def get_requirements(self):
        """
        Chiediamo a CLASS lo spettro EE fino a l_max specificato nello yaml
        """
        return {
            "Cl": {
                "ee": self.l_max 
            }
        }

    def logp(self, **params_values):
        """
        Calcola la log-likelihood usando Corecon.
        Accetta **params_values che contiene i parametri campionati.
        """
        # 0. Passa i parametri extra se necessario
        # (Qui potresti chiamare update_parameters come discusso, se serve)
        # self.corecon_worker.update_parameters(**params_values)
        
        # 1. Recupera lo spettro Cl da Cobaya (unitless)
        cl_dict = self.provider.get_Cl(ell_factor=False) 
        cl_ee = cl_dict['ee']
        
        # 2. Prepara gli ell e fai la conversione in D_ell [muK^2]
        l_min = 2
        l_curr_max = self.l_max 
        
        limit = min(len(cl_ee), l_curr_max)
        
        ell = np.arange(l_min, limit)
        cl_subset = cl_ee[l_min:limit]
        
        # Costanti
        T_CMB = 2.7255e6 # microkelvin
        
        # Conversione Cl -> Dl [muK^2]
        dl_factor = ell * (ell + 1) / (2 * np.pi)
        dl_ee_muk2 = dl_factor * cl_subset * (T_CMB**2)
        
        # 3. Chiama Corecon
        log_L = self.corecon_worker.log_likelihood(dl_ee_muk2)
        
        return log_L