from cobaya.component import CobayaComponent
from cobaya.tools import get_external_function, NumberWithUnits, load_DataFrame
from cobaya.emulator_cache import EmulatorCache, PCACache
from typing import  Dict
import copy
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.exceptions import ConvergenceWarning
import matplotlib.pyplot as plt
import scipy as sc
from sklearn.model_selection import train_test_split
import time
import os
import gc

class quantity_GP:
    name: str       # name of this GP
    dim: int        # dimensionality of this problem
    n: int          # number of GP for this qunatity

    def __init__(self, name, dim, n):
        self.name = name
        self.dim = dim
        self.n = n


#
# This Class generates a PCA Guassian process emulator
#
class Emulator(CobayaComponent):
    r"""
    Adaptive, PCA and gaussian based Emulator.
    """

    _at_resume_prefer_new = [ "postpone_learning", "learn_every", "training_size", "precision"]
    _at_resume_prefer_old = []
    _prior_rejections: int = 0
    file_base_name = 'emulator'

    parameter_prediction: Dict[str, quantity_GP]  # parameter name -> prediction type

    # instance variables from yaml
    postpone_learning: NumberWithUnits
    learn_every: NumberWithUnits
    precision: float

    def __init__(self, *args, **kwargs):
        self.set_logger("emulator")
        self._emulator = None
        self._emulator_ready = False
        self._emulator_trained = False
        self.theories = []
        self.must_provide = {}

        self.evalution_counter = 0

        self.postpone_learning = 0 if 'postpone_learning' not in args[1] else args[1]['postpone_learning']

        self.learn_every = 200 if 'learn_every' not in args[1] else args[1]['learn_every']
        self.precision = 1.e-1 if 'precision' not in args[1] else args[1]['precision'] # This is basicially the minimal error of the emulator

        # This is relevat error criterion if we leave the vicinity of the training set. It scales the allowed error linear in log10 of the loglike distance
        self.precision_linear = 1.e-1 if 'precision_linear' not in args[1] else args[1]['precision_linear']
        self.precision_quadratic = 0.0 if 'precision_quadratic' not in args[1] else args[1]['precision_quadratic']

        self._max_loglike = -np.inf # This is the minimal loglike value we witnessed so far


        self.N_validation_states = 5 if 'N_validation_states' not in args[1] else args[1]['N_validation_states']

        self.testset_fraction = 0.1 if 'testset_fraction' not in args[1] else args[1]['testset_fraction']

        self.N_pca_components = {} if 'N_pca' not in args[1] else args[1]['N_pca']
        self.N_pca_components_default = 15 if 'N_pca_default' not in args[1] else args[1]['N_pca_default']

        self.validation_loglikes = np.zeros(self.N_validation_states)
        self.validation_states = []
        self.predictors = None

        self.in_PCA_validation = False
        self.N_PCA_validation_states = 5 if 'N_PCA_validation_states' not in args[1] else args[1]['N_PCA_validation_states']
        self.PCA_validation_loglikes = np.zeros(self.N_PCA_validation_states)
        self.PCA_validation_states = []

        self._pca_update = 2 if 'pca_update' not in args[1] else args[1]['pca_update']

        self.in_validation = False
        self.debug = False if 'debug' not in args[1] else args[1]['debug']
        self.counter_emulator_used = 0
        self.counter_emulator_not_used = 0

        # During burnin phase we use a reduced emulator size
        self._in_burnin_phase = True
        self._last_loglike_update = 0
        self._N_burnin = 40 if 'training_size_burnin' not in args[1] else args[1]['training_size_burnin']
        self._burnin_trigger = 50 if 'burnin_trigger' not in args[1] else args[1]['burnin_trigger']

        self._gp_fit_size = 100 if 'gp_fit_size' not in args[1] else args[1]['gp_fit_size']

        self.last_evaluated_state = {}

        self.timings = {'training':0.0,
                        'evaluating':0.0,
                        'add_data':0.0}

        self.set_timing_on(True)

        # check if log file exists and delete it
        # if self.debug:
        if os.path.exists('log_file.txt'):
            os.remove('log_file.txt')

        self._gp_initial_minimization_states = 40 if 'gp_initial_minimization_states' not in args[1] else args[1]['gp_initial_minimization_states']
        self._gp_minimization_states = 5 if 'gp_minimization_states' not in args[1] else args[1]['gp_minimization_states']

        # Create a cache instance
        self.data_cache = EmulatorCache(
            cache_size = 200 if 'training_size' not in args[1] else args[1]['training_size'],
            proximity_threshold = 0.01 if 'proximity_threshold' not in args[1] else args[1]['proximity_threshold'],
            delta_loglike_cache = 100 if 'delta_loglike_cache' not in args[1] else args[1]['delta_loglike_cache'],
        )

        # Create a cache instance for the PCA cache
        self.pca_cache = PCACache(
            cache_size = 200 if 'pca_cache_size' not in args[1] else args[1]['pca_cache_size'],
            delta_loglike_cache = 100 if 'delta_loglike_cache' not in args[1] else args[1]['delta_loglike_cache'],
        )

        self.parameter_dimension = {}

        self.is_initialized = False



    def initialize_from_file(self):

        self.is_initialized = True
        return False
    
    def write_log_step(self, event, loglike=0.0):
        #if self.debug:  # Change this to be relevant soon TODO [SG]
        with open('log_file.txt', 'a') as f:
            f.write('%f %s %d %d %d %f ' % (loglike, event, self.counter_emulator_used, self.counter_emulator_not_used, self.evalution_counter ,time.time()))
            for val in self.validation_loglikes:
                f.write('%f ' % val)
            f.write('\n')

    def initialize(self,state):
        self._emulator = None
        self._emulator_ready = False
        self._emulator_trained = False

        self.state = state # dummy state which will be adapted for later

        # We need to remove some potential double counting of parameters in self.must_provide
        # This is a bit of a hack, but it works for now. Change this in future. TODO [SG]
        remove_keys = []
        for theory in self.theories:
            for key, value in self.must_provide[theory].items():
                remove = True
                for key_state, value_state in state[theory].items():
                    if key == key_state:
                        remove = False
                    if type(value_state) == dict:
                        for k,v in value_state.items():
                            if key == k:
                                remove = False
                if remove:
                    remove_keys.append(key)
        for key in remove_keys:
            for theory in self.theories:
                self.must_provide[theory].pop(key)
                    
                        


        # Create Gaussian Processes
        self.predictors = {}
        for theory in self.theories:
            self.predictors[theory] = {}
            for key, value in self.must_provide[theory].items():

                # Create a new GP
                if type(value) == int:
                    self.predictors[theory][key] = PCA_GPEmulator(name=str(key),
                                                                  out_dim=value,in_dim=len(state[theory]['params']), 
                                                                  testset_fraction = self.testset_fraction, 
                                                                  pca_cache=self.pca_cache, 
                                                                  debug=self.debug, 
                                                                  _pca_update=self._pca_update,
                                                                  _gp_fit_size=self._gp_fit_size,
                                                                  gp_initial_minimization_states=self._gp_initial_minimization_states,
                                                                  gp_minimization_states=self._gp_minimization_states,
                                                                  N_pca_components=self.N_pca_components,
                                                                  N_pca_components_default=self.N_pca_components_default)
                elif type(value) == dict:
                    if len(value) == 1:
                        self.predictors[theory][key] = PCA_GPEmulator(name=str(key),
                                                                      out_dim=value[list(value.keys())[0]],
                                                                      in_dim=len(state[theory]['params']), 
                                                                      testset_fraction = self.testset_fraction, 
                                                                      pca_cache=self.pca_cache, 
                                                                      debug=self.debug, 
                                                                      _pca_update=self._pca_update,
                                                                      _gp_fit_size=self._gp_fit_size,
                                                                      gp_initial_minimization_states=self._gp_initial_minimization_states,
                                                                      gp_minimization_states=self._gp_minimization_states,
                                                                      N_pca_components=self.N_pca_components,
                                                                      N_pca_components_default=self.N_pca_components_default)
                    else:
                        for k,v in value.items():   # TODO: super ugly, but works for now. Fix this
                            if key == 'Cl': 
                                self.predictors[theory][k] = PCA_GPEmulator(name=str(k),
                                                                            out_dim=v+1,
                                                                            in_dim=len(state[theory]['params']), 
                                                                            testset_fraction = self.testset_fraction, 
                                                                            pca_cache=self.pca_cache, 
                                                                            debug=self.debug,
                                                                            _pca_update=self._pca_update,
                                                                            _gp_fit_size=self._gp_fit_size,
                                                                            gp_initial_minimization_states=self._gp_initial_minimization_states,
                                                                            gp_minimization_states=self._gp_minimization_states,
                                                                            N_pca_components=self.N_pca_components,
                                                                            N_pca_components_default=self.N_pca_components_default)
              
                            else:
                                self.predictors[theory][k] = PCA_GPEmulator(name=str(k),
                                                                            out_dim=v,
                                                                            in_dim=len(state[theory]['params']), 
                                                                            testset_fraction = self.testset_fraction, 
                                                                            pca_cache=self.pca_cache, 
                                                                            debug=self.debug, 
                                                                            _pca_update=self._pca_update,
                                                                            _gp_fit_size=self._gp_fit_size,
                                                                            gp_initial_minimization_states=self._gp_initial_minimization_states,
                                                                            gp_minimization_states=self._gp_minimization_states,
                                                                            N_pca_components=self.N_pca_components,
                                                                            N_pca_components_default=self.N_pca_components_default)
                else:
                    self.log.error("Unknown type of prediction: %s" % type(value))
            
            # for testing add a loglike predictor
            self.predictors[theory]['loglike'] = PCA_GPEmulator(name='loglike',
                                                                out_dim=1,
                                                                in_dim=len(state[theory]['params']), 
                                                                testset_fraction = self.testset_fraction, 
                                                                pca_cache=self.pca_cache, 
                                                                debug=self.debug, 
                                                                _pca_update=self._pca_update,
                                                                _gp_fit_size=self._gp_fit_size,
                                                                gp_initial_minimization_states=self._gp_initial_minimization_states,
                                                                gp_minimization_states=self._gp_minimization_states,
                                                                N_pca_components=self.N_pca_components,
                                                                N_pca_components_default=self.N_pca_components_default)

        self.is_initialized = True
        return True

    def _create_validation_states(self, theory, state):
        #self.log.info("Creating validation states")

        # No need for validation if not requested
        if self.N_validation_states == 0:
            return True
        
        self.validation_states = []
        for i in range(self.N_validation_states):
            self.validation_states.append(copy.deepcopy(self.state))

        data_in = np.array([[value for key,value in state['params'].items()]])
        for name, GP in self.predictors[theory].items():

            test_pred = GP._sample(data_in,self.N_validation_states)

            # Add the prediction to the validation states
            for i in range(self.N_validation_states):
                self.validation_states[i][theory]['params'] = state['params']
                for key, value in self.validation_states[i][theory].items():
                    if key == name:
                        self.validation_states[i][theory][key] = test_pred[i]
                    if type(value) == dict:
                        for k,v in value.items():
                            if k == name:
                                self.validation_states[i][theory][key][k] = test_pred[i]
        

    def _create_PCA_validation_states(self, theory, state):
        #self.log.info("Creating PCA validation states")

        # No need for validation if not requested
        if self.N_PCA_validation_states == 0:
            return True
        
        self.PCA_validation_states = []
        for i in range(self.N_PCA_validation_states):
            self.PCA_validation_states.append(copy.deepcopy(self.state))

        data_in = np.array([[value for key,value in state['params'].items()]])

        for name, GP in self.predictors[theory].items():

            test_pred = GP._sample(data_in,self.N_PCA_validation_states)

            # Add the prediction to the validation states
            for i in range(self.N_PCA_validation_states):
                for key, value in self.validation_states[i][theory].items():
                    if key == name:
                        self.PCA_validation_states[i][theory][key] = test_pred[i]
                    if type(value) == dict:
                        for k,v in value.items():
                            if k == name:
                                self.PCA_validation_states[i][theory][key][k] = test_pred[i]

        #self.log.info(self.validation_states)


    def evaluate(self,theory, state, want_derived, loglike, **params_values_dict):
        self.evalution_counter += 1

        if self.timer:
            self.timer.start()

        # decide if we are still in burnin phase
        if ((self.counter_emulator_not_used+self.counter_emulator_used)>self.postpone_learning+self._burnin_trigger):
            if (self.evalution_counter - self._last_loglike_update) > self._burnin_trigger:
                if self._in_burnin_phase:
                    self._in_burnin_phase = False
                    self.write_log_step('burnin_end')

        #if (self.evalution_counter%100 == 0):
        if not self.in_validation:
            if ((self.counter_emulator_not_used+self.counter_emulator_used)%10 == 0):
                self.log.info("Emulator used %d; not used %d" % (self.counter_emulator_used,self.counter_emulator_not_used))
                self.log.info(self.timings)

        # If we are not initialized yet, we cannot calculate anything
        if not self.is_initialized:
            self.counter_emulator_not_used += 1
            self.in_validation = False
            self.write_log_step('not_used')
            return None, True
        
        #self.log.info("(self.data_cache._newly_added_points+1) self.learn_every")
        #self.log.info((self.data_cache._newly_added_points+1) % self.learn_every)

        # Should we train the emulator? Then first load data to GPs
        #if ((self.data_cache._newly_added_points+1) % self.learn_every == 0) and (self.counter_emulator_not_used>self.postpone_learning):
        #    self.log.info("_load_data_to_GP")
        #    if self.in_validation == False:
        #        #self._create_PCA_validation_states(theory, state)
        #        self.in_PCA_validation = True

        # Do some PCA validation TODO


        # Should we train the emulator? Then train I guess 
        if ((self.data_cache._newly_added_points+1) % self.learn_every == 0) and (self.counter_emulator_not_used>self.postpone_learning):
            self.log.debug("_train_emulator")
            if self.in_validation == False:
                self.write_log_step('train')
                t_start = time.time()
                self._load_data_to_GP(renormalize=True)
                self._train_emulator()
                self.timings['training'] += time.time()-t_start


        # If we are not ready yet, we cannot calculate anything
        if not self._emulator_trained:
            self.log.info("Emulator not trained yet")
            self.counter_emulator_not_used += 1
            self.in_validation = False
            self.write_log_step('not_used')
            return None, True
        
        # Check for validation_states
        if self.N_validation_states>0:
            # in first iteration, create validation states
            if self.in_validation == False:
                self.log.debug("Creating validation states")
                t_start = time.time()
                self._create_validation_states(theory, state)
                self.timings['evaluating'] += time.time()-t_start
                self.in_validation = True

            # Create precision criterion
            if loglike!=0.0:
                precision = self.precision + (self._max_loglike - loglike)*self.precision_linear + (self._max_loglike - loglike)**2 * self.precision_quadratic
                if self._max_loglike<loglike:
                    precision = self.precision
            else:
                precision = self.precision

            # Ensure that our debug loglike are not artificially veto the validation
            if self.debug:
                if (self.validation_loglikes[self.N_validation_states-2] != 0.0) and (self.validation_loglikes[self.N_validation_states-1] == 0.0):
                    precision = 9999999999.9

            if self.in_validation == True:
                for i in range(self.N_validation_states):
                    if i>0:
                        # Check for consistency
                        
                        if self.debug:
                            if i == (self.N_validation_states-1):
                                # we are in CLASS run here, thus dont complain pls
                                self.validation_loglikes[i] = loglike
                            else:
                                if abs(loglike-self.validation_loglikes[i-1]) > precision:
                                    self.validation_loglikes[i] = loglike
                                    self.log.debug("Validation loglikes are not consistent!")                            
                                    self.in_validation = False
                                    self.counter_emulator_not_used += 1
                                    self.write_log_step('not_used')
                                    self.validation_loglikes = np.zeros(self.N_validation_states)
                                    return None, True
                        else:
                            if abs(loglike-self.validation_loglikes[i-1]) > precision:
                                self.validation_loglikes[i] = loglike
                                self.log.debug("Validation loglikes are not consistent!")                            
                                self.in_validation = False
                                self.counter_emulator_not_used += 1
                                self.write_log_step('not_used')
                                self.validation_loglikes = np.zeros(self.N_validation_states)
                                return None, True
                            
                    if self.validation_loglikes[i] == 0.0:
                        self.validation_loglikes[i]=loglike
                        #self.log.info("Validation loglikes:" )
                        #self.log.info(self.validation_loglikes)
                        break
                
                # count the remaining states which are to be computed
                count = 0
                for i in range(self.N_validation_states):
                    if self.validation_loglikes[i] == 0.0:
                        count += 1

                if count == 0:
                    self.log.debug("Validation loglikes are consistent!")
                elif count == 1: # THis is only debug. Remove later
                    if self.debug:
                        self.log.debug("Validate with CLASS!")
                        self.in_validation = True
                        return None, False
                    else:
                        self.log.debug("Validation not complete! Rerun with different seed")
                        index = self.N_validation_states-count
                        self.in_validation = True
                        return copy.deepcopy(self.validation_states[index][theory]), False
                else:
                    self.log.debug("Validation not complete! Rerun with different seed")
                    index = self.N_validation_states-count
                    self.in_validation = True
                    return copy.deepcopy(self.validation_states[index][theory]), False
        
        self.in_validation = False

        t_start = time.time()

        # Now we can emulate the state and evaluate the accuracy
        data_in = np.array([[value for key,value in state['params'].items()]])
        for name, GP in self.predictors[theory].items():
            pred, unc = GP._predict(data_in)

            # Add the prediction to the state
            for key, value in self.state[theory].items():
                if key == name:
                    #self.log.info("Setting %s to %s" % (key, pred))
                    self.state[theory][key] = pred
                if type(value) == dict:
                    for k,v in value.items():
                        if k == name:
                            #self.log.info("Setting %s to %s" % (k, pred))
                            self.state[theory][key][k] = pred
        
        self.timings['evaluating'] += time.time()-t_start

        self.state[theory]['params'] = state['params']

        self.counter_emulator_used += 1

        if self.timer:
            self.timer.increment(self.log)
        self.write_log_step('used')

        self.validation_loglikes = np.zeros(self.N_validation_states)

        self.last_evaluated_state[theory] = copy.deepcopy(self.state[theory])
        return self.state[theory], True

    # This is the function that is called by the sampler
    def _evaluate(self):

        return False
    
    # This function trains a new set of emulators 
    def _train_emulator(self):

        for theory in self.theories:
            for name, GP in self.predictors[theory].items():
                # Create emulator by creating (if necessary) a PCA and train a GP
                GP.create_emulator()

        # reset the data cache
        self.data_cache._newly_added_points = 0
        self._emulator_trained = True

        return True
    
    # This function loads data to the GPs and potentially creates PCAs
    def _load_data_to_GP(self, renormalize=True):
        any_pca_created = False
        for theory in self.theories:
            for name, GP in self.predictors[theory].items():
                # Get the data from the cache
                if self._in_burnin_phase:
                    N = self._N_burnin
                else:
                    N = self.data_cache._size()
                data = self.data_cache.get_data(theory, keys = ['params',name,'loglike'], N=N)

                # Load the data into the GP
                pca_created = GP.load_training_data(data['params'],data[name],data['loglike'],renormalize=renormalize)

                if pca_created:
                    any_pca_created = True

        return any_pca_created


    # Set the required parameters for each theory code
    def _set_theories(self, theory):
        self.theories.append(theory)
        #self.log.info('theory')
        #self.log.info(theory)
        return False
    
    # Set the required parameters for each theory code
    def _set_must_provide(self, must_provide, theory):
        self.must_provide[theory] = {}

        for element in must_provide:
            if element.options is None:
                dim = 1
                self.must_provide[theory][element.name] = dim
            elif type(element.options)==dict:
                if element.name in self.must_provide[theory].keys():
                    for key in element.options:
                        if (type(element.options[key])==int) or (type(element.options[key])==np.int64):
                            dim = element.options[key]
                        else:
                            dim = len(element.options[key])
                        if key in self.must_provide[theory][element.name].keys():
                            if key in ['tt','te','ee','pp']:
                                if dim > self.must_provide[theory][element.name][key]:
                                    self.must_provide[theory][element.name][key] = dim
                            else:
                                self.must_provide[theory][element.name][key] += dim
                        else:
                            self.must_provide[theory][element.name][key] = dim
                else:
                    self.must_provide[theory][element.name] = {}
                    for key in element.options:
                        if type(element.options[key])==int:
                            dim = element.options[key]
                        else:
                            dim = len(element.options[key])
                        if key in self.must_provide[theory][element.name].keys():
                            if key in ['tt','te','ee','pp']:
                                if dim > self.must_provide[theory][element.name][key]:
                                    self.must_provide[theory][element.name][key] = dim
                            else:
                                self.must_provide[theory][element.name][key] += dim
                        else:
                            self.must_provide[theory][element.name][key] = dim

        # EBS work in the way that they calc all cls to the highest demanded ell
        if 'Cl' in self.must_provide[theory].keys():
            max_ell = 0
            for key in self.must_provide[theory]['Cl'].keys():
                if key in ['tt','te','ee','pp']:
                    if self.must_provide[theory]['Cl'][key] > max_ell:
                        max_ell = self.must_provide[theory]['Cl'][key]

            for key in self.must_provide[theory]['Cl'].keys():
                if key in ['tt','te','ee','pp']:
                    self.must_provide[theory]['Cl'][key] = max_ell

        return False

    def get_must_provide(self):
        return self.must_provide

    # This function adds a new state to the emulator
    def add_state(self, state, loglike):

        # update min loglike
        if (loglike > self._max_loglike) and (loglike != 0.0):
            self._max_loglike = loglike
            self._last_loglike_update = self.counter_emulator_not_used+self.counter_emulator_used

        # dont add very low likelihood points. Possibly -inf 
        if loglike < -1.e+20:
            return False


        theory_states = {}
        theory_name = []
        likelihood_states = {}

        for name, sub_state in state.items():
            if sub_state[0] is None:
                theory_states[name] = sub_state[1]

                # check whether the state was evaluated just before before
                if name in self.last_evaluated_state.keys():
                    
                    for key,val in self.last_evaluated_state[name]['params'].items():
                        if abs(sub_state[1]['params'][key]/val-1.0) <1.e-7: # This is a bit arbitrary nad not really safe
                            #self.log.info("State was predicted before!")
                            return False
                        else:
                            continue

                theory_name.append(name)
            else:
                likelihood_states[name] = sub_state[1]

        # Initialize if not already done so
        if not self.is_initialized:
            self.initialize(theory_states)

        t_start = time.time()

        cs_theory = self._condense_data(theory_states)
        cs_likelihood = self._condense_data(likelihood_states)

        # Initialize the data cache if not already done so
        self.log.debug("Try adding state to emulator")
        if self.data_cache.initialized is False:
            self.data_cache.initialize(cs_theory,loglike)
            added = True
        else:
            added = self.data_cache.add_data(cs_theory,loglike)

        if added:
            #self.log.info('ADDED STATE!!!')
            #self.log.info(state)
            self.write_log_step('added', loglike)
        
        ## Initialize the PCA cache if not already done so
        if self.pca_cache.initialized is False:
            self.pca_cache.initialize({**cs_theory, **cs_likelihood},loglike,theory_name)
        else:
            self.pca_cache.add_data({**cs_theory, **cs_likelihood},loglike,theory_name)

        self.timings['add_data'] += time.time()-t_start

        # Update the GP if a point was added and the emulator is actually trained:
        if added and self._emulator_trained:
            t_start = time.time()
            self._update_GP()
            self.timings['training'] += time.time()-t_start

        return True
    

    # function to update the GP without fitting the kernel. Also without renew the normalization
    def _update_GP(self):
        
        # Load the data into the GP
        self.log.debug("Loading data to GP")

        # Load the data into the GP
        self._load_data_to_GP(renormalize=False)

        # Update the GP
        self.log.debug("Updating GP")
        for theory in self.theories:
            for name, GP in self.predictors[theory].items():
                GP.update_emulator()

    # condense the data to only the required quantities
    # TODO: [SG] is this the best way to do this?
    def _condense_data(self, state):
        cs = copy.deepcopy(state)

        for theory in cs.keys():
            if 'derived' in cs[theory].keys(): del cs[theory]['derived'] 
            if 'dependency_params' in cs[theory].keys(): del cs[theory]['dependency_params'] 

            # Check if we have any extra derived quantities
            if 'derived_extra' in cs[theory].keys():
                for quant in list(cs[theory]['derived_extra'].keys()):
                    cs[theory][quant] = cs[theory]['derived_extra'][quant]

            if 'derived_extra' in cs[theory].keys(): del cs[theory]['derived_extra'] 

            # Check if we have any extra parameters
            if 'derived_extra' in cs[theory].keys():
                for quant in list(cs[theory].keys()):
                    if quant not in list(self.must_provide[theory].keys())+['derived_extra','params']:
                        del cs[theory][quant]

            # Flatten for nested dictionaries and merge parameters
            for quant in list(cs[theory].keys()):
                if quant == 'params':
                    cs[theory]['params'] = [cs[theory]['params'][key] for key in cs[theory]['params'].keys()]
                if type(cs[theory][quant]) is dict:
                    for key in list(cs[theory][quant].keys()):
                        if key in self.must_provide[theory][quant].keys():
                            cs[theory][key] = cs[theory][quant][key]
                    del cs[theory][quant]

        return cs
    

# This class is the gaussian process emulator
class PCA_GPEmulator(CobayaComponent):
    r"""
    PCA_GPEmulator.
    """

    _at_resume_prefer_new = []
    _at_resume_prefer_old = []
    _prior_rejections: int = 0
    file_base_name = 'emulator'

    def __init__(self, *args, **kwargs):
        self.name = 'name' if 'name' not in kwargs else kwargs['name']
        self.set_logger("PCA_GP_Emulator_"+self.name)
        self.out_dim = kwargs['out_dim']
        self.in_dim = kwargs['in_dim']

        self.debug= kwargs['debug']

        self.testset_fraction = kwargs['testset_fraction']

        self.pca_cache = kwargs['pca_cache']

        self.N_pca_components = kwargs['N_pca_components'] 
        self.N_pca_components_default = kwargs['N_pca_components_default'] 

        self._determine_n_pca()

        self._out_means = np.zeros(self.out_dim)
        self._out_stds = np.zeros(self.out_dim)

        self._out_means_pca = np.zeros(self.n_pca)
        self._out_stds_pca = np.zeros(self.n_pca)

        self._in_means = np.zeros(self.in_dim)
        self._in_stds = np.zeros(self.in_dim)

        # create a mask of parameters that are used for training the GP. Unrelevant parameters are set to zero
        if self.n_pca is not None:
            self._in_mask = np.ones((self.n_pca,self.in_dim), dtype=bool)
        else:
            self._in_mask = np.ones((self.out_dim,self.in_dim), dtype=bool)

        # array with indices of the parameters that are used for training the GP
        self._in_mask_indices = []
        if self.n_pca is not None:
            for i in range(self.n_pca):
                self._in_mask_indices.append(np.arange(self.in_dim))
        else:
            for i in range(self.out_dim):
                self._in_mask_indices.append(np.arange(self.in_dim))

        
        # PCA counter and parameter when to update the PCA
        self._pca_counter = 0
        self._pca_update = kwargs['_pca_update']

        self._gp_fit_size = kwargs['_gp_fit_size']

        self._gp_fitting_mode = 'random' # 'random' or 'bestfit' How to choose the data for the fitting process. Either sample randomly or choose best loglike values

        self.data_in_add = None # data that is added to the training data
        self.data_out_add = None # data that is added to the training data
        


        self._pca = None
        self._singular_values = None
        self._data_out_pca = None
        self._data_out_pca_add = None
        self._data_out_pca_fit = None

        self.set_timing_on(True)

        self._kernels = None
        self._gps = None

        # precision parameters regarding the GP
        self._theta_boundary_scale = 3.0 # how far to go in each direction when searching for the best theta
        self._N_restarts_initial = 40 if 'gp_initial_minimization_states' not in kwargs else kwargs['gp_initial_minimization_states'] # how many random restarts to do for the initial theta
        self._N_restarts = 5 if 'gp_minimization_states' not in kwargs else kwargs['gp_minimization_states'] # how many random restarts to do for the theta after the initial one

        self._use_reduced_input = False # whether to use the reduced input for the GP





    def _determine_n_pca(self):
        if self.out_dim == 1:
            self.n_pca = None
        elif self.out_dim > 1000:
            # some handwaving here. This is not really tested TODO: test this
            if self.name in self.N_pca_components.keys():
                self.n_pca = self.N_pca_components[self.name]
            else:
                self.n_pca = self.N_pca_components_default
        else:
            self.n_pca = None#self.out_dim

        return True
    
    def load_training_data(self, data_in, data_out, loglike, renormalize=True):
        self.log.debug("Loading data")
        self.data_in = data_in
        self.data_out = data_out
        self.loglike = loglike
        if len(self.data_out.shape)==1:
            self.data_out = np.array(self.data_out).astype('float').reshape(-1,1) # this weird shape is important otherwise it breaks for some reason 
        self._normalize_training_data(renormalize)


        if self.debug and renormalize:
            # Calculate the distance matrix of data_in
            self._data_in_dist = np.zeros((self.data_in.shape[0],self.data_in.shape[0]))
            for i in range(self.data_in.shape[0]):
                for j in range(self.data_in.shape[0]):
                    self._data_in_dist[i,j] = np.sqrt(np.sum((self.data_in[i,:]-self.data_in[j,:])**2))
            
            # Plot the distance matrix
            plt.figure()
            plt.title("Data in distance matrix")
            plt.imshow(self._data_in_dist)
            plt.colorbar()
            plt.savefig('./plots/data_in_dist.png')
            plt.close()

            # Select closest neighbours
            self._closest_neighbor = np.sort(self._data_in_dist, axis=1)[:,1]

            # Histogram of closest neighbours
            plt.figure()
            plt.title("Histogram of closest neighbours")
            plt.hist(self._closest_neighbor, bins=int(len(self._closest_neighbor)/2))
            plt.savefig('./plots/hist_closest_neighbor.png')
            plt.close()

        # a new PCA is required every new round
        pca_created = self._create_pca(renormalize)
        return pca_created
    
    def _normalize_training_data(self, renormalize=True):
        self.log.debug("Normalizing data")
        if renormalize:
            self._in_means = np.mean(self.data_in, axis=0)
            self._in_stds = np.std(self.data_in, axis=0)
            self._out_means = np.mean(self.data_out, axis=0)
            self._out_stds = np.std(self.data_out, axis=0)
        

            # check for zero stds
            if type(self._out_stds)==np.float64:
                if self._out_stds==0:
                    self._out_stds = 1
            else:
                self._out_stds[self._out_stds==0] = 1
            if type(self._in_stds)==np.float64:
                if self._in_stds==0:
                    self._in_stds = 1
            else:
                self._in_stds[self._in_stds==0] = 1

        if self.debug and renormalize:
            # plot the data
            if self.name in ['tt','te','ee','pp','angular_diameter_distance','Hubble']:
                for i in range(len(self.data_in[0])):
                    fig,ax = plt.subplots(figsize=(10,5))
                    for j in range(len(self.data_out)):
                        if self.name in ['tt','te','ee','pp']:
                            ax.plot(np.arange(len(self.data_out[0])), np.arange(len(self.data_out[0]))*np.arange(len(self.data_out[0]))*self.data_out[j])
                    ax.set_xlabel('Input')
                    ax.set_ylabel(self.name)
                    if self.name in ['tt','te','ee','pp']:
                        ax.set_xscale('log')
                    fig.savefig('./plots/data_'+self.name+'_'+str(i)+'.png')
            else:
                for i in range(len(self.data_in[0])):
                    fig,ax = plt.subplots(figsize=(10,5))
                    ax.plot(self.data_in[:,i], self.data_out, 'o')
                    ax.set_xlabel('Input')
                    ax.set_ylabel(self.name)
                    fig.savefig('./plots/data_'+self.name+'_'+str(i)+'.png')
            plt.figure().clear()
            plt.close('all')
            plt.close()
            plt.cla()
            plt.clf()
            gc.collect()

        self.data_in = (self.data_in - self._in_means)/self._in_stds
        self.data_out = (self.data_out - self._out_means)/self._out_stds

        if self.debug and renormalize:
            # plot the data
            if self.name in ['tt','te','ee','pp','angular_diameter_distance','Hubble']:
                for i in range(len(self.data_in[0])):
                    fig,ax = plt.subplots(figsize=(10,5))
                    for j in range(len(self.data_out)):
                        ax.plot(np.arange(len(self.data_out[0])), self.data_out[j])
                    ax.set_xlabel('Input')
                    ax.set_ylabel(self.name)
                    fig.savefig('./plots/data_'+self.name+'_'+str(i)+'_norm.png')
            else:
                for i in range(len(self.data_in[0])):
                    fig,ax = plt.subplots(figsize=(10,5))
                    ax.plot(self.data_in[:,i], self.data_out, 'o')
                    ax.set_xlabel('Input')
                    ax.set_ylabel(self.name)
                    fig.savefig('./plots/data_'+self.name+'_'+str(i)+'_norm.png')

            plt.figure().clear()
            plt.close('all')
            plt.close()
            plt.cla()
            plt.clf()
            gc.collect()

        return True
    
    def _create_pca(self, renormalize=True):

        if self.n_pca is None:
            return False
        else:
            if renormalize and ((self._pca_counter%self._pca_update)==0):
                #self.log.info("Creating PCA")
                self._pca = PCA(n_components=self.n_pca)

                data_pca_cache = self.pca_cache.get_data([self.name])[self.name]

                # normalize the data and remove nans from liklihoods
                data_pca_cache = np.array([(_[0]-self._out_means)/self._out_stds for _ in data_pca_cache if not np.isnan(_).any()])
                #self.log.info('data_pca_cache')
                #self.log.info(data_pca_cache)
                #self.log.info(data_pca_cache.shape)
                #self.log.info('self.data_out')
                #self.log.info(self.data_out)
                #self.log.info(self.data_out.shape)

                self._pca.fit(data_pca_cache)

                # a new PCA was created. We cannot be certain on the components and their dependencies. Thus, we need to recompute the kernels on all input dimensiopns
                self._use_reduced_input = False

                # update counter
                self._pca_counter += 1

                pca_created = True
            else:
                pca_created = False

            self._data_out_pca = self._pca.transform(self.data_out)

            # retranform the data to check
            reconst = self._pca.inverse_transform(self._data_out_pca)

            # get the residuals
            residuals = self.data_out - reconst

            # get the variance of the residuals and unnormalize
            self._pca_residual_std = np.std(residuals, axis=0)*self._out_stds

            #self.log.info('residual variance')
            #self.log.info(self._pca_residual_std)
            #self.log.info(self._pca_residual_std.shape)
            

            # normalize the pca compoenents
            self._out_means_pca = np.mean(self._data_out_pca, axis=0)
            self._out_stds_pca = np.std(self._data_out_pca, axis=0)

            self._data_out_pca = (self._data_out_pca - self._out_means_pca)/self._out_stds_pca

            self._singular_values = self._pca.singular_values_



            return pca_created


    def _create_kernel(self, theta_boundary_scale= 3.0, update_mask = False):
        #self.log.info("Creating kernel")

        # create kernel
        a = ([0.01,5.0],[0.01,5.0],[0.01,5.0],[0.01,5.0],[0.01,5.0],[0.01,5.0])
        # if we have already some working kernels, we can help ourself by constraing the new ones to be similar
        if self._gps is None:
            if self.n_pca is not None:
                #self._kernels = [ConstantKernel() * RBF() for i in range(self.n_pca)]
                self._kernels = [ConstantKernel() * RBF(np.ones(self.in_dim)) for i in range(self.n_pca)]
            else:
                #self._kernels = [ConstantKernel() * RBF() for i in range(self.out_dim)]
                self._kernels = [ConstantKernel() * RBF(np.ones(self.in_dim)) for i in range(self.out_dim)]
        else:
            thetas= []   
            bounds = []
            for i,GP in enumerate(self._gps):

                thetas.append([GP.kernel_.theta[0]])
                bounds.append([[max(np.exp(GP.kernel_.theta[0]-theta_boundary_scale),np.exp(-11.0)), min(np.exp(GP.kernel_.theta[0]+theta_boundary_scale),np.exp(11.0))]])
                

                # veto parameters that are not relevant for the kernel
                _ = 0
                for j in range(self.in_dim):
                    if self._use_reduced_input:
                        if not (self._in_mask[i,j] == False):
                            if GP.kernel_.theta[_+1]>10.9:
                                self._in_mask[i,j] = False  
                                _+=1              
                            else:
                                thetas[i].append(GP.kernel_.theta[_+1])
                                bounds[i].append([max(np.exp(GP.kernel_.theta[_+1]-theta_boundary_scale),np.exp(-11.0)), min(np.exp(GP.kernel_.theta[_+1]+theta_boundary_scale),np.exp(11.0))])
                                _+=1              
                    else:
                        # if the previous kernel had the same number of parameters, we can use the same
                        if len(GP.kernel_.theta)==self.in_dim+1:
                            thetas[i].append(GP.kernel_.theta[j+1])
                            bounds[i].append([max(np.exp(GP.kernel_.theta[j+1]-theta_boundary_scale),np.exp(-11.0)), min(np.exp(GP.kernel_.theta[j+1]+theta_boundary_scale),np.exp(11.0))])
                        else:
                            thetas[i] = np.append(thetas[i],1.0)
                            bounds[i] = np.append(bounds[i],np.array([[np.exp(-11.0), np.exp(11.0)]]), axis=0)
                        self._in_mask[i,j] = True

                # transform the input mask to indices
                self._in_mask_indices[i] = np.where(self._in_mask[i])[0]


                
                #self.log.info('masked in')
                #self.log.info(self._use_reduced_input)
                #self.log.info(self._in_mask[i])
                #self.log.info(sum(self._in_mask[i]))
                #self.log.info(self._in_mask_indices[i])
                #self.log.info('pre thetas')
                #self.log.info(GP.kernel_.theta)
                #self.log.info(thetas[i])
                #self.log.info(bounds[i])



            #self.log.info('self.n_pca')
            #self.log.info(self.n_pca)
            #self.log.info('self.out_dim')
            #self.log.info(self.out_dim)

            # special case at 0.0 due to some rounding errors
            if theta_boundary_scale == 0.0:
                if self.n_pca is not None:
                    for i in range(self.n_pca):
                        for j in range(len(thetas[i])):
                            bounds[i][j] = [np.exp(thetas[i][j]),np.exp(thetas[i][j])]
                else:
                    for i in range(self.out_dim):
                        for j in range(len(thetas[i])):
                            bounds[i][j] = [np.exp(thetas[i][j]),np.exp(thetas[i][j])]
                    

            if self.n_pca is not None:
                self._kernels = [ConstantKernel(constant_value=np.exp(thetas[i][0]), constant_value_bounds=tuple(bounds[i][0])) * RBF(np.exp(thetas[i][1:]),length_scale_bounds=tuple(bounds[i][1:])) for i in range(self.n_pca)]
            else:
                self._kernels = [ConstantKernel(constant_value=np.exp(thetas[i][0]), constant_value_bounds=tuple(bounds[i][0])) * RBF(np.exp(thetas[i][1:]),length_scale_bounds=tuple(bounds[i][1:])) for i in range(self.out_dim)]
        

        #self.log.info('kernels done')
        
        return True

    def _create_gp(self):

        if self.timer:
            self.timer.start()

        #self.log.info("Creating GP")

        # create GP if it is not already created
        if self._gps is None:
            if self.n_pca is not None:
                #self.log.info('self.n_pca')
                #self.log.info(self.n_pca)
                self._gps = [GaussianProcessRegressor(kernel=self._kernels[i], n_restarts_optimizer=self._N_restarts_initial, alpha=1.e-8) for i in range(self.n_pca)]

                if self.debug:
                    #do some GP input plots for PCA
                    for i in range(self.n_pca):
                        for j in range(self.in_dim):
                            fig,ax = plt.subplots()
                            ax.scatter(self.data_in[:,j],self._data_out_pca[:,i])
                            ax.set_xlabel('input')
                            ax.set_ylabel('PCA component %d' % i)
                            fig.savefig('./plots_pca_gp/%s_PCA_%d_%d.png' % (self.name,i,j))

                    plt.figure().clear()
                    plt.close('all')
                    plt.close()
                    plt.cla()
                    plt.clf()
                    gc.collect()



            else:

                self._gps = [GaussianProcessRegressor(kernel=self._kernels[i], n_restarts_optimizer=self._N_restarts_initial, alpha=1.e-8) for i in range(self.out_dim)]
        else:
            if self.n_pca is not None:
                self._gps = [GaussianProcessRegressor(kernel=self._kernels[i], n_restarts_optimizer=self._N_restarts, alpha=1.e-8) for i in range(self.n_pca)]


                if self.debug:
                    #do some GP input plots for PCA
                    for i in range(self.n_pca):
                        for j in range(self.in_dim):
                            fig,ax = plt.subplots()
                            ax.scatter(self.data_in[:,j],self._data_out_pca[:,i])
                            ax.set_xlabel('input')
                            ax.set_ylabel('PCA component %d' % i)
                            fig.savefig('./plots_pca_gp/%s_PCA_%d_%d.png' % (self.name,i,j))

                    plt.figure().clear()
                    plt.close('all')
                    plt.close()
                    plt.cla()
                    plt.clf()
                    gc.collect()

            else:

                self._gps = [GaussianProcessRegressor(kernel=self._kernels[i], n_restarts_optimizer=self._N_restarts, alpha=1.e-8) for i in range(self.out_dim)]


        # here we check the size of the dataset. If it is larger than we require to fit the kernels,
        # we split the dataset into a fitting and additional dataset. The fitting dataset is used to fit the GP
        # and the additional dataset is used to feed the GP with a fixed Kernel.

        if self.data_in.shape[0] > self._gp_fit_size:
            
            if self._gp_fitting_mode == 'random':

                # indices for fitting 
                indices = np.arange(len(self.data_in))
                np.random.shuffle(indices)
                fit_indices = indices[:self._gp_fit_size]
                add_indices = indices[self._gp_fit_size:]

                self.data_in_add = self.data_in[add_indices]
                self.data_out_add = self.data_out[add_indices]

                self.data_in_fit = self.data_in[fit_indices]
                self.data_out_fit = self.data_out[fit_indices]

                if self.n_pca is not None:
                    self._data_out_pca_add = self._data_out_pca[add_indices]
                    self._data_out_pca_fit = self._data_out_pca[fit_indices]




        else:
            self.data_in_fit = self.data_in
            self.data_out_fit = self.data_out
            self._data_out_pca_fit = self._data_out_pca


        self.train_indices, self.test_indices = train_test_split(np.arange(len(self.data_in_fit)), test_size=self.testset_fraction, random_state=42)

        #self.log.info("Test set size: %d" % len(self.test_indices))
        #self.log.info("Train set size: %d" % len(self.train_indices))


        # Train the GP
        #self.log.info("Training GP")
        start = time.time()
        for i,GP in enumerate(self._gps):
            #self.log.info("Training GP %d" % i)
            #self.log.info(self._in_mask_indices[i])
            #self.log.info(type(self.data_in))
            #self.log.info(type(self.train_indices))
            #self.log.info(self.train_indices)
            #self.log.info(self._in_mask_indices[i])
            #self.log.info(self.data_in.shape)
            #self.log.info(self.data_in[self.train_indices].shape)
            #self.log.info(self.data_in[np.ix_(self.train_indices, self._in_mask_indices[i])].shape)
            
            
            if self.n_pca is not None:
                GP.fit(self.data_in_fit[np.ix_(self.train_indices, self._in_mask_indices[i])], self._data_out_pca_fit[self.train_indices,i])
                score_train = GP.score(self.data_in_fit[np.ix_(self.train_indices, self._in_mask_indices[i])], self._data_out_pca_fit[self.train_indices,i])
                score_test = GP.score(self.data_in_fit[np.ix_(self.test_indices, self._in_mask_indices[i])], self._data_out_pca_fit[self.test_indices,i])
                self.log.debug("GP score train: %f" % score_train)
                self.log.debug("GP score test: %f" % score_test)
            else:
                if len(self._gps) == 1:
                    GP.fit(self.data_in_fit[np.ix_(self.train_indices, self._in_mask_indices[i])], self.data_out_fit[self.train_indices])
                    score_train = GP.score(self.data_in_fit[np.ix_(self.train_indices, self._in_mask_indices[i])], self.data_out_fit[self.train_indices,i])
                    score_test = GP.score(self.data_in_fit[np.ix_(self.test_indices, self._in_mask_indices[i])], self.data_out_fit[self.test_indices,i])
                    self.log.debug("GP score train: %f" % score_train)
                    self.log.debug("GP score test: %f" % score_test)
                else:
                    GP.fit(self.data_in_fit[np.ix_(self.train_indices, self._in_mask_indices[i])], self.data_out_fit[self.train_indices,i])
                    score_train = GP.score(self.data_in_fit[np.ix_(self.train_indices, self._in_mask_indices[i])], self.data_out_fit[self.train_indices,i])
                    score_test = GP.score(self.data_in_fit[np.ix_(self.test_indices, self._in_mask_indices[i])], self.data_out_fit[self.test_indices,i])
                    self.log.debug("GP score train: %f" % score_train)
                    self.log.debug("GP score test: %f" % score_test)
        


        #self.log.info("Time to fit kernel: %f" % (time.time() - start))


        # Once the kernel are fitted we can calculate the full GP including the additional dataset.
        # This is done by setting the kernel to fixed and using the additional dataset as input.
        # The output is then used to train the GP.
        if self.data_in.shape[0] > self._gp_fit_size:
            self.update_emulator()


        # use reduced input for the next GP
        self._use_reduced_input = True

        #if self.debug:
        #    for i,GP in enumerate(self._gps):
        #        self.log.info('POST GP')
        #        self.log.info(GP.kernel_.get_params())
        #        self.log.info(GP.kernel_.theta)



        if self.debug:


            # TEST SET!!!!!!!!!!!!!!!
            # Test the GP by predicting the test set
            self.log.info("Testing GP")
            if self.n_pca is not None:
                self._data_out_pca_test = np.zeros((len(self.test_indices), self.n_pca))
                self._data_out_pca_test_std = np.zeros((len(self.test_indices), self.n_pca))
                for i,GP in enumerate(self._gps):
                    self._data_out_pca_test[:,i], self._data_out_pca_test_std[:,i] = GP.predict(self.data_in_fit[np.ix_(self.test_indices, self._in_mask_indices[i])], return_std=True)
                    self._data_out_pca_test[:,i] = self._data_out_pca_test[:,i]*self._out_stds_pca[i]+self._out_means_pca[i]
                    self._data_out_pca_test_std[:,i] = self._data_out_pca_test_std[:,i]*self._out_stds_pca[i]
            else:
                self._data_out_test = np.zeros((len(self.test_indices), self.out_dim))
                self._data_out_test_std = np.zeros((len(self.test_indices), self.out_dim))
                for i,GP in enumerate(self._gps):
                    self._data_out_test[:,i], self._data_out_test_std[:,i] = GP.predict(self.data_in_fit[np.ix_(self.test_indices, self._in_mask_indices[i])], return_std=True)
            
            # Plot the test set
            if self.n_pca is not None:
                for i in range(len(self._data_out_pca_test[0])):
                    fig,ax = plt.subplots(figsize=(10,5))
                    ax.errorbar(self._data_out_pca_fit[self.test_indices,i], self._data_out_pca_fit[self.test_indices,i]-(self._data_out_pca_test[:,i]-self._out_means_pca[i])/self._out_stds_pca[i], yerr=(self._data_out_pca_test_std[:,i]-self._out_means_pca[i])/self._out_stds_pca[i], fmt='o', label='Predicted')
                    ax.set_xlabel('true')
                    ax.set_ylabel('predicted - true')
                    ax.set_title(self.name)
                    ax.grid(True)
                    ax.legend()
                    fig.savefig('./plots/test_'+self.name+'_'+str(i)+'_gp.png')
            else:
                for i in range(len(self._data_out_test[0])):
                    fig,ax = plt.subplots(figsize=(10,5))
                    ax.errorbar(self.data_out_fit[self.test_indices,i], self.data_out_fit[self.test_indices,i]-self._data_out_test[:,i], yerr=self._data_out_test_std[:,i], fmt='o', label='Predicted')
                    ax.set_xlabel('true')
                    ax.set_ylabel('predicted - true')
                    ax.set_title(self.name)
                    ax.grid(True)
                    ax.legend()
                    fig.savefig('./plots/test_'+self.name+'_'+str(i)+'_gp.png')


            # TRAIN SET !!!!!!!!!!!!!!!
            #self.log.info("Testing GP")
            if self.n_pca is not None:
                self._data_out_pca_train = np.zeros((len(self.train_indices), self.n_pca))
                self._data_out_pca_train_std = np.zeros((len(self.train_indices), self.n_pca))
                for i,GP in enumerate(self._gps):
                    self._data_out_pca_train[:,i], self._data_out_pca_train_std[:,i] = GP.predict(self.data_in_fit[np.ix_(self.train_indices, self._in_mask_indices[i])], return_std=True)
                    self._data_out_pca_train[:,i] = self._data_out_pca_train[:,i]*self._out_stds_pca[i]+self._out_means_pca[i]
                    self._data_out_pca_train_std[:,i] = self._data_out_pca_train_std[:,i]*self._out_stds_pca[i]
            else:
                self._data_out_train = np.zeros((len(self.train_indices), self.out_dim))
                self._data_out_train_std = np.zeros((len(self.train_indices), self.out_dim))
                for i,GP in enumerate(self._gps):
                    self._data_out_train[:,i], self._data_out_train_std[:,i] = GP.predict(self.data_in_fit[np.ix_(self.train_indices, self._in_mask_indices[i])], return_std=True)
            
            # Plot the train set
            if self.n_pca is not None:
                for i in range(len(self._data_out_pca_train[0])):
                    fig,ax = plt.subplots(figsize=(10,5))
                    ax.errorbar(self._data_out_pca_fit[self.train_indices,i], self._data_out_pca_fit[self.train_indices,i]-(self._data_out_pca_train[:,i]-self._out_means_pca[i])/self._out_stds_pca[i], yerr=(self._data_out_pca_train_std[:,i]-self._out_means_pca[i])/self._out_stds_pca[i], fmt='o', label='Predicted')
                    ax.set_xlabel('true')
                    ax.set_ylabel('predicted - true')
                    ax.set_title(self.name)
                    ax.grid(True)
                    ax.legend()
                    fig.savefig('./plots/train_'+self.name+'_'+str(i)+'_gp.png')
            else:
                for i in range(len(self._data_out_test[0])):
                    fig,ax = plt.subplots(figsize=(10,5))
                    ax.errorbar(self.data_out_fit[self.train_indices,i], self.data_out_fit[self.train_indices,i]-self._data_out_train[:,i], yerr=self._data_out_train_std[:,i], fmt='o', label='Predicted')
                    ax.set_xlabel('true')
                    ax.set_ylabel('predicted - true')
                    ax.set_title(self.name)
                    ax.grid(True)
                    ax.legend()
                    fig.savefig('./plots/train_'+self.name+'_'+str(i)+'_gp.png')

            plt.figure().clear()
            plt.close('all')
            plt.close()
            plt.cla()
            plt.clf()
            gc.collect()

            # Plot the PCA untransformed data
            if self.n_pca is not None:

                original_data = self.data_out_fit[self.test_indices] * self._out_stds + self._out_means
                test_data = np.zeros(original_data.shape)
                test_unc = np.zeros(original_data.shape)

                #self.log.info('original_data.shape')
                #self.log.info(original_data.shape)
                #self.log.info('self.data_out_fit')
                #self.log.info(self.data_out_fit.shape)
                #self.log.info('self._out_stds')
                #self.log.info(self._out_stds.shape)
                #self.log.info('self._out_means')
                #self.log.info(self._out_means.shape)
                #self.log.info('test_data.shape')
                #self.log.info(test_data.shape)
                #self.log.info('test_unc.shape')
                #self.log.info(test_unc.shape)

                test_data = self._pca.inverse_transform(self._data_out_pca_test)


                for i in range(len(self._data_out_pca_test_std[0])):
                    test_unc += abs(np.outer(self._data_out_pca_test_std[:,i], self._pca.components_[i]))

                # unstransform test data
                test_data = test_data * self._out_stds + self._out_means
                test_unc = test_unc * self._out_stds

                rel_index = np.arange(np.min([5,len(self.test_indices)]))

                index = self.test_indices[rel_index]

                for ind in rel_index:
                    fig,ax = plt.subplots(3,sharex=True,figsize=(10,10))
                    ax[2].set_xlabel('ell')
                    ax[0].set_ylabel(self.name)
                    ax[0].set_title(self.name + ' ' + str(ind))
                    ax[0].plot(np.arange(self.out_dim),np.arange(self.out_dim)*np.arange(self.out_dim)*original_data[ind], label='true')
                    ax[0].plot(np.arange(self.out_dim),np.arange(self.out_dim)*np.arange(self.out_dim)*test_data[ind], label='predicted')
                    ax[0].fill_between(np.arange(self.out_dim), np.arange(self.out_dim)*np.arange(self.out_dim)*(test_data[ind]-test_unc[ind]), np.arange(self.out_dim)*np.arange(self.out_dim)*(test_data[ind]+test_unc[ind]), alpha=0.5, label='SAMPLING uncertainty')
                    ax[0].fill_between(np.arange(self.out_dim), np.arange(self.out_dim)*np.arange(self.out_dim)*(test_data[ind]-self._pca_residual_std), np.arange(self.out_dim)*np.arange(self.out_dim)*(test_data[ind]+self._pca_residual_std),color='orange', alpha=0.5, label='PCA uncertainty')
                    ax[0].grid(True)
                    ax[0].legend()
                    ax[0].set_ylabel('l*(l+1)*C_l')
                    ax[1].plot(np.arange(self.out_dim),np.arange(self.out_dim)*np.arange(self.out_dim)*(original_data[ind]-test_data[ind]), label='difference')
                    ax[1].fill_between(np.arange(self.out_dim), np.arange(self.out_dim)*np.arange(self.out_dim)*(-test_data[ind]-test_unc[ind]+original_data[ind]), np.arange(self.out_dim)*np.arange(self.out_dim)*(-test_data[ind]+test_unc[ind]+original_data[ind]), alpha=0.5, label='SAMPLING uncertainty')
                    
                    ax[1].fill_between(np.arange(self.out_dim), np.arange(self.out_dim)*np.arange(self.out_dim)*(-test_data[ind]-self._pca_residual_std+original_data[ind]), np.arange(self.out_dim)*np.arange(self.out_dim)*(-test_data[ind]+self._pca_residual_std+original_data[ind]),color='orange' ,alpha=0.5, label='PCA uncertainty')
                    ax[1].grid(True)
                    ax[1].legend()
                    ax[1].set_ylabel('l*(l+1)*delta C_l')
                    cv = original_data[ind]/np.sqrt(np.arange(self.out_dim)+0.5)

                    ax[2].set_ylabel('delta C_l/cosmic variance')
                    ax[2].plot(np.arange(self.out_dim),(original_data[ind]-test_data[ind])/cv, label='difference')
                    ax[2].fill_between(np.arange(self.out_dim), (-test_data[ind]-test_unc[ind]+original_data[ind])/cv, (-test_data[ind]+test_unc[ind]+original_data[ind])/cv, alpha=0.5, label='SAMPLING uncertainty')
                    
                    ax[2].fill_between(np.arange(self.out_dim), (-test_data[ind]-self._pca_residual_std+original_data[ind])/cv, (-test_data[ind]+self._pca_residual_std+original_data[ind])/cv,color='orange', alpha=0.5, label='PCA uncertainty')
                    ax[2].grid(True)
                    ax[2].legend()

                    fig.savefig('./plots/test_'+self.name+'_'+str(ind)+'_gp_backtrafo.pdf')
 
                plt.figure().clear()
                plt.close('all')
                plt.close()
                plt.cla()
                plt.clf()
                gc.collect()



        if self.timer:
            self.timer.increment(self.log)
        
        return True
    
    def create_emulator(self):
        # Create the kernels if there are not allocated yet. Otherwise reuse the old ones
        self._create_kernel(self._theta_boundary_scale, update_mask = True)
        self._create_gp()
        return True

    # THis function takes the most recent kernel and fits it to all data
    def update_emulator(self):
        start = time.time()
        # create kernels again with fixed values
        self._create_kernel(0.0, update_mask=False)
        #self.log.info("Training GP on additional data")

        if self.n_pca is not None:
            self._gps = [GaussianProcessRegressor(kernel=self._kernels[i], n_restarts_optimizer=0, alpha=1.e-8) for i in range(self.n_pca)]
        else:
            self._gps = [GaussianProcessRegressor(kernel=self._kernels[i], n_restarts_optimizer=0, alpha=1.e-8) for i in range(self.out_dim)]

        for i,GP in enumerate(self._gps):
            # Train the GP on all data
            if self.n_pca is not None:
                GP.fit(self.data_in[:,self._in_mask_indices[i]], self._data_out_pca[:,i])
            else:
                GP.fit(self.data_in[:,self._in_mask_indices[i]], self.data_out[:,i])

        #self.log.info("Time to fit full GP: %f" % (time.time() - start))

        return True
    
    def _predict(self, data_in):
        # Normalize the data
        data_in = (data_in - self._in_means)/self._in_stds
        #self.log.info(self.name)
        # Predict the data
        if self.n_pca is not None:
            data_out = np.zeros(self.n_pca)
            std_out_pca = np.zeros(self.n_pca)
            std_out = np.zeros(self.out_dim)
            for i, GP in enumerate(self._gps):
                data_out[i], std_out_pca[i] = GP.predict(data_in[:,self._in_mask[i]], return_std=True)
                data_out[i] = data_out[i]*self._out_stds_pca[i] + self._out_means_pca[i]
                std_out_pca[i] = std_out_pca[i]*self._out_stds_pca[i] + self._out_means_pca[i]
                std_out += np.abs(np.outer(std_out_pca[i], self._pca.components_[i]))[0]


            data_out = self._pca.inverse_transform(data_out)
            #self.log.info(std_out.shape)
        else:
            data_out = np.zeros(self.out_dim)
            std_out = np.zeros(self.out_dim)
            for i, GP in enumerate(self._gps):
                data_out[i], std_out[i] = GP.predict(data_in[:,self._in_mask[i]], return_std=True)
                #self.log.info(data_out)
                #self.log.info(std_out)

        # Unnormalize the data
        data_out = data_out*self._out_stds + self._out_means
        std_out = std_out*self._out_stds
        return data_out, std_out

    # This function is used to sample from the emulator. The difference to the _predict function is
    # that the _predict function returns the mean and the standard deviation of the emulator. The
    # _sample function additionally samples the uncertainty of the emulator.
    def _sample(self, data_in, n_samples=1):
        # Normalize the data
        data_in = (data_in - self._in_means)/self._in_stds
        # Predict the data
        if self.n_pca is not None:
            data_out = np.zeros((n_samples, self.n_pca))
            for i, GP in enumerate(self._gps):
                data_out[:,i] = GP.sample_y(data_in[:,self._in_mask[i]], n_samples=n_samples, random_state=None)
                data_out[:,i] = data_out[:,i]*self._out_stds_pca[i] + self._out_means_pca[i]
            data_out = self._pca.inverse_transform(data_out)
        else:
            data_out = np.zeros((n_samples, self.out_dim))
            for i, GP in enumerate(self._gps):
                data_out[:,i] = GP.sample_y(data_in[:,self._in_mask[i]], n_samples=n_samples, random_state=None)

        # Unnormalize the data
        data_out = data_out*self._out_stds + self._out_means

        # Add PCA related uncertainty
        if self.n_pca is not None:
            data_out = data_out + np.outer(np.random.normal(np.zeros(n_samples)), self._pca_residual_std)

        return data_out
    
    # This function is used to sample from the emulator with the PCA uncertainty. The difference to the _predict function is
    # that the _predict function returns the mean and the standard deviation of the emulator. The
    # _sample function additionally samples the uncertainty of the emulator.
    def _sample_PCA(self, data_in, n_samples=1):
        # Normalize the data
        data_in = (data_in - self._in_means)/self._in_stds
        # Predict the data
        if self.n_pca is not None:
            data_out = np.zeros((n_samples, self.n_pca))
            for i, GP in enumerate(self._gps):
                data_out[:,i] = GP.predict(data_in[:,self._in_mask[i]])
                data_out[:,i] = data_out[:,i]*self._out_stds_pca[i] + self._out_means_pca[i]
            data_out = self._pca.inverse_transform(data_out)
        else:
            data_out = np.zeros((n_samples, self.out_dim))
            for i, GP in enumerate(self._gps):
                data_out[:,i] = GP.predict(data_in[:,self._in_mask[i]])

        # Unnormalize the data
        data_out = data_out*self._out_stds + self._out_means

        # Add PCA related uncertainty
        if self.n_pca is not None:
            data_out = data_out + np.outer(np.random.normal(np.zeros(n_samples)), self._pca_residual_std)

        return data_out
