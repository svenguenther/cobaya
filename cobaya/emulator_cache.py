# This class is storing and updating the training data for the emulator.  

# The training data is stored in a pandas dataframe.  The dataframe is saved to a file
# and loaded from a file.  The file name is stored in the class variable file_base_name.
# The file name is constructed by appending the emulator name to the file_base_name.
# The emulator name is stored in the class variable name.  The name is set in the
# yaml file.  The name is used to construct the file name.  The file name is used
# to save and load the training data.

from cobaya.component import CobayaComponent
import pandas as pd
import numpy as np
import copy 

#
# This Class generates a PCA Guassian process emulator
#
class EmulatorCache(CobayaComponent):

    _at_resume_prefer_new = ["training_size"]
    _at_resume_prefer_old = []
    _prior_rejections: int = 0
    file_base_name = 'emulator_cache'

    precision: int

    def __init__(self, *args, **kwargs):
        self.set_logger("emulator_cache")
        self.N = 200 if 'cache_size' not in kwargs else kwargs['cache_size']
        self.initialized = False

        self.theories = None

        self.dataframes = {} # a dict between theory and a dataframe

    def initialize(self,state,loglike):
        
        self._emulator_cache = None
        self._emulator_cache_ready = False
        self._emulator_cache_trained = False
        self._newly_added_points = 0

        for theory in state.keys():
            state[theory]['loglike']=loglike
            state[theory]['hash']=hash(tuple(state[theory]['params']))
            for key in state[theory].keys():
                state[theory][key] = [state[theory][key]]
            self.dataframes[theory] = pd.DataFrame(state[theory], columns=state[theory].keys())

        self.theories = list(state.keys())

        self.initialized = True

    # This function provides data from the cache
    def get_data(self, theory, keys):
        if self._size() > 0:
            # Get a random data point from the cache
            self.log.debug("Getting data from cache")
            data = {}
            for key in keys:
                data[key] = np.array([np.array(_) for _ in self.dataframes[theory][key].values])
            self._newly_added_points = 0
            return data
        else:
            self.log.error("Cache is empty. Training not possible")
    
    # This function adds data to the cache
    def add_data(self, data, loglike):
        # First check whether the cache is full
        if self._size() >= self.N:
            # Then check whether the new data is better than the worst data in the cache
            min_loglike = self.dataframes[self.theories[0]]['loglike'].min()
            self.log.debug("Min loglike in cache: {}".format(min_loglike))
            self.log.debug("New loglike: {}".format(loglike))
            if loglike > min_loglike:
                # Check whether the data point is already in the cache
                new_hash = hash(tuple(data[self.theories[0]]['params']))

                if new_hash in self.dataframes[self.theories[0]]['hash'].values:
                    self.log.debug("Data already in cache")
                    # select the data point with the same hast and replace its loglike
                    for theory in data.keys():
                        if self.dataframes[theory].loc[self.dataframes[theory]['hash'] == new_hash, 'loglike'].all() < loglike:
                            self.dataframes[theory].loc[self.dataframes[theory]['hash'] == new_hash, 'loglike'] = loglike
                    return False
                else:
                    self.log.debug("Add into in cache")
                    # Remove the worst data point from the cache and add the new data point
                    for theory in data.keys():
                        data[theory]['loglike']=loglike
                        data[theory]['hash']=hash(tuple(data[theory]['params']))
                        for key in data[theory].keys():
                            data[theory][key] = [data[theory][key]]
                        self.dataframes[theory]=pd.concat([self.dataframes[theory], pd.DataFrame(data[theory], columns=data[theory].keys())])
                        self.dataframes[theory]=self.dataframes[theory][self.dataframes[theory]['loglike'] != min_loglike]
                        self.log.debug("Adding data to cache")
                        self._newly_added_points += 1
                    return True
        else:
            # Check whether the data point is already in the cache
            new_hash = hash(tuple(data[self.theories[0]]['params']))

            if new_hash in self.dataframes[self.theories[0]]['hash'].values:
                self.log.debug("Data already in cache")
                for theory in data.keys():
                    if self.dataframes[theory].loc[self.dataframes[theory]['hash'] == new_hash, 'loglike'].all() < loglike:
                        self.dataframes[theory].loc[self.dataframes[theory]['hash'] == new_hash, 'loglike'] = loglike

                return False
            else:
                for theory in data.keys():
                    data[theory]['loglike']=loglike
                    data[theory]['hash']=hash(tuple(data[theory]['params']))
                    for key in data[theory].keys():
                        data[theory][key] = [data[theory][key]]
                    self.dataframes[theory]=pd.concat([self.dataframes[theory], pd.DataFrame(data[theory], columns=data[theory].keys())])
                    self.log.debug("Adding data to cache")
                    self._newly_added_points += 1
                return True
    
    # This function saves the cache to a file
    def _save_cache(self):
        return False
    
    # This function sets a new size for the cache
    def _set_new_cache_size(self, N):
        self.N = N
        return True

    # This function loads the cache from a file
    def _load_cache(self):
        return False
    
    # This function returns the size of the cache
    def _size(self):
        theory = list(self.dataframes.keys())[0]
        return self.dataframes[theory].shape[0] 
    





# PCA Cache
# This class is a cache for the PCA emulator.  It is used to store training data
# for the PCA emulator.  The cache is a pandas dataframe.  The dataframe is stored
# in a file.  The file name is constructed from the emulator name.  The emulator
# name is used to construct the file name.  The file name is used to save and load
# the training data.

class PCACache(CobayaComponent):

    _at_resume_prefer_new = ["training_size"]
    _at_resume_prefer_old = []
    _prior_rejections: int = 0
    file_base_name = 'emulator_pca_cache'

    precision: int

    def __init__(self, *args, **kwargs):
        self.set_logger("pca_cache")
        self.N = 200 if 'cache_size' not in kwargs else kwargs['cache_size']
        self.initialized = False

        self.theories = None

        self.dataframe = None # a dict between theory and a dataframe

    def initialize(self,state,loglike,theory_name):
            
        my_hash = 0.0
        for theory in theory_name:
            if 'params' in state[theory].keys():
                if type(state[theory]['params'][0]) is not list:
                    my_hash += hash(tuple([float(_) for _ in state[theory]['params']]))
                else:
                    my_hash += hash(tuple([float(_) for _ in state[theory]['params'][0]]))

        for key in state.keys():
            state[key]['loglike'] = loglike

        data = {my_hash: state}

        self.dataframe = pd.DataFrame.from_dict({(i,j): data[i][j] 
                           for i in data.keys() 
                           for j in data[i].keys()},
                       orient='index')

        self.theories = list(state.keys())
    
        self.initialized = True

    
    # This function returns the data from the cache
    def get_data(self, keys):
        if self.initialized:
            data = {}
            for key in keys:
                data[key] = np.array([np.array(_) for _ in self.dataframe[key].values])
            return data
        else:
            self.log.error("Cache is empty. Training not possible")

    # This function adds data to the cache
    def add_data(self, state, loglike, theory_name):
        # First check whether the cache is full

        my_hash = 0.0
        for theory in theory_name:
            if 'params' in state[theory].keys():
                if type(state[theory]['params'][0]) is not list:
                    my_hash += hash(tuple([float(_) for _ in state[theory]['params']]))
                else:
                    my_hash += hash(tuple([float(_) for _ in state[theory]['params'][0]]))

        for key in state.keys():
            state[key]['loglike'] = loglike

        data = {my_hash: state}

        new_dataframe = pd.DataFrame.from_dict({(i,j): data[i][j] for i in data.keys() for j in data[i].keys()}, orient='index')

        min_loglike = self.dataframe['loglike'].min()

        #self.log.info(self.dataframe)
        if self._size() >= self.N:
            # Then select a random data point from the cache and remove it
            if my_hash in self.dataframe.index:
                #self.log.info("Data already in cache")
                # update the loglikelihood
                for theory in state.keys():
                    if self.dataframe.loc[(my_hash, theory),'loglike'] < loglike:
                        self.log.info("Update loglike")
                        self.dataframe.loc[(my_hash, theory),'loglike'] = loglike
                return False
            else:
                if loglike > min_loglike:
                    self.dataframe=pd.concat([self.dataframe, new_dataframe])

                    # Remove the datapoint with the lowest loglikelihood
                    for _ in self.theories:
                        self.dataframe = self.dataframe.drop(self.dataframe['loglike'].idxmin())

                    return True
                return False
        else:
            # Check whether the data point is already in the cache
            if my_hash in self.dataframe.index.get_level_values(0):
                #self.log.info("Data already in cache")
                for theory in state.keys():
                    if self.dataframe.loc[(my_hash, theory),'loglike'] < loglike:
                        self.dataframe.loc[(my_hash, theory),'loglike'] = loglike
                return False
            else:
                self.dataframe=pd.concat([self.dataframe, new_dataframe])
                self.log.debug("Adding data to cache")
                return True

    # This function saves the cache to a file
    def _save_cache(self):
        return False
    
    # This function sets a new size for the cache
    def _set_new_cache_size(self, N):
        self.N = N
        return True

    # This function loads the cache from a file
    def _load_cache(self):
        return False

    # This function returns the size of the cache
    def _size(self):
        theory = list(self.dataframe.keys())[0]
        return self.dataframe[theory].shape[0]/len(self.theories) 
