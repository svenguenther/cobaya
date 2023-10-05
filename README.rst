*OLÉ*, a native online learning emulator for Cobaya
===================================================

*Code under progress!*

This code implements a native emulator for the Bayesian inference code Cobaya. This emulator is using active learning and accuracy estimation that ensures reliable predictions.

Find further information in this paper https://arxiv.org/abs/2307.01138.

The usage is as follows: You add to your cobaya input dict or ini file following the element 'emulator: {}' or 'emulator: None'. Everything else the emulator takes care of.

Additional parameters which might modify the performance are:

 **theory_codes:** [],            # list of theories codes which are to be emulated. If empty, all codes are emulated \

 **postpone_learning:** 80,            # number of simulation calls before the emulator is used. Required to get useful PCA and to not train the burn-in \
 
 **learn_every:** 20,                 # every N new data points the kernel of the GP is fitted. This is more expensive than just adding a new point to your GP \
 
 **training_size:** 1000,             # maximum datasize of the GP. When exceeded, points with the largest loglike will be removed \
 
 **gp_fit_size:** 60,                 # data size which is used to compute the GP kernel \
 
 **pca_cache_size:** 2000,            # size of the cache to compute the PCA. It scales better with dimensionality than the training size \
 
 **precision:** 0.1,                  # precision criterium for the emulator to be used \
 
 **precision_linear:** 0.1,            # linear precision criterium \
 
                                        # Total tot_precision = precision + precision_linear * (loglike_max - loglike)   # Thus, we allow data points which are further away from the bestfit point to be more noisy \
                                        
 **N_validation_states:** 10,         # number of validation to estimate the accuracy of the emulator \
 
 **testset_fraction:** 0.1,           # fraction of the training set to use for validation \
 
 **debug:** False,                     # expensive debug mode which makes a lot of plots \
 
 **pca_update:** 1,                   # update PCA every N training steps \
 
 **delta_loglike_cache:** 300,        # only data points with a delta_loglike + min_loglike are cached. All other points are removed from the cache   \  
 
 **min_training_size:** 30,        # minimum number of points in cache before emulator is getting trained   \  
 
 **gp_initial_minimization_states:** 80,        # number of kernel optimizations when learning new pca  \  
 
 **gp_minimization_states:** 10,        # number of kernel optimizations when NOT learning new pca   \  
 
 **input_parameter:** {},        # dictionary of emulated quantities with lists of the demanded input parameters. If left blank, all parameters are used to predict the quantity   \  


===================

Author of Modification: Sven Günther; 
Cobaya by Jesus Torrado and Antony Lewis

.. image:: ./img/logo_ttk.png
   :alt: RWTH Aachen
   :target: https://www.particle-theory.rwth-aachen.de/
   :height: 150px

