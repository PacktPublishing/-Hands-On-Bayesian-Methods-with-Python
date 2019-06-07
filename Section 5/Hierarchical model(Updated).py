# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 13:05:48 2019

@author: OK
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 17:31:58 2019

@author: OK
"""

%matplotlib inline

import matplotlib.pyplot as plt
import numpy as np
import pymc3 as pm 
import pandas as pd

# Importing the dataset


import os
os.chdir('D:\Machine Learning\A-Data Prediction - House')
data = pd.read_csv('radon.csv')

county_names = data.county.unique()
county_idx = data['county_code'].values

data[['county', 'log_radon', 'floor']].head()

    
#Hierarchical Model
    
with pm.Model() as hierarchical_model:
    # Hyperpriors
    mu_a = pm.Normal('mu_alpha', mu=0., sd=1)
    sigma_a = pm.HalfCauchy('sigma_alpha', beta=1)
    mu_b = pm.Normal('mu_beta', mu=0., sd=1)
    sigma_b = pm.HalfCauchy('sigma_beta', beta=1)
    
    # Intercept for each county, distributed around group mean mu_a
    a = pm.Normal('alpha', mu=mu_a, sd=sigma_a, shape=len(data.county.unique()))
    # Intercept for each county, distributed around group mean mu_a
    b = pm.Normal('beta', mu=mu_b, sd=sigma_b, shape=len(data.county.unique()))
    
    # Model error
    eps = pm.HalfCauchy('eps', beta=1)
    
    # Expected value
    radon_est = a[county_idx] + b[county_idx] * data.floor.values
    
    # Data likelihood
    y_like = pm.Normal('y_like', mu=radon_est, sd=eps, observed=data.log_radon)    
    
with hierarchical_model:
    hierarchical_trace = pm.sample(njobs=4)

pm.traceplot(hierarchical_trace);


# RESULT:The marginal posteriors in the left column are highly informative. mu_a tells us the group mean (log) radon levels. mu_b tells us that having no basement decreases radon levels significantly (no mass above zero).    