#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from surprise import Reader, Dataset, KNNBasic, SVD, NMF
from surprise.model_selection import GridSearchCV, cross_validate
from surprise import accuracy
from math import sqrt
from sklearn.metrics import mean_squared_error
from surprise.model_selection import train_test_split
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


movies_df = pd.read_csv(r'C:\Users\P Sai Deekshith\Recommender Systems\ml-100k\movies.csv')

ratings_df = pd.read_csv(r'C:\Users\P Sai Deekshith\Recommender Systems\ml-100k\ratings.csv')


# In[3]:


reader = Reader(rating_scale=(0.5, 5.0))

data = Dataset.load_from_df( ratings_df[['userId', 'movieId', 'rating']], reader = reader )



def rmseKNN():
    sim_options = {'name': 'msd'}

    algo = KNNBasic(k=20, sim_options=sim_options)
    cross_validate(algo=algo, data=data, measures=['RMSE'], cv=5, verbose=True)
    n_neighbours = [10, 20, 30]
    param_grid = {'n_neighbours': n_neighbours}

    gs = GridSearchCV(KNNBasic, measures=['RMSE'], param_grid=param_grid)
    gs.fit(data)
    print('\n\n###############')
    
    print('Best Score :', gs.best_score['rmse'])
    
    
    print('Best Parameters :', gs.best_params['rmse'])
    print('###############')
    
    return gs.best_score['rmse']



def rmseSVD():
    algo = SVD()
    cross_validate(algo=algo, data=data, measures=['RMSE'], cv=5, verbose=True)
    param_grid = {'n_factors': [50, 75], 'lr_all': [0.5, 0.05], 'reg_all': [0.06, 0.04]}

    gs = GridSearchCV(algo_class=SVD, measures=['RMSE'], param_grid=param_grid)
    gs.fit(data)
    print('\n###############')
    print('Best Score :', gs.best_score['rmse'])

    print('Best Parameters :', gs.best_params['rmse'])
    print('###############')

    return gs.best_score['rmse']







