#!/usr/bin/env python2.7

import os, re
import pickle
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.svm import SVR
from sklearn.grid_search import GridSearchCV
from sklearn import cross_validation

def get_site_name(fpath):
    return fpath.split("/")[-3]

def import_spa_outputs(fpath):
    dataset = pd.read_csv(fpath, sep=r',\s+', engine='python', na_values=["Infinity"])
    dataset.rename(columns=lambda x: x.strip(), inplace=True)
    return dataset

def main():

#----------------------------------------------------------------------
    # STAGE DATA

    # get the filepaths leading the hourly outputs from a designated set of
    # SPA simulations
    file_paths = [os.path.join(dp, f) for (dp, _, fn) in os.walk(DIRPATH) \
                  for f in fn if re.search(r"hourly", f)]

    # load these with pandas and save in a dictionary - site name as label
    spa_hourly = {get_site_name(fn): import_spa_outputs(fn) \
                  for fn in file_paths}

    # also load the pre-saved driver information that was used to run the simulations
    natt_datasets = pickle.load(open(DIRPATH + FILEPATH, 'rb'))

#----------------------------------------------------------------------
    # BUILD MODELS

    # practice set
    sname = spa_hourly.keys()[0]
    print sname

    # the target values it is trying to reproduce
    #target = spa_hourly[sname].ix[:, ["gpp", "lemod"]].values
    target = np.array(spa_hourly[sname]["gpp"])
    # the drivers of the emulator (crop to match length of target)
    data = natt_datasets[sname]["drivers"] \
        .ix[:len(target), 1:].values

    # hold back 40% of the dataset for testing
    X_train, X_test, Y_train, Y_test = \
        cross_validation.train_test_split(data, target, \
                                          test_size=0.4, random_state=0)

    param_DTR = { \
        #"min_samples_split": [2, 10, 20], \
        "max_depth": [2, 5, 10, 30], \
        #"min_samples_leaf": [1, 5, 10], \
        #"max_leaf_nodes": [5, 10, 20], \
        #"criterion": ["gini", "entropy"], \
        "max_features": [2, 4, 8], \
        "n_estimators": [5, 10, 20] \
        }
    #regmod = DecisionTreeRegressor()
    regmod = RandomForestRegressor()

#    param_DTR = { \
#        #"min_samples_split": [2, 10, 20], \
#        #"max_depth": [2, 5, 10, 30, 50], \
#        #"min_samples_leaf": [1, 5, 10], \
#        #"max_leaf_nodes": [5, 10, 20], \
#        }
#    regmod = SVR()

    emulator = GridSearchCV(regmod, param_grid=param_DTR, cv=5)

    emulator.fit(X_train, Y_train)
    pred_test = emulator.predict(X_test)

    plt.plot(pred_test, Y_test, 'ro')
    plt.show()

    return None

    gs = gridspec.GridSpec(1, 2)
    for i in range(2):
        ax = plt.subplot(gs[i])
        ax.plot(pred_test[:, i], Y_test[:, i], 'ro')
    plt.show()

# --------------------------------------------------------------------------------


    return None

if __name__ == "__main__":

    DIRPATH = os.path.expanduser("~/Savanna/Models/SPA1/outputs/baseline/")
    FILEPATH = "natt_datasets.pkl"

    main()
