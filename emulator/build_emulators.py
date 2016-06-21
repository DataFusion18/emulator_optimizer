#!/usr/bin/env python3

import os, re
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import datetime as dt
import sys
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn import cross_validation
from cStringIO import StringIO
from sklearn.grid_search import GridSearchCV

def make_data_index(df):
    dates = []
    for index, row in df.iterrows():
        s = str(int(float(row['YEAR']))) + " " + str(int(float(row['DOY'])))
        dates.append(dt.datetime.strptime(s, '%Y %j'))
    return dates

def date_converter(*args):
    return dt.datetime.strptime(str(int(float(args[0]))) + " " +\
                                str(int(float(args[1]))), '%Y %j')

def remove_comments_from_header(fname):
    """ I have made files with comments which means the headings can't be
    parsed to get dictionary headers for pandas! Solution is to remove these
    comments first """
    s = StringIO()
    with open(fname) as f:
        for line in f:
            if '#' in line:
                line = line.replace("#", "").lstrip(' ')
            s.write(line)
    s.seek(0) # "rewind" to the beginning of the StringIO object

    return s

def main():

    # GDAY inputs
    fdir = "/Users/mq20101267/Desktop/gday_simulations/DUKE/step_change/met_data"
    fname = os.path.join(fdir, "DUKE_met_data_amb_co2.csv")
    s = remove_comments_from_header(fname)
    df_met = pd.read_csv(s, parse_dates=[[0,1]], skiprows=4, index_col=0,
                           sep=",", keep_date_col=True,
                           date_parser=date_converter)

    met_data = df_met.ix[:,2:].values

    # GDAY outputs
    fdir = "/Users/mq20101267/Desktop/gday_simulations/DUKE/step_change/outputs"
    fname = os.path.join(fdir, "D1GDAYDUKEAMB.csv")
    df = pd.read_csv(fname, skiprows=3, sep=",", skipinitialspace=True)
    df['date'] = make_data_index(df)
    df = df.set_index('date')

    target = df["GPP"].values


#----------------------------------------------------------------------
    # BUILD MODELS

    # hold back 40% of the dataset for testing
    #X_train, X_test, Y_train, Y_test = \
    #    cross_validation.train_test_split(met_data, target, \
    #                                      test_size=0.99, random_state=0)

#    param_DTR = { \
#        #"min_samples_split": [2, 10, 20], \
#        #"max_depth": [2, 5, 10, 30, 50], \
#        #"min_samples_leaf": [1, 5, 10], \
#        #"max_leaf_nodes": [5, 10, 20], \
#        }
#
#    param_RFR = { \
#        #"min_samples_split": [2, 10, 20], \
#        "max_depth": [None], \
#        #"min_samples_leaf": [1, 5, 10], \
#        #"max_leaf_nodes": [5, 10, 20], \
#        #"criterion": ["gini", "entropy"], \
#        "max_features": [2, 4, 8], \
#        #"n_estimators": [5, 10, 20] \
#        "n_estimators": [300], \
#
#    param_SVR = { \
#        "C": [0.5, 1, 2], \
#        "kernel": ['rbf'], \
#        #"coeff": [0, 0.3, 0.5], \
#        "degree": [1, 2, 3], \
#        }
#
    param_KNR = { \
        "n_neighbors": [5, 10, 20], \
        "weights": ['uniform', 'distance'], \
        }

#    regmod = DecisionTreeRegressor()
#    regmod = RandomForestRegressor()
#    regmod = SVR()
    regmod = KNeighborsRegressor()

    pipeit3 = lambda model: make_pipeline(StandardScaler(), PCA(), model)
    pipeit2 = lambda model: make_pipeline(StandardScaler(), model)
    regmod_p = pipeit2(regmod)

    modlab = regmod_p.steps[-1][0]

    par_grid = {'{0}__{1}'.format(modlab, parkey): pardat \
                 for (parkey, pardat) in param_KNR.iteritems()}

    #emulator = GridSearchCV(regmod, param_grid=param_DTR, cv=5)
    emulator = GridSearchCV(regmod_p, param_grid=par_grid, cv=5)

    #emulator.fit(X_train, Y_train)
    emulator.fit(met_data, df.GPP)
    pred_test = emulator.predict(met_data)

    output_test = pd.DataFrame({'emu': pred_test, \
                                'spa': df.GPP})
#    sns.jointplot(x='emu', y='spa', data=output_test, kind='reg')
#    plt.show()

    output_emulated = pd.DataFrame({'DT': df.index, \
                                    'emu': emulator.predict(met_data), \
                                    'gday': df.GPP})
    output_emulated.set_index(['DT'], inplace=True)
    print(output_emulated.head())

    output_day = output_emulated.resample('D').mean()

    plt.plot_date(output_day.index, output_day['emu'], 'o', label='Emulator')
    plt.plot_date(output_day.index, output_day['gday'], '.', label='GDAY')
    plt.ylabel('GPP ($\mu$mol m$^{-2}$ s$^{-1}$)')
    plt.legend()
    plt.show()

    return None

#    gs = gridspec.GridSpec(1, 2)
#    for i in range(2):
#        ax = plt.subplot(gs[i])
#        ax.plot(pred_test[:, i], Y_test[:, i], 'ro')
#    plt.show()

# --------------------------------------------------------------------------------


    return None

if __name__ == "__main__":

    DIRPATH = os.path.expanduser("~/Savanna/Models/SPA1/outputs/baseline/")
    FILEPATH = "natt_datasets.pkl"

    main()
