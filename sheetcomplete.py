#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__      = "Noah Kruiper and Tyson Moore"
__license__     = "TBD"
__version__     = "0"
__email__       = "nkrui088@uottawa.ca, tmoor092@uottawa.ca"
__status__      = "Development"

# SheetComplete main program 
# by Noah Kruiper and Tyson Moore
# for Miodrag Bolic, CEG 4913 Fall 2018

# ------------------------------------------------------------------------------
#   Imports
# ------------------------------------------------------------------------------

from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
import sklearn.neural_network
import sklearn.utils
import csv
import sys
import pandas
import numpy

# ------------------------------------------------------------------------------
#   Functions
# ------------------------------------------------------------------------------

# Step -1: parse command-line arguments for things like:
# - input and output filenames
# - random number input for test/train split
# - verbosity
# TODO: method implementation
# NOTE: hard-coding now is fine

# open first argument as CSV
filename = sys.argv[1]
# indicates if the original format of the csv was column-oriented
columnOriented = True

# random state for regressors
# TODO: implement random chooser for this state and a corresponding flag
rand_state = 4 # chosen by dice roll, guaranteed to be random

# Step 0: get the data out
# Step 1: determine directionality
def parse_csv():
    """
    Parse the given CSV file to extract data in a usable format. At a high
    level, this involves the following steps:

    1. Determine directionality
    2. If directionality does not conform to column-wise grouping, transpose the data
    3. Import data to dataframe
    """
    with open(filename, newline='') as csvFile:
        # set seek position to the start
        # Note: possibly unnecessary
        csvFile.seek(0)
        # If the csv is determined to have a header, then it's ready for analysis
        if csv.Sniffer().has_header(csvFile.read(1024)):
            dataframe = pandas.read_csv(filename)
        else:
            # transpose the data to see if that reveals a header
            # todo: delete the transposed version after processing
            pandas.read_csv(filename).T.to_csv('transposed-' + filename, header=False)
            with open("transposed-" + filename, newline='') as csvFileFlipped:
                if csv.Sniffer().has_header(csvFileFlipped.read(1024)):
                    columnOriented = False
                    dataframe = pandas.read_csv('transposed-' + filename)
                else:
                    print('Unable to determine data orientation. Now exiting.')
                    # todo: delete transposed file here?
                    exit()
        return dataframe

    # TODO: method completion


# Step 2: determine data set type
def get_data_types():
    """
    Intuit the extracted data fields to determine what type of data they
    contain. For example, phone numbers, street names, and other discrete
    categorizable data types.

    This isn't a robust check, but it feeds into later steps to determine
    correctness of the chosen machine learning algorithm.
    """

    # TODO: method completion





# Step 3: sort and classify data sets
# Step 4: train networks
def evaluate_algo(xTrain, yTrain, xTest, yTest, regressor):
    """
    This function is the heart of SheetComplete. TODO: complete documentation
    """
    # Remove rows with missing data
    notnans = dataframe.notnull().all(axis=1)
    df_notnans = dataframe[notnans]

    # Split into 75% train and 25% test
    X_train, X_test, y_train, y_test = train_test_split(df_notnans,
                                                        train_size=0.75,
                                                        random_state=4)

    # Set up parameters of each classifier 
    # TODO: move hardcoded parameters into Step -1

    regressors = [
        KNeighborsRegressor(n_neighbors=5),
        # TODO: use a wide variety of generated K values.
        DecisionTreeRegressor(random_state=rand_state),
        ExtraTreesRegressor(n_estimators=100, random_state=rand_state),
        # TODO: determine if n_estimators needs to be varied, and implement
        MLPRegressor(hidden_layer_sizes=100, random_state=rand_state)
        # TODO: flesh out options -- this neural net will need a lot of them,
        #       maybe even a loop over multiple layer sizes, learning rates, etc.
        ]

    for algo in regressors:
        # fit the training data to the regressors
        algo.fit(X_train, y_train)

# Step 5: assess networks
def assess_networks():
    """
        For each column with missing data, determine which network produces the best predictions.
    """
    # TODO: complete method

# ------------------------------------------------------------------------------
#   Main function
# ------------------------------------------------------------------------------

# treat inf and empty string as NA
pandas.options.mode.use_inf_as_na = True

# STEP 1: Parse input CSV to dataframe
dataframe = parse_csv()
print('\nInput Dataframe:\n' + dataframe.to_string())

# STEP 2:
datatypes = dataframe.dtypes # where object, treat as string
print('\nDatatypes:\n' + datatypes.to_string())

    # STEP 2: possibly unnecessary

# Store rows which have NO NULLs
df_noNull = dataframe.dropna()
print('\nRows with NO NULL cells:\n' + df_noNull.to_string())

# Store rows which have NULLs
df_onlyNull = dataframe[~dataframe.index.isin(df_noNull.index)]
print('\nRows WITH NULL cells:\n' + df_onlyNull.to_string())

# Indicate which columns have NULLs
df_containNull = df_onlyNull.isna().any()
print('\nColumns with NULLs:\n' + df_containNull.to_string())

# NOTE: index SHOULD be preserved with df_onlyNull, meaning it should be possible to iterate through at the end and...
# NOTE: ...fill in missing data by iterating through those indexes, recreating the original order and shape of the data.



# Set up parameters of each classifier
# TODO: move hardcoded parameters into Step -1

# regressors = [
#     KNeighborsRegressor(n_neighbors=5)
#     # TODO: use a wide variety of generated K values.
#     ]

# Iterate through all columns
for i in range(len(df_containNull)):
    # if the column has missing data
    if df_containNull[i]:
        print('\n----------------\nColumn to train: ' + df_noNull.iloc[:, i].name)
        # strip the column from the dataframe
        y = df_noNull.iloc[:, i].copy()
        print('\nTarget Column Y:\n' + y.to_string())
        x = df_noNull.drop(df_noNull.iloc[:, i].name, axis=1).copy()
        print('\nInput Data X:\n' + x.to_string())
        # perform a data split
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
        print('\nTrain Data X:\n' + X_train.to_string())
        print('\nTest Data X:\n' + X_test.to_string())
        print('\nTrain Data Y:\n' + y_train.to_string())
        print('\nTest Data Y:\n' + y_test.to_string())
        # TODO: ITERATE THROUGH VARIOUS REGRESSORS
        regressor = sklearn.neural_network.MLPRegressor()
        # Todo: is it possible to avoid converting to numeric?
        X_train_num = X_train.apply(pandas.to_numeric, errors='coerce')
        y_train_num = y_train.apply(pandas.to_numeric, errors='coerce')
        # TODO: Note that these are filling in NaNs with 0, since there are problems converting String to float.
        # This might be related to the @ symbols in the emails, not sure
        X_train_num.fillna(0, inplace=True)
        y_train_num.fillna(0, inplace=True)
        regressor.fit(X_train_num, y_train_num)

#         # Evaluate each algorithm
#         for algorithm in regressors:
#             evaluate_algo(X_train, y_train, X_test, y_test, algorithm)
#         # Select the best algorithm and save it into an array or something


if __name__ == '__main__':
    exit(0)
