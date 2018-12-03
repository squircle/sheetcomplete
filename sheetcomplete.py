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
import sklearn.utils
import csv
import sys
import pandas
import numpy
import copy

# ------------------------------------------------------------------------------
#   Definitions
# ------------------------------------------------------------------------------

# random state for regressors
# TODO: implement random chooser for this state and a corresponding flag
rand_state = 4 # chosen by fair dice roll, guaranteed to be random

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
def run_algos(x, y):
    """
    This function is the heart of SheetComplete. TODO: complete documentation
    """
    # perform a data split
    X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.33, random_state=rand_state)

    # do some preliminary data processing
    X_train_num = X_train.apply(pandas.to_numeric, errors='coerce')
    Y_train_num = Y_train.apply(pandas.to_numeric, errors='coerce')
    X_test_num = X_train.apply(pandas.to_numeric, errors='coerce')
    Y_test_num = Y_train.apply(pandas.to_numeric, errors='coerce')
    # TODO: Note that these are filling in NaNs with 0, since there are problems converting String to float.
    # This might be related to the @ symbols in the emails, not sure
    X_train_num.fillna(0, inplace=True)
    Y_train_num.fillna(0, inplace=True)
    X_test_num.fillna(0, inplace=True)
    Y_test_num.fillna(0, inplace=True)

    # create a list of dictionaries
    # each dictionary contains the regressor itself and its input parameters
    regressors_list = list()

    for algo in regressor_factory():
        regressors_list.append({
            "regressor": algo,
            "X_train": X_train_num,
            "X_test": X_test_num,
            "Y_train": Y_train_num,
            "Y_test": Y_test_num
        })

    print('\nTrain Data X:\n' + X_train.to_string())
    print('\nTest Data X:\n' + X_test.to_string())
    print('\nTrain Data Y:\n' + Y_train.to_string())
    print('\nTest Data Y:\n' + Y_test.to_string())

    # run the regressors against the input data
    for algo in regressors_list:
        algo["regressor"].fit(X_train_num, Y_train_num)

    return regressors_list

# Step 5: evaluate algorithms
def evaluate_algos(reg_list):
    """
    For each column with missing data, determine which network produces the best predictions.
    Return a reference to the best-performing algorithm, and print a report to stdout.

    This currently uses the simple R^2 score, which is the default .score method of each regressor.
    In the future, it could be expanded to use the cross-validation score.
    """

    # score all the algorithms
    for algo in reg_list:
        algo["score"] = algo["regressor"].score(algo["X_test"], algo["Y_test"])

    # choose the best one and return it
    best_algo = None
    print("\nScore report:\n=============\n\n")

    for algo in reg_list:
        print("{}: {}".format(type(algo["regressor"]), algo["score"]))
        if best_algo == None:
            best_algo = algo
        elif algo["score"] > best_algo["score"]:
            best_algo = algo

    print("\nBest regressor: {}".format(type(best_algo["regressor"])))
    return best_algo

def regressor_factory():
    """
    Create a list of regressors that can be used with a data set. This is a 'deep copy' operation
    because it duplicates the objects, rather than simply creating references to the old ones
    (doing so would result in training the same regressors over and over again).
    """
    # list of regressors to use in evaluation
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
    
    return copy.deepcopy(regressors)

# Step 6: fill in missing data
def fill_missing(X, regressor):
    """
    Fills in the missing data using the chosen regressor. X is the input, and will output Y,
    given the specified regressor.
    """
    Y = regressor.predict(X) # FIXME: does this need to unpack the data frame somehow?

    return Y # FIXME: does this need to be of some dataframe type?

# ------------------------------------------------------------------------------
#   Main function
# ------------------------------------------------------------------------------

if __name__ == '__main__':
    
    # treat inf and empty string as NA
    pandas.options.mode.use_inf_as_na = True

    # STEP 1: Parse input CSV to dataframe
    dataframe = parse_csv()
    print('\nInput Dataframe:\n' + dataframe.to_string())

    # STEP 2:
    datatypes = dataframe.dtypes # where object, treat as string
    print('\nDatatypes:\n' + datatypes.to_string())

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
            
            # run all the regressors
            reg_list = run_algos(x, y)

            # evaluate which is best
            best_algo = evaluate_algos(reg_list)

            # fill missing data with predictions
            # FIXME: do data frame maniupulations to call fill_missing and put the data back
            # fill_missing(???, best_algo["regressor"])
