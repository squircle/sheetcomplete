#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import math

__author__ = "Noah Kruiper and Tyson Moore"
__license__ = "GPL-3.0"
__version__ = "1.0"
__email__ = "nkrui088@uottawa.ca, tmoor092@uottawa.ca"
__status__ = "Development"

# SheetComplete
# by Noah Kruiper and Tyson Moore
# for Miodrag Bolic, CEG 4913 Fall 2018

# ------------------------------------------------------------------------------
#   Imports
# ------------------------------------------------------------------------------

from sklearn.neighbors import KNeighborsRegressor, RadiusNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
import csv
import sys
import pandas
import numpy
import copy

# ------------------------------------------------------------------------------
#   Definitions
# ------------------------------------------------------------------------------

# random state for testing regressors, or None for proper randomness
rand_state = None

# test size for test/train split
test_split = 0.20

# ------------------------------------------------------------------------------
#   Functions
# ------------------------------------------------------------------------------


# Step 0: parse command-line arguments
def load_file():
    global filename, column_oriented
    # open first argument as CSV
    filename = sys.argv[1]
    print('\n========\nProcessing file: ' + filename + '\n========\n')
    # indicates if the original format of the csv was column-oriented
    column_oriented = True


# Step 1: determine directionality
def parse_csv():
    global column_oriented
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
        if csv.Sniffer().has_header(csvFile.read(2)):
            data_frame = pandas.read_csv(filename)
        else:
            # transpose the data to see if that reveals a header
            # todo: delete the transposed version after processing
            pandas.read_csv(filename).T.to_csv('transposed-' + filename, header=False)
            with open("transposed-" + filename, newline='') as csvFileFlipped:
                if csv.Sniffer().has_header(csvFileFlipped.read(1024)):
                    data_frame = pandas.read_csv('transposed-' + filename)
                    column_oriented = False
                else:
                    print('Unable to determine data orientation. Now exiting.')
                    exit()
        return data_frame


# Step 3: sort and classify data sets
# Step 4: train networks
def run_algos(x, y):
    """
    This function is the heart of SheetComplete. It does the test/train split for each dataset,
    and runs the regressors against them. It also creates a data structure to hold the instance
    of the regressor and the test/train data (for scoring in later steps).
    """
    # perform a data split
    X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=test_split, random_state=rand_state)

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

    # print('\nTrain Data X:\n' + X_train.to_string())
    # print('\nTest Data X:\n' + X_test.to_string())
    # print('\nTrain Data Y:\n' + Y_train.to_string())
    # print('\nTest Data Y:\n' + Y_test.to_string())

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
    print("\n-------- Score report: --------")

    for algo in reg_list:
        print("{}: {}".format(type(algo["regressor"]), algo["score"]))
        if best_algo is None:
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

    (n.b. All regressors listed below use R^2 as their score function. Modifications to the
    evaluate_algos function would be required to accept other regressors.)
    """
    # list of regressors to use in evaluation
    regressors = [
        KNeighborsRegressor(n_neighbors=5),
        # TODO: use a wide variety of generated K values.

        # RadiusNeighborsRegressor(),
        # FIXME: RadiusNeighbors results in blank predictions:
        # UserWarning: One or more samples have no neighbors within specified radius; predicting NaN.

        DecisionTreeRegressor(random_state=rand_state),

        ExtraTreesRegressor(n_estimators=100, random_state=rand_state),
        RandomForestRegressor(n_estimators=100, random_state=rand_state),
        AdaBoostRegressor(n_estimators=50, random_state=rand_state),
        # TODO: determine if n_estimators needs to be varied, and implement

        MLPRegressor(hidden_layer_sizes=100, random_state=rand_state),
        # TODO: flesh out options -- this neural net will need a lot of them,
        #       maybe even a loop over multiple layer sizes, learning rates, etc.

        GradientBoostingRegressor(random_state=rand_state),
        GaussianProcessRegressor(random_state=rand_state)
    ]

    return copy.deepcopy(regressors)


# Step 6: fill in missing data
def fill_missing():
    """
    Fills in the missing data using the chosen regressor. X is the input, and will output Y,
    given the specified regressor.
    """
    # fill missing data with predictions
    # for every row with nulls:
    for i in df_onlyNull.index:
        # for every column with nulls:
        for j in range(len(df_containNull)):
            # make a prediction if the cell is empty
            if df_containNull[j] and datatypes[j] != 'object':
                if math.isnan(filled.iloc[i, j]):
                    x = filled.drop(filled.iloc[:, j].name, axis=1).copy()
                    x = x.iloc[i]
                    # do some preliminary data processing to convert strings to float
                    x = x.apply(pandas.to_numeric, errors='coerce')
                    x.fillna(0, inplace=True)
                    prediction = selected_regressors[j]['regressor'].predict(numpy.array([x]).reshape(-1, len(df_containNull)-1))
                    # print('\nPredicted Data Y:\n' + str(prediction))
                    filled.iloc[i, j] = prediction

    print('Writing output file: ' + 'filled-' + filename)
    if not column_oriented:
        filled.T.to_csv('filled-' + filename, header=False, index=False)
    else:
        filled.to_csv('filled-' + filename, header=True, index=False)

# ------------------------------------------------------------------------------
#   Main function
# ------------------------------------------------------------------------------

if __name__ == '__main__':

    # treat inf and empty string as NA
    pandas.options.mode.use_inf_as_na = True

    # STEP 0: Load input file
    load_file()

    # STEP 1: Parse input CSV to dataframe
    input_dataframe = parse_csv()
    filled = copy.deepcopy(input_dataframe)
    # print('\nInput Dataframe:\n' + input_dataframe.to_string())

    # Step 2: determine data set type
    datatypes = input_dataframe.dtypes  # where object, treat as string
    # print('\nDatatypes:\n' + datatypes.to_string())

    # Store rows which have NO NULLs
    df_noNull = input_dataframe.dropna()
    # print('\nRows with NO NULL cells:\n' + df_noNull.to_string())

    # Store rows which have NULLs
    df_onlyNull = input_dataframe[~input_dataframe.index.isin(df_noNull.index)]
    # print('\nRows WITH NULL cells:\n' + df_onlyNull.to_string())

    # Indicate which columns have NULLs
    df_containNull = df_onlyNull.isna().any()
    # print('\nColumns with NULLs:\n' + df_containNull.to_string())

    # create list of regressors corresponding to columns
    selected_regressors = []

    # Iterate through all columns
    for i in range(len(df_containNull)):
        # if the column has missing data
        if df_containNull[i]:
            print('\n========================\nColumn to train: ' + df_noNull.iloc[:, i].name)
            # strip the column from the dataframe
            y = df_noNull.iloc[:, i].copy()
            # print('\nTarget Column Y:\n' + y.to_string())
            x = df_noNull.drop(df_noNull.iloc[:, i].name, axis=1).copy()
            # print('\nInput Data X:\n' + x.to_string())

            # run all the regressors
            reg_list = run_algos(x, y)

            # evaluate which is best
            best_algo = evaluate_algos(reg_list)
            selected_regressors.append(best_algo)
        else:
            selected_regressors.append(None)

    # Fill empty numeric cells
    fill_missing()
