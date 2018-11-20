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
from sklearn.model_selection import train_test_split
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
    # fit the training data to the regressors
    #algo.fit(X_train, y_train)

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
# print(dataframe.to_string())

# STEP 2:
datatypes = sklearn.utils.check_array(dataframe)
print(datatypes)
# STEP 3:

# Store rows which have NO NULLs
#df_noNull = dataframe.dropna()

# Store rows which have NULLs
#df_onlyNull = dataframe[~dataframe.index.isin(df_noNull.index)]

# Indicate which columns have NULLs
#df_containNull = df_onlyNull.isna().any()

# NOTE: index SHOULD be preserved with df_onlyNull, meaning it should be possible to iterate through at the end and...
# NOTE: ...fill in missing data by iterating through those indexes, recreating the original order and shape of the data.


# Set up parameters of each classifier
# TODO: move hardcoded parameters into Step -1

# regressors = [
#     KNeighborsRegressor(n_neighbors=5)
#     # TODO: use a wide variety of generated K values.
#     ]

# Iterate through all columns
# for column in range(0, df_containNull.size -1):
#     # If the column has missing data...
#     if df_containNull[column]:
#         # Copy dataframe without the column in question
#         df_noColumn = df_noNull.copy(deep=True).drop(df_noNull.columns[[column]], axis=1, inplace=True)
#         # Copy ONLY the column
#         df_column = df_noNull[[column]].copy()
#         # Split into train and test sets
#         X_train, X_test, y_train, y_test = train_test_split(df_noNull(df_noColumn, df_column, test_size=0.25, random_state=42))
#         # Evaluate each algorithm
#         for algorithm in regressors:
#             evaluate_algo(X_train, y_train, X_test, y_test, algorithm)
#         # Select the best algorithm and save it into an array or something


if __name__ == '__main__':
    exit(0)
