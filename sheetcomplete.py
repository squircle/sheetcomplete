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
import csv
import sys
import pandas

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
    # NOTE: this can wait till the end -- it's more of an extra check


# Step 3: sort and classify data sets
# TODO: method prototype & description

# Step 4: train networks
def train_algos():
    """
    This function is the heart of SheetComplete. TODO: complete documentation
    """

    # Set up parameters of each classifier 
    # TODO: move hardcoded parameters into 

    regressors = [
        KNeighborsRegressor(n_neighbors=5)
        # TODO: use a wide variety of generated K values.
        ]

    for algo in regressors:
        # fit the training data to the regressors
        algo.fit(X_train, Y_train) # TODO: fix variable names once step 3 done

# Step 5: assess networks
def assess_networks():
    """
        For each column with missing data, determine which network produces the best predictions.
    """
    # TODO: complete method

# ------------------------------------------------------------------------------
#   Main function
# ------------------------------------------------------------------------------

# STEP 1: Parse input CSV to dataframe
# for debugging
# print(parse_csv().to_string())

# STEP 2: possibly unnecessary

# STEP 3:

if __name__ == '__main__':
    exit(0)
