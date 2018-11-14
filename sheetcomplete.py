#!/usr/bin/env python3

# SheetComplete main program 
# by Noah Kruiper and Tyson Moore
# for Miodrag Bolic, CEG 4913 Fall 2018
# University of Ottawa, 2018.
# FIXME: turn this into proper pydoc format

# ------------------------------------------------------------------------------
#   Imports
# ------------------------------------------------------------------------------



# ------------------------------------------------------------------------------
#   Functions
# ------------------------------------------------------------------------------

# Step -1: parse command-line arguments for things like:
# - input and output filenames
# - random number input for test/train split
# - verbosity
# TODO: method implementation
# NOTE: hard-coding now is fine

# Step 0: get the data out
# Step 1: determine directionality
def parse_csv():
    """
    Parse the given CSV file to extract data in a usable format. At a high
    level, this involves X steps:

    1. 
    """

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
    This function is the heart of SheetComplete. 
    """

# Step 5: assess networks
# TODO: method prototype & description

# ------------------------------------------------------------------------------
#   Main function
# ------------------------------------------------------------------------------

if __name__ == '__main__':
    exit(0)
