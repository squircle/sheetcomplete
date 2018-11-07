# Testing CSV opening/classifying
# By Noah Kruiper
# For CEG4136 Fall 2018

import csv, sys
import pandas

# clear first line
print('')
# open first argument as CSV
filename = sys.argv[1]
with open(filename, newline='') as csvFile:
    # determine style of CSV
    dialect = csv.Sniffer().sniff(csvFile.read(1024))
    # Column Grouping means you should find data of the same type within a column
    # If this is false, then the original data format is the opposite (data of the same type within a row)
    columnGrouping = None
    # reset seek position from 1024 back to the start
    csvFile.seek(0)
    # If the csv is determined to have a header, then it's ready for analysis
    if csv.Sniffer().has_header(csvFile.read(1024)):
        columnGrouping = True
    else:
        # transpose the data to see if that reveals a header
        pandas.read_csv(filename).T.to_csv('transposed-' + filename,header=False)
        with open("transposed-" + filename, newline='') as csvFileFlipped:
            if csv.Sniffer().has_header(csvFileFlipped.read(1024)):
                columnGrouping = False
                csvFileFlipped.seek(0)
                dialect = csv.Sniffer().sniff(csvFileFlipped.read(1024))
            else:
                print('Unable to determine data orientation. Now exiting.')
                # delete transposed file here?
                exit()
    # print if the csv had to be transposed or not
    print('Column Grouping: ' + str(columnGrouping))
    csvFile.seek(0)
    # open csv reader
    if columnGrouping :
        reader = csv.reader(csvFile, dialect)
    else :
        filename = "transposed-" + filename
        reader = csv.reader(open(filename, newline=''), dialect)
    # read csv data
    print('Reading from: ' + filename)
    print('')
    try:
        for row in reader:
            print(row)
    except csv.Error as e:
        sys.exit('file {}, line {}: {}'.format(filename, reader.line_num, e))
