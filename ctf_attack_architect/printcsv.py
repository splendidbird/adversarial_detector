from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse

import pandas as pd
import numpy as np


def printcsv(csvfile):
    cc = pd.read_csv(csvfile, index_col = None)
    print(cc)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', help='Input csv file name')
    args = parser.parse_args()
    printcsv(args.filename)

if __name__ == '__main__':
    main()
