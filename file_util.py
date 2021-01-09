"""
Utility file for managing files and directories.

Due to the size of epigenetic data we will be working with, it is necessary to
purge files to storage and free up available RAM. This utility module creates
functions for outputing data to files and loading data from files to enable
efficient analysis.

By:     Joseph Wakim
Date:   January 7, 2020

"""

import os
from datetime import datetime

import pandas as pd
import numpy as np


def create_directories(paths):
    """
    Create directories if they do not exist.

    Parameters
    ----------
    paths : List[str]
        List of paths at which directories should be created. 
    """
    for path in paths:
        if not os.path.exists(path):
            os.mkdir(path)


def get_date_time_label():
    """
    Get a date and time label for aiding in file formation.

    Returns
    -------
    str
        String representation of the date and time.
    """
    now = datetime.now()
    dt = format_datetime(now).replace("/","-").replace(" ","_").replace(':','')
    return dt


def format_datetime(date_time):
    """
    Convert `datetime` object to a string expressing date and time.
    Parameters
    ----------
    date_time : DateTime object
        Object representation of the date and time from `datetime` package
    Returns
    -------
    str
        Date and time in human readable format
    """
    return date_time.strftime("%Y/%m/%d %H:%M:%S")


def df_to_csv(df, path):
    """
    Save Pandas dataframe to a csv file.

    Parameters
    ----------
    df : dataframe
        Pandas dataframe to be saved
    path : str
        Path to csv file into which df will be saved
    """
    df.to_csv(path, index=False, sep='\t')


def csv_to_df(path):
    """
    Load a csv file as a Pandas dataframe.

    Parameters
    ----------
    path : str
        File path to csv file

    Returns
    -------
    dataframe
        Pandas dataframe containing data from csv
    """
    return pd.read_csv(path, sep='\t')