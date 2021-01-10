"""
Rediscretize pandas dataframes showing signals for a single chromosome.

BED and WIG files report signals or categorical annotations for genomic
intervals within a chromosome. Due to experimental limitations, these signals
are often measured at kilo-base scales. This module provides tools for
rediscretizing signals or annotations at new scales more appropriate for
simulation.

By:     Joseph Wakim
Date:   January 7, 2020

"""

import sys

import pandas as pd
import numpy as np


def check_single_chrom(df):
    """
    Check if dataframe contains information from a singlechromosome.

    Parameters
    ----------
    df : dataframe
        Pandas dataframe characterizing chromosomal intervals. Must contain
        columns ['chrom', 'start_ind', 'end_ind', 'signal']. Rows must be
        sorted in order of increasing genomic position.

    Returns
    -------
    chrom : str
        Chromosome identifier
    """
    chrom = np.unique(df['chrom'].to_list())
    if len(chrom) != 1:
        raise ValueError("Rediscretize only one chromosome at a time.")
    return chrom[0]


def rebuild_df(headers, chrom, start, end, signal):
    """
    Construct rediscretized dataframe.

    Parameters
    ----------
    headers : List[str]
        List of column headers for the dataframe
    chrom : str
        Chromosome identifier
    start : array_like (m,)
        Starting indices of genomic intervals
    end : array_like (m,)
        Ending indices of genomic intervals
    signal : array_like (m,)
        Signal (numeric) or label (str) distributed into new bins

    Returns
    -------
    dataframe
        Rediscretized dataframe containing redistributed signals
    """
    data = {
        headers[0] : [chrom] * len(start),
        headers[1] : start,
        headers[2] : end,
        headers[3] : signal
    }
    return pd.DataFrame(data=data)


class Bin:
    """Class representation of rediscretized bin holding signal values."""

    def __init__(self, values, fractions, width, repeat=False):
        """
        Initialize `Bin` object.

        Parameters
        ----------
        values : float or str or array_like
            Values represented by the bin
        fractions : float or array_like
            Fraction of the bin associated with each value, len = len(values)
        width : int
            Number of base pairs represented by the bin
        repeat : bool, default False
            If true, values in genomic interval will be repeated into
            each rediscretized bin; else, values in each genomic interval will
            be evenly distributed within the  interval before rediscretization
        """
        self.check_attribute_compatability(values, fractions)
        self.values = values
        self.fractions = fractions
        self.width = width
        self.repeat = repeat

    def check_attribute_compatability(self, values, fractions):
        """
        Verify that the values and fractions in `Bin` are of equal length.

        Parameters
        ----------
        values : float or str or array_like
            Values represented by the bin
        fractions : float or array_like
            Fraction of the bin associated with each value, len = len(values)
        """
        if isinstance(values, (list, tuple, np.ndarray)):
            if not isinstance(fractions, (list, tuple, np.ndarray)) or \
                len(values) != len(fractions):
                raise ValueError(
                    "Bin values and fractions must be equal lengths.")
        else:
            if isinstance(fractions, (list, tuple, np.ndarray)):
                raise ValueError(
                    "Bin values and fractions must be equal lengths.")
    
    def get_value(self):
        """
        Get the value of the bin.
        
        If `self.repeat` == False, return a weighted sum of values in 
        `self.values`. Otherwise, convert fractions into proportional whole
        numbers and duplicate each value in `self.values` by the respective
        whole number; return the most common value, randomly breaking ties.

        Returns
        -------
        numeric or str
            Overall value representative of the bin
        """
        if not self.repeat:
            return np.sum(np.multiply(self.values, self.fractions))
        else:
            fractions = np.atleast_1d(np.array(self.fractions).astype(float))
            frac_str = fractions.astype(str)
            max_decimals = np.max([len(x.split('.')[-1]) for x in frac_str])
            fractions *= 10 ** max_decimals
            fractions = fractions.astype(int)
            expanded = [[self.values[i]] * j for i, j in enumerate(fractions)]
            expanded = np.array(expanded).flatten()
            unique, count = np.unique(expanded, return_counts=True)
            return unique[
                np.random.choice(np.argwhere(count==np.amax(count)).flatten())]


class Rediscretize:
    """Class representation of rediscretization algorithm."""

    def __init__(self, step, categorical=False, repeat=False):
        """
        Initialize `Rediscretize` object.

        Parameters
        ----------
        step : int
            Unit size following rediscretization
        categorical : bool, default False
            If true, treats signal as categorical variable and randomly takes
            most common signal when rediscretizing; else, treats signal as
            numerical and distributes signal evenly within bins
        repeat : bool, default False
            If true, values in genomic interval will be repeated into
            each rediscretized bin; else, values in each genomic interval will
            be evenly distributed within the  interval before rediscretization
        """
        self.step = step
        self.signals = None
        self.repeat = repeat
        self.categorical = categorical

    def rediscretize_chromosome(self, df):
        """
        Rediscretize chromosomal signals.

        For memory efficiency, only rediscretize one chromosome at a time. If 
        df contains more than one chromosome, first split the dataframe by 
        chromosome using `separate_chromosomes()`.

        Start by verifying that a single chromosome is included in df. Then get
        rediscretized bin edges based on the overall bounds of the genomic
        intervals and the desired size of the bins. Calculate how many bins of
        size `self.step` fit into each genomic interval. Characterize bins as
        having a homogeneous value or heterogeneous values.

        Parameters
        ----------
        df : dataframe
            Pandas dataframe characterizing chromosomal intervals. Must contain
            columns ['chrom', 'start_ind', 'end_ind', 'signal']. Rows must be
            sorted in order of increasing genomic position.

        Returns
        -------
        dataframe
            Rediscretized dataframe containing redistributed signals
        """
        headers = list(df.columns)
        chrom = check_single_chrom(df)
        bin_edges = self.get_rediscretized_bin_edges(df)
        start, end = self.get_rediscretized_intervals(bin_edges)
        interval_edges = self.get_genomic_interval_edges(df)
        bins_per_interval = np.divide(np.ediff1d(interval_edges), self.step)
        bins = self.categorize_bins(df, bins_per_interval)
        signal = [b.get_value() for b in bins]
        return rebuild_df(headers, chrom, start, end, signal)

    def categorize_bins(self, df, bins_per_interval):
        """
        Characterize rediscretized bins as being homogeneous or heterogeneous.

        Parameters
        ----------
        df : dataframe
            Pandas dataframe characterizing chromosomal intervals. Must contain
            columns ['chrom', 'start_ind', 'end_ind', 'signal']. Rows must be
            sorted in order of increasing genomic position.
        bins_per_interval : List[float]
            Number of bins in each genomic interval

        Returns
        -------
        bins : List[Bin objects]
            List of bin object corresponding to rediscretized genome
        """
        remainder = 0
        bins = []
        i = 0
        while i <= len(bins_per_interval)-1:
            current_val = self.get_val_at_index(df, bins_per_interval, i)
            interval = bins_per_interval[i] - remainder
            values, fractions = [], []
            remainder = 0
            previous_interval = 0
            j = 0   
            while interval >= 1:
                bins.append(Bin(current_val, 1, self.step, self.repeat))
                interval -= 1
            while interval < 1 and interval != 0:
                if i == len(bins_per_interval)-1:
                    bins.append(Bin(current_val, 1, self.step, self.repeat))
                    return bins
                fractions.append(interval)
                values.append(self.get_val_at_index(df,bins_per_interval,i+j))
                j += 1
                previous_interval = interval
                interval += bins_per_interval[i+j]        
            if interval != 0:
                remainder = 1 - previous_interval
                fractions.append(1-np.sum(fractions))
                values.append(self.get_val_at_index(df,bins_per_interval,i+j))
                bins.append(Bin(values, fractions, self.step, self.repeat))
                i += j
            else:
                i += j + 1
        return bins

    def get_val_at_index(self, df, bins_per_interval, ind):
        """
        Get the value of a signal distributed into a rediscretized bin.

        Parameters
        ----------
        df : dataframe
            Pandas dataframe characterizing chromosomal intervals. Must contain
            columns ['chrom', 'start_ind', 'end_ind', 'signal']. Rows must be
            sorted in order of increasing genomic position.
        bins_per_interval : List[float]
            Number of bins in each genomic interval
        ind : int
            Row in the data frame containing original value
        
        Returns
        -------
        value : numeric or str
            Value to be distributed into rediscretized bins
        """
        value = df.iloc[ind,3]
        if not self.repeat:
            value /= bins_per_interval[ind]
        return value
        
    def get_rediscretized_bin_edges(self, df):
        """
        Get the bin edges for the dataframe following rediscretization.

        Parameters
        ----------
        df : dataframe
            Pandas dataframe characterizing chromosomal intervals. Must contain
            columns ['chrom', 'start_ind', 'end_ind', 'signal']. Rows must be
            sorted in order of increasing genomic position.

        Returns
        -------
        bin_edges : array_like (n,)
            Bin edges following rediscretization.
        """
        ind0 = df['start_ind'].iloc[0]
        indf = df['end_ind'].iloc[-1]
        bin_edges = np.arange(ind0, np.floor((indf-ind0)/self.step)+1)
        bin_edges *= self.step
        if (indf - ind0) % self.step == 0:
            return bin_edges
        else:
            return np.append(bin_edges, indf)

    def get_rediscretized_intervals(self, bin_edges):
        """
        Get genomic intervals following rediscretization.

        Parameters
        ----------
        bin_edges : array_like (n,)
            Bin edges following rediscretization.

        Returns
        -------
        start_ind : array_like (n-1,)
            Starting position of genomic intervals
        end_ind : array_like (n-1,)
            Ending position of genomic intervals
        """
        start_ind = np.array(bin_edges[0:-1])
        end_ind = np.array(bin_edges[1:])
        return start_ind, end_ind

    def get_genomic_interval_edges(self, df):
        """
        Get the edges of the genomic intervals in a single vector.

        Parameters
        ----------
        df : dataframe
            Pandas dataframe characterizing chromosomal intervals. Must contain
            columns ['chrom', 'start_ind', 'end_ind', 'signal']. Rows must be
            sorted in order of increasing genomic position.

        Returns
        -------
        array_like (m,)
            Bin edges of genomic intervals
        """
        interval_edges = df['start_ind'].to_list()
        interval_edges.append(df['end_ind'].iloc[-1])
        return np.array(interval_edges)
