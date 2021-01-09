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
    """
    chrom = np.unique(df['chrom'].to_list())
    if len(chrom) != 1:
        raise ValueError("Rediscretize only one chromosome at a time.")


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
    return pd.dataframe(data=data)


class Bin:
    """Class representation of rediscretized bin holding signal values."""

    def __init__(self, values, fractions, width):
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
        """
        self.check_attribute_compatability(values, fractions)
        self.values = values
        self.fractions = fractions
        self.width = width

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
        # Calculate rediscretized genomic interval bounds for single chromosome
        headers = list(df.columns)
        check_single_chrom(df)
        bin_edges = self.get_rediscretized_bin_edges(df)
        start, end = self.get_rediscretized_intervals(bin_edges)
        interval_edges = self.get_genomic_interval_edges(df)
        bins_per_interval = np.divide(np.ediff1d(interval_edges), self.step)
        bins = self.categorize_bins(df, bins_per_interval)

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
        while i <= len(bins_per_interval):
            interval = bins_per_interval[i] - remainder
            values = []
            fractions = []
            remainder = 0
            j = 0
            while interval > 1:
                bins.append(Bin(df.iloc[i,3], 1, self.step))
                interval -= 1
            while interval < 1 and interval != 0:
                fractions.append(interval)
                values.append(df.iloc[i+j,3])
                j += 1
                interval += bins_per_interval[i+j]
            if interval != 0:
                remainder = interval - 1
                fractions.append(1-np.sum(fractions))
                values.append(df.iloc[i+j,3])
                bins.append(Bin(values, fractions, self.step))
                i += j
            else:
                i += j+1
        return bins











            
            remainder = interval - 1
            fractions.append(1 - np.sum(fractions))
            i += j

                
                
                
                
                
                fractions.append(interval)


                interval += 

















        # Classify homogeneous and heterogeneous bins
        classes
        for interval in bins_per_interval:
            for i







        # Classify homogeneous and heterogeneous bins
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        start, end = self.get_rediscretized_intervals(bin_edges)
        interval_edges = self.get_genomic_interval_edges(df)
        bins_per_interval = np.divide(np.ediff1d(interval_edges), self.step)

        # Rediscretize signal into new genomic intervals
        self.signals = np.empty(start.shape[0])
        remainder = 0
        remainder_signal = 0
        self.fill_ind = 0
        
        for index, row in df.iterrows():
            full, signal, remainder, remainder_signal = rediscretize_interval(
                row.iloc[3], bins_per_interval[i], remainder, remainder_signal
            )
            for i in range(full):
                self.signals[self.fill_ind] = signal
                self.fill_ind += 1

        if remainder != 0:
            self.signals[-1] += remainder_signal

        signals = self.signals
        self.signals = None

        return rebuild_df(headers, chrom, start, stop, signals)
        
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
        bin_edges = np.arange(ind0, floor((indf-ind0)/self.step))
        if (indf - ind0) % self.step == 0:
            return bin_edges
        else:
            return np.append(bin_edges, indf)

    def get_rediscretized_intervals(bin_edges):
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

    def rediscretize_interval(signal, bins_in_interval, remainder, remainder_signal):
        """
        Distribute signal from an interval into new bins.

        Parameters
        ----------
        signal : float
            Signal obtained for the original interval
        bins_in_interval : float
            Number of new bins fitting in the old interval
        remainder : int
            Number of base pairs hanging over from the previous interval
        remainder_signal : float
            Signal obtained from the remaining base pairs from previous interval

        Returns
        -------
        full : int
            Number of full bins from new discretization fitting in old signal
        new_signal : float
            Signal to assign to full bins at new discretization
        remainder : int
            Number of base pairs remaining after the current interval
        remainder_signal : float
            Signal obtained from the remaining base pairs in current interval
        """
        if bins_in_interval < 1:

            # If interval w/ remainder is < self.step, add to remainder
            if bins_in_interval * self.step + remainder < self.step:
                remainder_signal += signal
                remainder += bins_in_interval * self.steps
                return(0, None, remainder, remainder_signal)

            else:
                new_signal = signal + remainder_signal
        
        
        
        
        
        new_signal = signal / bins_in_interval
        
        if bins_in_interval * self.step + remainder < self.step:
            remainder_signal += signal * bins_in_interval * self.step

        if remainder != 0:
            bins_in_interval = self.add_overlapping_signal(new_signal, remainder, remainder_signal)



    def case1(self, signal, bins_in_interval, remainder, remainder_signal):
        """
        Rediscretized bin and remainder smaller than original genomic interval.

        Parameters
        ----------
        signal : float
            Signal obtained for the original interval
        bins_in_interval : float
            Number of new bins fitting in the old interval
        remainder : int
            Number of base pairs hanging over from the previous interval
        remainder_signal : float
            Signal obtained from the remaining base pairs from previous interval

        Returns
        -------
        full : int
            Number of full bins from new discretization fitting in old signal
        new_signal : float
            Signal to assign to full bins at new discretization
        remainder : int
            Number of base pairs remaining after the current interval
        remainder_signal : float
            Signal obtained from the remaining base pairs in current interval
        """
        remainder_signal = self.add_signal(remainder_signal, signal)






        remainder += bins_in_interval * self.

    def add_signal(self, signal1, len1, signal2, len2):
        """
        Add together two signals.

        Parameters
        ----------
        signal1 : float or List[str]
            First signal being added together
        len1 : int
            Number of base pairs represented by signal1
        signal2 : float or List[str]
            Second signal being added together
        len2 : int
            Number of base pairs represented by signal2
        
        Returns
        -------
        combined_signal : float or List[str]
            Combined result of the first and second signal
        """
        if self.categorical:
            return [signal1] * len1 + [signal2] * len2
        else:
            return signal1 + signal2
        
    


















def rediscretize_chromosome(df, step, repeat=False):
    """
    Rediscretize chromosomal signals.

    For memory efficiency, only rediscretize one chromosome at a time. If df
    contains more than one chromosome, first split the dataframe by chromosome
    using `separate_chromosomes()`.

    Parameters
    ----------
    df : dataframe
        Pandas dataframe characterizing genomic intervals w/ columns ['chrom',
        'start_ind', 'end_ind', 'signal']. Rows should be sorted in order of
        genomic intervals with increasing start/end indices
    step : int
        Unit size following rediscretization
    repeat : bool, default False
        If true, values repeated during expansion; else vals evenly distributed
    
    Returns
    -------
    dataframe
        Rediscretized dataframe containing redistributed signals
    """
    ind0 = df["start_ind"].iloc[0]
    indf = df["end_ind"].iloc[-1]
    headers = list(df.columns)
    chrom = np.unique(df['chrom'].to_list())
    if len(chrom) != 1:
        raise ValueError("Rediscretize only chromosome at a time.")
    
    print("expanding signal")
    expanded_signal = expand_bins(df, 3, repeat)
    
    print("distributing signal")
    start, end, signal = distribute_bins(expanded_signal, [ind0, indf], step)
    
    return rebuild_df(headers, chrom, start, stop, signal)
    







def expand_bins(df, col, repeat):
    """
    Distribute signals within bins of dataframe

    Parameters
    ----------
    df : dataframe
        Pandas dataframe characterizing genomic intervals w/ columns ['chrom',
        'start_ind', 'end_ind', 'signal']
    col : int
        Column index of signal or label to distribute
    repeat : bool
        If true, values repeated during expansion; else vals evenly distributed

    Returns
    -------
    array_like (n,)
        Signal (numeric) or label (str) expanded to base-pair discretization
    """
    expanded_signal = []
    num_rows = len(df.index)

    for i in range(num_rows):
        width = df.iloc[i,2] - df.iloc[i,1]
        seg = np.empty(width,dtype=str)
        for j in range(width):
            seg[j] = df.iloc[i, col]
        if not repeat:
            seg /= width
        expanded_signal.append(seg)
    
    return np.hstack(expanded_signal)


def distribute_bins(expanded_signal, ind_range, step):
    """
    Distribute signals or labels from expanded bins into new bins.

    When distributing signals, take sum of signals merged into new bin. When
    distributing labels, take most common label merged into new bin. When two
    or more labels occur at max frequency, randomly select from these labels.

    Parameters
    ----------
    expanded_signal : array_like (n,)
        Signal (numeric) or label (str) expanded to base-pair discretization
    ind_range : Tuple[int, int]
        Range of genomic indices across all bins, in form (ind0, indf)
    step : int
        Unit size following rediscretization

    Returns
    -------
    start_inds : array_like (m,)
        Starting indices of genomic intervals
    end_inds : array_like (m,)
        Ending indices of genomic intervals
    array_like (m,)
        Signal (numeric) or label (str) distributed into new bins
    """
    num_bp = ind_range[1] - ind_range[0]
    num_steps = ceil(num_bp / step)
    start_inds, end_inds = get_indices(ind_range, step, num_steps)
    remainder = num_bp % step
    trailing_zeros = np.zeros(remainder)
    expanded_signal = np.array(expanded_signal)
    expanded_signal = np.concatenate((expanded_signal, trailing_zeros), axis=0)
    expanded_signal = np.reshape(expanded_signal, (num_steps, step))

    if expanded_signal.dtype == float or expand_signal.dtype == int:
        return start_inds, end_inds, np.sum(expanded_signal, axis=1)
    else:
        labels = []
        for i in range(expanded_signal.shape[0]):
            unique, counts = np.unique(expanded_signal[i], return_counts=True)
            labels.append(
                unique[
                    np.random.choice(
                        np.argwhere(
                            counts==np.amax(counts)
                        ).flatten()
                    )
                ]
            )
        return start_inds, end_inds, np.array(labels)


def get_indices(ind_range, step, num_steps):
    """
    Get starting and ending indices at specified discretization.

    Parameters
    ----------
    ind_range : Tuple[int, int]
        Range of genomic indices across all bins, in form (ind0, indf)
    step : int
        Unit size following rediscretization
    num_step : int
        Number of steps following rediscretization
    
    Returns
    -------
    start_inds : array_like (m,)
        Starting indices of genomic intervals
    end_inds : array_like (m,)
        Ending indices of genomic intervals
    """
    start_inds = [ind_range[0] + step * i for i in range(num_steps)]
    end_inds = [ind_range[0] + step * (i+1) for i in range(num_steps-1)]
    end_inds.append(ind_range[1])
    return start_inds, end_inds


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
    return pd.dataframe(data=data)
