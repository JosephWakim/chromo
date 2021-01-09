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


def rediscretize_chromosome(df, step):
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
    
    expanded_signal = expand_bins(df, 3)
    start, end, signal = distribute_bins(expanded_signal, [ind0, indf], step)
    
    return rebuild_df(headers, chrom, start, stop, signal)
    

def expand_bins(df, col, repeat=False):
    """
    Distribute signals within bins of dataframe

    Parameters
    ----------
    df : dataframe
        Pandas dataframe characterizing genomic intervals w/ columns ['chrom',
        'start_ind', 'end_ind', 'signal']
    col : int
        Column index of signal or label to distribute
    repeat : bool, default False
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
        seg = np.empty(width)
        for j in range(width):
            seg[j] = df.iloc[i, col])
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
