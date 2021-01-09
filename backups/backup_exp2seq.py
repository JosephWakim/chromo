"""
Epigenetic Domain Classifier

Objective:
    
    Classify heterochromatin and euchromatin domains using experimental
    ChIP-seq. Generate sequences of epigenetic marks from ChIP-seq data.

Approach:
    
    Start by parsing WIG files to get sequences of ChIP-seq signal strengths.
    Distribute the signal across all base pairs in the window. Now parse BED
    files to identify the location of peaks in the ChIP-seq data. Obtain the
    distribution of ChIP-seq signals inside and outside of peaks.

    Parse the BED file containing epigenetic subcompartment annotations.
    Distribute the annotations to each base pair in the interval. Discretize 
    the genome into 200 base-pair bins, and label each bin with its most common
    subcompartment annotation. For each subcompartment, obtain the distribution 
    of ChIP-seq signals. Subtract the background signal obtained from the peak 
    analysis. Normalize by the mean and standard deviation. Train a classifier 
    to identify subcompartment types based on ChIP-seq signals.

    Within each subcompartment, identify the distribution of ChIP-seq signals
    for each epigenetic mark. Using cutoff strengths, we will assign zero, one
    or two tails of the nucleosome containing the mark. The exact cutoffs will
    be determined by grid search.

By:     Joseph Wakim
Date:   January 6, 2020

"""

import csv

import pandas as pd
import numpy as np

from rediscretize import *

autosomes = ['chr'+str(i) for i in range(23)]
nuc_scale = 200
WIG_headers = ['chrom', 'start_ind', 'end_ind', 'signal']
annotation_headers = ['chrom', 'start_ind', 'end_ind', 'annotation']
peak_headers = annotation_headers[0:3]


def separate_chromosomes(df):
    """
    Separate chromosomes specified in 'chrom' column of a Pandas dataframe.

    Parameters
    ----------
    df : dataframe
        Pandas dataframe containing chromosomal labels in a 'chrom' column
    
    Returns
    -------
    chromosomes : Dict[Chromosome objects]
        Original dataframe separated into a dictionary by chromosome
    """
    if 'chrom' in list(df.columns):
        chromosomes = {}
        for c in df['chrom'].unique():
			if c in autosomes:
                chromosomes[c] = Chromosome(c, df[df['chrom'] == c])
        return chromosomes

    else:
        raise ValueError("Data frame passed into `separate_chromosomes()` is \
            missing a 'chrom' column.")


class Genome:
    """Class representation of genome being characterized."""

    def __init__(self, name, annotation_path, peak_paths, signal_paths):
        """
        Initialize genome object.

        Parameters
        ----------
        name : str
            Name to identify the genome.
        annotation_path : str
            Path to the BED file containing nucleosome annotations
        peak_paths : List[str]
            List of paths to the BED files containing ChIP-seq peak annotations
            (must be the same length as `WIG_paths`)
        signal_paths : List[str]
            List of paths to the WIG files containing aligned ChIP-seq signals
        """
        self.name = name
        self.read_annotations(annotation_path)
        self.read_peaks(peak_paths)

    def read_annotations(self, path):
        """
        Read epigenetic subcompartment annotations from a BED file.

        Parameters
        ----------
        patth : str
            Path to the BED file containing subcompartment annotations.
        """
        name = 'annotation'
        cols = [i in range(4)]
        self.annotations = BEDParser(name, path, cols, annotation_headers)

    def read_peaks(self, paths):
        """
        Read ChIP-seq peak annotations from a BED file.

        Parameters
        ----------
        peak_paths : List[str]
            List of paths to the BED files containing ChIP-seq peak annotations
        """
        cols = [i in range(3)]
        self.peaks = {}
        for path in paths:
            name = path.split('.')[0]
            self.peaks[name] = BEDParser(name, path, cols, peak_headers)


class Chromosome:
    """Class representation of a chromosome."""

    def __init__(self, name, data):
        """
        Initialize the chromosome class.

        Parameters
        ----------
        name : str
            Name identifying the chromosome
        data : dataframe
            Data frame containing experimental data from the chromosome.
        """
        self.name = name
        self.data = data
        self.sort_genomic_intervals()

    def sort_genomic_intervals(self):
        """
        Sort data by genomic intervals within chromosome.

        This method requires that self.data contains a 'start_ind' column.
        """
        if 'start_ind' in list(self.data.columns):
            self.data.sort_values(by=['start_ind'])
        else:
            raise ValueError("Chromosome " + str(self.name) + " is missing a \
                starting index ('start_ind') column.")

    def rediscretize(self, step):
        """
        Rediscretize the chromosome by calling `rediscretize_chromosome()`.

        Parameters
        ----------
        step : int
            Unit size following rediscretization
        """
        self.data = rediscretize_chromosome(self.data, step)


class BEDParser:
    """Class representation of a BED file parser."""

    def __init__(self, name, path, cols, headers):
		"""
		Initialize Parser object.

        Open all the data contained in the BED file, then separate the data by
        chromosome.

		Parameters
		----------
		name : str
			Name to identify `Parser` object
		path : str
			File path to the BED file being parsed
		cols : List[int]
			List of column indices to read from the BED file
		headers : List[str]
			List of headers for dataframe
		"""
		self.name = name
        all_data = self.open_BED(path, cols, headers)
        self.chromosomes = separate_chromosomes(all_data)

    def open_BED(self, path, cols, headers):
		"""
		Open tab-delimited BED file.

		Parameters
		----------
		path : str
			Path to the BED file
		cols : List[int]
			List of column indices to read from the BED file
		headers : List[str]
			List of headers for dataframe

		Returns
		-------
		df : dataframe
			Pandas dataframe containing all rows in tab-delimited BED file
		"""
        df = pd.read_csv(path,header=None,sep='\t',usecols=cols,names=headers)
		return df

    def rediscretize_all_chromosomes(self, step):
        """
        Rediscretize all chromosomes in `self.chromosomes`.
        
        Parameters
        ----------
        step : int
            Unit size following rediscretization
        """
        for chrom in self.chromosomes:
            self.chromosomes[chrom].data = rediscretize_chromosome(
                self.chromosomes[chrom].data, step
            )


class WIGParser:
    """
    Class representation of a WIG file parser.
    
    The WIG file is parsed in a different way than the traditional BED file. 
    Due to it's large size, it is preferred not to store the entire WIG file
    in memory, so the file is parsed one chromosome at a time.
    """

    def __init__(self, name, path):
        """
        Initialize WIGParser object.

        Parameters
        ----------
        name : str
            Name to identify the `WIGParser` object
        path : str
            Path to the WIG file
        """
        self.chromosomes = {}
        self.open_all_autosomes(path)

    def open_all_autosomes(self, path):
        """
        Open ChIP-seq signal data from all autosomes.

        Parameters
        ----------
        path : str
            Path to the WIG file
        """
        for chrom in autosomes:
            self.chromosome[chrom] = self.open_chromosome(path, chrom)
    
    def open_chromosome(self, path, chrom):
        """
        Open ChIP-seq signal data from specified chromosome.

        Parameters
        ----------
         path : str
            Path to the WIG file
        chrom : str
            Chromosome identifier

        Returns
        -------
        chrom_obj : Chromosome object
            Data for chromosome in WIG file specified by chrom
        """
        start_inds, end_inds, signals = [], [], []
        found_chrom = False
        
        with open(path, mode='r') as f:
            csv_reader = csv.reader(f, delimiter='\t')
            for row in csv_reader:
                
                if row[0].startswith('#'):
                    continue
                
                elif: row[0] == chrom:
                    start_inds.append(row[1])
                    end_inds.append(row[2])
                    signals.append(row[3])
                    if not found_chrom:
                        found_chrom == True
                
                else:
                    if found_chrom:
                        df = rebuild_df(WIG_headers, chrom, start, end, signal)
                        chrom_obj = Chromosome(chrom, df)
                        chrom_obj.rediscretize(nuc_scale)
                        return chrom_obj
