"""
Parse BED and WIG files to construct CSV files with rediscretized signals.

The `BEDParser` is used to read BED files containing ChIP-seq peak and
epigenetic subccompartment annotatations. The subcompartment annotations are
then rediscretized to approximate nucleosome scale resolution. `WIGParser` is
used to read aligned ChIP-seq signals and rediscretize those signals to
approximate nucleosome resolution. CSV files are created and outputted to the
appropriate directory to free up RAM.

By:     Joseph Wakim
Date:   January 7, 2020

"""

import csv

import pandas as pd
import numpy as np

from rediscretize import *
from file_util import *

autosomes = ['chr'+str(i) for i in range(1, 23)]
autosomes = autosomes[0:2]  # For debugging.
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

    def __init__(self, name, annotation, peaks, signals, output):
        """
        Initialize genome object.

        Get the date and time 

        Parameters
        ----------
        name : str
            Name to identify the genome.
        annotation : str
            Path to the BED file containing nucleosome annotations
        peaks : List[str]
            List of paths to the BED files containing ChIP-seq peak annotations
            (must be the same length as `WIG_paths`)
        signals : List[str]
            List of paths to the WIG files containing aligned ChIP-seq signals
        output : str
            Path to output directory into which parsed signals will be saved
        """
        self.name = name
        self.input_paths = {
            'annotation' : annotation, 'peaks' : peaks, 'signals' : signals}
        
        self.initialize_directories(output)
        self.parse_annotations(annotation)

    def initialize_directories(self, parent_dir):
        """
        Create directories and return file paths for storing epigenetic data.

        Create directories for ChIP-seq alignment signals, peak annotations,
        and epigenetic subcompartment classifications. Return a dictionary with
        file paths to the directories.

        Start by generating the date and time as a string for use in naming the
        outer-level directory. Then create outer-level directory for genome,
        and in this directory, create folders 'annotations', 'signals', and
        'peaks'. In each of these three subdirectories, create a folder called
        'chromosomes' for storing CSV files containing data from chromosomes.

        Parameters
        ----------
        parent_dir : str
            Path to directory into which all other directories will be created.
        """
        self.output_paths = self.make_output_paths(parent_dir)
        create_directories(self.output_paths.values())

    def make_output_paths(self, parent_dir):
        """
        Make paths for output directoies.

        Get the current date and time as a string. Then create a time-stamped
        directory for the genome. Add directories for annotations, peaks, and
        signals.

        Parameters
        ----------
        parent_dir : str
            Path to directory into which all other directories will be created.

        Returns
        -------
        output_paths : Dict[str: str]
            Labeled paths to output directories
        """
        dt_string = get_date_time_label()
        genome = self.name
        
        output_paths = {"genome" : parent_dir + '/' + genome + '_' + dt_string}
        
        output_paths['annotations'] = output_paths['genome'] + '/annotations'
        output_paths['annotations_chr'] = output_paths['annotations'] + \
            '/chromosomes'
        
        output_paths['peaks'] = output_paths['genome'] + '/peaks'
        for peak in self.input_paths['peaks']:
            peak = peak.split('/')[-1].split('.')[0]
            output_paths["peaks_"+peak] = output_paths['peaks'] + '/' + peak
            output_paths['peaks_'+peak+'_chr']=output_paths["peaks_"+peak]+\
                '/chromosomes'
        
        output_paths['signals'] = output_paths['genome'] + '/signals'
        for sig in self.input_paths['signals']:
            sig = sig.split('/')[-1].split('.')[0]
            output_paths["signals_"+sig] = output_paths['signals'] + '/' + sig
            output_paths['signals_'+sig+'_chr']=output_paths["signals_"+sig]+\
                '/chromosomes'

        return output_paths
    
    def parse_annotations(self, path):
        """
        Read epigenetic subcompartment annotations from a BED file.

        Parameters
        ----------
        patth : str
            Path to the BED file containing subcompartment annotations.
        """
        annotations = BEDParser(
            'annotation', path, [i for i in range(4)], annotation_headers
        )
        annotations.fill_all_missing_annotations()
        annotations.rediscretize_all_chromosomes(nuc_scale, repeat=True)
        annotations.save_data(self.output_paths['annotations_chr'])

    def read_peaks(self, paths):
        """
        Read ChIP-seq peak annotations from a BED file.

        Parameters
        ----------
        peak_paths : List[str]
            List of paths to the BED files containing ChIP-seq peak annotations
        """
        cols = [i for i in range(3)]
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

    def save_data(self, path):
        """
        Save single chromosomal data to CSV file.

        Parameters
        ----------
        path : str
            File path at which to save csv file.
        """
        df_to_csv(self.data, path)

    def replace_missing_annotations(self):
        """
        Replace missing subcompartment annotations with 'NA' in `self.data`.
        
        In the future, consider using more advanced imputation approach to
        filling missing data.
        """
        annotations = self.data.iloc[:,3].to_list()
        for i in range(len(annotations)):
            if annotations[i] != annotations[i]:
                self.data.iloc[i,3] = "NA"


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

    def rediscretize_all_chromosomes(self, step, repeat=False):
        """
        Rediscretize all chromosomes in `self.chromosomes`.
        
        Parameters
        ----------
        step : int
            Unit size following rediscretization
        repeat : bool, default False
            If true, vals repeated in expansion; else vals evenly distributed
        """
        redisc = Rediscretize(step, categorical=repeat, repeat=repeat)
        for chrom in self.chromosomes:
            print("Rediscretizing Chromosome: " + chrom)
            self.chromosomes[chrom].data = \
                redisc.rediscretize_chromosome(self.chromosomes[chrom].data)

    def save_data(self, dir_path):
        """
        Save all chromosomal data to CSV files.

        Parameters
        ----------
        dir_path : str
            Path to 'chromosomes' directory
        """
        for chrom in self.chromosomes:
            self.chromosomes[chrom].save_data(dir_path+'/'+chrom)

    def fill_all_missing_annotations(self):
        """Replace all missing annotations in the `BEDParser` object."""
        for chrom in self.chromosomes:
            self.chromosomes[chrom].replace_missing_annotations()
