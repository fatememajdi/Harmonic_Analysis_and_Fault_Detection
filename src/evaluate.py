import os
import pandas as pd
import numpy as np
from preprocessing import load_signal_and_fs
from harmonic_analysis import analyze_harmonics_and_possible_faults


def batch_analyze_folder(folder_path, output_folder, fault_freqs):
    """
    Process multiple signal files in a folder, analyze harmonics and possible faults, 
    and save the results as CSV files.

    Parameters:
    -----------
    folder_path : str
        Path to the input folder containing signal text files (.txt).
    output_folder : str
        Path to the folder where analysis CSV files will be saved.
    fault_freqs : dict
        Dictionary defining fault frequencies and parameters (see analyze_harmonics_and_possible_faults).

    Returns:
    --------
    list of str
        List of file paths to the generated CSV analysis files.

    Description:
    -------------
    1. Checks if the output folder exists; creates it if not.
    2. Iterates over all '.txt' files in the input folder.
    3. For each file:
        - Loads the signal and sampling frequency using `load_signal_and_fs`.
        - Calls `analyze_harmonics_and_possible_faults` to perform the analysis.
        - Prints the resulting DataFrame.
        - Saves the analysis results to a CSV file in the output folder.
    4. Returns a list of all saved CSV file paths.
    """

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    output_files = []

    for filename in os.listdir(folder_path):
        if filename.endswith('.txt'):
            filepath = os.path.join(folder_path, filename)
            print(f"Processing file: {filename}")

            signal, fs = load_signal_and_fs(filepath)

            df = analyze_harmonics_and_possible_faults(signal, fs=fs, base_freq=16.5,
                                                       num_harmonics=10,
                                                       fault_freqs=fault_freqs,
                                                       freq_tolerance_factor=8)

            print(df)
            output_csv = os.path.join(
                output_folder, os.path.splitext(filename)[0] + '_analysis.csv')
            df.to_csv(output_csv, index=False)
            output_files.append(output_csv)

    return output_files
