import numpy as np
import re
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import find_peaks
from glob import glob


def load_signal_and_fs(path):
    """
    Load a vibration signal and compute its sampling frequency from a .txt file.

    The function reads header metadata (such as Max_X and NoOfItems) to extract
    the total duration and number of samples, and uses them to compute the sampling rate.
    It also loads the signal values, ignoring comment lines starting with '%'.

    Args:
        path (str): Path to the .txt file containing the signal and metadata.

    Returns:
        tuple:
            - signal (np.ndarray): The loaded 1D signal array.
            - fs (float): The computed sampling frequency (samples per second).

    Raises:
        ValueError: If the necessary metadata (duration or number of samples) cannot be extracted.
    """
    with open(path, 'r') as f:
        lines = f.readlines()

    duration = None
    N = None
    for line in lines:
        if line.startswith('%'):
            if 'Max_X' in line:
                match = re.search(r'Max_X:\s*=\s*([0-9.]+)', line)
                if match:
                    duration = float(match.group(1))
            elif 'NoOfItems' in line:
                match = re.search(r'NoOfItems:\s*=\s*([0-9]+)', line)
                if match:
                    N = int(match.group(1))
        else:
            break

    if duration is None or N is None:
        raise ValueError(
            "Failed to extract duration or number of samples from header")

    fs = N / duration
    signal = np.loadtxt(path, comments='%')

    return signal, fs
