import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import periodogram
from scipy.signal import find_peaks


def plot_signal_with_harmonics_and_faults(
    signal, fs, harmonics, faults,
    proximity_threshold=1.0,
    title="FFT with Harmonics and Faults"
):
    n = len(signal)
    freqs = np.fft.rfftfreq(n, d=1/fs)
    fft_vals = np.abs(np.fft.rfft(signal))

    fft_vals /= np.max(fft_vals)

    delta = 0.05
    harmonic_threshold = np.percentile(fft_vals, 90)
    neighborhood_width = 50 * fs / len(signal)

    plt.figure(figsize=(14, 6))
    plt.plot(freqs, fft_vals, label="FFT", color='gray')

    peaks, _ = find_peaks(fft_vals, height=0.05)

    result = []
    past_amp_h = 0
    for h in harmonics:
        mask = (freqs >= h - neighborhood_width) & (freqs <=
                                                    h + neighborhood_width)
        idx_peak = np.argmax(fft_vals[mask])
        peak_freq = freqs[mask][idx_peak]
        peak_amp = fft_vals[mask][idx_peak]

        amp_h = fft_vals[np.argmin(np.abs(freqs - h))]

        is_peak_stronger = peak_amp > amp_h + delta
        is_harmonic_strong = amp_h > past_amp_h

        matched_fault = None
        for fault_name, fault_freq in faults:
            if abs(peak_freq - fault_freq) <= proximity_threshold:
                matched_fault = fault_name
                break

        if (is_peak_stronger or is_harmonic_strong) and matched_fault:
            plt.plot(peak_freq, peak_amp, 'ro')
            plt.text(peak_freq, peak_amp * 1.05,
                     f"{matched_fault}", ha='center', color='red')
            result.append({
                "Type": matched_fault,
                "Freq": round(peak_freq, 2),
                "Amp": round(peak_amp, 3)
            })
        else:
            plt.axvline(h, color='b', linestyle='--', alpha=0.5)
            plt.text(h, 0.02, f"H({round(h, 1)})",
                     ha='center', fontsize=9, color='blue')
    past_amp_h = fft_vals[np.argmin(np.abs(freqs - h))]
    for name, f in faults:
        plt.axvline(f, color='r', linestyle=':', alpha=0.3)

    plt.title(title)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Normalized Amplitude")
    plt.grid(True)
    plt.xlim(0, harmonics[-1] * 1.2)
    plt.legend()
    plt.tight_layout()
    plt.show()

    return pd.DataFrame(result)
