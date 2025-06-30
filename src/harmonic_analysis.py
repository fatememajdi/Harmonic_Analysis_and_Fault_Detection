import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def analyze_harmonics_and_possible_faults(
    signal, fs, base_freq, num_harmonics=10,
        harmonics=None, fault_freqs=None,
        freq_tolerance_factor=5,
        default_prom_ratio=0.3
):
    """
    Analyze a time-domain signal to identify harmonics and detect possible fault-related frequency peaks.

    Parameters
    ----------
    signal : numpy.ndarray
        Input time-domain signal data array.
    fs : float
        Sampling frequency of the signal in Hz.
    base_freq : float
        Base frequency (e.g., rotational speed frequency) to calculate harmonics.
    num_harmonics : int, optional, default=10
        Number of harmonics to analyze starting from base_freq.
    harmonics : array-like or None, optional
        Specific array of harmonic frequencies to consider. If None, harmonics are generated as multiples of base_freq.
    fault_freqs : dict or None, optional
        Dictionary containing fault names with their fault characteristic frequencies and optional threshold ratios:
        {
            "FaultName": {"freq": float, "threshold_ratio": float (optional)},
            ...
        }
    freq_tolerance_factor : float, optional, default=5
        Factor to multiply with frequency resolution to define tolerance window for matching peaks to harmonics.
    default_prom_ratio : float, optional, default=0.3
        Default amplitude threshold ratio used for fault detection if not specified in fault_freqs.

    Returns
    -------
    pandas.DataFrame
        DataFrame listing suspicious peaks associated with faults. Columns include:
        - Fault: Name of the detected fault.
        - Match_Type: Either "Exact_On_Harmonic" (peak very close to harmonic) or "Near_Harmonic" (peak near harmonic).
        - Harmonic: Harmonic frequency associated with the peak.
        - Peak_Freq: Frequency of the detected peak.
        - Amplitude: Amplitude of the detected peak.
        - Delta_amp: Absolute difference between peak amplitude and fault threshold (ratio * harmonic amplitude).

    Detailed Explanation
    --------------------
    1. Compute the FFT of the input signal and normalize by the signal length.
    2. Extract only the positive frequency components and their corresponding amplitudes.
    3. Calculate frequency resolution and determine tolerance window based on freq_tolerance_factor.
    4. Generate harmonics as multiples of base_freq unless explicitly provided.
    5. For each harmonic:
       - Find the closest frequency bin in the FFT result.
       - Record the harmonic frequency and its corresponding amplitude.
    6. For each harmonic and each fault frequency:
       - Identify frequency indices within the tolerance window around the harmonic.
       - Find the peak frequency and amplitude within this window.
       - Calculate the frequency difference (delta_f) between the peak and harmonic.
       - Compute delta_amp as the absolute difference between the peak amplitude and the fault threshold.
       - If the peak lies very close to the harmonic (within frequency resolution) and is stronger than the previous harmonic's amplitude, mark it as "Exact_On_Harmonic".
         Skip the first harmonic as there is no previous harmonic.
       - Otherwise, if the peak amplitude exceeds the fault threshold (ratio * harmonic amplitude), mark it as "Near_Harmonic".
    7. Collect all suspicious peaks.
    8. From multiple faults detected at the same peak frequency, keep only the fault with the smallest delta_amp (closest to threshold).
    9. Plot the FFT spectrum with harmonics and detected fault peaks highlighted.
    10. Return a DataFrame of filtered suspicious fault peaks.

    """

    N = len(signal)
    fft_vals = np.abs(np.fft.fft(signal)) / N
    freqs = np.fft.fftfreq(N, 1/fs)
    idx = freqs >= 0
    freqs = freqs[idx]
    fft_vals = fft_vals[idx]

    freq_resolution = fs / N
    freq_tolerance = freq_tolerance_factor * freq_resolution

    if harmonics is None or len(harmonics) == 0:
        harmonics = np.array(
            [i * base_freq for i in range(1, num_harmonics + 1)])

    harmonic_info = []
    for h in harmonics:
        idx_closest = np.argmin(np.abs(freqs - h))
        peak_amp = fft_vals[idx_closest]
        harmonic_info.append((h, h, peak_amp))

    suspicious_peaks = []

    if fault_freqs:
        for i, (h, _, h_amp) in enumerate(harmonic_info):
            indices_in_range = np.where(
                (freqs >= h - freq_tolerance) & (freqs <= h + freq_tolerance))[0]
            if len(indices_in_range) == 0:
                continue

            peak_idx_in_range = indices_in_range[np.argmax(
                fft_vals[indices_in_range])]
            peak_freq = freqs[peak_idx_in_range]
            amp_fault = fft_vals[peak_idx_in_range]
            delta_f = abs(peak_freq - h)
            for name, info in fault_freqs.items():
                ratio = info.get("threshold_ratio", default_prom_ratio)
                delta_amp = abs(amp_fault - ratio * h_amp)

                if delta_f <= freq_resolution:
                    if i == 0:
                        continue
                    amp_prev = harmonic_info[i - 1][2] if i > 0 else 0
                    if amp_fault > amp_prev:
                        suspicious_peaks.append({
                            "Fault": name,
                            "Match_Type": "Exact_On_Harmonic",
                            "Harmonic": h,
                            "Peak_Freq": peak_freq,
                            "Amplitude": amp_fault,
                            "Delta_amp": delta_amp
                        })
                else:
                    if amp_fault >= ratio * h_amp:
                        suspicious_peaks.append({
                            "Fault": name,
                            "Match_Type": "Near_Harmonic",
                            "Harmonic": h,
                            "Peak_Freq": peak_freq,
                            "Amplitude": amp_fault,
                            "Delta_amp": delta_amp
                        })

    best_fault_for_peak = {}

    for peak in suspicious_peaks:
        peak_freq = peak['Peak_Freq']
        delta_amp = peak.get('Delta_amp', float('inf'))

        if peak_freq not in best_fault_for_peak or delta_amp < best_fault_for_peak[peak_freq]['Delta_amp']:
            best_fault_for_peak[peak_freq] = peak

    filtered_peaks = list(best_fault_for_peak.values())
    df = pd.DataFrame(filtered_peaks)

    plt.figure(figsize=(15, 7))
    plt.plot(freqs, fft_vals, label='FFT Spectrum', alpha=0.6)

    for h, _, pa in harmonic_info:
        plt.axvline(h, color='r', linestyle='--', alpha=0.3)
        plt.plot(h, pa, 'go')

    for _, row in df.iterrows():
        pf, pa = row['Peak_Freq'], row['Amplitude']
        label = f"{row['Fault']} ({pf:.1f} Hz)"
        plt.plot(pf, pa, 'ro')
        plt.text(pf, pa * 1.05, label, color='red', fontsize=9, rotation=45)

    plt.title(f'Harmonics and Fault-Related Peaks (±{freq_tolerance:.2f} Hz)')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.xlim(0, harmonics[-1] * 1.2)
    plt.tight_layout()
    plt.show()

    return df
