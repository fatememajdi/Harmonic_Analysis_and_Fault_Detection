
# Harmonic Analysis and Fault Detection

---

## 1. Introduction

This project implements signal analysis methods to detect harmonics and identify fault-related frequencies in vibration signals. The approach is based on FFT (Fast Fourier Transform) to analyze frequency domain features and compare them against known fault signatures using configurable thresholds.

---

## 2. Prerequisites

- Python 3.11 or higher  
- Required libraries (specified in `requirements.txt`):
  - numpy
  - pandas
  - scipy
  - matplotlib

Install dependencies with:  
```bash
pip install -r requirements.txt
```

---

## 3. Project Structure

```
signal_classification/
├── data/
│   ├── raw/                # Raw input signal files (.txt)
│   └── processed/          # Processed analysis results (.csv)
├── notebooks/              # Jupyter notebooks for running and evaluating analysis
├── src/                    # Source code modules
├── requirements.txt        # Python package requirements
└── README.md               # This file
```

---

## 4. How to Run

1. Clone this repository:

```bash
git clone https://github.com/fatememajdi/Harmonic_Analysis_and_Fault_Detection.git
cd signal_classification
```

2. Install required dependencies:

```bash
pip install -r requirements.txt
```

3. Place your vibration signal `.txt` files in the `data/raw/` folder.

4. Execute the analysis notebook located at `notebooks/evaluate.ipynb` to perform harmonic analysis and fault detection.

---

## 5. Outputs

- Results are saved as CSV files in the `data/processed/` directory.
- Visual plots are generated during the analysis to visualize the frequency spectrum, harmonics, and detected faults.

---

## 6. Maintenance and Updates

- Keep the `requirements.txt` updated with necessary packages and versions.
- Re-run the analysis when making significant changes to data preprocessing or detection logic.
- Regularly check input data quality to ensure consistent results.

---
