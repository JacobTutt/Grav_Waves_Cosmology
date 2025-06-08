# **Cosmology with Gravitational Waves**

## Author: Jacob Tutt, Department of Physics, University of Cambridge

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)


## Description
This project outlines some of the key statisitcal techniques used during Gravitational Wave detection. It looks at signal detection, sky localisation for multi-messenger astronomy and cosmological inference. The work is divided into three key stages:
1. **Matched Filtering for Signal Detection**
- The Time-of-arrival at detectors can be extracted by using matched filtering to determine the signal to noise ratio at all time points and statisticaly identify the coalescence time ($t_c^{\mathrm{geo}}$) of the true signal. Using multiple detectors (H1, L1, V1) the source's location on the sky can be determined through triangulation.

2.	**Bayesian Inference for Parameter Estimation**
- Using the likelihoods within the frequency domain, posterior distributions for the source parameters are drawn by using nested sampling: right ascension (RA), declination (Dec), polarization angle ($\psi$), coalescence time ($t_c^{\mathrm{geo}}$), and luminosity distance ($D_L$).

3.	**Cosmological Inference using Galaxy Redshifts**
- Posterior RA and Dec values can be used to indentify potential host galaxies with a mock catalog. The galaxies known redshifts can be combined with the inferred luminosity distances to determine the Hubble constant, $H_0$.



This repository forms part of the submission for the MPhil in Data Intensive Science's A5 Gravitational Waves Course at the University of Cambridge.


## Table of Contents
- [Data](#data)
- [Pipeline Functionalities](#pipelines)
- [Notebooks](#notebooks)
- [Results](#results)
- [Installation](#installation-and-usage)
- [License](#license)
- [Support](#support)
- [Author](#author)


## Data

- The [datasets](data) used to preform the analysis within this repository are included to allow reproducibility and for the work to be built upon by others. The aim of this work is the focus on the identification and analysis of the detected signals and therefore data was artificially generated using `bilby`. The main file structure is outlines below:

| Directory      | Description |
|----------------|-------------|
| [H1](data/H1)  | **Hanford Interferometer (LIGO), Washington:** The measured frequency-domain strain data and the detectors noise power spectral densities (PSD). |
| [L1](data/L1)  | **Livingston Interferometer (LIGO), Louisiana:** The measured frequency-domain strain data and the detectors noise power spectral densities (PSD). |
| [V1](data/V1)  | **Virgo Interferometer (EGO), Italy:** The measured frequency-domain strain data and the detectors noise power spectral densities (PSD). |
| [Waveform](data/Waveform)  | Frequency-domain gravitational waveform generated at a fiducial distance of $D_L$ = 1 Gpc. |
| [Catalog](data/Catalog)  | A mock galaxy catalog to cross reference posterior sky positions with to indentify potential matches for host galaxies|

### Notes
- All strain and PSD files are in plain text format (`.txt`).
- Units are provided in [$Hz/Hz^{-1}$] with real/imaginary strain components seperately.
- The `'IMRPhenomPv2'` waveform model was used to generate simulated signals and Gaussian noise consistent with the design PSD of each detector.

## Pipelines

To allow the analysis to be easy to follow the majority of code important to theory and implementation is defined within the notebooks. However, longer iterative functions and plotting routines are provided in the [`src`](src) directory:

| Files | Description |
|----------|-------------|
| [match filtering](src/matchfilter.py) | Functions for performing matched filtering over parameter grids to try and optimize detection. Plot skymaps with confidence levels for time delays using bilby.|
| [posterior plotting](src/posterior_plotting.py) | Provides plotting functions for `dynesty` result outputs. Includes tabuluar summaries, corner plots, RA and Dec posterior sky maps and convergence diagnostics. |

## Notebooks

The [notebooks](notebooks) in this repository serve as walkthroughs for the analysis performed. They include derivations of the mathematical implementations, explanations of key choices made, and present the main results. Three notebooks are provided:

| Notebook | Description |
|----------|-------------|
| [0_Matched_Filter.ipynb](notebooks/0_Matched_Filter.ipynb) | Implements matched filtering to detect the gravitational wave signal within noisy time-series data and isolate the time of coalescence using the signal-to-noise ratio (SNR). By analysing time delays across multiple detectors, the sky position of the source is also be constrained. |
| [1_Bayesian_Analysis.ipynb](notebooks/1_Bayesian_Analysis.ipynb) | Performs a more rigorous Bayesian analysis on the time-series data by employing nested sampling to obtain posterior distributions for the parameters RA, Dec, $\psi$, $t_c^{\mathrm{(geo)}}$, and $D_L$. Also compares results from different detector combinations. |
| [2_Cosmology.ipynb](notebooks/2_Cosmology.ipynb) | Uses the posterior samples to derive the Hubble constant from the inferred luminosity distance and redshift provided by the mock galaxy catalog |


## Results

Although the full runtime of all notebooks is $\approx$ 30 minutes, the repository contains stored [results](results) for the pipelines intermediatary results. This allows the more computationally demanding tasks such as Nested Sampling runs and optimisation to be skipped if desired. These process can be easily toggled on and off using flags like `run_sampling` defined at the beginning of each notebook.

---

## Installation and Usage

To reproduce the analysis in these notebooks, please follow these steps:

### 1. Clone the Repository

Clone the repository from the remote repository (GitLab) to your local machine.

```bash
git clone https://gitlab.developers.cam.ac.uk/phy/data-intensive-science-mphil/assessments/a5_coursework/jlt67.git
cd jlt67
```

### 2. Create a Fresh Virtual Environment
It is recommended to use a clean virtual environment to avoid dependency conflicts.
```bash
python -m venv env
source env/bin/activate   # For macOS/Linux
env\Scripts\activate      # For Windows
```

### 3. Install the dependencies
#### 3.1 Navigate to the repository’s root directory and install the PyPi package dependencies:
```bash
cd jlt67
pip install -r requirements.txt
```

### 4. Set Up a Jupyter Notebook Kernel
To ensure the virtual environment is recognised within Jupyter notebooks, set up a kernel:
```bash
python -m ipykernel install --user --name=env --display-name "Python (Grav Waves Venv)"
```

### 5. Run the Notebooks
Open the notebooks and select the created kernel **(Python (Grav Waves Venv))** to run the code.



## For Assessment
- The associated project report can be found under [Project Report](report/report.pdf). 

## License
This project is licensed under the [MIT License](https://opensource.org/license/mit/) - see the [LICENSE](LICENSE) file for details.

## Support
If you have any questions, run into issues, or just want to discuss the project, feel free to:
- Open an issue on the [GitHub Issues](https://github.com/JacobTutt/GravitationalWavesCW/issues) page.  
- Reach out to me directly via [email](mailto:jacobtutt@icloud.com).

## Authors
This project is maintained by Jacob Tutt 


## Declaration of Use of Autogeneration Tools
This report made use of Large Language Models (LLMs) to assist in the development of the project.
These tools have been employed for:
- Formatting plots to enhance presentation quality.
- Performing iterative changes to already defined code.
- Debugging code and identifying issues in implementation.
- Helping with Latex formatting for the report.
- Identifying grammar and punctuation inconsistencies within the report.
- Helping to generate the repository's metadata files.