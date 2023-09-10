# LoRa Signal Classification

Code and Data for the simulation results presented in the following paper: 
Tarun Rao Keshabhoina and Marcos M. Vasconcelos. “Data-driven classification of low-power communication signals by an unauthenticated user using a Software Radio” IEEE Asilomar Conference on Signals, Systems, and Computers - 2023 [Accepted].

## Table of Contents

- [Description](#description)
- [Installation](#installation)
- [Usage](#usage)
- [Contact](#contact)

## Description

This project implements a feed-forward neural network implementation for estimating the bandwidth and spreading factor configurations from LoRa signal preambles. We consider two separate datasets for training and validation. The training dataset features 9,000 files containing LoRa preamble data from SNR ranging 0 dB to 20 dB, and the validation dataset features 6,480 files containing LoRa preamble data from SNR ranging -15 dB to 20 dB. The code for running the simulation, plotting and storing results can be found in the notebook 'Code.ipynb'.

## Installation

### Prerequisites

- Ensure you have **Python 3.x** installed on your system.
- Required packages: 
  - TensorFlow
  - SciPy
  - Scikit-learn
  - Matplotlib
  - NumPy

### Installing on local machine

1. Clone the Repository:
   git clone https://github.com/MINDS-code/jammingSDR.git

2. Navigate to the Project Directory:
   cd jammingSDR

3. Install the Required Dependencies:
   pip install tensorflow scipy scikit-learn matplotlib numpy

4. Access the Asilomar 2023 Resources:
   If you're particularly interested in the resources dedicated for the Asilomar 2023 conference:
   cd Asilomar23

5. Update path variables:
   Please update all path varaibles in 'Code.ipynb' (variables in UPPERCASE, starting with 'PATH') to reflect the paths to your local file directories.

## Usage

Navigate to the notebook 'Code.ipynb'. Open and run all cells in the notebook using Jupyter Notebook or any platform that supports IPython notebooks.

Results generated from the simulation can be found in the Results directory, to regenerate plots.

Figures displayed in the simulation are automatically saved in the Figures directory after execution.

## Contact
GitHub: https://github.com/TarunRao-K
Email: tarunrao@vt.edu, m.vasconcelos@fsu.edu
