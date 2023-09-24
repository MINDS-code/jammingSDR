# jammingSDR

Readme for experimentation on SDR Dataset

step 1: Download SDR_Dataset2.

step 2: Update path vairables to match local file system.

step 3: Run the python notebook to perform simulation.

step 4: use files acc.py, and roc.py in 'Display Results' directory to plot results from simulation.

Notes: 
1. Make sure to match all path variables in python notebook before execution.

2. Dataset Directory Information:
    SDR_Dataset: A dataset of .bin files containing IQ data recorded using SDR.
    SDR_Dataset_trimmed_data: A dataset of .bin files containing data for the same packets in SDR_Dataset. But these files are of fixed length; Containing noise preceeding the preamble if packet was small, and contains trimmed portion of the preamble if packet was long.
    SDR_Dataset2: Instantaneous Frequency dataset, after feature extraction on SDR_Dataset_trimmed.