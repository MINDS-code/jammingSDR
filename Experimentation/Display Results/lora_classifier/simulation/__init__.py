from sklearn.model_selection import train_test_split
import numpy as np

from lora_classifier.dataset.utils import filter_files
from lora_classifier.model import create_model, train_model_from_files, evaluate_model_from_files

def monte_carlo(dataset_name = 'Sim_IF_Dataset2', num_iterations= 10, val_split= 0.2, num_samples= 50, snr_range=list(range(0, 21, 4)), num_epochs= 10, batch_size= 8):
    """
    Monte Carlo simulation for the LoRa classifier.
    
    Parameters
    ----------
    dataset_name : str
        Name of the dataset to use. Default is 'Sim_IF_Dataset2'. // Simulated samples of IF signals.
    num_iterations : int
        Number of iterations to run. Default is 10.
    val_split : float
        Fraction of the dataset to use for validation. Default is 0.2.
    num_samples : int
        Number of samples to use from each SNR. Default is 50.
    snr_range : list
        List of SNRs to use. Default is [0, 4, 8, 12, 16, 20].
    num_epochs : int
        Number of epochs to train for. Default is 10.
    batch_size : int
        Batch size to use for training. Default is 8.

    Returns
    -------
    results : np.array of shape (num_iterations, len(snr_range), 2) with last two dimensions (prediction, label).
        Results of the simulation.
    snr_range : list
        List of SNRs used.    
    """
    dataset = filter_files(dataset_name, filter_snr = snr_range, filter_count = num_samples)
    results = []
    for i in range(num_iterations):
        print(f'Iteration: {i}/{num_iterations}')
        train_data, val_data = train_test_split(dataset, test_size=val_split, random_state=42)
        model = create_model()
        model = train_model_from_files(model, train_data, val_data, num_epochs, batch_size)
        results.append(evaluate_model_from_files(model, val_data, snr_range))
    return np.array(results), snr_range