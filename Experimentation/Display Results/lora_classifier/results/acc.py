import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from scipy.stats import t
import numpy as np
import os

FONT_SIZE = 14
FONT_SIZE_LEGEND = 12
PATH_FIGURES = '/My Drive/LoRa_Detection/Simulation/Simulation Results/Figures/'

def select_configs(results, config_choice):
    return results[:, config_choice, :].reshape(-1, results.shape[-1])

def plot_results(results, snr_range, label=None, cf = 0.95):
    n = results.shape[0]
    means = np.mean(results, axis=0)
    stds = np.std(results, axis=0)
    h = stds * t.ppf((1 + cf) / 2, n - 1) / np.sqrt(n)
    cis = np.vstack((means - h, means + h))
    plt.plot(snr_range, means, 'o-', markersize= 3, label=label)
    plt.fill_between(snr_range, cis[0, :], cis[1, :], alpha=0.2, label=f'Confidence {int(cf*100)}%')

def display_results_BW(results, snr_range, path_save=None):

    BW_values = [125, 250, 500]
    plt.figure()
    plt.rc('font', size=FONT_SIZE)
    plt.grid()
    plt.yticks(range(0, 101, 10), fontsize=FONT_SIZE_LEGEND)
    plt.xticks(snr_range[::2], fontsize=FONT_SIZE_LEGEND)
    plt.title('Classification Accuracy by Bandwidth')
    plt.xlabel('Signal to Noise Ratio (dB)')
    plt.ylabel('Accuracy (%)')
    for idx, bw in enumerate(BW_values):
        config_choice = [idx*6+i for i in range(6)]
        reshaped_results = select_configs(results, config_choice)
        plot_results(reshaped_results, snr_range, label=f'BW {bw} kHz')
    plt.legend(loc='lower right', fontsize=FONT_SIZE_LEGEND)
    if path_save is not None:       
        plt.savefig(os.path.join(path_save, 'ACC_Vs_SNR_BW.pdf'))     
    plt.show()

def display_results_SF(results, snr_range, path_save=None):

    SF_values = [7, 8, 9, 10, 11, 12]
    plt.figure()
    plt.rc('font', size=FONT_SIZE)
    plt.grid()
    plt.yticks(range(0, 101, 10), fontsize=FONT_SIZE_LEGEND)
    plt.xticks(snr_range[::2], fontsize=FONT_SIZE_LEGEND)
    plt.title('Classification Accuracy by Spread Factor')
    plt.xlabel('Signal to Noise Ratio (dB)')
    plt.ylabel('Accuracy (%)')
    for idx, sf in enumerate(SF_values):
        config_choice = [i*6+idx for i in range(3)]
        reshaped_results = select_configs(results, config_choice)
        plot_results(reshaped_results, snr_range, label=f'SF {sf}')
    plt.legend(loc='lower right', fontsize=FONT_SIZE_LEGEND)
    if path_save is not None:       
        plt.savefig(os.path.join(path_save, 'ACC_Vs_SNR_SF.pdf'))     
    plt.show()

def display_results_DR(results, snr_range, path_save=None):

    data_rates = [
        [5],              # DR0
        [4, 11],          # DR1
        [3, 10, 17],      # DR2
        [2, 9, 16],       # DR3
        [1, 8, 15],       # DR4
        [0, 7, 14],       # DR5
        [6, 13],          # DR6
        [12]              # DR7
        ]

    plt.figure()
    plt.rc('font', size=FONT_SIZE)
    plt.grid()
    plt.yticks(range(0, 101, 10), fontsize=FONT_SIZE_LEGEND)
    plt.xticks(snr_range[::2], fontsize=FONT_SIZE_LEGEND)
    plt.title('Classification Accuracy by Data Rate (bits/sec)')
    plt.xlabel('Signal to Noise Ratio (dB)')
    plt.ylabel('Accuracy (%)')
    for idx, combination in enumerate(data_rates):
        reshaped_results = select_configs(results, combination)
        plot_results(reshaped_results, snr_range, label=f'DR {idx}')
    plt.legend(loc='lower right', fontsize=FONT_SIZE_LEGEND-3)
    if path_save is not None:
            plt.savefig(os.path.join(path_save, 'ACC_Vs_SNR_DR.pdf'))    
    plt.show()

def display_results(results, snr_range, path_save=None):

    plt.figure()
    plt.rc('font', size=FONT_SIZE)
    plt.grid()
    plt.yticks(range(0, 101, 10), fontsize=FONT_SIZE_LEGEND)
    plt.xticks(snr_range[::2], fontsize=FONT_SIZE_LEGEND)
    plt.title('Classification Accuracy')
    plt.xlabel('Signal to Noise Ratio (dB)')
    plt.ylabel('Accuracy (%)')
    results = results.reshape(-1, results.shape[-1])
    plot_results(results, snr_range, label='Mean Accuracy')
    plt.legend(loc='lower right', fontsize=FONT_SIZE_LEGEND)
    #plt.savefig('/Users/kt/Desktop/plot.pdf', dpi=300, bbox_inches='tight')
    if path_save is not None:
        plt.savefig(os.path.join(path_save, 'ACC_Vs_SNR.pdf'))
    plt.show()

def compute_accuracy_matrix(results, snr_range= None):

    if snr_range is None:
        snr_range = list(results[0].keys())
    accuracy_matrix = np.zeros((results.shape[0], 18, len(snr_range)))
    for idx, result in enumerate(results):
        for snr, value in result.items():
            for config in range(18):
                indices = np.where(value[:,1, config]==1)[0]
                
                preds, labels = value[indices,0,:], value[indices,1,:]

                accuracy_matrix[idx, config, snr_range.index(snr)] = accuracy_score(np.argmax(labels, axis=-1), np.argmax(preds, axis=-1))*100
    return accuracy_matrix, snr_range

def reformat_results(results):

    snr_range = list(range(-14, 21, 2))
    reformat_results = []
    for idx in range(results.shape[0]):
        reformat_results.append({})
        for snr_idx, snr in enumerate(snr_range):
            reformat_results[idx][snr] = np.array([results[idx, :, snr_idx]]).reshape(-1, 1, 1)
    return np.array(reformat_results)

def main():
    '''
    Results is a 3D array of shape (num_iterations, num_classes, num_snr_points)
    '''
    sim_results_path = '/Users/kt/Library/CloudStorage/GoogleDrive-k.tarun.rao@gmail.com/My Drive/LoRa_Detection/Simulation/Results'
    #simulation_results_N5_s40_V97.npy
    sim_results_name = 'new_results.npy'

    results = np.load(os.path.join(sim_results_path, sim_results_name), allow_pickle=True)

    snr_range = list(range(-20, 21, 4))
    #snr_range = list(range(-14, 21, 2))
    
    accuracy_matrix, snr_range = compute_accuracy_matrix(results)
    display_results(accuracy_matrix, snr_range)
    display_results_DR(accuracy_matrix, snr_range)
    display_results_BW(accuracy_matrix, snr_range)
    display_results_SF(accuracy_matrix, snr_range)  

if __name__ == '__main__':
    main()