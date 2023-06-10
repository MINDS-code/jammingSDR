'''
Python Script to plot ROC curves and AUC for multilabel classifier.
Expects NUM_CLASSES = 18
'''
from sklearn.metrics import roc_curve, auc
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
import numpy as np
import os

NUM_CLASSES = 18
BW_VALUES = [125000, 250000, 500000]
SF_VALUES = [7, 8, 9, 10, 11, 12]
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

FONT_SIZE = 14
FONT_SIZE_LEGEND = 12

def one_hot_encode(n, num_classes = NUM_CLASSES):
    '''
    Returns one-hot vector of length NUM_CLASSES with 1 at index n
    '''
    arr = np.zeros(num_classes)
    arr[n] = 1
    return arr

def reshape_data(predictions, labels):
    '''
    Reshape data to be of shape (num_samples, num_classes) if data is in shape (num_batches, batch_size, num_classes)
    '''
    if predictions.shape[0] != labels.shape[0]:
        raise ValueError('predictions.shape[0] and labels.shape[0] must be equal')
    if len(predictions.shape) == 3:
        predictions = np.reshape(predictions, (-1, NUM_CLASSES))
        labels = np.reshape(labels, (-1, NUM_CLASSES))
    return predictions, labels

def ROC_OVR_custom(predictions, labels, class_filter = list(range(NUM_CLASSES)), fpr = dict(), tpr = dict(), roc_auc = dict()):
    '''
    This program filters out the predictions and labels vectors to contain records corresponding only to classes in the class_filter list.
    Then, it computes the micro average ROC curve and AUC one vs rest.
    '''
    predictions = predictions[:, class_filter]
    labels = labels[:, class_filter]

    fpr["micro"], tpr["micro"], roc_auc["micro"] = ROC_micro_average(predictions, labels)

    return fpr, tpr, roc_auc

def ROC_OVR(predictions, labels, class_filter = list(range(NUM_CLASSES)), fpr = dict(), tpr = dict(), roc_auc = dict()):
    '''
    Compute ROC curve and AUC for each class, one vs rest
    '''
    for i in class_filter:
        fpr[i], tpr[i], _ = roc_curve(labels[:, i], predictions[:, i])
        #fpr[i], tpr[i], _ = roc_curve(np.argmax(labels, axis=1), np.argmax(predictions, axis=1))
        roc_auc[i] = auc(fpr[i], tpr[i])

    return fpr, tpr, roc_auc

def ROC_micro_average(predictions, labels):
    '''
    Compute micro-average ROC curve and AUC
    '''
    fpr, tpr, _ = roc_curve(labels.ravel(), predictions.ravel())
    roc_auc = auc(fpr, tpr)

    return fpr, tpr, roc_auc

def ROC_macro_average(fpr, tpr, num_classes=NUM_CLASSES):
    '''
    Compute macro-average ROC curve and AUC
    '''
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(num_classes)]))

    mean_tpr = np.zeros_like(all_fpr)
    for i in range(num_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    sorted_indices = np.argsort(all_fpr)
    all_fpr = all_fpr[sorted_indices]
    mean_tpr = mean_tpr[sorted_indices]
    mean_tpr /= num_classes

    return all_fpr, mean_tpr, auc(all_fpr, mean_tpr)

def compute_ROC_AUC(predictions, labels):
    '''
    Compute ROC curves and AUCs for each class, micro-average and macro-average
    '''
    predictions, labels = reshape_data(predictions, labels)
    fpr, tpr, roc_auc = ROC_OVR(predictions, labels)
    fpr["micro"], tpr["micro"], roc_auc["micro"] = ROC_micro_average(predictions, labels)
    fpr["macro"], tpr["macro"], roc_auc["macro"] = ROC_macro_average(fpr, tpr)

    return fpr, tpr, roc_auc

def display_results(fpr, tpr, roc_auc, num_classes= NUM_CLASSES):
    '''
    Display ROC curves and AUCs for each class, micro-average and macro-average
    '''

    plt.figure()
    lw = 2
    plt.plot(fpr["micro"], tpr["micro"],
            label='micro-average ROC curve (area = {0:4f})'
                ''.format(roc_auc["micro"]),
            color='deeppink', linestyle=':', linewidth=4)

    plt.plot(fpr["macro"], tpr["macro"],
            label='macro-average ROC curve (area = {0:4f})'
                ''.format(roc_auc["macro"]),
            color='navy', linestyle=':', linewidth=4)

    for i in range(num_classes):
        plt.plot(fpr[i], tpr[i], lw=lw,
                label='ROC curve of class {0} (area = {1:4f})'
                ''.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([-0.05, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.show()

def display_ROC_AUC_custom(fpr, tpr, roc_auc, config='micro', label='micro-average'):
    lw = 2
    plt.plot(fpr[config], tpr[config],
            label='{} (area = {})'
                ''.format(label, np.round(roc_auc[config], 4)), linestyle=':', linewidth=4)
    plt.legend(loc='lower right', fontsize=FONT_SIZE_LEGEND)
    plt.savefig('/Users/kt/Desktop/plot.pdf', dpi=300, bbox_inches='tight')
    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([-0.05, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    if label[:2] == 'SF':
        plt.title('ROC by Spreading Factor (SF)')
    elif label[:2] == 'BW':
        plt.title('ROC by Bandwidth (BW)')
    elif label[:2] == 'DR':
        plt.title('ROC by Data Rate (DR)')
    else:
        plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.grid(True)

def display_results_BW(predictions, labels, bw_list):
    plt.figure()
    #plt.grid()
    for bw in bw_list:
        class_filter = [BW_VALUES.index(bw)*6+i for i in range(6)]
        fpr, tpr, roc_auc = ROC_OVR_custom(predictions, labels, class_filter)
        display_ROC_AUC_custom(fpr, tpr, roc_auc, config='micro', label=f'BW: {bw} Hz')
    plt.show()

def display_results_SF(predictions, labels, sf_list):
    plt.figure(figsize=(8, 5))
    for sf in sf_list:
        class_filter = [i*6+SF_VALUES.index(sf) for i in range(3)]
        fpr, tpr, roc_auc = ROC_OVR_custom(predictions, labels, class_filter)
        display_ROC_AUC_custom(fpr, tpr, roc_auc, config='micro', label=f'SF: {sf}')
    plt.show()

def display_results_DR(predictions, labels, dr_list=data_rates):
    plt.figure()
    for idx, class_filter in enumerate(dr_list):
        fpr, tpr, roc_auc = ROC_OVR_custom(predictions, labels, class_filter)
        display_ROC_AUC_custom(fpr, tpr, roc_auc, config='micro', label=f'DR: {idx}')
    plt.show()

def load_results(path, file_name):
    results = np.load(os.path.join(path, file_name), allow_pickle=True)
    snr_range = results[0].keys()
    #   Select and concatenate iterations
    results_iter = {}
    filter_iterations = list(range(len(results)))
    for snr in snr_range:
        results_iter[snr] = np.concatenate([results[i][snr] for i in filter_iterations])
    #   Select and concatenate SNR values
    filter_snr = snr_range
    results_all = np.concatenate([results_iter[snr] for snr in filter_snr])

    return results_all[:, 0], results_all[:, 1]

def main():    
    path = '/Users/kt/Library/CloudStorage/GoogleDrive-k.tarun.rao@gmail.com/My Drive/LoRa_Detection/Code/'
    #file_name = 'simulation_results_N5_V98.npy'
    file_name = 'simulation_results_N15_S40_V97.npy'

    #   ROC curves and AUCs for each class, micro-average and macro-average
    predictions, labels = load_results(path, file_name)
    fpr, tpr, roc_auc = compute_ROC_AUC(predictions, labels)

    #   Display ROC curves
    plt.rc('font', size=FONT_SIZE)
    #display_ROC_AUC_custom(fpr, tpr, roc_auc)

    display_results_BW(predictions, labels, [125000, 250000, 500000])
    display_results_SF(predictions, labels, [7, 8, 9, 10, 11, 12])
    display_results_DR(predictions, labels, data_rates)

if __name__ == "__main__":
    main()