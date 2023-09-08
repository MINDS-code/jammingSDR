
import os
import re
import numpy as np

PATH_DATASETS = '/Users/kt/Library/CloudStorage/GoogleDrive-k.tarun.rao@gmail.com/My Drive/LoRa_Detection'
NUM_CLASSES = 18

def encode_label(config):
  # Compute one-hot encoding for label
  one_hot_label = np.zeros(NUM_CLASSES)
  one_hot_label[config] = 1
  return one_hot_label

def decode_snr(snr):
    # Decode SNR from file name (sign: 0/1, magnitude: 0.00-99.99)
    snr_sign, snr_mag = divmod(int(snr), 10000)
    snr = (-1)*(snr_mag/100) if snr_sign else snr_mag/100
    return snr

def parse_file_name(file_path):
    # Parse file name to get SNR, distance, configuration, and index
    file_name = os.path.basename(file_path) #check
    match = re.search(r'SNR(\d{5})D(\d{2})C(\d{2})T(\d{2}).npy', file_name)
    if match:
        snr = decode_snr(match.group(1))
        dist = int(match.group(2))
        config = int(match.group(3))
        index = int(match.group(4))
        return snr, dist, config, index
    else:
        return None

def filter_files(dataset_name, filter_snr=None, filter_config=None, filter_count=50, shuffle=True):
    # Filter files in dataset by SNR and/or configuration
    files_list = []
    path_dataset = os.path.join(PATH_DATASETS, dataset_name)
    all_files = os.listdir(path_dataset)
    for file_name in all_files:
        snr, _, config, index = parse_file_name(file_name)          
        if filter_snr is not None:
            if int(snr) not in filter_snr: #check int()
                continue
        if filter_config is not None:
            if config not in filter_config:
                continue
        if index > filter_count:
          continue
        files_list.append(os.path.join(path_dataset, file_name))
    if shuffle:
        np.random.shuffle(files_list)
    return files_list