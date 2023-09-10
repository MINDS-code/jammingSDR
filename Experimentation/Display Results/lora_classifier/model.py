
import os, numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten

from lora_classifier.dataset.utils import encode_label, parse_file_name
from lora_classifier.dataset.read import read_IF_sample
from lora_classifier.simulation import simulation_data_generator

SNR_MIN = -20
SNR_MAX = 20
SNR_STEP = 2

PATH_MODELS = '/Users/kt/Library/CloudStorage/GoogleDrive-k.tarun.rao@gmail.com/My Drive/LoRa_Detection/models'
NUM_CLASSES = 18
MODEL_INPUT_SHAPE = (1024, 1)
MODEL_OUTPUT_SHAPE = (NUM_CLASSES,)

def create_model():
  model = Sequential([
      Flatten(input_shape=MODEL_INPUT_SHAPE),
      Dense(16, activation='tanh'),
      Dense(16, activation='tanh'),
      Dropout(0.5),
      Dense(NUM_CLASSES, activation='softmax')
  ])
  model.compile(loss=tf.keras.losses.categorical_crossentropy,
                optimizer=tf.keras.optimizers.Adam(),
                metrics=['accuracy'])
  return model

def load_model(model_name):
  path_model = os.path.join(PATH_MODELS, model_name)
  return tf.keras.models.load_model(path_model)

def save_model(model, model_name):
  path_model = os.path.join(PATH_MODELS, model_name)
  model.save(path_model)

def data_generator(files_list, batch_size):
  while True:
    batch_data, batch_labels = [], []
    batch_sample_count = 0
    for file_path in files_list:
      sample = read_IF_sample(file_path)
      sample = sample * 2 if 'sim' not in file_path.lower() else sample
      config = parse_file_name(os.path.basename(file_path))[2]
      label = encode_label(config)
      if sample.shape==MODEL_INPUT_SHAPE and label.shape==MODEL_OUTPUT_SHAPE:
        batch_data.append(sample)
        batch_labels.append(label)
        batch_sample_count += 1
      if batch_sample_count >= batch_size:
        indices = np.arange(batch_sample_count)
        np.random.shuffle(indices)
        yield np.array(batch_data)[indices], np.array(batch_labels)[indices]
        batch_data, batch_labels = [], []
        batch_sample_count = 0

def simulation_data_generator(num_samples_per_case, snr_range, config_range, batch_size):
  while True:
    batch_data, batch_labels = [], []
    batch_sample_count = 0
    for _ in range(num_samples_per_case):
      for snr in snr_range:
        for config in config_range:
            sample = simulate_IF_sample(config, snr)
            label = encode_label(config)
            if sample.shape==MODEL_INPUT_SHAPE and label.shape==MODEL_OUTPUT_SHAPE:
              batch_data.append(sample)
              batch_labels.append(label)
              batch_sample_count += 1
            if batch_sample_count >= batch_size:
              indices = np.arange(batch_sample_count)
              np.random.shuffle(indices)
              yield np.array(batch_data)[indices], np.array(batch_labels)[indices]
              batch_data, batch_labels = [], []
              batch_sample_count = 0

def train_model_from_files(model, train_data, epochs=5, batch_size=8, validation_data=None):
  # Train the given model in batches using data generator functions.
  num_samples = len(train_data)
  steps_per_epoch = num_samples // batch_size
  train_data_gen = data_generator(train_data, batch_size)
  if validation_data is not None:
    num_val_samples = len(validation_data)
    validation_steps = num_val_samples // batch_size
    val_data_gen = data_generator(validation_data, batch_size)
  else:
    num_samples_per_case = 50
    snr_range = range(SNR_MIN, SNR_MAX+1, SNR_STEP)
    config_range = range(NUM_CLASSES)
    num_val_samples = num_samples_per_case*len(snr_range)*len(config_range)
    validation_steps = num_val_samples // batch_size
    val_data_gen = simulation_data_generator(num_samples_per_case, snr_range, 
                                          config_range, batch_size)
  model.fit(train_data_gen, epochs=epochs, steps_per_epoch=steps_per_epoch, 
            validation_data=val_data_gen, validation_steps=validation_steps,
            verbose=1)
  return model