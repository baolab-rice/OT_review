## General modules
import numpy as np
import pandas as pd
import random 
import time
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
import sklearn
import pickle
import subprocess
from numba import cuda
from tensorflow.keras import backend as K


## (4) CRISPR-M (OT_review)
from Programs.Programs.CRISPR_M.positional_encoding import PositionalEncoding
from Programs.Programs.CRISPR_M.transformer_utils import add_encoder_layer, add_decoder_layer
from Programs.Programs.CRISPR_M.test_model import m81212_n13
from Programs.Programs.CRISPR_M.encoding import encode_by_base_pair_vocabulary, encode_by_base_vocabulary
from Programs.Programs.CRISPR_M.mismatch_test import Trainer

# 1 Dataset prep
for N in ['ori_filtered']:
    training_sets = []
    label_train   = []

    def read_npz(file_path):
        data  = np.load(file_path, allow_pickle=True)
        ont   = data['ont']
        offt  = data['offt']
        label = data['label']
        return ont, offt, label

    data_set_files = ['./Datasets/training_sets_HEK293T_',
                    './Datasets/training_sets_K562_',
                    './Datasets/training_sets_II3_',
                    './Datasets/training_sets_II4_',
                    './Datasets/training_sets_II5_', 
                    './Datasets/training_sets_II6_'
    ]
    print(data_set_files)
    new_datasets_name = []

    for filename in data_set_files:
        new_datasets_name.append(filename.split('_')[-2] + '_' + str(N))
        filename = filename + str(N) + '.npz'
        ont, offt, label = read_npz(filename)
        training_sets.append([ont, offt])
        label_train.append(label)
    print(new_datasets_name)

    #3 Model training for each program

    batch_size_list = [20000]
    epochs_list = [200]
    early_stopping = tf.keras.callbacks.EarlyStopping(
                    monitor='loss', min_delta=0.0001,
                    patience=10, verbose=1, mode='auto')

    callbacks = [early_stopping]
    ###2.2.4 CRISPR-M

    encoded_training_sets_crispr_m = []
    encoded_training_sets_crispr_m_on = []
    encoded_training_sets_crispr_m_off = []


    for i in range(len(training_sets)):
        temp_set = []
        temp_set_on = []
        temp_set_off = []
        for j in range(len(training_sets[i][0])):
            temp_set.append(encode_by_base_pair_vocabulary(training_sets[i][0][j],training_sets[i][1][j]))
            temp_set_on.append(encode_by_base_vocabulary(training_sets[i][0][j]))
            temp_set_off.append(encode_by_base_vocabulary(training_sets[i][1][j]))

        encoded_training_sets_crispr_m.append(temp_set)
        encoded_training_sets_crispr_m_on.append(temp_set_on)
        encoded_training_sets_crispr_m_off.append(temp_set_off)

        temp_set = []
        temp_set_on = []
        temp_set_off = []


    ### 3.4 CRISPR-M
    PATH = "./Trained_models/public_datasets/CRISPR_M"

    for batch_size in batch_size_list:
        computation_time = []
        for epochs in epochs_list:
            for i in range(len(encoded_training_sets_crispr_m)):

                ## Training the model
                trainer = Trainer()

                TRAIN_X = np.array(encoded_training_sets_crispr_m[i])
                ont = np.array(encoded_training_sets_crispr_m_on[i])
                offt = np.array(encoded_training_sets_crispr_m_off[i])

                trainer.BATCH_SIZE = int(batch_size)
                trainer.N_EPOCHS = int(epochs)
                trainer.train_features = np.array(TRAIN_X, dtype=np.float32)
                trainer.train_feature_ont = np.array(ont, dtype=np.float32)
                trainer.train_feature_offt = np.array(offt, dtype=np.float32)
                trainer.train_labels = np.array(label_train[i], dtype=np.float32)

                print("train_features dtype:", trainer.train_features.dtype)
                print("train_feature_ont dtype:", trainer.train_feature_ont.dtype)
                print("train_feature_offt dtype:", trainer.train_feature_offt.dtype)
                print("train_labels dtype:", trainer.train_labels.dtype)


                start_time = time.time()
                model = trainer.train_model()
                end_time = time.time()
                elapsed_time = end_time - start_time

                ### save the model and weight
                model_jason = model.to_json()
                if os.path.isdir(PATH):
                    pass
                else:
                    os.mkdir(PATH)
                with open(PATH +  "/CRISPR_M_model_" + new_datasets_name[i]  + str(epochs)+ ".json", "w") as jason_file:
                    jason_file.write(model_jason)
                    model.save_weights(PATH + "/CRISPR_M_model_" + new_datasets_name[i] + str(epochs) + ".weights.h5")

                K.clear_session()
                print("Set: {}, Batch size: {}, Epochs: {}, Time: {}".format(new_datasets_name[i], batch_size, epochs, elapsed_time))
                computation_time.append([new_datasets_name[i], i, batch_size, epochs, elapsed_time])
                
                comp_timefile = './New_evaluation/crispr_m/formal_comp_b' + str(batch_size) + '_e' + str(epochs) + '_' + str(N) + '.csv'
                with open(comp_timefile, 'w') as f:
                    for element in computation_time:
                        f.write("{},{},{},{},{}\n".format(element[0], element[1], element[2], element[3], element[4]))


cuda.select_device(0)
cuda.close() 