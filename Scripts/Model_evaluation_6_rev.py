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


## (6) CRISPR-SGRU (OT_review)
from Programs.Evaluation import Encoder_sgRNA_off
from Programs.Programs.Crispr_SGRU.MODEL import Crispr_SGRU


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
    ###2.2.6 Crispr-SGRU
    ####2.2.6.1 For training data
    encoded_seq_crispr_sgru_list = []
    for i in range(len(training_sets)):
        encoded_seqs = []
        for j in range(len(training_sets[i][0])):
            on_target = training_sets[i][0][j]
            off_target = training_sets[i][1][j]
            e = Encoder_sgRNA_off.Encoder(on_seq=on_target, off_seq=off_target)
            encoded_seqs.append(e.on_off_code)
        encoded_seq_crispr_sgru_list.append(encoded_seqs)

    encoded_seq_crispr_sgru_train = encoded_seq_crispr_sgru_list

    ###2.3.6 Crispr-SGRU
    PATH = "./Trained_models/public_datasets/Crispr_SGRU"
    for batch_size in batch_size_list:
        computation_time = []
        for epochs in epochs_list:
            for i in range(len(data_set_files)):

                X_train = encoded_seq_crispr_sgru_train[i]
                X_train = np.array(X_train).reshape((len(X_train), 1, 24, 7))
                y_train = label_train[i]

                model = Crispr_SGRU()

                start_time = time.time()
                history = model.fit(
                                X_train, y_train,
                                batch_size=batch_size,
                                epochs=epochs,
                                shuffle=True
                            )

                end_time = time.time()
                elapsed_time = end_time - start_time

                ### save the model and weight
                model_jason = model.to_json()
                if os.path.isdir(PATH):
                    pass
                else:
                    os.mkdir(PATH)
                with open(PATH +  "/Crispr_SRGU_model_" + new_datasets_name[i]  + str(epochs)+ ".json", "w") as jason_file:
                    jason_file.write(model_jason)
                    model.save_weights(PATH + "/Crispr_SRGU_model_" + new_datasets_name[i] + str(epochs) +".weights.h5")

                K.clear_session()
                print("Set: {}, Batch size: {}, Epochs: {}, Time: {}".format(new_datasets_name[i], batch_size, epochs, elapsed_time))
                computation_time.append([new_datasets_name[i], i, batch_size, epochs, elapsed_time])
                
                comp_timefile = './New_evaluation/crispr_sgru/formal_comp_b' + str(batch_size) + '_e' + str(epochs) +'_' + str(N) +  '.csv'
                with open(comp_timefile, 'w') as f:
                    for element in computation_time:
                        f.write("{},{},{},{},{}\n".format(element[0], element[1], element[2], element[3], element[4]))


cuda.select_device(0)
cuda.close() 