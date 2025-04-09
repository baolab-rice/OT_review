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

## (3) R-CRISPR (OT_review) (From CRISPR-M)
from Programs.Programs.R_CRISPR.encoding import encode_by_r_crispr_method
from Programs.Programs.R_CRISPR.R_CRISPR import ConvBn, RepVGGBlock, R_CRISPR_model, R_CRISPR_training

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
    ###2.2.3 R-CRISPR

    encoded_training_sets_r_crispr = []

    for i in range(len(training_sets)):
        temp_set = []
        for j in range(len(training_sets[i][0])):
            temp_set.append(encode_by_r_crispr_method(training_sets[i][0][j],training_sets[i][1][j]))
        encoded_training_sets_r_crispr.append(temp_set)


    ### 3.3 R-CRISPR
    PATH = "./Trained_models/public_datasets/R_CRISPR"

    for batch_size in batch_size_list:
        computation_time = []
        for epochs in epochs_list:
            for i in range(len(encoded_training_sets_r_crispr)):

                TRAIN_X = np.array(encoded_training_sets_r_crispr[i])

                ## Training the model
                start_time = time.time()
                model = R_CRISPR_training(TRAIN_X, label_train[i], batch_size, epochs, callbacks)
                end_time = time.time()
                elapsed_time = end_time - start_time
                
                "/CRISPR_Net_model_" + new_datasets_name[i] + ".json"

                ### save the model and weight
                model_jason = model.to_json()
                if os.path.isdir(PATH):
                    pass
                else:
                    os.mkdir(PATH)
                with open(PATH +  "/R_CRISPR_model_" + new_datasets_name[i]  + str(epochs)+ ".json", "w") as jason_file:
                    jason_file.write(model_jason)
                    model.save_weights(PATH + "/R_CRISPR_model_" + new_datasets_name[i] + str(epochs) + ".weights.h5")

                K.clear_session()
                print("Set: {}, Batch size: {}, Epochs: {}, Time: {}".format(new_datasets_name[i], batch_size, epochs, elapsed_time))
                computation_time.append([new_datasets_name[i], i, batch_size, epochs, elapsed_time])
                
                comp_timefile = './New_evaluation/r_crispr/formal_comp_b' + str(batch_size) + '_e' + str(epochs) + '_' + str(N) + '.csv'
                with open(comp_timefile, 'w') as f:
                    for element in computation_time:
                        f.write("{},{},{},{},{}\n".format(element[0], element[1], element[2], element[3], element[4]))


cuda.select_device(0)
cuda.close() 