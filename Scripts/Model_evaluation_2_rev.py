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

## (2) CRISPR-IP (OT_review)
from Programs.Programs.CRISPR_IP.codes.encoding import my_encode_on_off_dim
from Programs.Programs.CRISPR_IP.codes.CRISPR_IP import transformIO,crispr_ip


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
    ###2.2.2 CRISPR-IP

    #### Encoding
    encoded_training_sets_crispr_ip = []

    for i in range(len(training_sets)):
        temp_list = []
        for j in range(len(training_sets[i][0])):
            ont  = training_sets[i][0][j]
            offt = training_sets[i][1][j]
            train_data_encodings = np.array(my_encode_on_off_dim(ont,offt))
            temp_list.append(train_data_encodings)
        encoded_training_sets_crispr_ip.append(temp_list)


    ###3.2 CRISPR-IP
    num_classes = 2
    retrain=False
    encoder_shape=(24,7)
    seg_len, coding_dim = encoder_shape
    PATH = "./Trained_models/public_datasets/CRISPR_IP"


    for batch_size in batch_size_list:
        computation_time = []
        for epochs in epochs_list:
            for i in range(len(encoded_training_sets_crispr_ip)):

                ## Model training 
                train_data_encodings = np.array(encoded_training_sets_crispr_ip[i])
                train_labels = label_train[i]

                start_time = time.time()
                TRAIN_X, TRAIN_Y, inputshape = transformIO(
                    train_data_encodings, 
                    train_labels, 
                    seg_len, 
                    coding_dim, 
                    num_classes
                )

                model = crispr_ip(TRAIN_X, TRAIN_Y, inputshape, num_classes, batch_size, epochs, callbacks, retrain)
                end_time = time.time()
                elapsed_time = end_time - start_time
                
                ### save the model and weight
                model_jason = model.to_json()
                if os.path.isdir(PATH):
                    pass
                else:
                    os.mkdir(PATH) 
                with open(PATH + "/CRISPR_IP_model_" + new_datasets_name[i]  + str(epochs)+ ".json", "w") as jason_file:
                    jason_file.write(model_jason)
                    model.save_weights(PATH + "/CRISPR_IP_model_" + new_datasets_name[i]  + str(epochs)+ ".weights.h5")


                K.clear_session()
                print("Set: {}, Batch size: {}, Epochs: {}, Time: {}".format(new_datasets_name[i], batch_size, epochs, elapsed_time))
                computation_time.append([new_datasets_name[i], i, batch_size, epochs, elapsed_time])
                
                comp_timefile = './New_evaluation/crispr_ip/formal_comp_b' + str(batch_size) + '_e' + str(epochs) + '_' + str(N) + '.csv'
                with open(comp_timefile, 'w') as f:
                    for element in computation_time:
                        f.write("{},{},{},{},{}\n".format(element[0], element[1], element[2], element[3], element[4]))


cuda.select_device(0)
cuda.close() 