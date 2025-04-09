## General modules
for N in ['ori_filtered']:
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
    tf.config.optimizer.set_jit(False)  # Turn off XLA


    ## (5) CrisprDNT (OT_review)
    from Programs.Programs.CrisprDNT.model_network import transformIO,PositionalEncoding,new_crispr_ip_rev2,validate_tensor
    from tensorflow.keras.models import model_from_json, load_model
    from tensorflow.keras.models import Model
    from keras_multi_head import MultiHeadAttention
    from tensorflow.keras.layers import GRU,Embedding,Activation,ReLU,AveragePooling2D,MaxPool2D,BatchNormalization,Conv1D,Attention, Dense, Conv2D, Bidirectional, LSTM, Flatten, Input, Activation, Reshape, Dropout, Concatenate, AveragePooling1D, MaxPool1D, BatchNormalization, Attention, GlobalAveragePooling1D, GlobalMaxPool1D, GRU, AdditiveAttention, AlphaDropout, LeakyReLU
    from tensorflow.keras.initializers import VarianceScaling,RandomUniform
    from tensorflow.keras.utils import to_categorical
    from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, roc_auc_score, average_precision_score
    from keras_layer_normalization import LayerNormalization
    from tensorflow.keras.initializers import glorot_normal
    import shutil
    from keras_bert import get_custom_objects
    from tensorflow.python.keras.layers.core import Reshape, Permute
    from tensorflow.python.keras.models import *
    from tensorflow.python.keras.layers import multiply
    from tensorflow.python.keras.layers.core import Dense, Dropout, Lambda, Flatten

# 1 Dataset prep

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
    
    for i in range(len(training_sets)):

        sgrna = training_sets[i][0]
        otdna = training_sets[i][1]
        label = label_train[i]
        data = {'sgrna': sgrna, 'otdna': otdna, 'label': label}
        df = pd.DataFrame(data)
        filename = 'training_set' + str(i) +'.csv'
        df.to_csv(filename, index=False)

        script_path = "/home/remote_guest/Desktop/Programs/Programs/CrisprDNT/create_coding_scheme.py"

        # Call the script using subprocess
        subprocess.run(["python", script_path, filename, str(i), "train"])
        #get_ipython().run_line_magic('run', "'/home/remote_guest/Desktop/Programs/Programs/CrisprDNT/create_coding_scheme.py' {filename} {i} train")

    ###2.3.5 CrisprDNT

    encoder_shape = (23, 14)
    seg_len, coding_dim = encoder_shape
    num_classes = 2
    retrain=False
    list_type = ['14x23']

    PATH = "./Trained_models/public_datasets/CrisprDNT"
    for batch_size in batch_size_list:
        computation_time = []
        for epochs in epochs_list:
            for i in range(len(data_set_files)):
                with open("encodedmismatchtype14x23cd33withoutTsai" + str(i) + "train.pkl", "rb") as f:
                    data = pickle.load(f, encoding = 'latin1')
                with open("encodedmismatchtype14x23cd33withoutTsai" + str(i) + "test.pkl", "rb") as f:
                    data_ = pickle.load(f, encoding = 'latin1')

                X_test = data_.images
                y_test = data_.target  

                X_train = data.images
                y_train = data.target

                ## Training the model 
                xtrain, xtest, ytrain, ytest, inputshape = transformIO(
                X_train, X_test, y_train, y_test, seg_len, coding_dim, num_classes)

                ds = tf.data.Dataset.from_tensor_slices((xtrain, ytrain))
                # Batch and prefetch the dataset
                batched_ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
                # Calculate steps per epoch based on the number of samples
                steps_per_epoch = int(np.ceil(len(ytrain) / batch_size))

                start_time = time.time()
                model = new_crispr_ip_rev2(batched_ds,steps_per_epoch,
                                          inputshape, 
                                          batch_size, 
                                          epochs, 
                                          callbacks)
                end_time = time.time()
                elapsed_time = end_time - start_time

                ### save the model and weight
                model_jason = model.to_json()
                if os.path.isdir(PATH):
                    pass
                else:
                    os.mkdir(PATH)
                with open(PATH +  "/CrisprDNT_model_" + new_datasets_name[i] + str(epochs) + ".json", "w") as jason_file:
                    jason_file.write(model_jason)
                    model.save_weights(PATH + "/CrisprDNT_model_" + new_datasets_name[i] + str(epochs) + ".weights.h5")

                K.clear_session()
                print("Set: {}, Batch size: {}, Epochs: {}, Time: {}".format(new_datasets_name[i], batch_size, epochs, elapsed_time))
                computation_time.append([new_datasets_name[i], i, batch_size, epochs, elapsed_time])
                
                comp_timefile = './New_evaluation/crisprdnt/formal_comp_b' + str(batch_size) + '_e' + str(epochs) +'_' + str(N) +  '.csv'
                with open(comp_timefile, 'w') as f:
                    for element in computation_time:
                        f.write("{},{},{},{},{}\n".format(element[0], element[1], element[2], element[3], element[4]))


    cuda.select_device(0)
    cuda.close() 