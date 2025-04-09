import numpy as np
import os
import pandas as pd
from keras.callbacks import *
from keras.layers import *
from keras.utils import np_utils

from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve, precision_recall_curve, auc, matthews_corrcoef
from sklearn.model_selection import train_test_split
<<<<<<< HEAD

import Shuffle,



from MODEL import Crispr_SGRU

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from Encoder-sgRNA_off import Encoder

def load_data(file_path):
    data_list = []
    label = []
    Negative = []
    Positive = []
    with open(file_path) as f:
        for line in f:
            ll = [i for i in line.strip().split(',')]  # strip()表示删除掉数据中的换行符，split（‘，’）则是数据中遇到‘,’ 就隔开。
            label_item = np.float64(ll[2])
            data_item = [int(i) for i in ll[3:]]
            if label_item == 0.0:
                Negative.append(ll)
            else:
                Positive.append(ll)
            data_list.append(data_item)
            label.append(label_item)
    return Negative, Positive, label
def encording(data):
    encode=[]
    for idx, row in data.iterrows():
        on_seq = row[0]
        off_seq = row[1]
        label = row[2]
        en = Encoder_sgRNA_off.Encoder(on_seq=on_seq, off_seq=off_seq, with_category=True, label=label)
        encode.append(en.on_off_code)
    return encode

def encordingXtest(Xtest):
    final_code = []
    for idx, row in Xtest.iterrows():
        on_seq = row[0]
        off_seq = row[1]
        en = Encoder_sgRNA_off.Encoder(on_seq=on_seq, off_seq=off_seq, with_category=True, label=label)
        final_code.append(en.on_off_code)
    return final_code


BATCH_SIZE=100
# Negative, Positive, label =loadData('change_CIRCLE_seq_10gRNA_wholeDataset.txt')
Negative, Positive, label =load_data('Dataset.csv')
Negative=Shuffle(Negative)
Positive=Shuffle(Positive)
Train_Validation_Negative, Test_Negative = train_test_split(Negative, test_size=0.15, random_state=42)
Train_Validation_Positive, Test_Positive = train_test_split(Positive, test_size=0.15, random_state=42)

Xtest = np.vstack((Test_Negative, Test_Positive))
Xtest = Shuffle(Xtest)

weighs_path = "weight/Dataset.h5"
model=Crispr_SGRU()
model.load_weights(weighs_path)

y_pred=model.predict_generator(test_D.__iter__(),steps=len(test_D))
y_test=[]
Xtest=pd.DataFrame(Xtest)
Xtest_oneHot = encordingXtest(Xtest)
y_test = [1 if float(i) > 0.0 else 0 for i in Xtest[:, 1]]
y_test = np_utils.to_categorical(y_test)
y_prob = y_pred[:, 1]
y_prob = np.array(y_prob)
y_pred = [int(i[1] > i[0]) for i in y_pred]
y_test = [int(i[1] > i[0]) for i in y_test]


fpr, tpr, au_thres = roc_curve(y_test, y_prob)
auroc = auc(fpr, tpr)
precision, recall, pr_thres = precision_recall_curve(y_test, y_prob)
prauc = auc(recall, precision)
f1score = f1_score(y_test, y_pred)
precision_scores = precision_score(y_test, y_pred)
recall_scores = recall_score(y_test, y_pred)
mcc=matthews_corrcoef(y_test, y_pred)
print("AUROC=%.3f, PRAUC=%.3f, F1score=%.3f, Precision=%.3f, Recall=%.3f,Mcc=%.3f" % (auroc, prauc, f1score, precision_scores, recall_scores,mcc))
