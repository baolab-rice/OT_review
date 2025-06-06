import os, time, sys, pickle
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, MultiHeadAttention, Dense, GlobalAvgPool1D, Conv1D, BatchNormalization, Activation, Dropout, LayerNormalization, Flatten, Conv2D, Reshape, Bidirectional, LSTM, Concatenate, AveragePooling1D, MaxPool1D, BatchNormalization, Attention, GlobalAveragePooling1D, GlobalMaxPool1D, GRU
from tensorflow.keras.initializers import VarianceScaling
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.metrics import AUC
from tensorflow.keras.losses import BinaryCrossentropy
from Programs.Programs.CRISPR_M.test_model import m81212_n13
sys.path.append("../../../codes")
sys.path.append("../../2encoding_test/mine")
#from positional_encoding import PositionalEncoding
#from encoding import encode_in_6_dimensions, encode_by_base_pair_vocabulary, encode_by_one_hot, encode_by_crispr_net_method, encode_by_crispr_net_method_with_isPAM, encode_by_crispr_ip_method, encode_by_crispr_ip_method_without_minus, encode_by_crispr_ip_method_without_isPAM
#from metrics_utils import compute_auroc_and_auprc
#from data_preprocessing_utils import load_PKD_encoded_by_both_base_and_base_pair, load_PDH_encoded_by_both_base_and_base_pair, load_SITE_encoded_by_both_base_and_base_pair, load_GUIDE_I_encoded_by_both_base_and_base_pair, load_GUIDE_II_encoded_by_both_base_and_base_pair, load_GUIDE_III_encoded_by_both_base_and_base_pair
#from test_model import m81212_n13

os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"

class Trainer:
    def __init__(self) -> None:
        self.VOCABULARY_SIZE = 50 # 词典的长度，就是句子里的词有几种。在这里是base pair有几种。口算应该是(4+8+12)*2=48种。再加两种__
        self.MAX_STEPS = 24 # 句子长度。21bpGRNA+3bpPAM = 24。
        self.EMBED_SIZE = 6 # 句子里每个词的向量长度。base pair的编码后的向量的长度，编码前是我设计的长度6的向量，编码后是embed_size。

        self.BATCH_SIZE = 20000
        self.N_EPOCHS = 100

        self.circle_feature = None
        self.circle_labels = None

        self.train_features = None
        self.train_feature_ont =  None
        self.train_feature_offt =  None
        self.validation_features = None
        self.validation_feature_ont =  None
        self.validation_feature_offt =  None
        self.train_labels = None
        self.validation_labels = None

        self.six_dataset_fold = dict()

    def train_model(self):
        print("[INFO] ===== Start train =====")

        model = m81212_n13()

        model.compile(optimizer=Adam(), loss=BinaryCrossentropy(), metrics=['acc', AUC(num_thresholds=4000, curve="ROC", name="auroc"), AUC(num_thresholds=4000, curve="PR", name="auprc")])

        eary_stopping = EarlyStopping(
                        monitor='loss', min_delta=0.0001,
                        patience=10, verbose=1, mode='auto')
        #model_checkpoint = ModelCheckpoint( # 在每轮过后保存当前权重
        #                filepath='tcrispr_model.h5', # 目标模型文件的保存路径
        #                # 这两个参数的含义是，如果 val_loss 没有改善，那么不需要覆盖模型文件。这就可以始终保存在训练过程中见到的最佳模型
        #                monitor='val_auprc',
        #                verbose=1,
        #                save_best_only=True,
        #                mode='max')
                        
        model_checkpoint = ModelCheckpoint(
            filepath='tcrispr_model.keras',  # Use `.keras` extension
            monitor='val_auprc',
            verbose=1,
            save_best_only=True,
            mode='max'
        )
        
        reduce_lr_on_plateau = ReduceLROnPlateau( # 如果验证损失在 4 轮内都没有改善，那么就触发这个回调函数
                        monitor='val_loss', # 监控模型的验证损失
                        factor=0.8, # 触发时将学习率除以 2
                        patience=4,
                        verbose=1,
                        mode='min',
                        min_lr=1e-7)
        ## for validation only
        #callbacks = [eary_stopping, model_checkpoint, reduce_lr_on_plateau]
        callbacks = []
        
        print("self.train_features:", self.train_features)
        print("self.train_feature_ont:", self.train_feature_ont)
        print("self.train_feature_offt:", self.train_feature_offt)
        print("self.train_labels:", self.train_labels)

        print("self.train_features shape:", self.train_features.shape if self.train_features is not None else "None")
        print("self.train_feature_ont shape:", self.train_feature_ont.shape if self.train_feature_ont is not None else "None")
        print("self.train_feature_offt shape:", self.train_feature_offt.shape if self.train_feature_offt is not None else "None")
        print("self.train_labels shape:", self.train_labels.shape if self.train_labels is not None else "None")

        history = model.fit(
            #x=self.train_features, y=self.train_labels,
            #x={"input_1": self.train_feature_ont, "input_2": self.train_feature_offt}, y={"output": self.train_labels},
            x={"input_1": self.train_features, "input_2": self.train_feature_ont, "input_3": self.train_feature_offt}, 
            y=self.train_labels,
            #y={"output": self.train_labels},
            batch_size=self.BATCH_SIZE,
            epochs=self.N_EPOCHS,
            shuffle=True
            # validation_data=(self.validation_features, self.validation_labels),
            # validation_data=({"input_1": self.validation_feature_ont, "input_2": self.validation_feature_offt}, {"output": self.validation_labels}),
            #validation_data=({"input_1": self.validation_features, "input_2": self.validation_feature_ont, "input_3": self.validation_feature_offt}, {"output": self.validation_labels}),
            #callbacks=callbacks
        )
        print("[INFO] ===== End train =====")
     
        return model
        
        #model = load_model('tcrispr_model.h5', custom_objects={"PositionalEncoding": PositionalEncoding})
        # test_loss, test_acc, auroc, auprc = model.evaluate(self.validation_features, self.validation_labels)
        #test_loss, test_acc, auroc, auprc = model.evaluate(x={"input_1": self.validation_features, "input_2": self.validation_feature_ont, "input_3": self.validation_feature_offt}, y=self.validation_labels)
        # accuracy, precision, recall, f1, fbeta, auroc_skl, auprc_skl, auroc_by_auc, auprc_by_auc, spearman_corr_by_pred_score, spearman_corr_by_pred_labels = compute_auroc_and_auprc(model=model, out_dim=1, test_features=self.validation_features, test_labels=self.validation_labels)
        #accuracy, precision, recall, f1, fbeta, auroc_skl, auprc_skl, auroc_by_auc, auprc_by_auc, spearman_corr_by_pred_score, spearman_corr_by_pred_labels, fpr, tpr, precision_point, recall_point = compute_auroc_and_auprc(model=model, out_dim=1, test_features={"input_1": self.validation_features, "input_2": self.validation_feature_ont, "input_3": self.validation_feature_offt}, test_labels=self.validation_labels)
        #return test_loss, test_acc, auroc, auprc, accuracy, precision, recall, f1, fbeta, auroc_skl, auprc_skl, auroc_by_auc, auprc_by_auc, spearman_corr_by_pred_score, spearman_corr_by_pred_labels, fpr, tpr, precision_point, recall_point

if __name__ == "__main__":
    time1 = time.time()

    # classification 0; regression 1;
    cla_or_reg = 0

    # Mismatch leave-one-dataset-out cross-validation (LODOCV)
    trainer = Trainer()
    test_loss_sum, test_acc_sum, auroc_sum, auprc_sum, accuracy_sum, precision_sum, recall_sum, f1_sum, fbeta_sum, auroc_skl_sum, auprc_skl_sum, auroc_by_auc_sum, auprc_by_auc_sum, spearman_corr_by_pred_score_sum, spearman_corr_by_pred_labels_sum = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    fpr_list, tpr_list, precision_point_list, recall_point_list = list(), list(), list(), list()
    trainer.six_dataset_fold["pkd"] = {"features": list(), "feature_on": list(), "feature_off": list(), "labels": list()}
    trainer.six_dataset_fold["pdh"] = {"features": list(), "feature_on": list(), "feature_off": list(), "labels": list()}
    trainer.six_dataset_fold["site"] = {"features": list(), "feature_on": list(), "feature_off": list(), "labels": list()}
    trainer.six_dataset_fold["guide_i"] = {"features": list(), "feature_on": list(), "feature_off": list(), "labels": list()}
    trainer.six_dataset_fold["guide_ii"] = {"features": list(), "feature_on": list(), "feature_off": list(), "labels": list()}
    trainer.six_dataset_fold["guide_iii"] = {"features": list(), "feature_on": list(), "feature_off": list(), "labels": list()}
    classification_data_abbr = ["site", "pkd", "guide_ii", "guide_iii"]
    classification_data_method = [load_SITE_encoded_by_both_base_and_base_pair, load_PKD_encoded_by_both_base_and_base_pair, load_GUIDE_II_encoded_by_both_base_and_base_pair, load_GUIDE_III_encoded_by_both_base_and_base_pair]
    regression_data_abbr = ["site", "pkd", "pdh", "guide_i"]
    regression_data_method = [load_SITE_encoded_by_both_base_and_base_pair, load_PKD_encoded_by_both_base_and_base_pair, load_PDH_encoded_by_both_base_and_base_pair, load_GUIDE_I_encoded_by_both_base_and_base_pair]
    
    if cla_or_reg == 0:
        data_abbr = classification_data_abbr
        data_method = classification_data_method
    else:
        data_abbr = regression_data_abbr
        data_method = regression_data_method
    
    for i in range(4):
        if i < 2:
            result = data_method[i](is_binarized=True)
            # result = classification_data_method[i]()
        else:
            result = data_method[i]()
        trainer.six_dataset_fold[data_abbr[i]]["features"] = result[0]
        trainer.six_dataset_fold[data_abbr[i]]["feature_on"] = result[1]
        trainer.six_dataset_fold[data_abbr[i]]["feature_off"] = result[2]
        trainer.six_dataset_fold[data_abbr[i]]["labels"] = result[3]
    for i in range(2):
        trainer.train_features = list()
        trainer.train_feature_ont = list()
        trainer.train_feature_offt = list()
        trainer.train_labels = list()
        trainer.validation_features = list()
        trainer.validation_feature_ont = list()
        trainer.validation_feature_offt = list()
        trainer.validation_labels = list()
        for j in range(4):
            if bool(j) == bool(i):
                trainer.validation_features.extend(trainer.six_dataset_fold[data_abbr[j]]["features"])
                trainer.validation_feature_ont.extend(trainer.six_dataset_fold[data_abbr[j]]["feature_on"])
                trainer.validation_feature_offt.extend(trainer.six_dataset_fold[data_abbr[j]]["feature_off"])
                trainer.validation_labels.extend(trainer.six_dataset_fold[data_abbr[j]]["labels"])
            else:
                trainer.train_features.extend(trainer.six_dataset_fold[data_abbr[j]]["features"])
                trainer.train_feature_ont.extend(trainer.six_dataset_fold[data_abbr[j]]["feature_on"])
                trainer.train_feature_offt.extend(trainer.six_dataset_fold[data_abbr[j]]["feature_off"])
                trainer.train_labels.extend(trainer.six_dataset_fold[data_abbr[j]]["labels"])
        trainer.train_features = np.array(trainer.train_features, dtype=np.float32)
        trainer.train_feature_ont = np.array(trainer.train_feature_ont, dtype=np.float32)
        trainer.train_feature_offt = np.array(trainer.train_feature_offt, dtype=np.float32)
        trainer.train_labels = np.array(trainer.train_labels, dtype=np.float32)
        print("[INFO] Encoded dataset train_features with size of", trainer.train_features.shape)
        print("[INFO] Encoded dataset train_feature_ont with size of", trainer.train_feature_ont.shape)
        print("[INFO] Encoded dataset train_feature_offt with size of", trainer.train_feature_offt.shape)
        print("[INFO] The labels number of active off-target sites in dataset train_labels is {0}, the active+inactive is {1}.".format(len(trainer.train_labels[trainer.train_labels>0]), len(trainer.train_labels)))
        trainer.validation_features = np.array(trainer.validation_features, dtype=np.float32)
        trainer.validation_feature_ont = np.array(trainer.validation_feature_ont, dtype=np.float32)
        trainer.validation_feature_offt = np.array(trainer.validation_feature_offt, dtype=np.float32)
        trainer.validation_labels = np.array(trainer.validation_labels, dtype=np.float32)
        print("[INFO] Encoded dataset validation_features with size of", trainer.validation_features.shape)
        print("[INFO] Encoded dataset validation_feature_ont with size of", trainer.validation_feature_ont.shape)
        print("[INFO] Encoded dataset validation_feature_offt with size of", trainer.validation_feature_offt.shape)
        print("[INFO] The labels number of active off-target sites in dataset validation_labels is {0}, the active+inactive is {1}.".format(len(trainer.validation_labels[trainer.validation_labels>0]), len(trainer.validation_labels)))
        if os.path.exists("tcrispr_model.h5"):
            os.remove("tcrispr_model.h5")
        result = trainer.train_model()
        if os.path.exists("tcrispr_model.h5"):
            os.remove("tcrispr_model.h5")
        test_loss_sum += result[0]
        test_acc_sum += result[1]
        auroc_sum += result[2]
        auprc_sum += result[3]
        accuracy_sum += result[4]
        precision_sum += result[5]
        recall_sum += result[6]
        f1_sum += result[7]
        fbeta_sum += result[8]
        auroc_skl_sum += result[9]
        auprc_skl_sum += result[10]
        auroc_by_auc_sum += result[11]
        auprc_by_auc_sum += result[12]
        spearman_corr_by_pred_score_sum += result[13]
        spearman_corr_by_pred_labels_sum += result[14]
        fpr_list.append(result[15])
        tpr_list.append(result[16])
        precision_point_list.append(result[17])
        recall_point_list.append(result[18])
    with open("fpr.csv", "wb") as f:
        pickle.dump(fpr_list, f)
    with open("tpr.csv", "wb") as f:
        pickle.dump(tpr_list, f)
    with open("precision_point.csv", "wb") as f:
        pickle.dump(precision_point_list, f)
    with open("recall_point.csv", "wb") as f:
        pickle.dump(recall_point_list, f)
    print("=====")
    print("average_test_loss=%s, average_test_acc=%s, average_auroc=%s, average_auprc=%s, average_accuracy=%s, average_precision=%s, average_recall=%s, average_f1=%s, average_fbeta=%s, average_auroc_skl=%s, average_auprc_skl=%s, average_auroc_by_auc=%s, average_auprc_by_auc=%s, average_spearman_corr_by_pred_score=%s, average_spearman_corr_by_pred_labels=%s" % (test_loss_sum/2, test_acc_sum/2, auroc_sum/2, auprc_sum/2, accuracy_sum/2, precision_sum/2, recall_sum/2, f1_sum/2, fbeta_sum/2, auroc_skl_sum/2, auprc_skl_sum/2, auroc_by_auc_sum/2, auprc_by_auc_sum/2, spearman_corr_by_pred_score_sum/2, spearman_corr_by_pred_labels_sum/2))

    print(time.time()-time1)
