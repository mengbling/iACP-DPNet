# 导入MINE类，这是minepy库的一部分，专门用于计算MIC和其他相关统计量
from minepy import MINE
from keras import backend as K
def attention_pooling(inputs):
    dense_layer = tf.keras.layers.Dense(1, activation='tanh')(inputs)
    dense_layer = tf.keras.layers.Flatten()(dense_layer)
    attention_weights = tf.keras.layers.Dense(inputs.shape[1], activation='softmax')(dense_layer)
    attention_weights = tf.keras.layers.Lambda(lambda x: K.expand_dims(x, axis=-1))(attention_weights)
    # attention_weights = tf.keras.layers.Permute([2, 1])(attention_weights)
    weighted_sum = tf.keras.layers.Multiply()([inputs, attention_weights])
    weighted_sum = tf.keras.layers.Lambda(lambda x: K.sum(x, axis=1))(weighted_sum)
    return weighted_sum


import tensorflow as tf

import numpy as np
from keras.callbacks import EarlyStopping
from Bio import SeqIO
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, accuracy_score, f1_score, matthews_corrcoef
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
import sys
sys.path.append('.')

# #############################
# 创建氨基酸到索引的映射
amino_acids = 'ACDEFGHIKLMNPQRSTVWYB'
amino_acid_to_index = {amino_acid: index for index, amino_acid in enumerate(amino_acids)}

# 将蛋白质序列转换为数值表示
def sequence_to_numeric(sequence, mapping):
    return [mapping[amino_acid] for amino_acid in sequence]

# protein_sequences = [
#     "ACDEFGHIKLMNPQRSTVWY",
#     "LMNPQRSTVWYACDEFGHI",
#     "KLMNPQRSTVWYACDEFGHIKLMN"
# ]
Xtest_file = "F:/Desk/新找数据/新数据/补齐样本/x-testB.fasta"
Xtrain_file = "F:/Desk/新找数据/新数据/补齐样本/x-trainB.fasta"
# Xtrain_file = "E:/抗癌肽预测/补齐50数据/trainB.txt"
# y_train = np.array([1] * 874+[0] * 4253)
# 读取FASTA文件内容
def read_fasta(file_path):
    sequences = []
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if line and not line.startswith('>'):
                sequences.append(line)
    return sequences
# #####################
# 读取训练集和测试集FASTA文件内容
Xtrain_sequences = read_fasta(Xtrain_file)
Xtest_sequences = read_fasta(Xtest_file)

xtrain_sequences = [sequence_to_numeric(seq, amino_acid_to_index) for seq in Xtrain_sequences]  # 多条蛋白质序列
# print(xtrain_sequences)
xtrain_sequences = np.array(xtrain_sequences)
xtest_sequences = [sequence_to_numeric(seq, amino_acid_to_index) for seq in Xtest_sequences]  # 多条蛋白质序列
# print(xtrain_sequences)
xtest_sequences = np.array(xtest_sequences)

# 填充序列，使它们具有相同的长度
# padded_sequences = pad_sequences(numeric_sequences, padding='post')
# print(padded_sequences)
# 输出： [[ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19  0  0  0]
#         [ 9 10 11 12 13 14 15 16 17 18 19  0  1  2  3  4  5  6  7  8  0  0  0]
#         [ 8  9 10 11 12 13 14 15 16 17 18 19  0  1  2  3  4  5  6  7  8  9 10]]

# 参数设置
d_model = 1024
max_len = xtrain_sequences.shape[1]
dropout = 0.1

# 创建嵌入层
embedding_layer = tf.keras.layers.Embedding(input_dim=len(amino_acids), output_dim=d_model)

# 将数值表示转换为嵌入表示
xtrain_embeddedes = embedding_layer(tf.convert_to_tensor(xtrain_sequences))
xtest_embeddedes = embedding_layer(tf.convert_to_tensor(xtest_sequences))
# 将嵌入表示扩展维度以匹配位置编码的输入形状 [batch size, sequence length, embed dim]
print(xtrain_embeddedes.shape)
print(xtest_embeddedes.shape)
# 输出： (3, 24, 512)
# tAPE 位置编码类
class tAPE(tf.keras.layers.Layer):
    def __init__(self, d_model, dropout=0.1, max_len=1024, scale_factor=1.0):
        super(tAPE, self).__init__()
        self.dropout = tf.keras.layers.Dropout(dropout)
        pe = np.zeros((max_len, d_model))
        position = np.expand_dims(np.arange(0, max_len, dtype=np.float32), 1)
        div_term = np.exp(np.arange(0, d_model, 2, dtype=np.float32) * (-np.log(10000.0) / d_model))

        pe[:, 0::2] = np.sin((position * div_term) * (d_model / max_len))
        pe[:, 1::2] = np.cos((position * div_term) * (d_model / max_len))
        pe = scale_factor * np.expand_dims(pe, 0)
        self.pe = tf.constant(pe, dtype=tf.float32)

    def call(self, x):
        x = x + self.pe
        return self.dropout(x)

# AbsolutePositionalEncoding 位置编码类
class AbsolutePositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, d_model, dropout=0.1, max_len=1024, scale_factor=1.0):
        super(AbsolutePositionalEncoding, self).__init__()
        self.dropout = tf.keras.layers.Dropout(dropout)
        pe = np.zeros((max_len, d_model))
        position = np.expand_dims(np.arange(0, max_len, dtype=np.float32), 1)
        div_term = np.exp(np.arange(0, d_model, 2, dtype=np.float32) * (-np.log(10000.0) / d_model))

        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        pe = scale_factor * np.expand_dims(pe, 0)
        self.pe = tf.constant(pe, dtype=tf.float32)

    def call(self, x):
        x = x + self.pe
        return self.dropout(x)

# LearnablePositionalEncoding 位置编码类
class LearnablePositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, d_model, dropout=0.1, max_len=1024):
        super(LearnablePositionalEncoding, self).__init__()
        self.dropout = tf.keras.layers.Dropout(dropout)
        self.pe = self.add_weight("pe", shape=[max_len, d_model], initializer=tf.keras.initializers.RandomUniform(-0.02, 0.02), trainable=True)

    def call(self, x):
        x = x + self.pe
        return self.dropout(x)

# 使用 tAPE
tape_layer = tAPE(d_model, dropout, max_len, scale_factor=1.0)
xtrain_tape = tape_layer(xtrain_embeddedes)
xtrain_tape = np.array(xtrain_tape)
xtest_tape = tape_layer(xtest_embeddedes)
xtest_tape = np.array(xtest_tape)
# 使用 AbsolutePositionalEncoding
absolute_positional_encoding_layer = AbsolutePositionalEncoding(d_model, dropout, max_len)
xtrain_absolute = absolute_positional_encoding_layer(xtrain_embeddedes)
xtrain_absolute =np.array(xtrain_absolute)
xtest_absolute = absolute_positional_encoding_layer(xtest_embeddedes)
xtest_absolute =np.array(xtest_absolute)
# 使用 LearnablePositionalEncoding
learnable_positional_encoding_layer = LearnablePositionalEncoding(d_model, dropout, max_len)
xtrain_learnable = learnable_positional_encoding_layer(xtrain_embeddedes)
xtrain_learnable = np.array(xtrain_learnable)
xtest_learnable = learnable_positional_encoding_layer(xtest_embeddedes)
xtest_learnable = np.array(xtest_learnable)
# 输出结果
print("tAPE encoded train sequences shape:", xtrain_tape.shape)
print("tAPE encoded test sequences shape:", xtest_tape.shape)
print("AbsolutePositionalEncoding encoded sequences shape:", xtrain_absolute.shape)
print("LearnablePositionalEncoding encoded sequences shape:", xtrain_learnable.shape)
#
# np.save('xtrain_tape.npy', xtrain_tape)
# np.save('xtest_tape.npy', xtest_tape)
#
# np.save('xtrain_absolute.npy', xtrain_absolute)
# np.save('xtest_absolute.npy', xtest_absolute)
#
# np.save('xtrain_learnable.npy', xtrain_learnable)
# np.save('xtest_learnable.npy', xtest_learnable)
# ############################

y_train = np.array([1] * 1128+[0] * 1128)
y_test = np.array([1] * 282+[0] * 282)
# Xtrain_features_A = np.load('F:/a濛/New-data-ACPs/新数据1128APACB_train.npy')
# Xtest_features_A = np.load('F:/a濛/New-data-ACPs/新数据282APACB_test.npy')

protbert_Xtrain_reshaped = np.load('F:/a濛/New-data-ACPs/新数据1128protbert_Xtrain.npy')
protbert_Xtest_reshaped = np.load('F:/a濛/New-data-ACPs/新数据282protbert_Xtest.npy')
# x_train_PE = np.load('F:/a濛/New-data-ACPs/ATT/位置编码/xtrain_tape.npy')  # (2256,50,512)
# x_test_PE = np.load('F:/a濛/New-data-ACPs/ATT/位置编码/xtest_tape.npy')
# x_train_PE = np.load('F:/a濛/New-data-ACPs/ATT/位置编码/xtrain_absolute.npy')
# x_test_PE = np.load('F:/a濛/New-data-ACPs/ATT/位置编码/xtest_absolute.npy')
# x_train_PE = np.load('F:/a濛/New-data-ACPs/ATT/位置编码/xtrain_learnable.npy')
# x_test_PE = np.load('F:/a濛/New-data-ACPs/ATT/位置编码/xtest_learnable.npy')
x_train_PE = xtrain_learnable.reshape(2256, -1)
x_test_PE = xtest_learnable.reshape(564, -1)
# ########################
Xtrain_features = protbert_Xtrain_reshaped + x_train_PE  # 拼接）
Xtest_features = protbert_Xtest_reshaped + x_test_PE
# Xtrain_features_A = Xtrain_features_A + x_train_PE  # 拼接）
# Xtest_features_A = Xtest_features_A + x_test_PE

print(Xtrain_features.shape)
print(Xtest_features.shape)
# #####################  StandardScaler 标准化数据
from sklearn.preprocessing import StandardScaler
# 初始化StandardScaler
scaler = StandardScaler()
# 对训练集进行拟合并标准化
Xtrain_features = scaler.fit_transform(Xtrain_features)
Xtest_features = scaler.transform(Xtest_features)
# Xtrain_features_A = scaler.fit_transform(Xtrain_features_A)
# Xtest_features_A = scaler.transform(Xtest_features_A)
# ###########################
import pandas as pd
import lightgbm as lgb
# 假设特征 i 有 150 个特征，特征 j 有 100 个特征
# A_i=2195
porbert_i= 51200
# 生成特征名称列表并合并数据
feature_namesp = [f'{i}' for i in range(porbert_i)]
# feature_namesa = [f'{i}' for i in range(A_i)]
# 创建 DataFrame，并设置列名为特征名称列表
p_train = pd.DataFrame(Xtrain_features, columns=feature_namesp)
p_test = pd.DataFrame(Xtest_features, columns=feature_namesp)

model = lgb.LGBMClassifier(
    boosting_type='gbdt',  # 设置提升类型为梯度提升决策树gbdt
    objective='binary',  # 对于分类问题，设置为二分类
    num_leaves=31,  # 定义叶子节点的数量
    learning_rate=0.05,  # 学习率
    n_estimators=100  # 迭代次数
)
model.fit(p_train, y_train)
feature_importancep = model.feature_importances_

# 创建一个 DataFrame 来存储特征重要性及其对应的列名
# feature_importance_df = pd.DataFrame({'Feature': df_train.columns, 'Importance': feature_importance})
# 按照重要性降序排列特征
# sorted_features = feature_importance_df.sort_values(by='Importance', ascending=False)
feature_importance_dictp = dict(zip(p_train.columns, feature_importancep))
sorted_featuresp = sorted(feature_importance_dictp.items(), key=lambda x: x[1], reverse=True)
# 可以打印排序后的特征及其重要性
# for feature, importance in sorted_featuresp:
#     print(f"Feature: {feature}, Importance: {importance}")
# 根据需要选择保留的特征数量或重要性阈值，进行特征选择
# selected_featuresp = [f[0] for f in sorted_featuresp[:150]]  # 指定个数
# selected_featuresa = [f[0] for f in sorted_featuresa[:50]]
selected_featuresp = [f[0] for f in sorted_featuresp if f[1] > 2]  # 设置重要性阈值
# 使用选择的特征来重新定义训练集和测试集
x_train_pro = p_train[selected_featuresp]
x_test_pro = p_test[selected_featuresp]
# 定义一个名为compute_mic的函数，接受两个参数：X（特征矩阵）和y（目标变量）
def compute_mic(X, y):
    # 初始化一个空列表，用于存储每一列与y之间的MIC得分
    mic_scores = []
    # 创建一个MINE实例，用于进行MIC的计算
    mine = MINE()
    # 遍历DataFrame X的所有列
    for column in X.columns:
        # 使用MINE的compute_score方法来计算当前列与目标变量y之间的MIC
        mine.compute_score(X[column], y)
        # 将计算得到的MIC得分追加到mic_scores列表中
        mic_scores.append(mine.mic())
    # 返回一个Series，其中索引为X的列名，值为相应的MIC得分
    return pd.Series(mic_scores, index=X.columns)
# 调用compute_mic函数，传入特征矩阵X和目标变量y，计算MIC得分
# 1. 计算 MIC 得分
mic_scores = compute_mic(x_train_pro, y_train)

# # 2. 根据 MIC 得分进行特征选择
mic_threshold = 0.1  # 设置阈值
selected_features = mic_scores[mic_scores > mic_threshold].index  # 筛选特征

# k = 128  # 选择前 50 个特征
# selected_features = mic_scores.sort_values(ascending=False).index[:k]

# 3. 获取筛选后的特征数据集
p_train_selected =x_train_pro[selected_features]
p_test_selected = x_test_pro[selected_features]

# 输出选择后的特征信息
print(f"Selected {len(selected_features)} important features.")

np.save('xtrain_pro_PE.npy', p_train_selected)
np.save('xtest_pro_PE.npy', p_test_selected)

# x_train_pro_PE = np.expand_dims(p_train_selected, axis=2)
# x_test_pro_PE = np.expand_dims(p_test_selected, axis=2)
#
#
# # x_test = {"pro_PE": x_test_pro_PE, "A15PACB": x_test_A}
# num_features1 = x_train_pro_PE.shape[1]
# # 定义一个函数来创建并行卷积层
# def parallel_convolution():
#     global encode
#     kernel_num =128
#     kernel_size_1 = 1
#     kernel_size_2 = 3
#     kernel_size_3 = 5
#     # input_PE = tf.keras.Input(shape=(num_features2, 1))  #  padding='causal'
#     input_pro_PE = tf.keras.Input(shape=(num_features1,1), name='pro_PE')
#     # attention_probs = tf.keras.layers.Dense(num_features1, activation='softmax', name="ATT12")(input_pro_PE)
#     # attention_mul = tf.keras.layers.Multiply()([input_pro_PE, attention_probs])
#     # # ATTENTION PART FINISHES HERE
#     # x= tf.keras.layers.Dense(256)(attention_mul)  # 原始的全连接
#     # # x = tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=1))(x)
#     # # x = tf.keras.layers.GlobalAveragePooling1D()(x)
#     # x = tf.keras.layers.Dense(128, activation='relu')(x)
#     # x1 = tf.expand_dims(x, axis=-1)
#
#     y = tf.keras.layers.Conv1D(128, kernel_size=kernel_size_1, strides=1, padding='causal', dilation_rate=3, activation=tf.nn.relu,
#                                kernel_regularizer=tf.keras.regularizers.l2(0.001))(input_pro_PE)
#     y = tf.keras.layers.Conv1D(128, kernel_size=kernel_size_1, strides=1, padding='causal', dilation_rate=3, activation=tf.nn.relu,
#                                kernel_regularizer=tf.keras.regularizers.l2(0.001))(y)
#     y = tf.keras.layers.BatchNormalization()(y)
#     y1 = attention_pooling(y)
#     y11 = tf.keras.layers.GlobalAveragePooling1D()(y)  # GlobalAveragePooling1D
#     o1 = tf.keras.layers.Concatenate()([y1, y11])
#     y = tf.keras.layers.Conv1D(128, kernel_size=kernel_size_2, strides=1, padding='causal', dilation_rate=3, activation=tf.nn.relu,
#                                kernel_regularizer=tf.keras.regularizers.l2(0.001))(input_pro_PE)
#     y = tf.keras.layers.Conv1D(128, kernel_size=kernel_size_2, strides=1, padding='causal', dilation_rate=3, activation=tf.nn.relu,
#                                kernel_regularizer=tf.keras.regularizers.l2(0.001))(y)
#     y = tf.keras.layers.BatchNormalization()(y)
#     y2 = attention_pooling(y)
#     y22 = tf.keras.layers.GlobalAveragePooling1D()(y)
#     o2 = tf.keras.layers.Concatenate()([y2, y22])
#     # block2_output = tf.keras.layers.add([y, block1_output])
#     y = tf.keras.layers.Conv1D(128, kernel_size=kernel_size_3, padding='causal', dilation_rate=3, activation=tf.nn.relu,
#                                kernel_regularizer=tf.keras.regularizers.l2(0.001))(input_pro_PE)
#     y = tf.keras.layers.Conv1D(kernel_num, kernel_size=kernel_size_3, padding='causal', dilation_rate=3, activation=tf.nn.relu,
#                                kernel_regularizer=tf.keras.regularizers.l2 (0.01))(y)
#     y = tf.keras.layers.BatchNormalization()(y)
#     # block3_output = tf.keras.layers.add([y, block2_output])
#     # y = tf.keras.layers.Conv1D(kernel_num, kernel_size=3, padding='same', activation=tf.nn.relu,
#     #                            kernel_regularizer=tf.keras.regularizers.l2(0.001))(block3_output)
#     y3 = attention_pooling(y)
#     y33 = tf.keras.layers.GlobalAveragePooling1D()(y)
#     o3 = tf.keras.layers.Concatenate()([y3, y33])
#     output1 = tf.keras.layers.Concatenate()([o1, o2,o3])
#     y = tf.keras.layers.Dense(128, activation='relu')(output1)
#
#     # # #####################################
#     # input_A= tf.keras.Input(shape=(num_features,1), name='A15PACB')
#     # x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True))(input_A)
#     # x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, return_sequences=True))(x)
#     # x = tf.keras.layers.Flatten()(x)
#     #
#     # # attention_probs = tf.keras.layers.Dense(num_features, activation='softmax', name="ATT12")(input_A)
#     # # attention_mul = tf.keras.layers.Multiply()([input_A, attention_probs])
#     # # # ATTENTION PART FINISHES HERE
#     # # x= tf.keras.layers.Dense(256)(attention_mul)  # 原始的全连接
#     # x = tf.keras.layers.Dense(128, activation='relu')(x)  # 输出层
#     # #####################################
#     feature_layer = tf.keras.layers.concatenate([y, y])  # 把三个通道的编码都拼接起来
#     att = tf.keras.layers.Attention()([feature_layer, feature_layer, feature_layer])
#     # att = tf.keras.layers.Attention()([y, y, y])
#     d = tf.keras.layers.Dense(128, activation='relu')(att)
#     d = tf.keras.layers.Dropout(0.1)(d)
#     d = tf.keras.layers.Dense(256, activation='relu')(d)  # 256
#     d = tf.keras.layers.Dropout(0.5)(d)
#     output = tf.keras.layers.Dense(1, activation='sigmoid')(d)
#     # model = tf.keras.Model([input_text_5, input_text, input_con], output)
#     model = tf.keras.Model(input_pro_PE, output)
#     model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss='binary_crossentropy',
#                   metrics=['accuracy'])
#     return model
#
# num_folds = 10
# skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)
# best_accuracy = 0.0
# best_model = None
# best_params = None
# fold_accuracies = []
# fold_aucs = []
# fold_sensitivities = []
# fold_specificities = []
# fold_mccs = []
# for fold, (train_idx, val_idx) in enumerate(skf.split(x_train_pro_PE, y_train)):
#     print(f"Fold {fold+1}/{num_folds}")
#     x_train_fold = x_train_pro_PE[train_idx]
#     x_val_fold = x_train_pro_PE[val_idx]
#     y_train_fold, y_val_fold = y_train[train_idx], y_train[val_idx]
#     # 创建模型
#     # output_layer = parallel_convolution(x_train_A)
#     # 创建模型
#     model = parallel_convolution()
#     # model = Model(inputs=x_train_A, outputs=output_layer)
#     # model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])  # 默认lr=0.001,binary_crossentropy
#     early_stopping = EarlyStopping(monitor='val_loss', patience=35, restore_best_weights=True)
#     # 训练模型
#     model.summary()
#     history = model.fit(x_train_fold, y_train_fold, epochs=300, batch_size=100, validation_data=(x_val_fold, y_val_fold),
#                         callbacks=[early_stopping], verbose=1)
#     # history = model.fit(x_train_fold, y_train_fold, epochs=300, batch_size=100, validation_data=(x_val_fold, y_val_fold),verbose=1)
#     # 评估模型
#     val_predictions = (model.predict(x_val_fold) > 0.5).astype(int)
#     val_accuracy = accuracy_score(y_val_fold, val_predictions)
#     fold_accuracies.append(val_accuracy)
#     val_proba = model.predict(x_val_fold)
#     val_auc = roc_auc_score(y_val_fold, val_proba)
#
#     fold_aucs.append(val_auc)
#     conf_matrix = confusion_matrix(y_val_fold, val_predictions)
#     tn, fp, fn, tp = conf_matrix.ravel()
#     sp = tn / (tn + fp)
#     fold_specificities.append(sp)
#     sn = tp / (tp + fn)
#     fold_sensitivities.append(sn)
#     mcc = (tp * tn - fp * fn) / ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
#     fold_mccs.append(mcc)
#
#     # loss = history.history['loss']
#     # val_loss = history.history['val_loss']
#     # plt.plot(loss, label='loss')
#     # plt.plot(val_loss, label='val_loss')
#     # plt.title('model loss')
#     # plt.ylabel('loss')
#     # plt.xlabel('epoch')
#     # plt.legend(['train', 'valid'], loc='upper left')
#     # plt.savefig('./loss-base1.png')
#     # plt.show()
#
#     # 保存最佳模型和参数
#     if val_accuracy > best_accuracy:
#         best_accuracy = val_accuracy
#         best_model = model
#         best_params = model.get_weights()
#     best_model.save('model11.h5')
# # 输出每个折的验证准确率和AUC
# for i, (acc, auc) in enumerate(zip(fold_accuracies, fold_aucs)):
#     print(f"Fold {i + 1} Validation Accuracy: {acc:.4f}, AUC: {auc:.4f}")
#
# # 输出平均准确率和AUC
# mean_accuracy = np.mean(fold_accuracies)
# mean_auc = np.mean(fold_aucs)
# mean_sensitivity = np.mean(fold_sensitivities)
# mean_specificity = np.mean(fold_specificities)
# mean_mcc = np.mean(fold_mccs)
# print(f"Mean Validation Accuracy: {mean_accuracy:.4f}, Mean AUC: {mean_auc:.4f}")
# print(f"Mean Validation SP: {mean_specificity:.4f}, Mean SN: {mean_sensitivity:.4f}")
# print(f"Mean Validation MCC: {mean_mcc:.4f}")
#
# # 使用最佳模型和参数进行验证
# best_model.set_weights(best_params)
# test_proba = best_model.predict(x_test_pro_PE)
# test_predictions = (test_proba > 0.5).astype(int)
# test_accuracy = accuracy_score(y_test, test_predictions)
# test_auc = roc_auc_score(y_test, test_proba)
# print(f"Best Model Test Accuracy: {test_accuracy:.4f}, Test AUC: {test_auc:.4f}")
# print("-------Best Model result-------")
# cm = confusion_matrix(y_test, test_predictions)
# tn, fp, fn, tp = cm.ravel()
# SP = tn / (tn + fp)
# SN = tp / (tp + fn)
# precision = tp / (tp + fp)
# F1 = f1_score(y_test, test_predictions)
# MCC = matthews_corrcoef(y_test, test_predictions)
# # Print evaluation metrics
# print("Confusion Matrix:\n", cm)
# print("Specificity (SP):", SP)
# print("Sensitivity (SN):", SN)
# print("Precision:", precision)
# print("F1-score:", F1)
# print("Matthews Correlation Coefficient (MCC):", MCC)
#
# # Plot ROC curve
# fpr, tpr, thresholds = roc_curve(y_test, test_proba)
# plt.plot(fpr, tpr)
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver Operating Characteristic (ROC) Curve')
# plt.show()