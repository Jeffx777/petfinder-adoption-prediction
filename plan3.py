#!/usr/bin/python
# -*- coding: utf-8 -*-

# 导入必要的库
import glob
import json
import string
import re
import numpy as np
import pandas as pd
import Levenshtein as lv
from joblib import Parallel, delayed
from PIL import Image
import os
import random
import warnings
import fastText as ft
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD, NMF, PCA
from sklearn.metrics import mean_squared_error, classification_report
from sklearn.model_selection import StratifiedKFold, train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

import cv2
from tqdm import tqdm
from keras.applications.densenet import preprocess_input, DenseNet121
from keras.models import Model
from keras.layers import GlobalAveragePooling2D, Input, Lambda, AveragePooling1D
import keras.backend as K

import scipy as sp
import tensorflow as tf
from collections import Counter
from functools import partial
from math import sqrt

# 忽略警告信息
warnings.filterwarnings("ignore")


# 设置随机种子，保证结果可复现
def seed_everything(seed=1337):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    tf.set_random_seed(seed)
    np.random.seed(seed)


seed_everything()

# 1. 加载核心数据
train = pd.read_csv('C:/Users/23206/PycharmProjects/petfinder/petfinder-adoption-prediction/train/train.csv')
test = pd.read_csv('C:/Users/23206/PycharmProjects/petfinder/petfinder-adoption-prediction/test/test.csv')
sample_submission = pd.read_csv('C:/Users/23206/PycharmProjects/petfinder/petfinder-adoption-prediction/test'
                                '/sample_submission.csv')

labels_breed = pd.read_csv('C:/Users/23206/PycharmProjects/petfinder/petfinder-adoption-prediction/input/petfinder'
                           '-adoption-prediction/breed_labels.csv')
labels_color = pd.read_csv('C:/Users/23206/PycharmProjects/petfinder/petfinder-adoption-prediction/input/petfinder'
                           '-adoption-prediction/color_labels.csv')
labels_state = pd.read_csv('C:/Users/23206/PycharmProjects/petfinder/petfinder-adoption-prediction/input/petfinder'
                           '-adoption-prediction/state_labels.csv')

# 提取ID和目标变量
target = train['AdoptionSpeed']
train_id = train['PetID']
test_id = test['PetID']

# 2. 特征工程

# 填充缺失值
imp = SimpleImputer(missing_values=np.nan, strategy="median")
train["AdoptionSpeed"] = imp.fit_transform(train[["AdoptionSpeed"]])
train["AdoptionSpeed"] = train["AdoptionSpeed"].astype(int)

# 增加一些新特征
train['DescriptionLength'] = train['Description'].apply(lambda x: len(str(x)))
train['NameLength'] = train['Name'].apply(lambda x: len(str(x)))
train['HasVideo'] = train['VideoAmt'].apply(lambda x: 1 if x > 0 else 0)
train['HasPhoto'] = train['PhotoAmt'].apply(lambda x: 1 if x > 0 else 0)

# 更新特征列表
features = ['Type', 'Age', 'Breed1', 'Breed2', 'Gender', 'Color1', 'Color2', 'Color3',
            'MaturitySize', 'FurLength', 'Vaccinated', 'Dewormed', 'Sterilized', 'Health',
            'Quantity', 'Fee', 'State', 'VideoAmt', 'PhotoAmt', 'DescriptionLength', 'NameLength', 'HasVideo',
            'HasPhoto']

x = train[features]
y = train['AdoptionSpeed']

# 标准化
scaler = StandardScaler()
x[features] = scaler.fit_transform(x[features])

# 拆分训练和测试集
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)

# 使用决策树
dtc = DecisionTreeClassifier(random_state=42)
dtc.fit(x_train, y_train)
dt_predict = dtc.predict(x_test)
print("Decision Tree Classifier Score:", dtc.score(x_test, y_test))
print(classification_report(y_test, dt_predict, target_names=["0", "1", "2", "3", "4"]))

# 使用随机森林
rfc = RandomForestClassifier(n_estimators=100, random_state=42)
rfc.fit(x_train, y_train)
rfc_y_predict = rfc.predict(x_test)
print("Random Forest Classifier Score:", rfc.score(x_test, y_test))
print(classification_report(y_test, rfc_y_predict, target_names=["0", "1", "2", "3", "4"]))

# 使用XGBoost
xgb_model = xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42)
xgb_model.fit(x_train, y_train)
xgb_y_predict = xgb_model.predict(x_test)
print("XGBoost Classifier Score:", xgb_model.score(x_test, y_test))
print(classification_report(y_test, xgb_y_predict, target_names=["0", "1", "2", "3", "4"]))

# 使用交叉验证评估模型
scores = cross_val_score(rfc, x, y, cv=5)
print(f"Random Forest Cross-validation scores: {scores}")
print(f"Mean cross-validation score: {scores.mean()}")

# 使用随机森林全量学习和全量预测
rfc.fit(x, y)
test = pd.read_csv("../input/petfinder-adoption-prediction/test/test.csv")
test['DescriptionLength'] = test['Description'].apply(lambda x: len(str(x)))
test['NameLength'] = test['Name'].apply(lambda x: len(str(x)))
test['HasVideo'] = test['VideoAmt'].apply(lambda x: 1 if x > 0 else 0)
test['HasPhoto'] = test['PhotoAmt'].apply(lambda x: 1 if x > 0 else 0)

x_test = test[features]
x_test[features] = scaler.transform(x_test[features])
final_result = rfc.predict(x_test)

# 生成提交文件
submission_df = pd.DataFrame(data={'PetID': test['PetID'].tolist(), 'AdoptionSpeed': final_result})
submission_df.to_csv('submission.csv', index=False)
