import numpy as np
import pandas as pd

from sklearn import preprocessing
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
import os, sys
# import time, timeit
from sklearn.utils import shuffle

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

import pre_train.costmodel_data_gen as dg
from joblib import dump, load

def encode_onehot(df, column_name):
    feature_df = pd.get_dummies(df[column_name], prefix=column_name)
    all = pd.concat([df.drop([column_name], axis=1), feature_df], axis=1)
    return all
# class Record_Info:
#     def __init__(self, cdf_changing, relative_depth, relative_cardinaltiy, insertion_ratio, distribution):
#         self.cdf_changing = cdf_changing
#         self.relative_depth = relative_depth
#         self.relative_cardinaltiy = relative_cardinaltiy
#         self.insertion_ratio = insertion_ratio
#         self.distribution = distribution
#         self.insertion_time = 0
#         self.old_query_time = 0
#         self.new_query_time = 0

# #  CDF
# #  relative depth 
# #  cardinality
# #  insertion ratio
# def read_from_insertion():
#     insert_path = "../files/records/insert/"
#     insert_query_path = "../files/records/insertPoint/"
#     build_path = "../files/records/build/"
#     files= os.listdir(insert_path)
#     for file in files:
#         if not os.path.isdir(file):
#             # TODO method cardinality
#             file_names = file.split('_')
#             cardinality = file_names[2]
#             structure = file_names[0]
#             with open(build_path + file) as f:
#                 lines = f.readlines()
#                 for line in lines:
#                     print(line)


#             with open(insert_path + file) as f:
#                 lines = f.readlines()
#                 for line in lines:
#                     print(line)
#         break

# if __name__ == '__main__':
#     # generate_training_dataset()
#     read_from_insertion()



# #  cardinality
# #  CDF
# #  relative depth 
# #  insertion ratio
def insert_data():
    data_arr = []

    data_arr.append([1, 0.008, 1.1, 0.01, 'skew', 0])
    data_arr.append([1, 0.015, 1.1, 0.02, 'skew', 0])
    data_arr.append([1, 0.024, 1.1, 0.03, 'skew', 0])
    data_arr.append([1, 0.033, 1.1, 0.04, 'skew', 0])
    data_arr.append([1, 0.041, 1.1, 0.05, 'skew', 0])
    data_arr.append([1, 0.050, 1.1, 0.06, 'skew', 0])
    data_arr.append([1, 0.059, 1.1, 0.07, 'skew', 0])
    data_arr.append([1, 0.068, 1.1, 0.08, 'skew', 0])
    data_arr.append([1, 0.077, 1.2, 0.09, 'skew', 0])
    data_arr.append([1, 0.086, 1.2, 0.10, 'skew', 0])
    data_arr.append([1, 0.095, 1.2, 0.11, 'skew', 0])
    data_arr.append([1, 0.104, 1.2, 0.12, 'skew', 1])
    data_arr.append([1, 0.113, 1.2, 0.13, 'skew', 1])
    data_arr.append([1, 0.122, 1.2, 0.14, 'skew', 1])
    data_arr.append([1, 0.131, 1.2, 0.15, 'skew', 1])
    data_arr.append([1, 0.139, 1.2, 0.16, 'skew', 1])
    data_arr.append([1, 0.143, 1.2, 0.17, 'skew', 1])
    data_arr.append([1, 0.150, 1.3, 0.18, 'skew', 1])
    data_arr.append([1, 0.155, 1.3, 0.19, 'skew', 1])
    data_arr.append([1, 0.158, 1.3, 0.20, 'skew', 1])
    data_arr.append([1, 0.164, 1.3, 0.21, 'skew', 1])
    data_arr.append([1, 0.168, 1.4, 0.22, 'skew', 1])

    data_arr.append([1.28, 0.008, 1.1, 0.01, 'skew', 0])
    data_arr.append([1.28, 0.015, 1.1, 0.02, 'skew', 0])
    data_arr.append([1.28, 0.024, 1.1, 0.03, 'skew', 0])
    data_arr.append([1.28, 0.033, 1.1, 0.04, 'skew', 0])
    data_arr.append([1.28, 0.041, 1.1, 0.05, 'skew', 0])
    data_arr.append([1.28, 0.050, 1.1, 0.06, 'skew', 0])
    data_arr.append([1.28, 0.059, 1.1, 0.07, 'skew', 0])
    data_arr.append([1.28, 0.068, 1.1, 0.08, 'skew', 0])
    data_arr.append([1.28, 0.077, 1.2, 0.09, 'skew', 0])
    data_arr.append([1.28, 0.086, 1.2, 0.10, 'skew', 0])
    data_arr.append([1.28, 0.095, 1.2, 0.11, 'skew', 0])
    data_arr.append([1.28, 0.104, 1.2, 0.12, 'skew', 0])
    data_arr.append([1.28, 0.113, 1.2, 0.13, 'skew', 0])
    data_arr.append([1.28, 0.122, 1.2, 0.14, 'skew', 0])
    data_arr.append([1.28, 0.131, 1.2, 0.15, 'skew', 1])
    data_arr.append([1.28, 0.139, 1.2, 0.16, 'skew', 1])
    data_arr.append([1.28, 0.143, 1.2, 0.17, 'skew', 1])
    data_arr.append([1.28, 0.150, 1.3, 0.18, 'skew', 1])
    data_arr.append([1.28, 0.155, 1.3, 0.19, 'skew', 1])
    data_arr.append([1.28, 0.158, 1.3, 0.20, 'skew', 1])
    data_arr.append([1.28, 0.164, 1.3, 0.21, 'skew', 1])
    data_arr.append([1.28, 0.168, 1.4, 0.22, 'skew', 1])
    data_arr.append([1.28, 0.168, 1.4, 0.23, 'skew', 1])
    data_arr.append([1.28, 0.168, 1.4, 0.24, 'skew', 1])
    data_arr.append([1.28, 0.168, 1.4, 0.25, 'skew', 1])
    data_arr.append([1.28, 0.168, 1.4, 0.26, 'skew', 1])
    data_arr.append([1.28, 0.168, 1.4, 0.27, 'skew', 1])
    data_arr.append([1.28, 0.168, 1.4, 0.28, 'skew', 1])
    data_arr.append([1.28, 0.168, 1.4, 0.29, 'skew', 1])
    data_arr.append([1.28, 0.168, 1.4, 0.30, 'skew', 1])

    # data_arr.append([1.28, 0.001, 1.1, 0.01, 'normal', 0])
    # data_arr.append([1.28, 0.002, 1.1, 0.02, 'normal', 0])
    # data_arr.append([1.28, 0.004, 1.1, 0.03, 'normal', 0])
    # data_arr.append([1.28, 0.005, 1.1, 0.04, 'normal', 0])
    # data_arr.append([1.28, 0.007, 1.1, 0.05, 'normal', 0])
    # data_arr.append([1.28, 0.010, 1.1, 0.06, 'normal', 0])
    # data_arr.append([1.28, 0.011, 1.1, 0.07, 'normal', 0])
    # data_arr.append([1.28, 0.013, 1.1, 0.08, 'normal', 0])
    # data_arr.append([1.28, 0.014, 1.2, 0.09, 'normal', 0])
    # data_arr.append([1.28, 0.017, 1.2, 0.10, 'normal', 0])
    # data_arr.append([1.28, 0.019, 1.2, 0.11, 'normal', 0])
    # data_arr.append([1.28, 0.021, 1.2, 0.12, 'normal', 0])
    # data_arr.append([1.28, 0.025, 1.2, 0.13, 'normal', 0])
    # data_arr.append([1.28, 0.027, 1.2, 0.14, 'normal', 0])
    # data_arr.append([1.28, 0.030, 1.2, 0.15, 'normal', 0])
    # data_arr.append([1.28, 0.031, 1.2, 0.16, 'normal', 0])
    # data_arr.append([1.28, 0.033, 1.2, 0.17, 'normal', 0])
    # data_arr.append([1.28, 0.034, 1.3, 0.18, 'normal', 0])
    # data_arr.append([1.28, 0.036, 1.3, 0.19, 'normal', 0])
    # data_arr.append([1.28, 0.039, 1.3, 0.20, 'normal', 0])
    # data_arr.append([1.28, 0.041, 1.3, 0.21, 'normal', 1])
    data_arr.append([1.28, 0.044, 1.4, 0.22, 'normal', 1])

    data_arr.append([1.28, 0.000, 1.1, 0.01, 'uniform', 0])
    # data_arr.append([1.28, 0.001, 1.1, 0.02, 'uniform', 0])
    # data_arr.append([1.28, 0.001, 1.1, 0.03, 'uniform', 0])
    # data_arr.append([1.28, 0.002, 1.1, 0.04, 'uniform', 0])
    # data_arr.append([1.28, 0.003, 1.1, 0.05, 'uniform', 0])
    # data_arr.append([1.28, 0.003, 1.1, 0.06, 'uniform', 0])
    # data_arr.append([1.28, 0.003, 1.1, 0.07, 'uniform', 0])
    # data_arr.append([1.28, 0.003, 1.1, 0.08, 'uniform', 0])
    # data_arr.append([1.28, 0.004, 1.2, 0.09, 'uniform', 0])
    # data_arr.append([1.28, 0.004, 1.2, 0.10, 'uniform', 0])
    # data_arr.append([1.28, 0.005, 1.2, 0.11, 'uniform', 0])
    # data_arr.append([1.28, 0.005, 1.2, 0.12, 'uniform', 0])
    # data_arr.append([1.28, 0.006, 1.2, 0.13, 'uniform', 0])
    # data_arr.append([1.28, 0.006, 1.2, 0.14, 'uniform', 0])
    # data_arr.append([1.28, 0.007, 1.2, 0.15, 'uniform', 0])
    # data_arr.append([1.28, 0.008, 1.2, 0.16, 'uniform', 0])
    # data_arr.append([1.28, 0.008, 1.2, 0.17, 'uniform', 0])
    # data_arr.append([1.28, 0.008, 1.3, 0.18, 'uniform', 0])
    # data_arr.append([1.28, 0.008, 1.3, 0.19, 'uniform', 0])
    # data_arr.append([1.28, 0.009, 1.3, 0.20, 'uniform', 0])
    # data_arr.append([1.28, 0.009, 1.3, 0.21, 'uniform', 1])
    # data_arr.append([1.28, 0.009, 1.4, 0.22, 'uniform', 1])

    np_data = np.array(data_arr)

    pd_data = pd.DataFrame(np_data,columns=['cardinality', 'cdf_change', 'relative_depth', 'update_ratio', 'distribution', 'label'])
    print(pd_data)
    pd_data.to_csv('training_set_raw.csv')

def deal_data():
    df = pd.read_csv('training_set_raw.csv')

    df = shuffle(df)
    train_x = df[['cardinality', 'cdf_change', 'relative_depth', 'update_ratio', 'distribution']]

    train_y = df[['label']]

    train_x = encode_onehot(train_x, 'distribution')

    temp = pd.concat([train_x, train_y], axis=1)
    temp.to_csv("train_set_formatted.csv", index=False, header=None)

if __name__ == '__main__':
    insert_data()
    deal_data()