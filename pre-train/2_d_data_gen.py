# !/usr/bin/python
# coding=utf-8

import sys, getopt
import os

import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
import matplotlib.patches as patches
from matplotlib.ticker import MultipleLocator
import configparser
import tensorflow as tf
import time

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

H_xs = [
[0.0, 0.25],[0.25,0.5],[0.25,0.5],[0.0, 0.25],[0.0, 0.25],[0.0, 0.25],[0.25,0.5],[0.25,0.5],
[0.5,0.75],[0.5,0.75],[0.75,1.0],[0.75,1.0],[0.75,1.0],[0.5,0.75],[0.5,0.75],[0.75,1.0]
]
H_ys = [
[0.0, 0.25],[0.0, 0.25],[0.25, 0.5],[0.25, 0.5],[0.5, 0.75],[0.75,1.0],[0.75,1.0],[0.5, 0.75],
[0.5, 0.75],[0.75,1.0],[0.75,1.0],[0.5, 0.75],[0.25, 0.5],[0.25, 0.5],[0.0, 0.25],[0.0, 0.25]
]

Z_xs = [
[0.0, 0.25],[0.25,0.5],[0.0,0.25],[0.25,0.5],[0.5,0.75],[0.75,1.0],[0.5,0.75],[0.75,1.0],
[0.0, 0.25],[0.25,0.5],[0.0,0.25],[0.25,0.5],[0.5,0.75],[0.75,1.0],[0.5,0.75],[0.75,1.0]
]

Z_ys = [
[0.0, 0.25],[0.0, 0.25],[0.25,0.5],[0.25,0.5],[0.0, 0.25],[0.0, 0.25],[0.25,0.5],[0.25,0.5],
[0.5,0.75],[0.5,0.75],[0.75,1.0],[0.75,1.0],[0.5,0.75],[0.5,0.75],[0.75,1.0],[0.75,1.0]
]

def getUniformPoints(num, filename, dim):
    locations = []
    for i in range(num):
        temp = []
        for j in range(dim):
            temp.append(random.uniform(0, 1))
        locations.append(temp)
    sizes = []
    temp_size = 1000000
    # temp_size = 2
    while temp_size <= num:
        sizes.append(temp_size)
        temp_size *= 2
    # print(sizes)
    for size in sizes:
        temp = locations[0:size]
        np.savetxt(filename % (size, dim), np.array(temp), delimiter=",")

    # all_result = {}
    # for i in range(dim - 1):
    #     all_result[i+2] = []
    # for i in range(num):
    #     node_string = ''
    #     for j in range(dim):
    #         val = random.uniform(0, 1)
    #         node_string = node_string + str(val) + ","
    #         if j >= 1:
    #             all_result[j + 1].append(node_string + str(i*1.0/num) + "\n")
    #     # node_string = node_string + str(i) + "\n"
    #     # all_result.append(node_string)
    # for j in range(dim - 1):
    #     name = filename % (num, j + 2)
    #     all_fo = open(name, "w")
    #     for i in range(num):
    #         all_fo.write(all_result[j+2][i])
    #     all_fo.close()

def getNormalPoints(num, filename, dim, mean=0.5, stddev=0.125):
    locations_tf = []
    for i in range(dim):
        locations_tf.append(tf.random.truncated_normal([num, 1], mean=0.5, stddev=0.125, dtype=tf.float32))
        # locations_tf.append(tf.random_normal([num * 2, 1], mean=mean, stddev=stddev, dtype=tf.float32))
    with tf.compat.v1.Session() as sees:
        locations = []
        for i in range(dim):
            locations.append(sees.run(locations_tf[i]))
        name = filename % (num, dim, mean, stddev)
        index = 0
        with open(name, "w") as fo:

            # for i in range(num * 2):
            while index < num and i < num:
                while True:
                    iswritable = True
                    node_string = ''
                    for j in range(dim):
                        if locations[j][i][0] < 0 or locations[j][i][0] > 1:
                            iswritable = False
                            break
                        node_string = node_string + str(locations[j][index][0]) + ","
                    if iswritable:
                        node_string = node_string + str(i*1.0/num) + "\n"
                        fo.write(node_string)
                        i += 1
                        index += 1
                        break
                    else:
                        i += 1

def getSkewedPoints(num, a, filename, dim):
    locations_tf = tf.random.truncated_normal([num, dim], mean=0.5, stddev=0.25, dtype=tf.float32)
    with tf.compat.v1.Session() as sees:
        locations = sees.run(locations_tf)
    for i in range(num):
        locations[i][1] = locations[i][1] ** a 
        # for i in range(dim):
        # locations.append(sees.run(locations_tf[i]))
    # for a in range(1, 9, 2):
    # name = filename % (num, a, dim)
    # with open(name, "w") as fo:
    #     for i in range(num):
    #         node_string = ''
    #         for j in range(dim - 1):
    #             node_string = node_string + str(locations[j][i][0]) + ","
    #         node_string = node_string + str(locations[dim - 1][i][0] ** a) + "," + str(i*1.0/num) + "\n"
    #         fo.write(node_string)
    # sizes = [1000000, 2000000, 4000000, 8000000, 16000000, 32000000, 64000000, 128000000]
    sizes = []
    temp_size = 1000000
    # temp_size = 2
    while temp_size <= num:
        sizes.append(temp_size)
        temp_size *= 2
    # print(sizes)
    for size in sizes:
        temp = locations[0:size]
        np.savetxt(filename % (size, a, dim), np.array(temp), delimiter=",")
    # all_result = {}
    # for i in range(dim - 1):
    #     all_result[i+2] = []
    # for i in range(num):
    #     node_string = ''
    #     for j in range(dim):
    #         val = random.uniform(0, 1)
    #         if j == dim - 1:
    #             val = val ** a
    #         node_string = node_string + str(val) + ","
    #         if j >= 1:
    #             all_result[j + 1].append(node_string + str(i) + "\n")
        # node_string = node_string + str(i) + "\n"
        # all_result.append(node_string)

    # for j in range(dim - 1):
    #     name = filename % (num, a, dim)
    #     all_fo = open(name, "w")
    #     for i in range(num):
    #         all_fo.write(all_result[j+2][i])
    #     all_fo.close()

def parser(argv):
    try:
        opts, args = getopt.getopt(argv, "d:s:n:f:m:")
    except getopt.GetoptError:
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-d':
            distribution = arg
        elif opt == '-s':
            size = int(arg)
        elif opt == '-n':
            skewness = int(arg)
        elif opt == '-f':
            filename = arg
        elif opt == '-m':
            dim = int(arg)
    return distribution, size, skewness, filename, dim

# python pre_train/data_gen.py -d uniform -s 10000 -n 1 -f /pre_train/data/uniform_10000_1_2_.csv -m 2
# python pre_train/data_gen.py -d normal -s 10000 -n 1 -f pre_train/data/normal_10000_1_2_.csv -m 2
# python pre_train/data_gen.py -d skewed -s 10000 -n 12 -f pre_train/data/skewed_10000_12_2_.csv -m 2

cells4_sfcs = []
synthetic_sfcs = []
def enumerate_4cells_sfc(sfc, b0_num, b1_num, length):
    if len(sfc) == length:
        cells4_sfcs.append(sfc)
        # print(sfc)
        return
    if b0_num > 0:
        sfc_copy = list(sfc)
        sfc_copy.append(0)
        enumerate_4cells_sfc(sfc_copy, b0_num - 1, b1_num, length)
    if b1_num > 0:
        sfc_copy = list(sfc)
        sfc_copy.append(1)
        enumerate_4cells_sfc(sfc_copy, b0_num, b1_num - 1, length)

def enumerate_synthetic_sfc(sfc, length):
    if length == 4:
        synthetic_sfcs.append(sfc)
        # print(sfc)
        return
    for i in range(len(cells4_sfcs)):
        sfc_copy = list(sfc)
        sfc_copy.extend(cells4_sfcs[i])
        enumerate_synthetic_sfc(sfc_copy, length + 1)

def gen_4cells_SFC():
    total_points = [2, 3, 4]
    sfc = []
    for i in range(len(total_points)):
        enumerate_4cells_sfc(sfc, 4 - total_points[i], total_points[i], 4)

def gen_2d_dataset_via_sfc(sfc, sfc_type, num = 100):
    result = []
    index = 0
    for i in range(len(sfc)):
        if sfc[i] != 0:
            if sfc_type == "Z":
                x_range = Z_xs[i]
                y_range = Z_ys[i]
            if sfc_type == "H":
                x_range = H_xs[i]
                y_range = H_ys[i]
            for j in range(num):
                x = random.uniform(x_range[0], x_range[1])
                y = random.uniform(y_range[0], y_range[1])
                result.append(str(x) + "," + str(y) + "," + str(index) + "\n")
                index += 1
    return result

def cal_sfc_dist(sfc1, sfc2):
    # if len(sfc1) != len(sfc2):
    #     return -1
    # sum1 = sum(sfc1)
    # sum2 = sum(sfc2)
    # if sum1 == 0 or sum2 == 0:
    #     return -1
    # sfc1 = [x / sum1 for x in sfc1]
    # sfc2 = [x / sum2 for x in sfc2]
    temp_sum_1 = 0
    temp_sum_2 = 0
    dist = 0
    is_upper = False
    is_Lower = False
    for i in range(len(sfc1)):
        temp_sum_1 += sfc1[i]
        temp_sum_2 += sfc2[i]
        if temp_sum_1 >= temp_sum_2:
            dist = temp_sum_1 - temp_sum_2 if temp_sum_1 - temp_sum_2 > dist else dist
            if i == len(sfc1) - 1:
                is_upper = True
                return dist
        else:
            break
    dist = 0
    temp_sum_1 = 0
    temp_sum_2 = 0
    for i in range(len(sfc1)):
        temp_sum_1 += sfc1[i]
        temp_sum_2 += sfc2[i]
        if temp_sum_1 <= temp_sum_2:
            dist = temp_sum_1 - temp_sum_2 if temp_sum_1 - temp_sum_2 < dist else dist
            if i == len(sfc1) - 1:
                is_Lower = True
                return dist
        else:
            break
    return -1

def save_synthetic_datasets(sfc_type, synthetic_sfcs):
    for i in range(len(synthetic_sfcs)):
        result = gen_2d_dataset_via_sfc(synthetic_sfcs[i], sfc_type, 20)
        if len(result) == 0:
            continue
        all_fo = open("/home/liuguanli/Documents/pre_train/2D_data/%s/dataset_%d_.csv" % (sfc_type, i), "w")
        for item in result:
            all_fo.write(item)
        all_fo.close()

def gen_SFC():
    sfc = []
    enumerate_synthetic_sfc(sfc, 0)
    print(len(synthetic_sfcs))

def synthetic_sfcs_pruning(threshold, synthetic_sfcs):
    synthetic_sfcs_prob = []
    for sfc1 in synthetic_sfcs:
        sum1 = sum(sfc1)
        if sum1 == 0:
            continue
        sfc1 = [x / sum1 for x in sfc1]
        synthetic_sfcs_prob.append(sfc1)
    synthetic_sfcs = synthetic_sfcs_prob
    synthetic_sfcs_delete = []
    counter = 0
    for synthetic_sfc in synthetic_sfcs:
        is_upper_exist = False
        is_lower_exist = False
        for temp in synthetic_sfcs:
            if synthetic_sfc != temp:
                dist = cal_sfc_dist(synthetic_sfc, temp)
                if dist > 0:
                    if threshold > dist:
                        is_lower_exist = True
                else:
                    if -threshold < dist:
                        is_upper_exist = True
                if is_upper_exist and is_lower_exist:
                    synthetic_sfcs_delete.append(synthetic_sfc)
                    break
        # if counter % 100 == 0:
        #     print(counter, len(synthetic_sfcs_delete))
        counter += 1
    print(len(synthetic_sfcs_delete))
    synthetic_sfcs = [i for i in synthetic_sfcs if i not in synthetic_sfcs_delete]
    print(len(synthetic_sfcs))
    return synthetic_sfcs

def gen(distribution, size, dim):
    if distribution == 'uniform':
        filename = "/home/liuguanli/Documents/pre_train/2D_data/uniform_%d_1_%d_.csv"
        getUniformPoints(size, filename, dim)
    elif distribution == 'normal':
        means = [0.0, 0.25, 0.5, 0.75, 1.0]
        stddevs = [0.125, 0.25, 0.5]
        filename = "/home/liuguanli/Documents/pre_train/2D_data/normal_%d_1_%d_%f_%f_.csv"
        for mean in means:
            for stddev in stddevs:
                getNormalPoints(size, filename, dim, mean, stddev)
    elif distribution == 'skewed':
        filename = "/home/liuguanli/Documents/pre_train/2D_data/skewed_%d_%d_%d_.csv"
        skewnesses = [4]
        # skewnesses = [2,3,4,5,6,7,8,9]
        for skewness in  skewnesses:
            getSkewedPoints(size, skewness, filename, dim)

if __name__ == '__main__':
    # distribution, size, skewness, filename, dim = parser(sys.argv[1:])
    # distribution, size, dim = 'normal', 1000, 2
    # gen(distribution, size, dim)
    # distribution, size, dim = 'uniform', 1000000, 2
    # gen(distribution, size, dim)
    # distribution, size, dim = 'uniform', 2000000, 2
    # gen(distribution, size, dim)
    # distribution, size, dim = 'uniform', 4000000, 2
    # gen(distribution, size, dim)
    # distribution, size, dim = 'uniform', 8000000, 2
    # gen(distribution, size, dim)
    # distribution, size, dim = 'uniform', 16000000, 2
    # gen(distribution, size, dim)
    # distribution, size, dim = 'uniform', 32000000, 2
    # gen(distribution, size, dim)
    # distribution, size, dim = 'uniform', 64000000, 2
    # gen(distribution, size, dim)
    distribution, size, dim = 'uniform', 128000000, 2
    gen(distribution, size, dim)
    # distribution, size, dim = 'skewed', 128000000, 2
    # distribution, size, dim = 'skewed', 10, 2
    # gen(distribution, size, dim)
    # save_synthetic_datasets("Z")

    # gen_4cells_SFC()
    # gen_SFC()

    # 14785 pruned
    # synthetic_sfcs = synthetic_sfcs_pruning(0.1, synthetic_sfcs)

    # save_synthetic_datasets("Z", synthetic_sfcs)
    # save_synthetic_datasets("H", synthetic_sfcs)
    
    # a = [1, 1, 1, 1, 0]
    # b = [1, 1, 1, 1, 1]
    # print(cal_sfc_dist(a, b))
    # print(cal_sfc_dist(b, a))