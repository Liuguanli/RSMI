# !/usr/bin/python
# coding=utf-8

import os
import random
import sys

import pandas as pd
import tensorflow as tf

from sympy import *
import math

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

filename = "/home/liuguanli/Documents/pre_train/1D_data/%.1f/index_%d.csv"
dirname = "/home/liuguanli/Documents/pre_train/1D_data/%.1f/"

cardinality = 100

class DataGenerator:

    def __init__(self, size, scale=1, epsilon=0.1):
        self.size = size
        self.scale = scale
        self.epsilon = epsilon
        self.composition_num = 0
        self.compositions = []

    def getUniformPoints(self, filename):
        num = self.size
        name = filename % (self.epsilon, num, int(self.scale))
        if os.path.exists(name):
            return name
        values = []
        for i in range(num):
            val = random.uniform(0, 1)
            values.append(val)
        values.sort()
        with open(name, "w") as all_fo:
            for i in range(num):
                node_string = str(values[i]) + "," + str(float(i * self.scale) / num) + "\n"
                all_fo.write(node_string)
        return name

    def getSkewPoints(self, filename, skewness=3):
        num = self.size
        name = filename % (self.epsilon, num, skewness, int(self.scale))
        if os.path.exists(name):
            return name
        values = []
        for i in range(num):
            val = random.uniform(0, 1)
            values.append(val)
        values.sort()
        out_values = []
        # write size
        with open(name, "w") as all_fo:
            all_fo.write(str(num))
            for i in range(num):
                # node_string = str(int(values[i] ** skewness * 1000000000000) + i) + "," + str(float(i * self.scale) / num) + "\n"
                node_string = str(int(values[i] ** skewness * 1000000000000) + i)
                all_fo.write(node_string)
                # out_values.append(int(values[i] ** skewness * 1000000000000) + i)
        return name

    # def getNormalPoints(self, filename):
    #     num = self.size
    #     name = filename % (num, int(self.scale))
    #     if os.path.exists(name):
    #         return name
    #     with tf.compat.v1.Session() as sees:
    #         tf_random = tf.random.truncated_normal([num, 1], mean=0.5, stddev=0.25, dtype=tf.float32)
    #         values = []
    #         locations = sees.run(tf_random)
    #         for item in locations:
    #             values.append(item[0])
    #         values.sort()
    #         with open(name, "w") as fo:
    #             for i in range(num):
    #                 node_string = str(values[i] * self.size) + "," + str(float(i * self.scale) / num) + "\n"
    #                 fo.write(node_string)
    #     return name

    def normalize(self, values):
        min_val = values[0]
        max_val = values[-1]
        gap = max_val - min_val
        for i in range(len(values)):
            values[i] = (values[i] - min_val) / gap
        return values

    def getNormalPoints(self, filename, mean=0, stddev=0.25):
        num = self.size
        name = filename % (self.epsilon, num, mean, stddev, int(self.scale))
        if os.path.exists(name):
            return name
        with tf.compat.v1.Session() as sees:
            # tf_random = tf.random.truncated_normal([num, 1], mean=mean, stddev=stddev, dtype=tf.float32)
            tf_random = tf.random.normal([num, 1], mean=mean, stddev=stddev, dtype=tf.float32)
            values = []
            locations = sees.run(tf_random)
            for item in locations:
                values.append(abs(item[0]))
            # values.append(1)
            values.sort()
            values = self.normalize(values)
            with open(name, "w") as fo:
                for i in range(num):
                    node_string = str(values[i]) + "," + str(float(i * self.scale) / num) + "\n"
                    fo.write(node_string)
            # data_set_size = 0
            # for i in range(len(values)):
            #     if (values[i] <= 1):
            #         data_set_size += 1
            # if data_set_size > 100:
            #     with open(name, "w") as fo:
            #         for i in range(data_set_size):
            #             node_string = str(values[i]) + "," + str(float(i * self.scale) / data_set_size) + "\n"
            #             fo.write(node_string)
        return name

    def getCluster_one(self, filename, cluster=10, stddev=1.0):
        num_of_cluster = cluster
        num = self.size // num_of_cluster
        name = filename % (self.epsilon, num * num_of_cluster, num_of_cluster, stddev, int(self.scale))
        if os.path.exists(name):
            return name
        gap = 1.0 / num_of_cluster
        with tf.compat.v1.Session() as sees:
            locations_tf = []
            for i in range(num_of_cluster):
                locations_tf.append(
                    tf.random.normal([num, 1], mean=gap * i + gap / 2.0, stddev=stddev, dtype=tf.float32))
        
            locations = []
            for i in range(num_of_cluster):
                locations.append(sees.run(locations_tf[i]))
            values = []
            for location in locations:
                for item in location:
                    values.append(item[0])
            values.sort()
            with open(name, "w") as fo:
                length = len(values)
                for i in range(length):
                    node_string = str(values[i]) + ',' + str(float(i * self.scale) / length) + "\n"
                    fo.write(node_string)
        return name

    def cal_miu_sigma(self):
        t = Symbol('t')
        miu_j = Symbol('miu_j')
        sigma_j = Symbol('sigma_j')

        miu_i = Symbol('miu_i')
        sigma_i = Symbol('sigma_i')

        miu_j = 0
        miu_i = 0
        sigma_i = 1
        sigma_j = 1

        mius = []
        sigmas = []
        mius.append(miu_j)
        sigmas.append(sigma_j)

        x_range = 20

        miu_range = int(1.0 / self.epsilon) + 1
        miu_gap = self.epsilon

        sigma_range = 20
        sigma_gap = 0.05

        for k in range(sigma_range):
            sigma_j = round((k + 1) * sigma_gap, 2)
            miu_i = 0
            for i in range(miu_range):
                miu_j = round(i * miu_gap, 2)
                flag = True
                for j in range(x_range):
                    x = j * 0.1
                    m = 1/(sigma_i * sqrt(2*pi)) * integrate(exp(-(t-miu_i)**2/(2*(sigma_i**2))), (t,-oo,x))
                    n = 1/(sigma_j * sqrt(2*pi)) * integrate(exp(-(t-miu_j)**2/(2*(sigma_j**2))), (t,-oo,x))
                    f = m.__float__() - n.__float__()
                    if abs(f) > self.epsilon:
                        flag = False
                if flag:
                    print("miu_i:", miu_i, " miu_j:", miu_j)
                    miu_i = miu_j
                    mius.append(miu_j)
                    sigmas.append(sigma_j)
            sigma_i = sigma_j
            print(mius)
            print(sigmas)

        return mius, sigmas

    def cal_boundary(self):
        min_m = int(1 / self.epsilon)
        if min_m * self.epsilon < 1:
            gap = 2
        else:
            gap = 2
        # gap = min_m
        max_m = min_m + gap if min_m > gap else 2 * min_m
        if max_m < 10:
            max_m = 10
        return min_m, max_m
    
    def composition(self, res, b0_num, b1_num, b2_num, bin_num):
        if len(res) == bin_num:
            # print(res)
            self.compositions.append(res)
            self.composition_num += 1
            return res
        if b0_num > 0:
            a = list(res)
            a.append(0)
            self.composition(a, b0_num - 1, b1_num, b2_num, bin_num)
        if b1_num > 0:
            a = list(res)
            a.append(1)
            self.composition(a, b0_num, b1_num - 1, b2_num, bin_num)
        if b2_num > 0:
            a = list(res)
            a.append(2)
            self.composition(a, b0_num, b1_num, b2_num - 1, bin_num)

def gen_1d_synthetic_datasets():
    # epsilons = [ 0.1,0.2,0.3,0.4,0.5]
    epsilons = [0.1]
    for epsilon in epsilons:
        dg = DataGenerator(100, 1, epsilon)
        min_m, max_m = dg.cal_boundary()
        for i in range(max_m - min_m + 1):
            b2_num = min_m - i
            if b2_num < 0:
                continue
            if i == 0:
                b1_num = 0
                b0_num = max_m - b1_num - b2_num
                print(b0_num, b1_num, b2_num)
                dg.composition([], b0_num, b1_num, b2_num, max_m)
                # print("composition_num: ", dg.composition_num)
            else:
                b1_num = i * 2
                b0_num = max_m - b1_num - b2_num
                if b0_num >= 0:
                    print(b0_num, b1_num, b2_num)
                    dg.composition([], b0_num, b1_num, b2_num, max_m)
                        # print("composition_num: ", dg.composition_num)
        print("-------------------------------------------")
        print("composition_num: ", dg.composition_num)
        item = int(cardinality * epsilon / 2)
        gap = 1.0 / max_m
        for dataset_index, composition in enumerate(dg.compositions):
            dataset = []
            begin = 0
            for index, num in enumerate(composition):
                begin = index * gap
                for j in range(num * item):
                    dataset.append(gap * j / (num * item)  + begin)
            dataset.sort()
            # filename = "./data/%.1f/index_%d.csv"
            # dirname = "./data/%.1f/"
            name = filename % (dg.epsilon, dataset_index)
            length = len(dataset)
            dirs = dirname % (dg.epsilon)
            if not os.path.exists(dirs):
                os.makedirs(dirs)
            with open(name, "w") as all_fo:
                for i in range(length):
                    node_string = str(dataset[i]) + "," + str(float(i) / length) + "\n"
                    all_fo.write(node_string)

if __name__ == '__main__':
    gen_1d_synthetic_datasets()
    gen_2d_synthetic_datasets()
    # get_data_from_tpc_h()
    # scales = [1, 128, 1024]
    # data_sizes = [1000]
    data_sizes = [111]
    # data_sizes = [200000000]
    scales = [1]
    # thresholds = [0.5,0.3,0.1]
    thresholds = [0.3]
    for epsilon in thresholds:
        for data_size in data_sizes:
            for scale in scales:
                dg = DataGenerator(data_size, scale, epsilon)

                filename = "./1D_data/%.1f/uniform_%d_scale_%d.csv"
                dg.getUniformPoints(filename)
                filename = "./1D_data/%.1f/skewed_%d_%.1f_scale_%d.csv"
                skewnesses = [1,3,5,7,9]
                for skewness in skewnesses:
                    dg.getSkewPoints(filename, skewness=skewness)

                # means = [0, 0.01, 0.02, 0.03, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.4, 1.5]

                # stddevs = [0.005, 0.01, 0.0125, 0.025, 0.05, 0.08, 0.09, 0.1, 0.11, 0.12, 0.125, 0.15, 0.16, 0.2, 0.25, 0.5, 1, 1.5, 2, 3, 4]

                # means, stddevs = dg.cal_miu_sigma()

                # # means = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
                # # stddevs = [1, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.25, 1.25, 1.25, 1.25, 1.25, 1.25, 1.25, 1.25, 1.25, 1.25, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.75, 1.75, 1.75, 1.75, 1.75, 1.75, 1.75, 1.75, 1.75, 1.75, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.75, 2.75, 2.75, 2.75, 2.75, 2.75, 2.75, 2.75, 2.75, 2.75, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0]
                # for mean in means:
                #     for stddev in stddevs:
                #         filename = "./1D_data/%.6f/normal_%d_mean_%.1f_stddev_%.3f_scale_%d.csv"
                #         dg.getNormalPoints(filename, mean=mean, stddev=stddev)