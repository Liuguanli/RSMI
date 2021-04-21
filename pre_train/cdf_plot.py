# from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
import os
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor

import scipy.stats as stats

import torch


def plot_cdf(xs, ys, labels):
    plt.subplot(341)
    plt.xticks([])
    plt.ylabel("CDF value")
    plt.title(labels[0])
    plt.plot(xs[0], ys[0])
    plt.subplot(342)
    plt.xticks([])
    plt.title(labels[1])
    plt.plot(xs[1], ys[1])
    plt.subplot(343)
    plt.xticks([])
    plt.title(labels[2])
    plt.plot(xs[2], ys[2])
    plt.subplot(344)
    plt.xticks([])
    plt.title(labels[3])
    plt.plot(xs[3], ys[3])

    plt.subplot(345)
    plt.xticks([])
    plt.title(labels[4])
    plt.plot(xs[4], ys[4])

    plt.ylabel("CDF value")
    plt.subplot(346)
    plt.xticks([])
    plt.title(labels[5])
    plt.plot(xs[5], ys[5])

    plt.subplot(347)
    plt.xticks([])
    plt.title(labels[6])
    plt.plot(xs[6], ys[6])

    plt.subplot(348)
    plt.xticks([])
    plt.title(labels[7])
    plt.plot(xs[7], ys[7])

    # plt.ylabel("CDF value")
    plt.subplot(349)
    plt.title(labels[8])
    plt.plot(xs[8], ys[8])

    plt.subplot(3,4,10) 
    plt.title(labels[9])
    plt.plot(xs[9], ys[9])

    plt.subplot(3,4,11) 
    plt.title(labels[10])
    plt.plot(xs[10], ys[10])

    plt.subplot(3,4,12) 
    plt.title(labels[11])
    plt.plot(xs[11], ys[11])

    plt.savefig("datasets.eps", format='eps', bbox_inches='tight')
    plt.show()

def load_data(name):
    data = pd.read_csv(name, header=None)
    x = data[0].values.reshape(-1, 1)
    y = data[1].values.reshape(-1, 1)
    return x, y

def plot_osm_cdf_zm_for_paper_figure():
    resolution = 54
    x, y = load_data('./features_zm/' + str(resolution) + '_OSM_100000000_1_2_.csv')
    side = int(pow(2, resolution / 2))
    plt.plot(x, y)
    plt.subplots_adjust(hspace=0.2)
    plt.savefig("OSM_cdfs_high_resolution.png", format='png', bbox_inches='tight')
    plt.savefig("OSM_cdfs_high_resolution.eps", format='eps', bbox_inches='tight')
    plt.show()

def plot_osm_cdf_zm():
    xs = []
    ys = []
    labels = []
    resolutions = [6, 22, 38, 54]
    for resolution in resolutions:
        x, y = load_data('./features_zm/' + str(resolution) + '_OSM_100000000_1_2_.csv')
        xs.append(x)
        ys.append(y)
        side = int(pow(2, resolution / 2))
        labels.append("$2 ^{" + str(int(resolution / 2)) + "}$*" + "$2 ^{" + str(int(resolution / 2)) + "}$" + ' cells')
    for i in range(2):
        for j in range(2):
            k = i * 2 + j
            # print(k)
            if k <= 2:
                plt.xticks([])
            else:
                print(k)
            plt.subplot(2, 2, k + 1) 
            plt.title(labels[k], fontsize=10)
            plt.plot(xs[k], ys[k])
    plt.subplots_adjust(hspace=0.2)
    plt.savefig("OSM_cdfs_zm_2t2.png", format='png', bbox_inches='tight')
    plt.savefig("OSM_cdfs_zm_2t2.eps", format='eps', bbox_inches='tight')
    plt.show()

# def plot_osm_cdf_zm():
#     xs = []
#     ys = []
#     labels = []
#     resolutions = [6, 22, 38, 54]
#     for i in range(25):
#         resolutions.append(int(pow(2, i)))
#     for resolution in resolutions:
#         x, y = load_data('./features_zm/' + str(resolution) + '/OSM_100000000_1_2_.csv')
#         xs.append(x)
#         ys.append(y)
#         labels.append('cell width=' + str(resolution))
#     for i in range(5):
#         for j in range(5):
#             k = i * 5 + j
#             # print(k)
#             if k <= 20:
#                 plt.xticks([])
#             else:
#                 print(k)
#             plt.subplot(5, 5, k + 1) 
#             plt.title(labels[k], fontsize=8)
#             plt.plot(xs[k], ys[k])
#     plt.subplots_adjust(hspace=0.5)
#     plt.savefig("OSM_cdfs_zm.png", format='png', bbox_inches='tight')
#     plt.show()

def plot_osm_cdf_rsmi():
    xs = []
    ys = []
    labels = []
    resolutions = []
    for i in range(25):
        resolutions.append(int(pow(2, i)))
    for resolution in resolutions:
        x, y = load_data('./features_rsmi/' + str(resolution) + '/OSM_100000000_1_2_.csv')
        xs.append(x)
        ys.append(y)
        labels.append('cell width=' + str(resolution))
    for i in range(5):
        for j in range(5):
            k = i * 5 + j
            # print(k)
            if k <= 20:
                plt.xticks([])
            else:
                print(k)
            plt.subplot(5, 5, k + 1) 
            plt.title(labels[k], fontsize=8)
            plt.plot(xs[k], ys[k])
    plt.subplots_adjust(hspace=0.5)
    plt.savefig("OSM_cdfs_rsmi.png", format='png', bbox_inches='tight')
    plt.show()

def plot_cdf_zm_synthetic():
    xs = []
    ys = []
    labels = []
    x, y = load_data('./features_zm/1/0.100000/normal_1000_mean_0.0_stddev_1.000_scale_1.csv')
    xs.append(x)
    ys.append(y)
    x, y = load_data('./features_zm/1/0.100000/normal_1000_mean_0.0_stddev_1.250_scale_1.csv')
    xs.append(x)
    ys.append(y)
    x, y = load_data('./features_zm/1/0.100000/normal_1000_mean_0.0_stddev_1.500_scale_1.csv')
    xs.append(x)
    ys.append(y)
    x, y = load_data('./features_zm/1/0.100000/normal_1000_mean_0.0_stddev_1.750_scale_1.csv')
    xs.append(x)
    ys.append(y)
    x, y = load_data('./features_zm/1/0.100000/normal_1000_mean_0.0_stddev_2.000_scale_1.csv')
    xs.append(x)
    ys.append(y)
    x, y = load_data('./features_zm/1/0.100000/normal_1000_mean_0.0_stddev_2.500_scale_1.csv')
    xs.append(x)
    ys.append(y)
    x, y = load_data('./features_zm/1/0.100000/normal_1000_mean_0.0_stddev_2.750_scale_1.csv')
    xs.append(x)
    ys.append(y)
    x, y = load_data('./features_zm/1/0.100000/normal_1000_mean_0.0_stddev_3.000_scale_1.csv')
    xs.append(x)
    ys.append(y)
    x, y = load_data('./features_zm/1/0.100000/normal_1000_mean_0.0_stddev_3.750_scale_1.csv')
    xs.append(x)
    ys.append(y)
    x, y = load_data('./features_zm/1/0.100000/normal_1000_mean_0.0_stddev_4.000_scale_1.csv')
    xs.append(x)
    ys.append(y)
    x, y = load_data('./features_zm/1/0.100000/normal_1000_mean_0.0_stddev_4.250_scale_1.csv')
    xs.append(x)
    ys.append(y)
    x, y = load_data('./features_zm/1/0.100000/normal_1000_mean_0.0_stddev_4.750_scale_1.csv')
    xs.append(x)
    ys.append(y)
    x, y = load_data('./features_zm/1/0.100000/normal_1000_mean_0.0_stddev_5.000_scale_1.csv')
    xs.append(x)
    ys.append(y)
    x, y = load_data('./features_zm/1/0.100000/normal_1000_mean_0.5_stddev_1.000_scale_1.csv')
    xs.append(x)
    ys.append(y)
    x, y = load_data('./features_zm/1/0.100000/normal_1000_mean_0.5_stddev_1.500_scale_1.csv')
    xs.append(x)
    ys.append(y)
    x, y = load_data('./features_zm/1/0.100000/normal_1000_mean_0.5_stddev_1.750_scale_1.csv')
    xs.append(x)
    ys.append(y)
    x, y = load_data('./features_zm/1/0.100000/normal_1000_mean_0.5_stddev_2.000_scale_1.csv')
    xs.append(x)
    ys.append(y)
    x, y = load_data('./features_zm/1/0.100000/normal_1000_mean_0.5_stddev_3.000_scale_1.csv')
    xs.append(x)
    ys.append(y)
    x, y = load_data('./features_zm/1/0.100000/normal_1000_mean_0.5_stddev_4.000_scale_1.csv')
    xs.append(x)
    ys.append(y)
    x, y = load_data('./features_zm/1/0.100000/normal_1000_mean_0.5_stddev_5.000_scale_1.csv')
    xs.append(x)
    ys.append(y)
    x, y = load_data('./features_zm/1/0.100000/normal_1000_mean_1.0_stddev_1.000_scale_1.csv')
    xs.append(x)
    ys.append(y)
    x, y = load_data('./features_zm/1/0.100000/normal_1000_mean_1.0_stddev_1.500_scale_1.csv')
    xs.append(x)
    ys.append(y)
    x, y = load_data('./features_zm/1/0.100000/normal_1000_mean_1.0_stddev_2.000_scale_1.csv')
    xs.append(x)
    ys.append(y)
    x, y = load_data('./features_zm/1/0.100000/normal_1000_mean_1.0_stddev_2.500_scale_1.csv')
    xs.append(x)
    ys.append(y)
    # x, y = load_data('./features_zm/1/0.100000/uniform_1000_scale_1.csv')
    x, y = load_data('./features_zm/1/0.100000/uniform_1000000_1_2_.csv')
    xs.append(x)
    ys.append(y)
    for i in range(5):
        for j in range(5):
            k = i * 5 + j
            # print(k)
            if k <= 20:
                plt.xticks([])
            else:
                print(k)
            plt.subplot(5, 5, k + 1) 
            plt.plot(xs[k], ys[k])
    plt.subplots_adjust(hspace=0.5)
    plt.show()


if __name__ == "__main__":
    # plot_osm_cdf_zm()
    plot_osm_cdf_zm_for_paper_figure()
    # plot_osm_cdf_rsmi()
    # plot_cdf_zm_synthetic()