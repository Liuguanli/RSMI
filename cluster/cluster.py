from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
import numpy as np
import pandas as pd
import time, timeit
import sys, getopt

def clock(func):
    def clocked(*args, **kwargs):
        t0 = timeit.default_timer()
        result = func(*args, **kwargs)
        elapsed = timeit.default_timer() - t0
        name = func.__name__
        arg_str = ', '.join(repr(arg) for arg in args)
        # print('[%0.8fs] %s(%s)' % (elapsed, name, arg_str))
        print('[%0.8fs] %s' % (elapsed, name))
        return result
    return clocked

@clock
def load_data(name):
    data = pd.read_csv(name, header=None)
    x = data[0].values.reshape(-1, 1)
    y = data[1].values.reshape(-1, 1)
    retult = np.hstack((x,y))
    return retult

# @clock
def cluster(x, method):
    return method.fit(x)

@clock
def test_mini_batch_kmeans_manual(X):
    batch_size = 100000
    n_clusters = 10000
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=0, batch_size=batch_size, max_iter=10)
    nums = int(X.shape[0] / batch_size)
    for i in range(nums):
        print("----", i, "----")
        x = X[(0+i)*batch_size:(1+i)*batch_size]
        kmeans = kmeans.partial_fit(x)
    # print(kmeans.cluster_centers_)
    np.savetxt("/home/research/datasets/OSM_100000000_1_2_minibatchkmeans_manual.csv", kmeans.cluster_centers_, delimiter=",")


@clock
def test_mini_batch_kmeans_auto(X, k, save_path):
    batch_size = 100000
    n_clusters = 10000
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=0, batch_size=batch_size, max_iter=10).fit(X)
    # print(kmeans.cluster_centers_)
    np.savetxt(save_path, kmeans.cluster_centers_, delimiter=",")


def test_kmeans(X):
    gap = 1000000
    nums = int(X.shape[0] / gap)
    centers = []
    for i in range(nums):
        x = X[(0+i)*gap:(1+i)*gap]
        # dbscan = cluster(x, DBSCAN(eps=0.1, min_samples=100))
        # centers.append(dbscan.core_sample_indices_)
        kmeans = cluster(x, KMeans(n_clusters=10, random_state=0))
        centers.append(kmeans.cluster_centers_)

    for i in range(len(centers)):
        if i == 0:
            res = centers[i]
        else:
            res = np.concatenate((res, centers[i]))
    print(res)
    # np.savetxt("/home/research/datasets/OSM_100000000_1_2_DBSCAN.csv", res, delimiter=",")

@clock
def test_DBSCAN(eps, min_samples, X):
    # X = X[0:10000000]
    dbscan = cluster(X, DBSCAN(eps=eps, min_samples=min_samples))
    # centers.append(dbscan.core_sample_indices_)
    np.savetxt("/home/research/datasets/OSM_100000000_1_2_DBSCAN.csv", dbscan.core_sample_indices_, delimiter=",")
    # print(dbscan.core_sample_indices_)
    # print(dbscan.core_sample_indices_.shape)
    label_set = {}
    # print(dbscan.labels_)
    for label in dbscan.labels_:
        if label in label_set.keys():
            label_set[label] += 1
        else:
            label_set[label] = 1
    # print(label_set)
    # print(label_set.keys())
    print(len(label_set.keys()))
    print(label_set[-1])
    
def parser(argv):
    try:
        opts, args = getopt.getopt(argv, "d:s:n:m:k:f:")
    except getopt.GetoptError:
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-d':
            distribution = arg
        elif opt == '-s':
            size = int(arg)
        elif opt == '-n':
            skewness = int(arg)
        elif opt == '-m':
            dim = int(arg)
        elif opt == '-k':
            k = int(arg)
        elif opt == '-f':
            filename = arg
    return distribution, size, skewness, dim, k, filename

if __name__ == "__main__":
    distribution, size, skewness, dim, k, filename = parser(sys.argv[1:])
    print(distribution, size, skewness, dim, k, filename)
    X = load_data("/home/research/datasets/%s_%d_%d_%d_.csv" % (distribution, size, skewness, dim))
    # test_mini_batch_kmeans_manual(X)
    test_mini_batch_kmeans_auto(X, k, filename % (distribution, size, skewness, dim))
    # for i in range(4):
    # 23854
    # 99457101
    # test_DBSCAN(0.00001, 10, X)
    # test_DBSCAN(0.0001, 10, X)
    # test_DBSCAN(0.001, 10, X)
    # test_DBSCAN(0.01, 10, X)
    # print("-----------------")
    # test_DBSCAN(0.000001, 100, X)
    # # 85
    # # 99973097
    # test_DBSCAN(0.00001, 100, X)
    # # 6797
    # # 98083068
    # test_DBSCAN(0.0001, 100, X)
    # test_DBSCAN(0.001, 100, X)
    # test_DBSCAN(0.01, 100, X)