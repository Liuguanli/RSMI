import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
import os
def load_data(name):
    data = pd.read_csv(name, header=None)
    x = data[0].values.reshape(-1, 1)
    y = data[1].values.reshape(-1, 1)
    res = []
    for i in range(x.shape[0]):
        if i % 2 == 0:
            res.append([x[i][0],y[i][0]])
    return res

def load_real_datasets():
    xs = []
    ys = []
    labels = []
    res  = load_data("/home/research/datasets/SA_367442158_1_2_.csv")
    np.savetxt("/home/research/datasets/SA_180000000_1_2_.csv", np.array(res), delimiter=",")
    xs.append(x)
    ys.append(y)
    labels.append("SA")

    # x, y  = load_data("/home/research/AS_1343101984_1_2_.csv")
    # xs.append(x)
    # ys.append(y)
    # labels.append("AS")

    # x, y  = load_data("/home/research/AF_766776444_1_2_.csv")
    # xs.append(x)
    # ys.append(y)
    # labels.append("AF")

    # x, y  = load_data("/home/research/NA_1322975397_1_2_.csv")
    # xs.append(x)
    # ys.append(y)
    # labels.append("NA")

    return xs, ys, labels

def plot(xs, ys, labels):
    plt.subplot(2,2,1) 
    # plt.title(labels[10])
    print(len(xs[0]))
    plt.plot(xs[0], ys[0])

    # plt.subplot(2,2,2) 
    # # plt.title(labels[10])
    # plt.plot(xs[1], ys[1])

    # plt.subplot(2,2,3) 
    # # plt.title(labels[10])
    # plt.plot(xs[2], ys[2])

    # plt.subplot(2,2,4) 
    # # plt.title(labels[10])
    # plt.plot(xs[3], ys[3])
    plt.show()

if __name__ == "__main__":
    xs, ys, labels = load_real_datasets()
    # plot(xs, ys, labels)