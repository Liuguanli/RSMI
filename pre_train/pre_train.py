from scipy.interpolate import CubicSpline
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
from pytorch_NN import Net

import torch

# np.load('./data/books_200M_uint32',allow_pickle=True)

def load_data(name):
    data = pd.read_csv(name, header=None)
    x = data[0].values.reshape(-1, 1)
    y = data[1].values.reshape(-1, 1)
    return x, y

def is_monotonic(model, xs):
    old = model.predict([xs[0]])
    for x in xs:
        result = model.predict([x])
        if old > result:
            return False
        old = result
    return True

def train(x, y, model_type="linear", width=50):
    xs = None
    parameters = []
    if model_type == "linear":
        model = LinearRegression()
        model.fit(x, y)
        parameters.append(model.coef_.tolist()[0][0])
        parameters.append(model.intercept_.tolist()[0])
        xs = x
    elif model_type == "cubic":
        degrees = [3, 2, 1]
        for degree in degrees:
            parameters = []
            poly = PolynomialFeatures(degree=degree)
            poly.fit(x)
            xs = poly.transform(x)
            model = LinearRegression()
            model.fit(xs, y)
            parameters = model.coef_.tolist()[0]
            # print(parameters)
            parameters[0] = model.intercept_.tolist()[0]
            if is_monotonic(model, xs):
                for i in range(3 - degree):
                    parameters.append(0)
                parameters.reverse()
                print(parameters)
                break

        # is_linear = True
        # for degree in degrees:
        #     if degree == 3:
        #         parameters = np.polyfit(x.flatten(), y.flatten(), degree)
        #     elif degree == 2:
        #         parameters=[]
        #         parameters.append(0)
        #         parameters.extend(np.polyfit(x.flatten(), y.flatten(), degree))
        #     if parameters[0] > 0:
        #         print('f1 is :\n',parameters)
        #         model = np.poly1d(parameters)
        #         is_linear = False
        # if is_linear:
        #     parameters = []
        #     model = LinearRegression()
        #     model.fit(x, y)
        #     parameters.append(0)
        #     parameters.append(0)
        #     parameters.append(model.coef_.tolist()[0][0])
        #     parameters.append(model.intercept_.tolist()[0])
    elif model_type == "nn":
        # model = MLPRegressor(hidden_layer_sizes=(52,), tol=1e-3, max_iter=500, random_state=0, activation='relu')
        # model.fit(x, y.flatten())
        # # print(model.coefs_[0][0])
        # # print(model.intercepts_[0][0])
        # parameters = model.coefs_[0][0].tolist()
        # parameters.append(model.intercepts_[0][0])
        # parameters.extend(model.coefs_[1][0].tolist())
        # parameters.append(model.intercepts_[1][0])
        # parameters.appendmodel.intercepts_.tolist()[0])
        model = Net(width)
        tensor_x = torch.from_numpy(x)
        tensor_y = torch.from_numpy(y)
        model.train(tensor_x, tensor_y)

        # module = torch.jit.script(net)
        # module.save("/home/research/code/SOSD/pre_train/xx.pt")
        xs = tensor_x
    return parameters, model, xs

def get_features(x, hist=10):
    res = stats.relfreq(x, numbins=hist, defaultreallimits=(0,1))
    # res = stats.relfreq(x, numbins=hist)
    return res.frequency

def get_errors(model, x, y, model_type="linear"):
    min_err = 0
    max_err = 0
    for i in range(len(x)):
        predict_index = 0
        if model_type == "linear":
            predict_index = model.predict([x[i]])[0][0]
        elif model_type == "cubic":
            predict_index = model.predict([x[i]])[0][0]
        elif model_type == "nn":
            # predict_index = model.forward(x[i].float())
            predict_index = model.forward(x[i].double())
        gap = y[i][0] - max(min(predict_index, 1), 0)
        if gap < 0:
            if gap < min_err:
                min_err = gap
        elif gap > 0:
            if gap > max_err:
                max_err = gap
    return min_err, max_err

def get_json(x, y, features, model_type, parameters):
    conf_content = {}
    conf_content['features'] = ','.join(map(str, features.tolist()))
    conf_content['model_type'] = model_type
    conf_content['parameters'] = ','.join(map(str, parameters))
    conf_content['x_min'] = float(x[0][0])
    conf_content['x_max'] = float(x[-1][0])
    conf_content['y_min'] = float(y[0][0])
    conf_content['y_max'] = float(y[-1][0])
    return conf_content

# def pre_train_synthetic_cubic():
#     pass

# /home/liuguanli/Documents/pre_train/features_zm/
def pre_train_synthetic(model_type="linear", width=50, epsilon=0.1):
    path = "/home/liuguanli/Documents/pre_train/data/" + str(epsilon)
    files= os.listdir(path)
    conf_dist = {}
    for file in files:
        if not os.path.isdir(file):
            print(file)

            x, y = load_data(path + '/' + file)
            features = get_features(x)
            # print(features)

            parameters, model, xs = train(x, y, model_type, width)
            min_err, max_err = get_errors(model, xs, y, model_type)
            print("------------gap------------: ",(max_err.item() - min_err.item()) * len(xs))
            json_item = get_json(x, y, features, model_type, parameters)
            prefix = file.split('.csv')[0]
            if model_type == "nn":
                # model_cpp = torch.jit.trace(model, xs[0].float())
                if x.shape[0] != 0:
                    model_cpp = torch.jit.trace(model, xs[0].double())
                    model_cpp.save('/home/liuguanli/Documents/pre_train/models_zm/1/' + str(epsilon) + '.pt')
                    # if type(min_err) != int:
                    #     min_err = min_err.item()
                    # if type(max_err) != int:
                    #     max_err = max_err.item()
            # json_item['min_err'] = min_err
            # json_item['max_err'] = max_err
            # if model_type == "nn":
            #     f = open('./trained_models/' + model_type + '/' + str(width) + '/' + prefix + '.json','w')
            # else:
            #     f = open('./trained_models/' + model_type + '/' + prefix + '.json','w')
            # f.write(json.dumps(json_item))
            # f.close()

        # conf_dist[prefix] = json_item
    
    # f = open('./trained_models/synthetic.json','w')
    # f.write(json.dumps(conf_dist))
    # f.close()

def pre_train_real():
    path = "../data"
    files= os.listdir(path)
    conf_dist = {}
    for file in files:
        if file.find("lookups") == -1:
            if file.find("uint64") != -1:
                print(file)
                xbash = np.fromfile(path + '/' + file, dtype='uint64')
            if file.find("uint32") != -1:
                print(file)
                xbash = np.fromfile(path + '/' + file, dtype='uint32')
                xbash = np.delete(xbash, 0)
            x = np.delete(xbash, 0)
            n = x.shape[0]
        
            y = np.arange(n)
            scale = 1.0
            y = y / n * scale
            x = x.reshape(-1, 1)
            y = y.reshape(-1, 1)

            prefix = file.split('.')[0]
            parameters, model = train(x, y)
            print(parameters)
            features = get_features(x)
            print(features)
            json_item = get_json(x, y, features, 'linear', parameters)

            f = open('./trained_models/' + prefix + '.json','w')
            f.write(json.dumps(json_item))
            f.close()

            conf_dist[prefix] = json_item
    
    f = open('./trained_models/real.json','w')
    f.write(json.dumps(conf_dist))
    f.close()

if __name__ == "__main__":
    torch.set_default_tensor_type('torch.DoubleTensor')
    # pre_train_synthetic(model_type="linear")
    # pre_train_synthetic(model_type="cubic")
    pre_train_synthetic(model_type="nn", width=8, epsilon=0.1)
    pre_train_synthetic(model_type="nn", width=8, epsilon=0.3)
    pre_train_synthetic(model_type="nn", width=8, epsilon=0.5)
    # pre_train_synthetic(model_type="nn", width=12)
    # pre_train_synthetic(model_type="nn", width=20)
    # pre_train_synthetic(model_type="nn", width=24)
    # pre_train_synthetic(model_type="nn", width=32)
    # pre_train_real()

    # xbash = np.fromfile('../data/books_200M_uint32', dtype='uint32')
    # xbash = np.delete(xbash, 0)
    # xs = np.delete(xbash, 0)
    # print(xs[0])
    # print(xs[199999999])
    # n = xs.shape[0]
    # scale = 1024.0
    # ys = np.arange(n)
    # ys = ys / n * scale
    # xs = xs.reshape(-1, 1)
    # ys = ys.reshape(-1, 1)
    # model = LinearRegression()
    # model.fit(xs, ys)
    # print(model.coef_)  # 1.21791541e-06
    # print(model.intercept_) # 135.90037025
    # print(model.predict([[0]]))


# rmi_data = np.fromfile('./rmi_data/books_200M_uint32_8_L1_PARAMETERS', dtype='uint32')
# print(rmi_data)

# print('---------------------------------')

# rmi_mr_data = np.fromfile('./rmi_mr_data/books_200M_uint32_8_L1_PARAMETERS', dtype='uint32')
# print(rmi_mr_data)

# print('---------------------------------')

# xbash = np.fromfile('./data/books_200M_uint32', dtype='uint32')
# xbash = np.delete(xbash, 0)
# xs = np.delete(xbash, 0)
# n = xs.shape[0]
# scale = 1024.0
# ys = np.arange(n)
# ys = ys / n * scale
# print(xs)
# print(ys)

# print(np.where(xs == 940075014))
# result = np.where(xs == 1073741824)
# xs[result[0][1]] = 1073741825

# xs = xs.reshape(-1, 1)
# ys = ys.reshape(-1, 1)

# # model = make_pipeline(PolynomialFeatures(2), LinearRegression())

# model = LinearRegression()
# model.fit(xs, ys)
# print(model.coef_)  # 1.21791541e-06
# print(model.intercept_) # 135.90037025
# print(model.predict([[0]]))
# y_plot = model.predict(xs)
# err_all = 0
# for i in range(n):
#     err_all += abs(y_plot[i] - ys[i])
#     y_plot[i] = min(y_plot[i], scale)
#     y_plot[i] = max(y_plot[i], 0)

# print("err_all: ", err_all)

# # res = stats.relfreq(xs, numbins=n)
# # x_cdf = res.lowerlimit + np.linspace(0, res.binsize*res.frequency.size, res.frequency.size)
# # y_cdf = np.cumsum(res.frequency)
# # plt.plot(x_cdf, y_cdf)
# xs = xs/xs[n-1]
# # ys = ys/n
# plt.plot(xs, ys)
# plt.plot(xs, y_plot)
# plt.show()
