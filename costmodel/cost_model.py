# https://www.kaggle.com/plantsgo/solution-public-0-471-private-0-505
import pandas as pd
from sklearn import preprocessing
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
import os, sys
import time, timeit
from sklearn.utils import shuffle

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

import pre_train.costmodel_data_gen as dg
from joblib import dump, load

import pydotplus
from IPython.display import Image, display
from six import StringIO
from sklearn.tree import export_graphviz

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

def generate_training_dataset():
    # sizes = [10000, 20000, 40000, 80000, 160000, 320000, 640000, 1280000, 2000000, 8000000, 32000000, 64000000]
    # sizes = [10000, 20000, 40000, 80000, 160000, 320000, 640000, 1280000]
    sizes = [2000000, 4000000, 8000000, 16000000, 32000000, 64000000]
    dim = 2
    distribution = 'uniform'
    for size in sizes:
        dg.gen(distribution, size, dim)
    distribution = 'normal'
    for size in sizes:
        dg.gen(distribution, size, dim)
    distribution = 'skewed'
    for size in sizes:
        dg.gen(distribution, size, dim)

def encode_onehot(df, column_name):
    feature_df = pd.get_dummies(df[column_name], prefix=column_name)
    all = pd.concat([df.drop([column_name], axis=1), feature_df], axis=1)
    return all

@clock
def build_cost_model():
    df = pd.read_csv('train_set.csv')

    df = shuffle(df)
    train_x = df[['cardinality', 'distribution', 'method']]

    train_y_1 = df[['build_cost']]
    train_y_2 = df[['query_cost']]

    # clf = RandomForestRegressor(max_depth=2, n_estimators=10, random_state=0, bootstrap = True,min_samples_leaf=3,min_samples_split=5)
    clf_build = RandomForestRegressor(max_depth=2, n_estimators=20, random_state=0)
    clf_query = RandomForestRegressor(max_depth=2, n_estimators=20, random_state=0)

    # estimator = clf._make_estimator()
    train_x = encode_onehot(train_x, 'distribution')
    train_x = encode_onehot(train_x, 'method')

    clf_build.fit(train_x.values, train_y_1.values.ravel())
    clf_query.fit(train_x.values, train_y_2.values.ravel())
    dump(clf_build, 'cost_model_build.joblib')
    dump(clf_query, 'cost_model_query.joblib') 
    temp = pd.concat([train_x, train_y_1], axis=1)
    temp = pd.concat([temp, train_y_2], axis=1)
    temp.to_csv("train_set_formatted.csv",index=False)

@clock
def predict(lamb, cardinality, distribution):
    clf_build = load('cost_model_build.joblib')
    clf_query = load('cost_model_query.joblib')
    # todo methods ... 
    # score = lamb * clf_build.predict() + (1 - lamb) * clf_query.predict()
    distributions = {'normal':[1,0,0],'skewed':[0,1,0],'uniform':[0,0,1]}
    # original , rs, rl, mr, sp, cl
    methods = {'cl':[1,0,0,0,0,0],'mr':[0,1,0,0,0,0],'original':[0,0,1,0,0,0],
                'rl':[0,0,0,1,0,0],'rs':[0,0,0,0,1,0],'sp':[0,0,0,0,0,1]}
    max_score = 0
    targer_method = ""
    for method in methods:
        test_item = []
        test_item.append(cardinality)
        test_item.extend(distributions[distribution])
        test_item.extend(methods[method])
        build_score = clf_build.predict([test_item])
        query_score = clf_query.predict([test_item])
        score = lamb * build_score + (1 - lamb) * query_score
        if score > max_score:
            max_score = score
            targer_method = method
    print("max_score", max_score)
    print("targer_method", targer_method)
    return targer_method

def plot(clf):
    estimator = clf.estimators_[1]

    os.environ['PATH'] = os.environ['PATH']+';'+os.environ['CONDA_PREFIX']+r"\Library\bin\graphviz"
    dot_data = StringIO()

    # export_graphviz(estimator, out_file=dot_data)
    export_graphviz(estimator, out_file=dot_data, filled=True, rounded=True, special_characters=True)
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    graph.write_pdf("tree.pdf")
    Image(graph.create_png())



if __name__ == '__main__':
    # generate_training_dataset()
    build_cost_model()
    # predict(0.5, 10000000, "normal")
    # predict(0.5, 100000000, "normal")
    # predict(0.5, 1000000, "normal")
    # pass