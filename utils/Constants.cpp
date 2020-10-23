#include "Constants.h"

#ifdef __APPLE__
// const string Constants::RECORDS = "/Users/guanli/Dropbox/records/VLDB20/";
// const string Constants::QUERYPROFILES = "/Users/guanli/Documents/datasets/RLRtree/queryprofile/";
// const string Constants::DATASETS = "/Users/guanli/Documents/datasets/RLRtree/raw/";
const string Constants::RECORDS = "./files/records/";
const string Constants::QUERYPROFILES = "./files/queryprofile/";
const string Constants::DATASETS = "./datasets/";
#else
// const string Constants::RECORDS = "/home/liuguanli/Dropbox/records/VLDB20/";
// const string Constants::QUERYPROFILES = "/home/liuguanli/Documents/datasets/RLRtree/queryprofile/";
// const string Constants::DATASETS = "/home/liuguanli/Documents/datasets/RLRtree/raw/";
const string Constants::RECORDS = "./files/records/";
const string Constants::QUERYPROFILES = "./files/queryprofile/";
const string Constants::DATASETS = "./datasets/";
#endif
const string Constants::DEFAULT_DISTRIBUTION = "skewed";
const string Constants::BUILD = "build/";
const string Constants::UPDATE = "update/";
const string Constants::POINT = "point/";
const string Constants::WINDOW = "window/";
const string Constants::ACCWINDOW = "accwindow/";
const string Constants::KNN = "knn/";
const string Constants::ACCKNN = "accknn/";
const string Constants::INSERT = "insert/";
const string Constants::DELETE = "delete/";
const string Constants::INSERTPOINT = "insertPoint/";
const string Constants::INSERTWINDOW = "insertWindow/";
const string Constants::INSERTACCWINDOW = "insertAccWindow/";
const string Constants::INSERTKNN = "insertKnn/";
const string Constants::INSERTACCKNN= "insertAccKnn/";
const string Constants::DELETEPOINT= "delete_point/";
const string Constants::DELETEWINDOW= "deleteWindow/";
const string Constants::DELETEACCWINDOW= "deleteAccWindow/";
const string Constants::DELETEKNN= "deleteKnn/";
const string Constants::DELETEACCKNN= "deleteAccKnn/";

const string Constants::TORCH_MODELS = "./torch_models/";
const double Constants::LEARNING_RATE = 0.05;
Constants::Constants()
{
}
