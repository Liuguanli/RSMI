#ifndef CONSTANTS_H
#define CONSTANTS_H
#include <string>
using namespace std;
class Constants
{
public:
    static const int DIM = 2;
    static const int PAGESIZE = 100;
    static const int EACH_DIM_LENGTH = 8;
    static const int INFO_LENGTH = 8;
    static const int MAX_WIDTH = 16;
    static const int EPOCH = 500;
    static const int START_EPOCH = 300;
    static const int EPOCH_ADDED = 100;
    static const int HIDDEN_LAYER_WIDTH = 50;
    static const int THRESHOLD = 20000;

    static const int DEFAULT_SIZE  = 16000000;
    static const int DEFAULT_SKEWNESS  = 4;

    static const double LEARNING_RATE;
    static const string RECORDS;
    static const string QUERYPROFILES;
    static const string DATASETS;

    static const string DEFAULT_DISTRIBUTION;

    static const string BUILD;
    static const string UPDATE;
    static const string POINT;
    static const string WINDOW;
    static const string ACCWINDOW;
    static const string KNN;
    static const string ACCKNN;
    static const string INSERT;
    static const string DELETE;
    static const string INSERTPOINT;
    static const string INSERTWINDOW;
    static const string INSERTACCWINDOW;
    static const string INSERTKNN;
    static const string INSERTACCKNN;
    static const string DELETEPOINT;
    static const string DELETEWINDOW;
    static const string DELETEACCWINDOW;
    static const string DELETEKNN;
    static const string DELETEACCKNN;

    static const string TORCH_MODELS;
    Constants();
};

#endif