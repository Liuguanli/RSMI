#ifndef EXPRECORDER_H
#define EXPRECORDER_H

#include <vector>
#include "../entities/Point.h"
#include <string>
#include "Constants.h"
#include "SortTools.h"
#include <queue>
using namespace std;

class ExpRecorder
{

public:

    priority_queue<Point , vector<Point>, sortForKNN2> pq;

    long long index_high;
    long long index_low;

    long long leaf_node_num;
    long long non_leaf_node_num;

    int max_error = 0;
    int min_error = 0;

    int depth = 0;

    long long total_depth;

    int N = Constants::THRESHOLD;

    long long average_max_error = 0;
    long long average_min_error = 0;

    int last_level_model_num = 0;

    string structure_name;
    string distribution;
    long dataset_cardinality;

    long long insert_num;
    long delete_num;
    float window_size;
    float window_ratio;
    int k_num;
    int skewness = 1;

    long time;
    long insert_time;
    long delete_time;
    long long rebuild_time;
    int rebuild_num;
    double page_access = 1.0;
    double accuracy;
    long size;

    int window_query_result_size;
    int acc_window_query_qesult_size;
    vector<Point> knn_query_results;
    vector<Point> acc_knn_query_results;

    vector<Point> window_query_results;
    ExpRecorder();
    string get_time();
    string get_time_pageaccess();
    string get_time_accuracy();
    string get_time_pageaccess_accuracy();
    string get_insert_time_pageaccess_rebuild();
    string get_size();
    string get_time_size();
    string get_time_size_errors();

    string get_insert_time_pageaccess();
    string get_delete_time_pageaccess();
    void cal_size();
    void clean();
};

#endif