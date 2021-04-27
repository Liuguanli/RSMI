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
    priority_queue<Point, vector<Point>, sortForKNN2> pq;

    long long point_not_found = 0;

    long long index_high;
    long long index_low;

    long long leaf_node_num;
    long long non_leaf_node_num;

    long long max_error = 0;
    long long min_error = 0;

    int depth = 0;

    long long total_depth;

    int N = Constants::THRESHOLD;

    long long top_error = 0;
    long long bottom_error = 0;
    float loss = 0;

    int last_level_model_num = 0;
    long long non_leaf_model_num = 0;

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
    long top_level_time;
    long bottom_level_time;
    long insert_time;
    long delete_time;
    long long rebuild_time;
    int rebuild_num;
    double page_access = 1.0;
    double accuracy;
    long size;
    long top_rl_time;
    long extra_time;

    long prediction_time = 0;
    long search_time = 0;
    long sfc_cal_time = 0;
    long ordering_cost = 0;
    long training_cost = 0;

    long cost_model_time = 0;

    long search_steps = 0;

    float sampling_rate = 1.0;

    int representative_threshold_m = 64;
    double model_reuse_threshold = 0.1;

    string cluster_method = "kmeans";
    int cluster_size = 10000;

    int window_query_result_size;
    int acc_window_query_qesult_size;
    vector<Point> knn_query_results;
    vector<Point> acc_knn_query_results;
    vector<Point> window_query_results;

    bool is_sp = false;
    bool is_sp_first = false;
    bool is_model_reuse = false;
    bool is_rl = false;
    bool is_cluster = false;
    bool is_rs = false;
    bool is_cost_model = false;

    int level = 1;

    double upper_level_lambda = 0.8;
    double lower_level_lambda = 0.8;

    int rs_threshold_m = 10000;

    ExpRecorder();
    string get_time();
    string get_time_pageaccess();
    string get_time_accuracy();
    string get_time_pageaccess_accuracy();
    string get_insert_time_pageaccess_rebuild();
    string get_size();
    string get_time_size();
    string get_time_size_errors();
    // string get_time_size_errors_sp();
    // string get_time_size_errors_mr();

    string get_insert_time_pageaccess();
    string get_delete_time_pageaccess();
    void cal_size();
    void clean();
    string get_file_name();
    string get_dataset_name();
    void set_structure_name(string prefix);

    ExpRecorder* test_sp();
    ExpRecorder* test_sp_mr();
    ExpRecorder* test_sp_first();
    ExpRecorder* test_model_reuse();
    ExpRecorder* test_rl();
    ExpRecorder* test_rl_mr();
    ExpRecorder* test_cluster();
    ExpRecorder* test_rs();
    ExpRecorder* test_rs_mr();
    ExpRecorder* set_cost_model(bool);
    void test_reset();

    ExpRecorder* set_level(int level);

};

#endif