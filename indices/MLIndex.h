#include <iostream>
#include <fstream>
#include <boost/algorithm/string.hpp>
#include <vector>
#include <iterator>
#include <string>
#include <algorithm>
#include <random>
#include <math.h>
#include <cmath>
#include <iomanip>
#include <sstream>
#include <cstdlib>
#include <ctime>
#include "../utils/ExpRecorder.h"
#include "../entities/Mbr.h"
#include "../entities/Point.h"
#include "../entities/LeafNode.h"
#include "../entities/NonLeafNode.h"
#include "../utils/Constants.h"
#include "../utils/SortTools.h"
#include "../utils/ModelTools.h"
#include "../entities/NodeExtend.h"
#include "../utils/PreTrainZM.h"
#include "../utils/Rebuild.h"
#include <torch/script.h>
#include <ATen/ATen.h>
#include <torch/torch.h>
#include <torch/nn/module.h>
#include <torch/nn/modules/linear.h>
#include <torch/optim.h>
#include <torch/types.h>
#include <torch/utils.h>

using namespace std;
using namespace at;
using namespace torch::nn;
using namespace torch::optim;

class MLIndex
{
private:
    int k = 100;
    long N;
    int page_size;

    vector<vector<std::shared_ptr<Net>>> index;

    vector<int> stages;

    // how to choose these points!!!  // kmeans!!!
    vector<Point> reference_points;

    // TODO for insertion
    vector<Point> inserted_points;
    vector<LeafNode *> inserted_leafnodes;
    Point reference_point_of_inserted_point;

    vector<vector<Point>> partitions;

    vector<double> offsets;
    vector<int> partition_size;

    vector<LeafNode *> leafnodes;

    int zm_max_error = 0;
    int zm_min_error = 0;

    double gap;
    double min_key_val;
    double max_key_val;
    long long top_error;
    long long bottom_error;
    double loss;
    int bit_num = 0;

    long long error_shift = 0;

    string model_path;
    string model_path_root;

    int added_num = 0;

public:
    MLIndex();
    MLIndex(int k);

    void build(ExpRecorder &exp_recorder, vector<Point> points);
    void get_reference_points(ExpRecorder &exp_recorder);
    int get_partition_id(Point point);
    void point_query(ExpRecorder &exp_recorder, Point query_point);
    void point_query(ExpRecorder &exp_recorder, vector<Point> query_points);

    long long get_point_index(ExpRecorder &exp_recorder, Point &query_point);
    long long get_point_index(ExpRecorder &exp_recorder, double key);
    int predict_closest_position(ExpRecorder &exp_recorder, double key, bool is_upper);
    vector<Point> find_furthest_points(Mbr query_window);
    vector<Point> find_closet_points(Mbr query_window);

    void window_query(ExpRecorder &exp_recorder, vector<Mbr> query_windows);
    vector<Point> window_query(ExpRecorder &exp_recorder, Mbr query_window);
    void acc_window_query(ExpRecorder &exp_recorder, vector<Mbr> query_windows);
    vector<Point> acc_window_query(ExpRecorder &exp_recorder, Mbr query_windows);

    void kNN_query(ExpRecorder &exp_recorder, vector<Point> query_points, int k);
    vector<Point> kNN_query(ExpRecorder &exp_recorder, Point query_point, int k);
    void acc_kNN_query(ExpRecorder &exp_recorder, vector<Point> query_points, int k);
    vector<Point> acc_kNN_query(ExpRecorder &exp_recorder, Point query_point, int k);

    void search_inward(int &lnode_index, int lower_bound, double key, vector<Point> &S, int k, Point query_point);
    void search_outward(int &lnode_index, int upper_bound, double key, vector<Point> &S, int k, Point query_point);
    void search_inserted_outward(int &lnode_index, int upper_bound, double key, vector<Point> &S, int k, Point query_point);

    void insert(ExpRecorder &exp_recorder, Point);
    bool insert(ExpRecorder &exp_recorder, vector<Point> &inserted_points);

    void remove(ExpRecorder &exp_recorder, Point);
    void remove(ExpRecorder &exp_recorder, vector<Point>);

    void clear(ExpRecorder &exp_recorder);
};

MLIndex::MLIndex()
{
    this->page_size = Constants::PAGESIZE;
    partitions.resize(k);
    offsets.resize(k + 1);
    partition_size.resize(k + 1);
}

MLIndex::MLIndex(int k)
{
    this->page_size = Constants::PAGESIZE;
    this->k = k;
    partitions.resize(k);
    offsets.resize(k + 1);
    partition_size.resize(k + 1);
}

void MLIndex::get_reference_points(ExpRecorder &exp_recorder)
{
    auto start_cluster = chrono::high_resolution_clock::now();
    string python_path = "/home/liuguanli/Dropbox/shared/VLDB20/codes/rsmi/cluster/cluster.py";
    string result_points_path = "/home/research/datasets/" + exp_recorder.distribution + "_" + to_string(exp_recorder.dataset_cardinality) + "_" + to_string(exp_recorder.skewness) + "_k_" + to_string(k) + "_minibatchkmeans_auto.csv";
    string commandStr = "python " + python_path + " -d " + exp_recorder.distribution + " -s " + to_string(exp_recorder.dataset_cardinality) + " -n " +
                        to_string(exp_recorder.skewness) + " -m 2 -k " + to_string(k) + " -f" + result_points_path;
    cout << "commandStr: " << commandStr << endl;
    char command[1024];
    strcpy(command, commandStr.c_str());
    int res = system(command);
    this->reference_points = pre_train_zm::get_cluster_point(result_points_path);
    cout << "points.size(): " << reference_points.size() << endl;
}

int MLIndex::get_partition_id(Point point)
{
    int partition_index = 0;
    double minimal_dist = numeric_limits<float>::max();
    for (size_t j = 0; j < k; j++)
    {
        double temp_dist = reference_points[j].cal_dist(point);
        if (temp_dist < minimal_dist)
        {
            minimal_dist = temp_dist;
            partition_index = j;
        }
        // cout << "temp_dist: " << temp_dist << endl;
    }
    return partition_index;
}

void MLIndex::build(ExpRecorder &exp_recorder, vector<Point> points)
{
    cout << "ML_index build: " << points.size() << endl;
    auto start = chrono::high_resolution_clock::now();
    // 1 prepare all data points
    // 2 use kmeans to find k reference points
    auto start_z_cal = chrono::high_resolution_clock::now();
    get_reference_points(exp_recorder);
    // 3 for the partitions
    N = points.size();

    for (size_t i = 0; i < N; i++)
    {
        int partition_index = get_partition_id(points[i]);
        points[i].partition_id = partition_index;
        // if (N > 10000000)
        // {
        //     cout<< "partitions.size():" << partitions.size() << endl;
        // }
        partitions[partition_index].push_back(points[i]);
    }

    // 4 calcualte the offsets
    offsets[0] = 0;
    partition_size[0] = 0;
    for (size_t i = 0; i < k; i++)
    {
        double maximal_dist = numeric_limits<float>::min();
        for (Point point : partitions[i])
        {
            double temp_dist = reference_points[i].cal_dist(point);
            if (maximal_dist < temp_dist)
            {
                maximal_dist = temp_dist;
            }
        }
        partition_size[i + 1] = partition_size[i] + partitions[i].size();
        offsets[i + 1] = offsets[i] + maximal_dist;
        // cout << "partitions[i] size: " << partitions[i].size() << endl;
        // cout << "offsets[i]: " << i << " " << offsets[i] << endl;
    }

    // 5 order the data
    for (size_t i = 0; i < N; i++)
    {
        points[i].key = offsets[points[i].partition_id] + points[i].cal_dist(reference_points[points[i].partition_id]);
    }
    auto end_z_cal = chrono::high_resolution_clock::now();
    exp_recorder.sfc_cal_time += chrono::duration_cast<chrono::nanoseconds>(end_z_cal - start_z_cal).count();
    auto start_sort = chrono::high_resolution_clock::now();
    sort(points.begin(), points.end(), sort_key());
    auto end_sort = chrono::high_resolution_clock::now();
    exp_recorder.ordering_cost += chrono::duration_cast<chrono::nanoseconds>(end_sort - start_sort).count();
    min_key_val = points[0].key;
    // min_key_val = points[0].ml_normalized_curve_val;
    max_key_val = points[N - 1].key;
    // max_key_val = points[N - 1].ml_normalized_curve_val;
    gap = max_key_val - min_key_val;
    for (size_t i = 0; i < N; i++)
    {
        // cout << "points[i]: " << points[i].key << endl;
        points[i].index = i * 1.0 / N;
        points[i].ml_normalized_curve_val = (points[i].key - min_key_val) / gap;
    }
    // 6 use ZM index structure to train the data
    // ---------------------------ZM---------------------------
    int leaf_node_num = points.size() / page_size;
    for (int i = 0; i < leaf_node_num; i++)
    {
        LeafNode *leaf_node = new LeafNode();
        auto bn = points.begin() + i * page_size;
        auto en = points.begin() + i * page_size + page_size;
        vector<Point> vec(bn, en);
        leaf_node->add_points(vec);
        LeafNode *temp = leaf_node;
        leafnodes.push_back(temp);
    }
    exp_recorder.leaf_node_num += leaf_node_num;
    if (N > page_size * leaf_node_num)
    {
        LeafNode *leafNode = new LeafNode();
        auto bn = points.begin() + page_size * leaf_node_num;
        auto en = points.end();
        vector<Point> vec(bn, en);
        leafNode->add_points(vec);
        leafnodes.push_back(leafNode);
        exp_recorder.leaf_node_num++;
    }

    stages.push_back(1);
    if (exp_recorder.level > 1)
    {
        if (points.size() > Constants::THRESHOLD)
        {
            if (exp_recorder.level >= 3)
            {
                stages.push_back((int)(sqrt(N / Constants::THRESHOLD)));
            }
            stages.push_back(N / Constants::THRESHOLD);
        }
    }

    vector<vector<vector<Point>>> tmp_records;

    vector<vector<Point>> stage1;
    stage1.push_back(points);
    tmp_records.push_back(stage1);

    stage1.clear();
    stage1.shrink_to_fit();

    for (size_t i = 0; i < stages.size(); i++)
    {
        // initialize
        auto start_0 = chrono::high_resolution_clock::now();
        vector<std::shared_ptr<Net>> temp_index;
        vector<vector<Point>> temp_points;
        int next_stage_length = 0;
        if (i < stages.size() - 1)
        {
            next_stage_length = stages[i + 1];
            for (size_t k = 0; k < next_stage_length; k++)
            {
                vector<Point> stage_temp_points;
                temp_points.push_back(stage_temp_points);
            }
            tmp_records.push_back(temp_points);
        }
        else
        {
            next_stage_length = N;
        }
        long long total_errors = 0;
        for (size_t j = 0; j < stages[i]; j++)
        {
            model_path = to_string(i) + "_" + to_string(j);
            auto net = std::make_shared<Net>(1);
#ifdef use_gpu
            net->to(torch::kCUDA);
#endif
            if (tmp_records[i][j].size() == 0)
            {
                temp_index.push_back(net);
                continue;
            }
            try
            {
                vector<float> locations;
                vector<float> labels;

                if (exp_recorder.is_cost_model)
                {
                    vector<float> locations_;
                    for (Point point : tmp_records[i][j])
                    {
                        locations_.push_back(point.ml_normalized_curve_val);
                    }
                    Histogram histogram(pow(2, Constants::UNIFIED_Z_BIT_NUM), locations_);
                    if (i == 0)
                    {
                        exp_recorder.ogiginal_histogram = histogram;
                        exp_recorder.changing_histogram = histogram;
                    }
                    float lambda = i == 0 ? exp_recorder.upper_level_lambda : exp_recorder.lower_level_lambda;
                    pre_train_zm::cost_model_predict(exp_recorder, lambda, locations_.size() * 1.0 / 10000, pre_train_zm::get_distribution(histogram));
                }

                if (i == 0)
                {
                    if (exp_recorder.is_sp)
                    {
                        exp_recorder.sp_num++;
                        auto start_sp = chrono::high_resolution_clock::now();
                        if (exp_recorder.is_sp_first)
                        {
                            int sample_num = N * exp_recorder.sampling_rate;
                            vector<Point> sampled_points;
                            for (size_t i = 0; i < sample_num; i++)
                            {
                                long long number = rand() % N;
                                sampled_points.push_back(points[number]);
                            }
                            sort(sampled_points.begin(), sampled_points.end(), sort_curve_val());
                            for (Point point : sampled_points)
                            {
                                locations.push_back(point.ml_normalized_curve_val);
                                labels.push_back(point.index);
                            }
                        }
                        else
                        {
                            int sample_gap = 1 / exp_recorder.sampling_rate;
                            long long counter = 0;
                            for (Point point : tmp_records[i][j])
                            {
                                if (counter % sample_gap == 0)
                                {
                                    locations.push_back(point.ml_normalized_curve_val);
                                    labels.push_back(point.index);
                                }
                                counter++;
                            }
                            if (sample_gap >= tmp_records[i][j].size())
                            {
                                Point last_point = tmp_records[i][j][tmp_records[i][j].size() - 1];
                                locations.push_back(last_point.ml_normalized_curve_val);
                                labels.push_back(last_point.index);
                            }
                        }
                        auto finish_sp = chrono::high_resolution_clock::now();
                        exp_recorder.extra_time += chrono::duration_cast<chrono::nanoseconds>(finish_sp - start_sp).count();
                        auto start_train = chrono::high_resolution_clock::now();
                        net->train_model(locations, labels);
                        auto end_train = chrono::high_resolution_clock::now();
                        exp_recorder.training_cost += chrono::duration_cast<chrono::nanoseconds>(end_train - start_train).count();
                    }
                    // else if (exp_recorder.is_cluster)
                    // {
                    //     auto start_cl = chrono::high_resolution_clock::now();
                    //     int k = 10000;
                    //     auto start_cluster = chrono::high_resolution_clock::now();
                    //     string commandStr = "python /home/liuguanli/Dropbox/shared/VLDB20/codes/rsmi/cluster/cluster.py -d " +
                    //                         exp_recorder.distribution + " -s " + to_string(exp_recorder.dataset_cardinality) + " -n " +
                    //                         to_string(exp_recorder.skewness) + " -m 2 -k " + to_string(k) +
                    //                         " -f /home/liuguanli/Documents/pre_train/cluster/%s_%d_%d_%d_minibatchkmeans_auto.csv";
                    //     // string commandStr = "python /home/liuguanli/Documents/pre_train/rl_4_sfc/RL_4_SFC.py";
                    //     cout << "commandStr: " << commandStr << endl;
                    //     char command[1024];
                    //     strcpy(command, commandStr.c_str());
                    //     int res = system(command);
                    //     vector<Point> clustered_points = pre_train_zm::get_cluster_point("/home/research/datasets/OSM_100000000_1_2_minibatchkmeans_auto.csv");
                    //     cout << "clustered_points.size(): " << clustered_points.size() << endl;
                    //     for (Point point : clustered_points)
                    //     {
                    //         locations.push_back(point.ml_normalized_curve_val);
                    //         labels.push_back(point.index);
                    //     }
                    //     auto finish_cl = chrono::high_resolution_clock::now();
                    //     exp_recorder.extra_time += chrono::duration_cast<chrono::nanoseconds>(finish_cl - start_cl).count();
                    //     auto start_train = chrono::high_resolution_clock::now();
                    //     net->train_model(locations, labels);
                    //     auto end_train = chrono::high_resolution_clock::now();
                    //     exp_recorder.training_cost += chrono::duration_cast<chrono::nanoseconds>(end_train - start_train).count();
                    // }
                    else if (exp_recorder.is_rs)
                    {
                        exp_recorder.rs_num++;
                        auto start_rs = chrono::high_resolution_clock::now();
                        vector<Point> tmp_records_0_0 = pre_train_zm::get_rep_set_space(exp_recorder.rs_threshold_m, 0, 0, 0.5, 0.5, tmp_records[0][0]);
                        int temp_N = tmp_records_0_0.size();
                        sort(tmp_records_0_0.begin(), tmp_records_0_0.end(), sort_key());
                        for (long long i = 0; i < temp_N; i++)
                        {
                            labels.push_back(tmp_records_0_0[i].index);
                            locations.push_back(tmp_records_0_0[i].ml_normalized_curve_val);
                        }
                        auto finish_rs = chrono::high_resolution_clock::now();
                        exp_recorder.extra_time += chrono::duration_cast<chrono::nanoseconds>(finish_rs - start_rs).count();
                        auto start_train = chrono::high_resolution_clock::now();
                        net->train_model(locations, labels);
                        auto end_train = chrono::high_resolution_clock::now();
                        exp_recorder.training_cost += chrono::duration_cast<chrono::nanoseconds>(end_train - start_train).count();
                    }
                    else if (exp_recorder.is_rl)
                    {
                        exp_recorder.rl_num++;
                        cout << "RL_SFC begin" << endl;
                        auto start_rl = chrono::high_resolution_clock::now();
                        int bit_num = exp_recorder.bit_num;
                        // pre_train_zm::write_approximate_SFC(Constants::DATASETS, exp_recorder.distribution + "_" + to_string(exp_recorder.dataset_cardinality) + "_" + to_string(exp_recorder.skewness) + "_2_.csv", bit_num);
                        pre_train_zm::write_approximate_SFC(points, exp_recorder.get_file_name(), bit_num);
                        string commandStr = "python /home/liuguanli/Dropbox/shared/VLDB20/codes/rsmi/pre_train/rl_4_sfc/RL_4_SFC.py -d " +
                                            exp_recorder.distribution + " -s " + to_string(exp_recorder.dataset_cardinality) + " -n " +
                                            to_string(exp_recorder.skewness) + " -m 2 -b " + to_string(bit_num) +
                                            " -f /home/liuguanli/Documents/pre_train/sfc_z_weight/bit_num_%d/%s_%d_%d_%d_.csv";
                        // string commandStr = "python /home/liuguanli/Documents/pre_train/rl_4_sfc/RL_4_SFC.py";
                        char command[1024];
                        strcpy(command, commandStr.c_str());
                        int res = system(command);
                        // todo save data
                        vector<int> sfc;
                        // vector<float> labels; //cdf
                        FileReader RL_SFC_reader("", ",");
                        labels.push_back(0);
                        RL_SFC_reader.read_sfc("/home/liuguanli/Documents/pre_train/sfc_z/" + to_string(bit_num) + "_" + exp_recorder.distribution + "_" + to_string(exp_recorder.dataset_cardinality) + "_" + to_string(exp_recorder.skewness) + "_2_.csv", sfc, labels);
                        locations.push_back(0);
                        for (size_t i = 1; i <= sfc.size(); i++)
                        {
                            // temp_sum += sfc[i];
                            locations.push_back(i * 1.0 / sfc.size());
                        }
                        // cout << "locations: " << locations << endl;
                        // cout << "labels: " << labels << endl;
                        auto finish_rl = chrono::high_resolution_clock::now();
                        exp_recorder.top_rl_time = chrono::duration_cast<chrono::nanoseconds>(finish_rl - start_rl).count();
                        exp_recorder.extra_time += exp_recorder.top_rl_time;
                        auto start_train = chrono::high_resolution_clock::now();
                        net->train_model(locations, labels);
                        auto end_train = chrono::high_resolution_clock::now();
                        exp_recorder.training_cost += chrono::duration_cast<chrono::nanoseconds>(end_train - start_train).count();
                        cout << "RL_SFC finish" << endl;
                    }
                    else if (exp_recorder.is_model_reuse)
                    {
                        exp_recorder.model_reuse_num++;
                        auto start_mr = chrono::high_resolution_clock::now();
                        for (Point point : tmp_records[i][j])
                        {
                            locations.push_back(point.ml_normalized_curve_val);
                        }
                        Histogram histogram(pow(2, Constants::UNIFIED_Z_BIT_NUM), locations);
                        std::stringstream stream;
                        stream << std::fixed << std::setprecision(1) << exp_recorder.model_reuse_threshold;
                        string threshold = stream.str();
                        if (net->is_reusable_zm(histogram, threshold, model_path))
                        {
                            cout << "model_path: " << model_path << endl;
                            torch::load(net, model_path);
                            auto finish_mr = chrono::high_resolution_clock::now();
                            exp_recorder.extra_time += chrono::duration_cast<chrono::nanoseconds>(finish_mr - start_mr).count();
                        }
                        else
                        {
                            cout << "train model: " << endl;
                            auto start_train = chrono::high_resolution_clock::now();
                            net->train_model(locations, labels);
                            auto end_train = chrono::high_resolution_clock::now();
                            exp_recorder.training_cost += chrono::duration_cast<chrono::nanoseconds>(end_train - start_train).count();
                            model_path = to_string(i) + "_" + to_string(j);
                        }
                    }
                    else
                    {
                        exp_recorder.original_num++;
                        for (Point point : tmp_records[i][j])
                        {
                            locations.push_back(point.ml_normalized_curve_val);
                            labels.push_back(point.index);
                        }
                        model_path = to_string(i) + "_" + to_string(j);
                        auto start_train = chrono::high_resolution_clock::now();
                        net->train_model(locations, labels);
                        auto end_train = chrono::high_resolution_clock::now();
                        exp_recorder.training_cost += chrono::duration_cast<chrono::nanoseconds>(end_train - start_train).count();
                    }
                }
                else
                {
                    if (exp_recorder.is_model_reuse)
                    {
                        exp_recorder.model_reuse_num++;
                        auto start_mr = chrono::high_resolution_clock::now();
                        for (Point point : tmp_records[i][j])
                        {
                            locations.push_back(point.ml_normalized_curve_val);
                        }
                        Histogram histogram(pow(2, Constants::UNIFIED_Z_BIT_NUM), locations);
                        std::stringstream stream;
                        stream << std::fixed << std::setprecision(1) << exp_recorder.model_reuse_threshold;
                        string threshold = stream.str();
                        if (net->is_reusable_zm(histogram, threshold, model_path))
                        {
                            // cout << "model_path: " << model_path << endl;
                            auto finish_mr = chrono::high_resolution_clock::now();
                            exp_recorder.extra_time += chrono::duration_cast<chrono::nanoseconds>(finish_mr - start_mr).count();
                            torch::load(net, model_path);
                        }
                        else
                        {
                            // cout << "train model 1" << endl;
                            auto start_train = chrono::high_resolution_clock::now();
                            net->train_model(locations, labels);
                            auto end_train = chrono::high_resolution_clock::now();
                            exp_recorder.training_cost += chrono::duration_cast<chrono::nanoseconds>(end_train - start_train).count();
                            model_path = to_string(i) + "_" + to_string(j);
                            // torch::save(net, model_path_root + model_path);
                        }
                    }
                    else if (exp_recorder.is_sp)
                    {
                        exp_recorder.sp_num++;
                        auto start_sp = chrono::high_resolution_clock::now();
                        int sample_gap = 1 / sqrt(exp_recorder.sampling_rate);
                        long long counter = 0;
                        for (Point point : tmp_records[i][j])
                        {
                            if (counter % sample_gap == 0)
                            {
                                locations.push_back(point.ml_normalized_curve_val);
                                labels.push_back(point.index);
                            }
                            counter++;
                        }
                        if (sample_gap >= tmp_records[i][j].size())
                        {
                            Point last_point = tmp_records[i][j][tmp_records[i][j].size() - 1];
                            locations.push_back(last_point.ml_normalized_curve_val);
                            labels.push_back(last_point.index);
                        }
                        auto finish_sp = chrono::high_resolution_clock::now();
                        exp_recorder.extra_time += chrono::duration_cast<chrono::nanoseconds>(finish_sp - start_sp).count();
                        auto start_train = chrono::high_resolution_clock::now();
                        net->train_model(locations, labels);
                        auto end_train = chrono::high_resolution_clock::now();
                        exp_recorder.training_cost += chrono::duration_cast<chrono::nanoseconds>(end_train - start_train).count();
                    }
                    else if (exp_recorder.is_rs)
                    {
                        exp_recorder.rs_num++;
                        auto start_rs = chrono::high_resolution_clock::now();
                        // cout << "tmp_records[i][j].size(): " << tmp_records[i][j].size() << endl;
                        // vector<Point> tmp_records_i_j = pre_train_zm::get_rep_set(sqrt(m), bit_num, 0, tmp_records[i][j]);
                        int sub_threshold = exp_recorder.rs_threshold_m;
                        if (tmp_records[i][j].size() < Constants::THRESHOLD)
                        {
                            sub_threshold = sqrt(exp_recorder.rs_threshold_m);
                        }
                        vector<Point> tmp_records_i_j = pre_train_zm::get_rep_set_space(sub_threshold, 0, 0, 0.5, 0.5, tmp_records[i][j]);
                        int temp_N = tmp_records_i_j.size();
                        auto start_sort = chrono::high_resolution_clock::now();
                        sort(tmp_records_i_j.begin(), tmp_records_i_j.end(), sort_key());
                        auto end_sort = chrono::high_resolution_clock::now();
                        exp_recorder.ordering_cost = chrono::duration_cast<chrono::nanoseconds>(end_sort - start_sort).count();
                        for (long long i = 0; i < temp_N; i++)
                        {
                            labels.push_back(tmp_records_i_j[i].index);
                            locations.push_back(tmp_records_i_j[i].ml_normalized_curve_val);
                        }
                        auto finish_rs = chrono::high_resolution_clock::now();
                        exp_recorder.extra_time += chrono::duration_cast<chrono::nanoseconds>(finish_rs - start_rs).count();
                        auto start_train = chrono::high_resolution_clock::now();
                        net->train_model(locations, labels);
                        auto end_train = chrono::high_resolution_clock::now();
                        exp_recorder.training_cost += chrono::duration_cast<chrono::nanoseconds>(end_train - start_train).count();
                    }
                    else
                    {
                        exp_recorder.original_num++;
                        for (Point point : tmp_records[i][j])
                        {
                            locations.push_back(point.ml_normalized_curve_val);
                            labels.push_back(point.index);
                        }
                        model_path = to_string(i) + "_" + to_string(j);
                        auto start_train = chrono::high_resolution_clock::now();
                        net->train_model(locations, labels);
                        auto end_train = chrono::high_resolution_clock::now();
                        exp_recorder.training_cost += chrono::duration_cast<chrono::nanoseconds>(end_train - start_train).count();
                    }
                }
                net->get_parameters_ZM();
                int max_error = 0;
                int min_error = 0;
                temp_index.push_back(net);
                for (Point point : tmp_records[i][j])
                {
                    int pos = 0;
                    float pred = net->predict_ZM(point.ml_normalized_curve_val);
                    if (i == stages.size() - 1)
                    {
                        pos = pred * N;
                    }
                    else
                    {
                        pos = pred * stages[i + 1];
                    }
                    if (pos < 0)
                    {
                        pos = 0;
                    }
                    if (pos >= next_stage_length)
                    {
                        pos = next_stage_length - 1;
                    }
                    if (i < stages.size() - 1)
                    {
                        tmp_records[i + 1][pos].push_back(point);
                        if (i == 0)
                        {
                            total_errors += abs((long long)point.index * stages[i + 1] - pos);
                        }
                    }
                    else
                    {
                        long long error = (long long)(point.index * N) - pos;
                        total_errors += abs(error);
                        if (error > 0)
                        {
                            if (error > max_error)
                            {
                                max_error = error;
                            }
                        }
                        else
                        {
                            if (error < min_error)
                            {
                                min_error = error;
                            }
                        }
                    }
                }
                if (i == 0)
                {
                    top_error = total_errors;
                }
                if (i == stages.size() - 1)
                {
                    bottom_error = total_errors;
                    net->max_error = max_error;
                    net->min_error = min_error;
                }
                if (i == stages.size() - 1 && (max_error - min_error) > (zm_max_error - zm_min_error))
                {
                    zm_max_error = max_error;
                    zm_min_error = min_error;
                }
                // cout << net->parameters() << endl;
                // cout << "stage:" << i << " size:" << tmp_records[i][j].size() << endl;
                tmp_records[i][j].clear();
                tmp_records[i][j].shrink_to_fit();
            }
            catch (const std::exception &e)
            {
                std::cerr << e.what() << '\n';
            }
        }
        auto finish_0 = chrono::high_resolution_clock::now();
        if (i == 0)
        {
            exp_recorder.top_level_time = chrono::duration_cast<chrono::nanoseconds>(finish_0 - start_0).count();
        }
        else
        {
            exp_recorder.bottom_level_time = chrono::duration_cast<chrono::nanoseconds>(finish_0 - start_0).count();
        }
        index.push_back(temp_index);
    }
    auto finish = chrono::high_resolution_clock::now();
    exp_recorder.max_error = zm_max_error;
    exp_recorder.min_error = zm_min_error;
    exp_recorder.top_error = top_error;
    exp_recorder.bottom_error = bottom_error;
    exp_recorder.loss = loss;
    exp_recorder.time = chrono::duration_cast<chrono::nanoseconds>(finish - start).count();
    exp_recorder.size = (1 * Constants::HIDDEN_LAYER_WIDTH + 1 * Constants::HIDDEN_LAYER_WIDTH + Constants::HIDDEN_LAYER_WIDTH * 1 + 1) * Constants::EACH_DIM_LENGTH * exp_recorder.non_leaf_node_num + (Constants::DIM * Constants::PAGESIZE * Constants::EACH_DIM_LENGTH + Constants::PAGESIZE * Constants::INFO_LENGTH + Constants::DIM * Constants::DIM * Constants::EACH_DIM_LENGTH) * exp_recorder.leaf_node_num;
    cout << "build time: " << exp_recorder.time << endl;
}

void MLIndex::point_query(ExpRecorder &exp_recorder, Point query_point)
{
    // TODO calculate the key
    auto start_sfc = chrono::high_resolution_clock::now();
    int partition_id = get_partition_id(query_point);
    // cout << "partition_id: " << partition_id << endl;
    double dist = query_point.cal_dist(reference_points[partition_id]);
    query_point.key = offsets[partition_id] + dist;
    // cout << "query_point.key: " << query_point.key << endl;
    query_point.ml_normalized_curve_val = (query_point.key - min_key_val) / gap;
    auto finish_sfc = chrono::high_resolution_clock::now();
    exp_recorder.sfc_cal_time += chrono::duration_cast<chrono::nanoseconds>(finish_sfc - start_sfc).count();

    double key = query_point.key;
    int predicted_index = 0;
    int next_stage_length = 1;
    int min_error = 0;
    int max_error = 0;

    auto start = chrono::high_resolution_clock::now();
    for (int i = 0; i < stages.size(); i++)
    {
        if (i == stages.size() - 1)
        {
            next_stage_length = N;
            min_error = index[i][predicted_index]->min_error;
            max_error = index[i][predicted_index]->max_error;
        }
        else
        {
            next_stage_length = stages[i + 1];
        }
        // cout << "next_stage_length: " << next_stage_length << " i: " << i << endl;
        predicted_index = index[i][predicted_index]->predict_ZM(query_point.ml_normalized_curve_val) * next_stage_length;
        // cout << "predicted_index: " << predicted_index << " i: " << i << endl;
        if (predicted_index < 0)
        {
            predicted_index = 0;
        }
        if (predicted_index >= next_stage_length)
        {
            predicted_index = next_stage_length - 1;
        }
    }
    auto finish = chrono::high_resolution_clock::now();
    exp_recorder.prediction_time += chrono::duration_cast<chrono::nanoseconds>(finish - start).count();
    // cout << "prediction time: " << chrono::duration_cast<chrono::nanoseconds>(finish - start).count() << endl;
    // cout << "final predicted_index: " << predicted_index << endl;

    auto start_1 = chrono::high_resolution_clock::now();
    long front = predicted_index + min_error;
    front = front < 0 ? 0 : front;
    long back = predicted_index + max_error;
    front = front / page_size;
    back = back / page_size;
    back = back >= leafnodes.size() ? leafnodes.size() - 1 : back;
    // cout << "predicted_index: " << predicted_index << " min_error: " << min_error << " max_error: " << max_error << endl;
    // cout << "front: " << front << " back: " << back << endl;
    // if (key)
    // {
    //     /* code */
    // }

    while (front <= back)
    {
        int mid = (front + back) / 2;
        LeafNode *leafnode = leafnodes[mid];
        // cout << "front: " << front << " back: " << back << " key: " << key << endl;
        exp_recorder.page_access += 1;
        if ((*leafnode->children)[0].key <= key && key <= (*leafnode->children)[leafnode->children->size() - 1].key)
        {
            // query_point.print();
            vector<Point>::iterator iter = find(leafnode->children->begin(), leafnode->children->end(), query_point);
            if (iter != leafnode->children->end())
            {
                // cout << "find it" << endl;
                break;
            }
            else
            {
                exp_recorder.point_not_found++;
                break;
            }
        }
        else
        {
            if ((*leafnode->children)[0].key < key)
            {
                front = mid + 1;
            }
            else
            {
                back = mid - 1;
            }
        }
        if (front > back)
        {
            exp_recorder.point_not_found++;
            // cout << "not found!" << endl;
            break;
        }
    }
    if (front <= back)
    {
        exp_recorder.search_length += (back - front + 1);
    }

    if (added_num > 0)
    {
        front = front < 0 ? 0 : front;
        back = inserted_leafnodes.size() - 1;
        while (front <= back)
        {
            int mid = (front + back) / 2;
            LeafNode *leafnode = inserted_leafnodes[mid];
            exp_recorder.page_access += 1;
            if ((*leafnode->children)[0].key <= key && key <= (*leafnode->children)[leafnode->children->size() - 1].key)
            {
                // query_point.print();
                vector<Point>::iterator iter = find(leafnode->children->begin(), leafnode->children->end(), query_point);
                if (iter != leafnode->children->end())
                {
                    break;
                }
                exp_recorder.point_not_found++;
                break;
            }
            else
            {
                if ((*leafnode->children)[0].key < key)
                {
                    front = mid + 1;
                }
                else
                {
                    back = mid - 1;
                }
            }
            if (front > back)
            {
                exp_recorder.point_not_found++;
                break;
            }
        }
    }

    auto finish_1 = chrono::high_resolution_clock::now();
    exp_recorder.search_time += chrono::duration_cast<chrono::nanoseconds>(finish_1 - start_1).count();
    // cout<< "search time: " << chrono::duration_cast<chrono::nanoseconds>(finish_1 - start_1).count() << endl;
}

void MLIndex::point_query(ExpRecorder &exp_recorder, vector<Point> query_points)
{
    cout << "point_query:" << query_points.size() << endl;
    // the_net = index[0][0];
    auto start = chrono::high_resolution_clock::now();
    long res = 0;
    for (long i = 0; i < query_points.size(); i++)
    {
        auto start1 = chrono::high_resolution_clock::now();
        point_query(exp_recorder, query_points[i]);
        auto finish1 = chrono::high_resolution_clock::now();
        res += chrono::duration_cast<chrono::nanoseconds>(finish1 - start1).count();
    }
    auto finish = chrono::high_resolution_clock::now();
    exp_recorder.time = res / query_points.size();
    exp_recorder.page_access /= query_points.size();
    exp_recorder.search_time /= query_points.size();
    exp_recorder.prediction_time /= query_points.size();
    exp_recorder.sfc_cal_time /= query_points.size();
    exp_recorder.search_steps /= query_points.size();
    exp_recorder.search_length /= query_points.size();
    cout << "finish point_query time: " << exp_recorder.time << endl;
}

long long MLIndex::get_point_index(ExpRecorder &exp_recorder, double key)
{
    long long predicted_index = 0;
    long long length_of_next_stage = 1;
    int min_error = 0;
    int max_error = 0;
    for (int i = 0; i < stages.size(); i++)
    {
        if (i == stages.size() - 1)
        {
            length_of_next_stage = N;
            min_error = index[i][predicted_index]->min_error;
            max_error = index[i][predicted_index]->max_error;
        }
        else
        {
            length_of_next_stage = stages[i + 1];
        }
        predicted_index = index[i][predicted_index]->predict_ZM(key) * length_of_next_stage;
        if (predicted_index < 0)
        {
            predicted_index = 0;
        }
        if (predicted_index >= length_of_next_stage)
        {
            predicted_index = length_of_next_stage - 1;
        }
    }
    exp_recorder.index_high = predicted_index + max_error;
    exp_recorder.index_high = exp_recorder.index_high > N - 1 ? N - 1 : exp_recorder.index_high;
    exp_recorder.index_low = predicted_index + min_error;
    exp_recorder.index_low = exp_recorder.index_low < 0 ? 0 : exp_recorder.index_low;
    return predicted_index;
}

long long MLIndex::get_point_index(ExpRecorder &exp_recorder, Point &query_point)
{
    int partition_id = get_partition_id(query_point);
    double dist = query_point.cal_dist(reference_points[partition_id]);
    query_point.key = offsets[partition_id] + dist;
    query_point.ml_normalized_curve_val = (query_point.key - min_key_val) / gap;
    double key = query_point.ml_normalized_curve_val;
    return get_point_index(exp_recorder, key);
}

int MLIndex::predict_closest_position(ExpRecorder &exp_recorder, double key, bool is_upper)
{
    get_point_index(exp_recorder, (key - min_key_val) / gap);

    int s = (int)exp_recorder.index_low;
    int t = (int)exp_recorder.index_high;
    int mid = 0;
    LeafNode *leafnode_first = leafnodes[0];
    LeafNode *leafnode_last = leafnodes[leafnodes.size() - 1];

    if (key <= (*leafnode_first->children)[0].key)
    {
        return 0;
    }
    if (key >= (*leafnode_last->children)[leafnode_last->children->size() - 1].key)
    {
        return N - 1;
    }

    while (s <= t)
    {
        mid = ceil((s + t) / 2);
        int leaf_node_index = mid / page_size;
        LeafNode *leafnode = leafnodes[leaf_node_index];
        if ((*leafnode->children)[0].key <= key && key <= (*leafnode->children)[leafnode->children->size() - 1].key)
        {
            int offset = 0;
            for (size_t i = 0; i < leafnode->children->size() - 1; i++)
            {
                if ((*leafnode->children)[i].key <= key && key <= (*leafnode->children)[i + 1].key)
                {
                    if (is_upper)
                    {
                        return leaf_node_index * page_size + i;
                    }
                    else
                    {
                        return leaf_node_index * page_size + i + 1;
                    }
                }
            }
        }
        int offset = mid - page_size * (leaf_node_index);
        Point mid_point = (*leafnode->children)[offset];
        if (key < mid_point.key)
        {
            t = mid - 1;
        }
        else
        {
            s = mid + 1;
        }
    }
    return 0;
}

vector<Point> MLIndex::find_furthest_points(Mbr query_window)
{
    vector<Point> vertexes = query_window.get_corner_points();
    vector<Point> furthest_points;

    for (size_t i = 0; i < k; i++)
    {
        double max_dist = numeric_limits<float>::min();
        Point furthest_point;
        for (Point point : vertexes)
        {
            double temp_dist = point.cal_dist(reference_points[i]);
            if (max_dist < temp_dist)
            {
                max_dist = temp_dist;
                furthest_point = point;
            }
        }
        furthest_points.push_back(furthest_point);
    }
    return furthest_points;
}

vector<Point> MLIndex::find_closet_points(Mbr query_window)
{
    vector<Point> closest_points;
    for (size_t i = 0; i < k; i++)
    {
        Point point;
        Point reference_point = reference_points[i];
        if (reference_point.x < query_window.x1)
        {
            point.x = query_window.x1;
        }
        else if (reference_point.x < query_window.x2)
        {
            point.x = reference_point.x;
        }
        else
        {
            point.x = query_window.x2;
        }

        if (reference_point.y < query_window.y1)
        {
            point.y = query_window.y1;
        }
        else if (reference_point.y < query_window.y2)
        {
            point.y = reference_point.y;
        }
        else
        {
            point.y = query_window.y2;
        }
        closest_points.push_back(point);
    }
    return closest_points;
}

vector<Point> MLIndex::window_query(ExpRecorder &exp_recorder, Mbr query_window)
{
    vector<Point> window_query_results;
    vector<Point> closest_points = find_closet_points(query_window);
    vector<Point> furthest_points = find_furthest_points(query_window);
    for (size_t i = 0; i < k; i++)
    {
        double lower_key = reference_points[i].cal_dist(closest_points[i]) + offsets[i];
        double upper_key = reference_points[i].cal_dist(furthest_points[i]) + offsets[i];
        // if (lower_key >= upper_key)
        // {
        //     cout << "lower_key: " << lower_key << " upper_key: " << upper_key << endl;
        //     cout<< "closest_points: " << closest_points[i].get_self();
        //     cout<< "furthest_points: " << furthest_points[i].get_self();
        //     cout<< "reference_points: " << reference_points[i].get_self();
        //     cout<< "query_window: " << query_window.get_self();
        // }
        auto start = chrono::high_resolution_clock::now();
        int lower_i = predict_closest_position(exp_recorder, lower_key, false);
        int upper_i = predict_closest_position(exp_recorder, upper_key, true);
        lower_i = lower_i < partition_size[i] ? partition_size[i] : lower_i;
        upper_i = upper_i > partition_size[i + 1] ? partition_size[i + 1] : upper_i;
        long front = lower_i / page_size - 1;
        long back = upper_i / page_size + 1;
        front = front < 0 ? 0 : front;
        back = back >= leafnodes.size() ? leafnodes.size() - 1 : back;
        auto finish = chrono::high_resolution_clock::now();
        exp_recorder.prediction_time += chrono::duration_cast<chrono::nanoseconds>(finish - start).count();
        start = chrono::high_resolution_clock::now();
        // cout << "front: " << front << endl;
        // cout << "back: " << back << endl;
        // cout << "lower_i: " << lower_i << endl;
        // cout << "upper_i: " << upper_i << endl;
        for (size_t j = front; j <= back; j++)
        {
            LeafNode *leafnode = leafnodes[j];
            if (leafnode->mbr.interact(query_window))
            {
                exp_recorder.page_access += 1;
                for (Point point : *(leafnode->children))
                {
                    if (query_window.contains(point))
                    {
                        window_query_results.push_back(point);
                    }
                }
            }
        }
        if (back >= front)
        {
            exp_recorder.search_length += (back - front + 1);
        }

        finish = chrono::high_resolution_clock::now();
        exp_recorder.search_time += chrono::duration_cast<chrono::nanoseconds>(finish - start).count();
    }
    if (added_num > 0)
    {
        for (size_t i = 0; i < inserted_leafnodes.size(); i++)
        {
            LeafNode *leafnode = inserted_leafnodes[i];
            if (leafnode->mbr.interact(query_window))
            {
                exp_recorder.page_access += 1;
                for (Point point : *(leafnode->children))
                {
                    if (query_window.contains(point))
                    {
                        window_query_results.push_back(point);
                    }
                }
            }
        }
    }
    return window_query_results;
}

void MLIndex::window_query(ExpRecorder &exp_recorder, vector<Mbr> query_windows)
{
    exp_recorder.is_window = true;
    cout << "MLIndex::window_query" << endl;
    int length = query_windows.size();
    for (int i = 0; i < length; i++)
    {
        auto start = chrono::high_resolution_clock::now();
        vector<Point> window_query_results = window_query(exp_recorder, query_windows[i]);
        auto finish = chrono::high_resolution_clock::now();
        exp_recorder.time += chrono::duration_cast<chrono::nanoseconds>(finish - start).count();
        exp_recorder.window_query_result_size += window_query_results.size();
    }
    cout << "exp_recorder.window_query_result_size: " << exp_recorder.window_query_result_size << endl;
    exp_recorder.time /= query_windows.size();
    exp_recorder.search_time /= query_windows.size();
    exp_recorder.search_length /= query_windows.size();
    exp_recorder.prediction_time /= query_windows.size();
    exp_recorder.page_access = (double)exp_recorder.page_access / query_windows.size();
}

vector<Point> MLIndex::acc_window_query(ExpRecorder &exp_Recorder, Mbr query_Window)
{
    vector<Point> window_Query_Results;

    for (LeafNode *leafnode : leafnodes)
    {
        if (leafnode->mbr.interact(query_Window))
        {
            exp_Recorder.page_access += 1;
            for (Point point : *(leafnode->children))
            {
                if (query_Window.contains(point))
                {
                    window_Query_Results.push_back(point);
                }
            }
        }
    }
    if (added_num > 0)
    {
        for (LeafNode *leafnode : inserted_leafnodes)
        {
            if (leafnode->mbr.interact(query_Window))
            {
                exp_Recorder.page_access += 1;
                for (Point point : *(leafnode->children))
                {
                    if (query_Window.contains(point))
                    {
                        window_Query_Results.push_back(point);
                    }
                }
            }
        }
    }

    // cout<< windowQueryResults.size() <<endl;
    return window_Query_Results;
}

void MLIndex::acc_window_query(ExpRecorder &exp_Recorder, vector<Mbr> query_Windows)
{
    cout << "ML::accWindowQuery" << endl;
    auto start = chrono::high_resolution_clock::now();
    for (int i = 0; i < query_Windows.size(); i++)
    {
        exp_Recorder.acc_window_query_result_size += acc_window_query(exp_Recorder, query_Windows[i]).size();
    }
    auto finish = chrono::high_resolution_clock::now();
    // cout << "end:" << end.tv_nsec << " begin" << begin.tv_nsec << endl;
    exp_Recorder.time = chrono::duration_cast<chrono::nanoseconds>(finish - start).count() / query_Windows.size();
    exp_Recorder.page_access = (double)exp_Recorder.page_access / query_Windows.size();
    cout << "finish ML::accWindowQuery" << endl;
}

void MLIndex::search_inward(int &lnode_index, int lower_bound, double key, vector<Point> &S, int kk, Point query_point)
{
    // std::cout << "begin: " << lnode_index << std::endl;
    while (true)
    {
        // cout << "lnode_index: " << lnode_index << " lower_bound: " << lower_bound << endl;
        if (lnode_index < lower_bound)
        {
            // std::cout << "end:" << lower_bound << std::endl;
            lnode_index = -1;
            return;
        }
        LeafNode *leafnode = leafnodes[lnode_index];
        for (Point point : *(leafnode->children))
        {
            if (S.size() == kk)
            {
                double dist_furthest = 0;
                int dist_furthest_i = 0;
                for (size_t i = 0; i < kk; i++)
                {
                    double temp_dist = S[i].cal_dist(query_point);
                    if (temp_dist > dist_furthest)
                    {
                        dist_furthest = temp_dist;
                        dist_furthest_i = i;
                    }
                }
                if (query_point.cal_dist(point) < dist_furthest)
                {
                    S.erase(S.begin() + dist_furthest_i);
                    S.push_back(point);
                    S[dist_furthest_i] = point;
                }
            }
            else if (S.size() < kk)
            {
                S.push_back(point);
            }
        }
        if ((*leafnode->children)[0].key > key)
        {
            if (lnode_index > lower_bound)
            {
                lnode_index--;
            }
            else
            {
                // std::cout << "end:" << lower_bound << std::endl;
                lnode_index = -1;
                break;
            }
        }
        else
        {
            // std::cout << "end:" << lnode_index << std::endl;
            lnode_index = -1;
            break;
        }
    }
}

void MLIndex::search_outward(int &lnode_index, int upper_bound, double key, vector<Point> &S, int kk, Point query_point)
{
    // std::cout << "begin: " << lnode_index << std::endl;
    while (true)
    {
        // cout << "lnode_index: " << lnode_index << " upper_bound: " << upper_bound << endl;
        if (lnode_index > upper_bound)
        {
            // std::cout << "end:" << upper_bound << std::endl;
            lnode_index = -1;
            return;
        }
        if (lnode_index >= leafnodes.size())
        {
            return;
        }

        LeafNode *leafnode = leafnodes[lnode_index];
        for (Point point : *(leafnode->children))
        {
            if (S.size() == kk)
            {
                double dist_furthest = 0;
                int dist_furthest_i = 0;
                for (size_t i = 0; i < kk; i++)
                {
                    double temp_dist = S[i].cal_dist(query_point);
                    if (temp_dist > dist_furthest)
                    {
                        dist_furthest = temp_dist;
                        dist_furthest_i = i;
                    }
                }
                if (query_point.cal_dist(point) < dist_furthest)
                {
                    S[dist_furthest_i] = point;
                }
            }
            else if (S.size() < kk)
            {
                S.push_back(point);
            }
        }
        if ((*leafnode->children)[leafnode->children->size() - 1].key < key)
        {
            if (lnode_index < upper_bound)
            {
                lnode_index++;
            }
            else
            {
                // std::cout << "end: " << upper_bound << std::endl;
                lnode_index = -1;
                break;
            }
        }
        else
        {
            // std::cout << "end: " << lnode_index << std::endl;
            lnode_index = -1;
            break;
        }
    }
}

vector<Point> MLIndex::kNN_query(ExpRecorder &exp_recorder, Point query_point, int kk)
{
    vector<int> lp(k, -1); // stores the index of node
    vector<int> rp(k, -1);
    vector<bool> oflag(k, false);
    double delta_r = sqrt((float)kk / N);
    double r = delta_r;
    vector<Point> S;
    while (true)
    {
        if (S.size() == kk)
        {
            double dist_furthest = 0;
            int dist_furthest_i = 0;
            for (size_t i = 0; i < kk; i++)
            {
                double temp_dist = S[i].cal_dist(query_point);
                if (temp_dist > dist_furthest)
                {
                    dist_furthest = temp_dist;
                    dist_furthest_i = i;
                }
            }
            if (dist_furthest < r)
            {
                break;
            }
        }
        for (size_t i = 0; i < k; i++)
        {
            // cout << "[i]: " << i << endl;
            double dis = reference_points[i].cal_dist(query_point);
            // cout << "dis[i]: " << dis << endl;
            if (oflag[i] == false)
            {
                if (offsets[i + 1] - offsets[i] >= dis) // shpere contains q
                {
                    auto start = chrono::high_resolution_clock::now();
                    oflag[i] = true;
                    int lnode_index = ceil(predict_closest_position(exp_recorder, dis + offsets[i], false) / page_size);
                    int upper_bound = ceil(partition_size[i + 1] / page_size);
                    // upper_bound = upper_bound < leafnodes.size() - 1 ? upper_bound + 1 : upper_bound;
                    // upper_bound = upper_bound > leafnodes.size() - 1 ? leafnodes.size() - 1 : upper_bound;
                    int lower_bound = floor(partition_size[i] / page_size);
                    // lower_bound = lower_bound > 0 ? lower_bound - 1 : lower_bound;

                    lnode_index = lnode_index < lower_bound ? lower_bound : lnode_index;
                    lnode_index = lnode_index > upper_bound ? upper_bound : lnode_index;

                    lp[i] = lnode_index;
                    rp[i] = lnode_index;
                    double key = dis + offsets[i] - r;
                    auto finish = chrono::high_resolution_clock::now();
                    exp_recorder.prediction_time += chrono::duration_cast<chrono::nanoseconds>(finish - start).count();

                    start = chrono::high_resolution_clock::now();
                    // cout << "key 1: " << key << endl;
                    search_inward(lp[i], lower_bound, key, S, kk, query_point);
                    // cout << "key 1: " << key << endl;
                    key = dis + offsets[i] + r;
                    // cout << "key 2: " << key << endl;
                    // cout << "upper_bound: " << upper_bound << endl;
                    search_outward(rp[i], upper_bound, key, S, kk, query_point);
                    finish = chrono::high_resolution_clock::now();
                    // cout << "key 2: " << key << endl;
                    exp_recorder.search_time += chrono::duration_cast<chrono::nanoseconds>(finish - start).count();
                }
                else if (offsets[i + 1] - offsets[i] + r >= dis)
                {
                    auto start = chrono::high_resolution_clock::now();
                    oflag[i] = true;
                    int lnode_index = predict_closest_position(exp_recorder, offsets[i + 1], false) / page_size;
                    int lower_bound = partition_size[i] / page_size;
                    lower_bound = lower_bound > 0 ? lower_bound - 1 : lower_bound;
                    if (lnode_index < lower_bound)
                    {
                        lnode_index = lower_bound;
                    }
                    lp[i] = lnode_index;
                    double key = dis + offsets[i] - r;
                    // cout << "key 3: " << key << endl;
                    auto finish = chrono::high_resolution_clock::now();
                    exp_recorder.prediction_time += chrono::duration_cast<chrono::nanoseconds>(finish - start).count();
                    start = chrono::high_resolution_clock::now();
                    search_inward(lp[i], lower_bound, key, S, kk, query_point);
                    finish = chrono::high_resolution_clock::now();
                    // cout << "key 3: " << key << endl;
                    exp_recorder.search_time += chrono::duration_cast<chrono::nanoseconds>(finish - start).count();
                }
            }
            else
            {
                if (lp[i] != -1)
                {
                    double key = dis + offsets[i] - r;
                    int lower_bound = partition_size[i] / page_size;
                    lower_bound = lower_bound > 0 ? lower_bound - 1 : lower_bound;
                    // cout << "key 4: " << key << endl;
                    auto start = chrono::high_resolution_clock::now();
                    search_inward(lp[i], lower_bound, key, S, kk, query_point);
                    // cout << "key 4: " << key << endl;
                    auto finish = chrono::high_resolution_clock::now();
                    exp_recorder.search_time += chrono::duration_cast<chrono::nanoseconds>(finish - start).count();
                }
                if (rp[i] != -1)
                {
                    double key = dis + offsets[i + 1] + r;
                    int upper_bound = partition_size[i + 1] / page_size;
                    upper_bound = upper_bound < leafnodes.size() - 1 ? upper_bound + 1 : upper_bound;
                    upper_bound = upper_bound > leafnodes.size() - 1 ? leafnodes.size() - 1 : upper_bound;
                    // cout << "key 5: " << key << endl;
                    auto start = chrono::high_resolution_clock::now();
                    search_outward(rp[i], upper_bound, key, S, kk, query_point);
                    auto finish = chrono::high_resolution_clock::now();
                    // cout << "key 5: " << key << endl;
                    exp_recorder.search_time += chrono::duration_cast<chrono::nanoseconds>(finish - start).count();
                }
            }
            // for (size_t j = 0; j < S.size(); j++)
            // {
            //     cout << "dist: " << query_point.cal_dist(S[j]) << " position: " << S[j].get_self();
            // }
        }
        // cout<< "added_num: " << added_num << endl;
        if (added_num > 0)
        {
            double knn_r = S[kk - 1].cal_dist(query_point);
            double query_to_center = query_point.cal_dist(reference_point_of_inserted_point);
            for (LeafNode *leafnode : inserted_leafnodes)
            {
                double first_dist = (*leafnode->children)[0].key - offsets[k];
                double last_dist = (*leafnode->children)[leafnode->children->size() - 1].key - offsets[k];
                if (last_dist < (query_to_center - knn_r))
                {
                    continue;
                }
                if (first_dist > (query_to_center + knn_r))
                {
                    break;
                }
                for (Point point : *(leafnode->children))
                {
                    double dist_furthest = 0;
                    int dist_furthest_i = 0;
                    for (size_t i = 0; i < kk; i++)
                    {
                        double temp_dist = S[i].cal_dist(query_point);
                        if (temp_dist > dist_furthest)
                        {
                            dist_furthest = temp_dist;
                            dist_furthest_i = i;
                        }
                    }
                    if (query_point.cal_dist(point) < dist_furthest)
                    {
                        S.erase(S.begin() + dist_furthest_i);
                        S.push_back(point);
                        S[dist_furthest_i] = point;
                    }
                }
            }
        }
        // r += delta_r;
        exp_recorder.knn_r_enlarged_num++;
        r *= 2;
    }
    return S;
    // vector<Point> result;
    // float knn_query_side = sqrt((float)kk / N);

    // while (true)
    // {
    //     Mbr mbr = Mbr::get_mbr(query_point, knn_query_side);
    //     vector<Point> temp_result = window_query(exp_recorder, mbr);
    //     // cout << "mbr: " << mbr->getSelf() << "size: " << temp_result.size() << endl;
    //     if (temp_result.size() >= kk)
    //     {
    //         sort(temp_result.begin(), temp_result.end(), sort_for_kNN(query_point));
    //         Point last = temp_result[kk - 1];
    //         // cout << " last dist : " << last->calDist(query_point) << " knn_query_side: " << knn_query_side << endl;
    //         if (last.cal_dist(query_point) <= knn_query_side)
    //         {
    //             // TODO get top K from the vector.
    //             auto bn = temp_result.begin();
    //             auto en = temp_result.begin() + kk;
    //             vector<Point> vec(bn, en);
    //             result = vec;
    //             break;
    //         }
    //     }
    //     knn_query_side = knn_query_side * 2;
    //     // cout << " knn_query_side: " << knn_query_side << endl;
    // }
    // return result;
}

void MLIndex::kNN_query(ExpRecorder &exp_recorder, vector<Point> query_points, int kk)
{
    exp_recorder.is_knn = true;
    cout << "MLIndex::kNN_query" << endl;

    for (int i = 0; i < query_points.size(); i++)
    {
        // cout << "query point:" << query_points[i].get_self();
        auto start = chrono::high_resolution_clock::now();
        vector<Point> knn_result = kNN_query(exp_recorder, query_points[i], kk);
        auto finish = chrono::high_resolution_clock::now();
        exp_recorder.time += chrono::duration_cast<chrono::nanoseconds>(finish - start).count();
        exp_recorder.knn_query_results.insert(exp_recorder.knn_query_results.end(), knn_result.begin(), knn_result.end());
        // for (size_t j = 0; j < knn_result.size(); j++)
        // {
        //     cout << "dist: " << query_points[i].cal_dist(knn_result[j]) << " position: " << knn_result[j].get_self();
        // }
        // break;
    }
    exp_recorder.knn_r_enlarged_num /= query_points.size();
    exp_recorder.time /= query_points.size();
    exp_recorder.search_length /= query_points.size();
    exp_recorder.search_time /= query_points.size();
    exp_recorder.prediction_time /= query_points.size();
    exp_recorder.page_access = (double)exp_recorder.page_access / query_points.size();
}

void MLIndex::acc_kNN_query(ExpRecorder &exp_recorder, vector<Point> query_points, int kk)
{
    cout << "MLIndex::accKNNQuery" << endl;
    for (int i = 0; i < query_points.size(); i++)
    {
        auto start = chrono::high_resolution_clock::now();
        vector<Point> knn_result = acc_kNN_query(exp_recorder, query_points[i], kk);
        auto finish = chrono::high_resolution_clock::now();
        exp_recorder.time += chrono::duration_cast<chrono::nanoseconds>(finish - start).count();
        exp_recorder.acc_knn_query_results.insert(exp_recorder.acc_knn_query_results.end(), knn_result.begin(), knn_result.end());
        // for (size_t j = 0; j < knn_result.size(); j++)
        // {
        //     cout << "dist: " << query_points[i].cal_dist(knn_result[j]) << " position: " << knn_result[j].get_self();
        // }
        // break;
    }
    exp_recorder.time /= query_points.size();
    exp_recorder.page_access = (double)exp_recorder.page_access / query_points.size();
}

vector<Point> MLIndex::acc_kNN_query(ExpRecorder &exp_Recorder, Point query_Point, int k)
{
    vector<Point> result;
    float knnquery_Side = sqrt((float)k / N);
    while (true)
    {
        Mbr mbr = Mbr::get_mbr(query_Point, knnquery_Side);
        vector<Point> tempResult = acc_window_query(exp_Recorder, mbr);
        if (tempResult.size() >= k)
        {
            sort(tempResult.begin(), tempResult.end(), sort_for_kNN(query_Point));
            Point last = tempResult[k - 1];
            if (last.cal_dist(query_Point) <= knnquery_Side)
            {
                // TODO get top K from the vector.
                auto bn = tempResult.begin();
                auto en = tempResult.begin() + k;
                vector<Point> vec(bn, en);
                result = vec;
                break;
            }
        }
        knnquery_Side = knnquery_Side * 2;
    }
    return result;
}

bool MLIndex::insert(ExpRecorder &exp_recorder, vector<Point> &inserted_points)
{
    cout << "MLIndex::insert" << endl;
    reference_point_of_inserted_point.x = 0.5;
    reference_point_of_inserted_point.y = 0.5;
    // vector<Point> inserted_points = Point::get_inserted_points(exp_recorder.insert_num, exp_recorder.insert_points_distribution);
    auto start = chrono::high_resolution_clock::now();
    for (int i = 0; i < inserted_points.size(); i++)
    {
        insert(exp_recorder, inserted_points[i]);
        // rebuild_index::is_rebuild(exp_recorder, "Z");
    }
    // cout << "is_rebuild: " << rebuild_index::is_rebuild(exp_recorder, "Z") << endl;
    auto finish = chrono::high_resolution_clock::now();
    cout << "added_num: " << added_num << endl;
    exp_recorder.previous_insert_num += inserted_points.size();
    bool is_rebuild = rebuild_index::is_rebuild(exp_recorder, "Z");
    long long previous_time = exp_recorder.insert_time * exp_recorder.previous_insert_num;
    exp_recorder.insert_time = (previous_time + chrono::duration_cast<chrono::nanoseconds>(finish - start).count()) / exp_recorder.previous_insert_num;
    return is_rebuild;
}

void MLIndex::insert(ExpRecorder &exp_recorder, Point point)
{
    inserted_points.push_back(point);
    exp_recorder.update_num++;
    long front = 0;
    long back = inserted_leafnodes.size() - 1;
    double dist = point.cal_dist(reference_point_of_inserted_point);
    double key = offsets[k] + dist;
    point.key = key;
    point.ml_normalized_curve_val = (point.key - min_key_val) * 1.0 / gap;
    exp_recorder.changing_histogram.update(point.ml_normalized_curve_val);
    if (inserted_leafnodes.size() > 0)
    {
        // cout << "--------insert: 1" << endl;
        LeafNode *last_leafnode = inserted_leafnodes[back];
        if ((*last_leafnode->children)[last_leafnode->children->size() - 1].key <= key)
        {
            // cout << "--------insert: 2" << endl;
            if (last_leafnode->is_full())
            {
                LeafNode *leafNode = new LeafNode();
                leafNode->add_point(point);
                inserted_leafnodes.insert(inserted_leafnodes.begin() + inserted_leafnodes.size(), leafNode);
            }
            else
            {
                last_leafnode->add_point(point);
                sort(last_leafnode->children->begin(), last_leafnode->children->end(), sort_key());
            }
            added_num++;
            return;
        }
        // cout << "--------insert: 11" << endl;
        LeafNode *first_leafnode = leafnodes[front];
        if ((*first_leafnode->children)[0].key > key)
        {
            // cout << "--------insert: 3" << endl;
            if (first_leafnode->is_full())
            {
                LeafNode *leafNode = new LeafNode();
                leafNode->add_point(point);
                inserted_leafnodes.insert(inserted_leafnodes.begin(), leafNode);
            }
            else
            {
                first_leafnode->children->insert(first_leafnode->children->begin(), point);
            }
            added_num++;
            return;
        }
    }
    else
    {
        // cout << "--------insert: 4" << endl;
        LeafNode *leafNode = new LeafNode();
        leafNode->add_point(point);
        added_num++;
        inserted_leafnodes.push_back(leafNode);
        return;
    }

    while (front <= back)
    {
        // cout << "--------insert: 5" << endl;
        int mid = (front + back) / 2;
        LeafNode *leafnode = inserted_leafnodes[mid];
        exp_recorder.page_access += 1;
        if ((*leafnode->children)[0].key <= key && key <= (*leafnode->children)[leafnode->children->size() - 1].key)
        {
            // cout << "--------insert: 6" << endl;
            if (leafnode->is_full())
            {
                leafnode->add_point(point);
                sort(leafnode->children->begin(), leafnode->children->end(), sort_key());
                LeafNode *right = leafnode->split();
                inserted_leafnodes.insert(inserted_leafnodes.begin() + mid + 1, right);
            }
            else
            {
                leafnode->add_point(point);
                sort(leafnode->children->begin(), leafnode->children->end(), sort_key());
            }
            added_num++;
            return;
        }
        else
        {
            if ((*leafnode->children)[0].key > key)
            {
                if (mid == 0)
                {
                    LeafNode *leafNode = new LeafNode();
                    leafNode->add_point(point);
                    inserted_leafnodes.insert(inserted_leafnodes.begin(), leafNode);
                    added_num++;
                    return;
                }
                else
                {
                    LeafNode *leafnode = inserted_leafnodes[mid - 1];
                    if ((*leafnode->children)[0].key <= key)
                    {
                        if (leafnode->is_full())
                        {
                            leafnode->add_point(point);
                            sort(leafnode->children->begin(), leafnode->children->end(), sort_key());
                            LeafNode *right = leafnode->split();
                            inserted_leafnodes.insert(inserted_leafnodes.begin() + mid - 1, right);
                        }
                        else
                        {
                            leafnode->add_point(point);
                            sort(leafnode->children->begin(), leafnode->children->end(), sort_key());
                        }
                        added_num++;
                        return;
                    }
                    back = mid - 1;
                }
            }
            if ((*leafnode->children)[leafnode->children->size() - 1].key < key)
            {
                if (mid == inserted_leafnodes.size() - 1)
                {
                    LeafNode *leafNode = new LeafNode();
                    leafNode->add_point(point);
                    inserted_leafnodes.insert(inserted_leafnodes.begin() + inserted_leafnodes.size(), leafNode);
                    added_num++;
                    return;
                }
                else
                {
                    LeafNode *leafnode = inserted_leafnodes[mid + 1];
                    if (key <= (*leafnode->children)[leafnode->children->size() - 1].key)
                    {
                        if (leafnode->is_full())
                        {
                            leafnode->add_point(point);
                            sort(leafnode->children->begin(), leafnode->children->end(), sort_key());
                            LeafNode *right = leafnode->split();
                            inserted_leafnodes.insert(inserted_leafnodes.begin() + mid + 1, right);
                        }
                        else
                        {
                            leafnode->add_point(point);
                            sort(leafnode->children->begin(), leafnode->children->end(), sort_key());
                        }
                        added_num++;
                        return;
                    }
                    front = mid + 1;
                }
            }
        }
    }
    // cout << "front: " << front << " back: " << back << endl;
}

// TODO change knn + window query search range after insertion and calculate the number of inserted points!!!
// void MLIndex::insert(ExpRecorder &exp_recorder, Point point)
// {
//     int partition_id = get_partition_id(point);
//     exp_recorder.update_num++;
//     double dist = point.cal_dist(reference_points[partition_id]);
//     point.key = offsets[partition_id] + dist;
//     // TODO update partition size
//     for (size_t i = partition_id + 1; i < partition_size.size(); i++)
//     {
//         partition_size[i]++;
//     }
//     // TODO update offset ?? cannot update offset, it causes rebuild the whole strucutre
//     point.ml_normalized_curve_val = (point.key - min_key_val) * 1.0 / gap;
//     float key = point.ml_normalized_curve_val;
//     exp_recorder.changing_histogram.update(key);
//     long long predicted_index = 0;
//     long long length_next_stage = 1;
//     int min_error = 0;
//     int max_error = 0;
//     std::shared_ptr<Net> *net;
//     int last_model_index = 0;
//     for (int i = 0; i < stages.size(); i++)
//     {
//         if (i == stages.size() - 1)
//         {
//             length_next_stage = N;
//             last_model_index = predicted_index;
//             min_error = index[i][predicted_index]->min_error;
//             max_error = index[i][predicted_index]->max_error;
//         }
//         else
//         {
//             length_next_stage = stages[i + 1];
//         }
//         predicted_index = index[i][predicted_index]->predict_ZM(key) * length_next_stage;
//         net = &index[i][predicted_index];
//         // predictedIndex = net->forward(torch::tensor({key})).item().toFloat() * lengthOfNextStage;
//         if (predicted_index < 0)
//         {
//             predicted_index = 0;
//         }
//         if (predicted_index >= length_next_stage)
//         {
//             predicted_index = length_next_stage - 1;
//         }
//     }
//     exp_recorder.index_high = predicted_index + max_error;
//     exp_recorder.index_low = predicted_index + min_error;

//     long front = predicted_index + min_error - error_shift;
//     front = front < 0 ? 0 : front;
//     long back = predicted_index + max_error + error_shift;
//     back = back >= N ? N - 1 : back;
//     front = 0;
//     back = N - 1;
//     front = front / page_size;
//     back = back / page_size;
//     N++;
//     LeafNode *last_leafnode = leafnodes[leafnodes.size() - 1];
//     if ((*last_leafnode->children)[last_leafnode->children->size() - 1].key < key)
//     {
//         if (last_leafnode->is_full())
//         {
//             LeafNode *leafNode = new LeafNode();
//             leafNode->add_point(point);
//             leafnodes.insert(leafnodes.begin() + leafnodes.size(), leafNode);
//             error_shift += page_size;
//         }
//         else
//         {
//             last_leafnode->add_point(point);
//         }

//         added_num++;
//         // max_error = max_error > (N - predicted_index) ? max_error : (N - predicted_index);
//         // if (error_shift < (N - predicted_index - max_error))
//         // {
//         //     error_shift = N - predicted_index - max_error;
//         // }
//         return;
//     }
//     LeafNode *first_leafnode = leafnodes[0];
//     if ((*first_leafnode->children)[0].key > key)
//     {
//         if (first_leafnode->is_full())
//         {
//             LeafNode *leafNode = new LeafNode();
//             leafNode->add_point(point);
//             leafnodes.insert(leafnodes.begin(), leafNode);
//             error_shift += page_size;
//         }
//         else
//         {
//             first_leafnode->children->insert(first_leafnode->children->begin(), point);
//         }

//         added_num++;
//         // min_error = min_error < -predicted_index ? min_error : -predicted_index;
//         // if (predicted_index + min_error > error_shift)
//         // {
//         //     error_shift = predicted_index + min_error;
//         // }
//         return;
//     }

//     while (front <= back)
//     {
//         int mid = (front + back) / 2;
//         LeafNode *leafnode = leafnodes[mid];
//         exp_recorder.page_access += 1;
//         if ((*leafnode->children)[0].key <= key && key <= (*leafnode->children)[leafnode->children->size() - 1].key)
//         {
//             if (leafnode->is_full())
//             {
//                 leafnode->add_point(point);
//                 sort(leafnode->children->begin(), leafnode->children->end(), sort_key());
//                 LeafNode *right = leafnode->split();
//                 leafnodes.insert(leafnodes.begin() + mid + 1, right);
//                 error_shift += page_size;
//             }
//             else
//             {
//                 leafnode->add_point(point);
//                 sort(leafnode->children->begin(), leafnode->children->end(), sort_key());
//             }
//             added_num++;
//             return;
//         }
//         else
//         {
//             if ((*leafnode->children)[0].key > key)
//             {
//                 if (mid == 0)
//                 {
//                     LeafNode *leafNode_new = new LeafNode();
//                     leafNode_new->add_point(point);
//                     leafnodes.insert(leafnodes.begin(), leafNode_new);
//                     added_num++;
//                     error_shift += page_size;
//                     return;
//                 }
//                 else
//                 {
//                     LeafNode *leafnode_front = leafnodes[mid - 1];
//                     if ((*leafnode_front->children)[0].key <= key)
//                     {
//                         if (leafnode_front->is_full())
//                         {
//                             leafnode_front->add_point(point);
//                             sort(leafnode_front->children->begin(), leafnode_front->children->end(), sort_key());
//                             LeafNode *right = leafnode_front->split();
//                             leafnodes.insert(leafnodes.begin() + mid, right);
//                             error_shift += page_size;
//                         }
//                         else
//                         {
//                             leafnode_front->add_point(point);
//                             sort(leafnode_front->children->begin(), leafnode_front->children->end(), sort_key());
//                         }
//                         added_num++;
//                         return;
//                     }
//                     back = mid - 1;
//                 }
//             }
//             if ((*leafnode->children)[leafnode->children->size() - 1].key < key)
//             {
//                 if (mid == leafnodes.size() - 1)
//                 {
//                     LeafNode *leafNode_new = new LeafNode();
//                     leafNode_new->add_point(point);
//                     leafnodes.insert(leafnodes.begin() + leafnodes.size(), leafNode_new);
//                     error_shift += page_size;
//                     added_num++;
//                     return;
//                 }
//                 else
//                 {
//                     // cout << "--------insert: 9" << endl;
//                     // cout << "--------insert: leafnodes.size(): " << leafnodes.size() << endl;
//                     // cout << "--------insert: mid: " << mid << endl;
//                     LeafNode *leafNode_back = leafnodes[mid + 1];
//                     if (key <= (*leafNode_back->children)[leafNode_back->children->size() - 1].key)
//                     {
//                         // cout << "--------insert: 10" << endl;
//                         if (leafNode_back->is_full())
//                         {
//                             leafNode_back->add_point(point);
//                             sort(leafNode_back->children->begin(), leafNode_back->children->end(), sort_key());
//                             LeafNode *right = leafNode_back->split();
//                             leafnodes.insert(leafnodes.begin() + mid + 2, right);
//                             error_shift += page_size;
//                         }
//                         else
//                         {
//                             leafNode_back->add_point(point);
//                             sort(leafNode_back->children->begin(), leafNode_back->children->end(), sort_key());
//                         }
//                         added_num++;
//                         return;
//                     }
//                     front = mid + 1;
//                 }
//             }
//         }
//     }
// }

void MLIndex::clear(ExpRecorder &exp_recorder)
{
    stages.clear();
    stages.shrink_to_fit();

    index.clear();
    index.shrink_to_fit();

    reference_points.clear();
    reference_points.shrink_to_fit();

    partitions.clear();
    partitions.shrink_to_fit();
    partitions.resize(k);

    offsets.clear();
    offsets.shrink_to_fit();
    offsets.resize(k);

    leafnodes.clear();
    leafnodes.shrink_to_fit();

    inserted_leafnodes.clear();
    inserted_leafnodes.shrink_to_fit();

    inserted_points.clear();
    inserted_points.shrink_to_fit();

    zm_max_error = 0;
    zm_min_error = 0;

    gap = 0;
    min_key_val = 0;
    max_key_val = 0;
    top_error = 0;
    bottom_error = 0;
    loss = 0;
    bit_num = 0;

    error_shift = 0;
    added_num = 0;
}