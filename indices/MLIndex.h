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

    vector<vector<Point>> partitions;

    vector<float> offsets;

    vector<LeafNode *> leafnodes;

    int zm_max_error = 0;
    int zm_min_error = 0;

    float gap;
    float min_key_val;
    float max_key_val;
    long long top_error;
    long long bottom_error;
    float loss;
    int bit_num = 0;

    string model_path;
    string model_path_root;

public:
    MLIndex();
    MLIndex(int k);

    void build(ExpRecorder &exp_recorder, vector<Point> points);
    void get_reference_points(ExpRecorder &exp_recorder);
    int get_partition_id(Point point);
    void point_query(ExpRecorder &exp_recorder, Point query_point);
    void point_query(ExpRecorder &exp_recorder, vector<Point> query_points);

    void insert(ExpRecorder &exp_recorder, Point);
    void insert(ExpRecorder &exp_recorder);

    void remove(ExpRecorder &exp_recorder, Point);
    void remove(ExpRecorder &exp_recorder, vector<Point>);
};

MLIndex::MLIndex()
{
    this->page_size = Constants::PAGESIZE;
    partitions.resize(k);
    offsets.resize(k);
}

MLIndex::MLIndex(int k)
{
    this->page_size = Constants::PAGESIZE;
    this->k = k;
    partitions.resize(k);
    offsets.resize(k);
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
    float minimal_dist = numeric_limits<float>::max();
    for (size_t j = 0; j < k; j++)
    {
        float temp_dist = reference_points[j].cal_dist(point);
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
    cout << "ML_index build: " << endl;
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
        partitions[partition_index].push_back(points[i]);
    }
    // 4 calcualte the offsets
    offsets[0] = 0;
    for (size_t i = 0; i < k - 1; i++)
    {
        float maximal_dist = numeric_limits<float>::min();
        for (Point point : partitions[i])
        {
            float temp_dist = reference_points[i].cal_dist(point);
            if (maximal_dist < temp_dist)
            {
                maximal_dist = temp_dist;
            }
        }
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
    // min_key_val = points[0].normalized_curve_val;
    max_key_val = points[N - 1].key;
    // max_key_val = points[N - 1].normalized_curve_val;
    gap = max_key_val - min_key_val;
    for (size_t i = 0; i < N; i++)
    {
        // cout << "points[i]: " << points[i].key << endl;
        points[i].index = i * 1.0 / N;
        points[i].normalized_curve_val = (points[i].key - min_key_val) / gap;
    }
    // 6 use ZM index structure to train the data
    // ---------------------------ZM---------------------------
    int leaf_node_num = points.size() / page_size;
    // cout << "leafNodeNum:" << leafNodeNum << endl;
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
                        locations_.push_back(point.normalized_curve_val);
                    }
                    Histogram histogram(pow(2, Constants::UNIFIED_Z_BIT_NUM), locations_);
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
                                locations.push_back(point.normalized_curve_val);
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
                                    locations.push_back(point.normalized_curve_val);
                                    labels.push_back(point.index);
                                }
                                counter++;
                            }
                            if (sample_gap >= tmp_records[i][j].size())
                            {
                                Point last_point = tmp_records[i][j][tmp_records[i][j].size() - 1];
                                locations.push_back(last_point.normalized_curve_val);
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
                    //         locations.push_back(point.normalized_curve_val);
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
                            locations.push_back(tmp_records_0_0[i].normalized_curve_val);
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
                        int bit_num = 6;
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
                            locations.push_back(point.normalized_curve_val);
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
                            locations.push_back(point.normalized_curve_val);
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
                            locations.push_back(point.normalized_curve_val);
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
                                locations.push_back(point.normalized_curve_val);
                                labels.push_back(point.index);
                            }
                            counter++;
                        }
                        if (sample_gap >= tmp_records[i][j].size())
                        {
                            Point last_point = tmp_records[i][j][tmp_records[i][j].size() - 1];
                            locations.push_back(last_point.normalized_curve_val);
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
                            locations.push_back(tmp_records_i_j[i].normalized_curve_val);
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
                            locations.push_back(point.normalized_curve_val);
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
                    float pred = net->predict_ZM(point.normalized_curve_val);
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
    query_point.key = offsets[partition_id] + query_point.cal_dist(reference_points[partition_id]);
    query_point.normalized_curve_val = (query_point.key - min_key_val) / gap;
    auto finish_sfc = chrono::high_resolution_clock::now();
    exp_recorder.sfc_cal_time += chrono::duration_cast<chrono::nanoseconds>(finish_sfc - start_sfc).count();

    float key = query_point.normalized_curve_val;
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
        predicted_index = index[i][predicted_index]->predict_ZM(key) * next_stage_length;
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
    auto start_1 = chrono::high_resolution_clock::now();
    long front = predicted_index + min_error;
    front = front < 0 ? 0 : front;
    long back = predicted_index + max_error;
    back = back >= N ? N - 1 : back;
    front = front / page_size;
    back = back / page_size;
    // cout << "predicted_index: " << predicted_index << " min_error: " << min_error << " max_error: " << max_error << endl;
    while (front <= back)
    {
        int mid = (front + back) / 2;
        LeafNode *leafnode = leafnodes[mid];
        // cout << "front: " << front << " back: " << back << " key: " << key << endl;
        exp_recorder.page_access += 1;
        if ((*leafnode->children)[0].normalized_curve_val <= key && key <= (*leafnode->children)[leafnode->children->size() - 1].normalized_curve_val)
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
            }
            // for (size_t i = 0; i < (*leafnode->children).size(); i++)
            // {
            //     (*leafnode->children)[i].print();
            // }
            return;
        }
        else
        {
            if ((*leafnode->children)[0].normalized_curve_val < key)
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
    exp_recorder.time = res / N;
    exp_recorder.page_access /= N;
    exp_recorder.search_time /= N;
    exp_recorder.prediction_time /= N;
    exp_recorder.sfc_cal_time /= N;
    exp_recorder.search_steps /= N;
    cout << "finish point_query time: " << exp_recorder.time << endl;
}

void MLIndex::insert(ExpRecorder &exp_recorder, Point point)
{
    int partition_id = get_partition_id(point);

    point.key = offsets[partition_id] + point.cal_dist(reference_points[partition_id]);
    point.normalized_curve_val = (point.key - min_key_val) * 1.0 / gap;
    float key = point.normalized_curve_val;
    long long predicted_index = 0;
    long long length_next_stage = 1;
    int min_error = 0;
    int max_error = 0;
    std::shared_ptr<Net> *net;
    int last_model_index = 0;
    for (int i = 0; i < stages.size(); i++)
    {
        if (i == stages.size() - 1)
        {
            length_next_stage = N;
            last_model_index = predicted_index;
            min_error = index[i][predicted_index]->min_error;
            max_error = index[i][predicted_index]->max_error;
        }
        else
        {
            length_next_stage = stages[i + 1];
        }
        predicted_index = index[i][predicted_index]->predict_ZM(key) * length_next_stage;

        net = &index[i][predicted_index];
        // predictedIndex = net->forward(torch::tensor({key})).item().toFloat() * lengthOfNextStage;
        if (predicted_index < 0)
        {
            predicted_index = 0;
        }
        if (predicted_index >= length_next_stage)
        {
            predicted_index = length_next_stage - 1;
        }
    }
    exp_recorder.index_high = predicted_index + max_error;
    exp_recorder.index_low = predicted_index + min_error;

    int inserted_index = predicted_index / Constants::PAGESIZE;

    LeafNode *leafnode = leafnodes[inserted_index];

    if (leafnode->is_full())
    {
        leafnode->add_point(point);
        LeafNode *right = leafnode->split();
        leafnodes.insert(leafnodes.begin() + inserted_index + 1, right);
        index[stages.size() - 1][last_model_index]->max_error += 1;
        index[stages.size() - 1][last_model_index]->min_error -= 1;
    }
    else
    {
        leafnode->add_point(point);
    }
}

void MLIndex::insert(ExpRecorder &exp_recorder)
{
    vector<Point> points = Point::get_inserted_points(exp_recorder.insert_num, exp_recorder.insert_points_distribution);
    auto start = chrono::high_resolution_clock::now();
    for (int i = 0; i < points.size(); i++)
    {
        insert(exp_recorder, points[i]);
    }
    auto finish = chrono::high_resolution_clock::now();
    long long previous_time = exp_recorder.insert_time * exp_recorder.previous_insert_num;
    exp_recorder.previous_insert_num += points.size();
    exp_recorder.insert_time = (previous_time + chrono::duration_cast<chrono::nanoseconds>(finish - start).count()) / exp_recorder.previous_insert_num;
}