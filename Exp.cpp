#ifndef use_gpu
#define use_gpu
#include <iostream>
#include <fstream>
#include <iterator>
#include <algorithm>
#include <vector>
#include <string>
#include <boost/algorithm/string.hpp>
#include "utils/FileReader.h"
#include "indices/MLIndex.h"
#include "indices/ZM.h"
#include "indices/LISA.h"
#include "indices/RSMI.h"
#include "indices/HRR.h"
#include "indices/Grid.h"
#include "indices/KDBTree.h"
#include "utils/ExpRecorder.h"
#include "utils/Constants.h"
#include "utils/FileWriter.h"
#include "utils/util.h"
#include "utils/PreTrainZM.h"
#include "utils/PreTrainRSMI.h"
#include "utils/Rebuild.h"
#include "utils/ModelTools.h"
#include "entities/SFC.h"
#include <torch/torch.h>

#include <xmmintrin.h>
#include <stdlib.h>
#include <stdio.h>
#include <getopt.h>
#include "curves/z.H"

using namespace std;

int ks[] = {1, 5, 25, 125, 625};
float areas[] = {0.000006, 0.000025, 0.0001, 0.0004, 0.0016};
float ratios[] = {0.25, 0.5, 1, 2, 4};
int Ns[] = {5000, 2500, 500};

int insertion_num[] = {0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512};

int k_length = sizeof(ks) / sizeof(ks[0]);
int window_length = sizeof(areas) / sizeof(areas[0]);
int ratio_length = sizeof(ratios) / sizeof(ratios[0]);

int n_length = sizeof(Ns) / sizeof(Ns[0]);

int query_window_num = 1000;
int query_k_num = 1000;

long long cardinality = 10000;
long long inserted_num = cardinality / 10;
string distribution = Constants::DEFAULT_DISTRIBUTION;
int inserted_partition = 5;
int skewness = 1;
float lambda = 0;
string name = "";
string insert_distribution = "";
bool is_cost = false;
bool is_rebuildable = false;
bool is_update = false;
int param_index = 0;

double knn_diff(vector<Point> acc, vector<Point> pred)
{
    int num = 0;
    for (Point point : pred)
    {
        for (Point point1 : acc)
        {
            if (point.x == point1.x && point.y == point1.y)
            {
                num++;
                break;
            }
        }
    }
    return num * 1.0 / pred.size();
}

// void exp_binary_search(FileWriter file_writer, ExpRecorder exp_recorder, vector<Point> points)
// {
//     cout << "exp_binary_search: " << endl;
//     std::stringstream stream;
//     stream << std::fixed << std::setprecision(1) << exp_recorder.model_reuse_threshold;
//     string threshold = stream.str();
//     if (exp_recorder.is_model_reuse)
//     {
//         pre_train_zm::pre_train_1d_Z(Constants::RESOLUTION, threshold);
//         Net::load_pre_trained_model_zm(threshold);
//     }
//     exp_recorder.clean();
//     exp_recorder.set_structure_name("BS");
//     ZM *zm = new ZM();
//     exp_recorder.clean();
//     zm->binary_search(exp_recorder, points);
//     file_writer.write_build(exp_recorder);
//     file_writer.write_point_query(exp_recorder);
//     exp_recorder.clean();
// }

void exp_RSMI(FileWriter file_writer, ExpRecorder &exp_recorder, vector<Point> &points, map<string, vector<Mbr>> mbrs_map, vector<Point> query_points, string model_path)
{
    cout << "exp_RSMI: " << endl;
    std::stringstream stream;
    stream << std::fixed << std::setprecision(1) << exp_recorder.model_reuse_threshold;
    string threshold = stream.str();

    auto start = chrono::high_resolution_clock::now();
    pre_train_rsmi::pre_train_2d_H(Constants::RESOLUTION, threshold);
    pre_train_rsmi::pre_train_2d_Z(Constants::RESOLUTION, threshold);
    Net::load_pre_trained_model_rsmi(threshold);
    auto finish = chrono::high_resolution_clock::now();
    cout << "load time: " << chrono::duration_cast<chrono::nanoseconds>(finish - start).count() << endl;
    exp_recorder.clean();
    exp_recorder.set_structure_name("RSMI");
    RSMI::model_path_root = model_path;
    RSMI *partition = new RSMI(0, Constants::MAX_WIDTH);
    start = chrono::high_resolution_clock::now();
    partition->model_path = model_path;
    partition->build(exp_recorder, points);
    finish = chrono::high_resolution_clock::now();
    exp_recorder.time = chrono::duration_cast<chrono::nanoseconds>(finish - start).count();
    cout << "build time: " << exp_recorder.time << endl;
    exp_recorder.size = (2 * Constants::HIDDEN_LAYER_WIDTH + Constants::HIDDEN_LAYER_WIDTH * 1 + Constants::HIDDEN_LAYER_WIDTH * 1 + 1) * Constants::EACH_DIM_LENGTH * exp_recorder.non_leaf_node_num + (Constants::DIM * Constants::PAGESIZE + Constants::PAGESIZE + Constants::DIM * Constants::DIM) * Constants::EACH_DIM_LENGTH * exp_recorder.leaf_node_num;
    file_writer.write_build(exp_recorder);
    exp_recorder.clean();
    cout << "begin point query: " << endl;
    partition->point_query(exp_recorder, points);
    file_writer.write_point_query(exp_recorder);
    exp_recorder.clean();
    cout << "finish point query: " << endl;
    // cout << "exp_recorder.upper_level_lambda: " << exp_recorder.upper_level_lambda << endl;

    // if (exp_recorder.upper_level_lambda == 0.8)
    // {
    //     // TODO window
    //     for (size_t i = 0; i < 5; i++)
    //     {
    //         exp_recorder.window_size = areas[i];
    //         exp_recorder.window_ratio = ratios[2];
    //         partition->acc_window_query(exp_recorder, mbrs_map[to_string(areas[i]) + to_string(ratios[2])]);
    //         file_writer.write_acc_window_query(exp_recorder);
    //         partition->window_query(exp_recorder, mbrs_map[to_string(areas[i]) + to_string(ratios[2])]);
    //         exp_recorder.accuracy = ((double)exp_recorder.window_query_result_size) / exp_recorder.acc_window_query_result_size;
    //         file_writer.write_window_query(exp_recorder);
    //         exp_recorder.clean();
    //     }

    //     // TODO knn
    //     for (size_t i = 0; i < 5; i++)
    //     {
    //         exp_recorder.k_num = ks[i];
    //         partition->acc_kNN_query(exp_recorder, query_points, exp_recorder.k_num);
    //         file_writer.write_acc_kNN_query(exp_recorder);
    //         partition->kNN_query(exp_recorder, query_points, exp_recorder.k_num);
    //         exp_recorder.accuracy = knn_diff(exp_recorder.acc_knn_query_results, exp_recorder.knn_query_results);
    //         file_writer.write_kNN_query(exp_recorder);
    //         exp_recorder.clean();
    //     }
    // }
    // else
    // {
    //     exp_recorder.window_size = areas[2];
    //     exp_recorder.window_ratio = ratios[2];
    //     partition->acc_window_query(exp_recorder, mbrs_map[to_string(areas[2]) + to_string(ratios[2])]);
    //     file_writer.write_acc_window_query(exp_recorder);
    //     partition->window_query(exp_recorder, mbrs_map[to_string(areas[2]) + to_string(ratios[2])]);
    //     exp_recorder.accuracy = ((double)exp_recorder.window_query_result_size) / exp_recorder.acc_window_query_result_size;
    //     file_writer.write_window_query(exp_recorder);
    //     exp_recorder.clean();

    //     exp_recorder.k_num = ks[2];
    //     partition->acc_kNN_query(exp_recorder, query_points, exp_recorder.k_num);
    //     file_writer.write_acc_kNN_query(exp_recorder);
    //     partition->kNN_query(exp_recorder, query_points, exp_recorder.k_num);
    //     exp_recorder.accuracy = knn_diff(exp_recorder.acc_knn_query_results, exp_recorder.knn_query_results);
    //     file_writer.write_kNN_query(exp_recorder);
    //     exp_recorder.clean();
    // }

    long inserted_all_size = exp_recorder.inserted_points.size();

    int inserted_time = sizeof(insertion_num) / sizeof(insertion_num[0]) - 1;

    if (is_update)
    {
        // vector<Point> original_points = points;

        auto insert_bn = exp_recorder.inserted_points.begin();
        auto insert_en = exp_recorder.inserted_points.begin();
        int scale = exp_recorder.dataset_cardinality / 100;
        for (size_t i = 0; i < inserted_time; i++)
        {
            auto bn = exp_recorder.inserted_points.begin() + insertion_num[i] * scale;
            auto en = exp_recorder.inserted_points.begin() + insertion_num[i + 1] * scale;

            vector<Point> vec(bn, en);
            bool is_rebuild = partition->insert(exp_recorder, vec);
            file_writer.write_insert(exp_recorder);
            exp_recorder.clean();

            if (is_rebuildable && is_rebuild)
            {
                insert_en = en;
                points.insert(points.end(), insert_bn, insert_en);
                insert_bn = insert_en;
                cout << "rebuild: " << points.size() << endl;
                partition->clear(exp_recorder);
                start = chrono::high_resolution_clock::now();
                partition->build(exp_recorder, points);
                finish = chrono::high_resolution_clock::now();
                exp_recorder.time = chrono::duration_cast<chrono::nanoseconds>(finish - start).count();
                file_writer.write_build(exp_recorder);
                exp_recorder.clean();
            }

            partition->point_query(exp_recorder, points);
            file_writer.write_insert_point_query(exp_recorder);
            exp_recorder.clean();

            // partition->acc_window_query(exp_recorder, mbrs_map[to_string(areas[2]) + to_string(ratios[2])]);
            // file_writer.write_insert_acc_window_query(exp_recorder);
            // partition->window_query(exp_recorder, mbrs_map[to_string(areas[2]) + to_string(ratios[2])]);

            // exp_recorder.window_size = areas[2];
            // exp_recorder.window_ratio = ratios[2];
            // exp_recorder.accuracy = ((double)exp_recorder.window_query_result_size) / exp_recorder.acc_window_query_result_size;
            // file_writer.write_insert_window_query(exp_recorder);
            // exp_recorder.clean();

            // exp_recorder.k_num = ks[2];
            // partition->acc_kNN_query(exp_recorder, query_points, exp_recorder.k_num);
            // file_writer.write_insert_acc_kNN_query(exp_recorder);

            // partition->kNN_query(exp_recorder, query_points, exp_recorder.k_num);
            // exp_recorder.accuracy = knn_diff(exp_recorder.acc_knn_query_results, exp_recorder.knn_query_results);
            // file_writer.write_insert_kNN_query(exp_recorder);
            // exp_recorder.clean();
           
        }
    }

    // for (size_t i = 0; i < exp_recorder.insert_times; i++)
    // {
    //     cout << " insert" << endl;
    //     partition->insert(exp_recorder, exp_recorder.insert_rebuild_index);
    //     // partition->insert(exp_recorder);
    // file_writer.write_insert(exp_recorder);
    //     cout << " after insert query" << endl;
    //     partition->point_query(exp_recorder, points);
    //     file_writer.write_insert_point_query(exp_recorder);
    // }

    // TODO redesign the insertion part!!!
    // string dataset = Constants::QUERYPROFILES + "update/skewed_128000000_4_12800000_" + to_string(1) + "_.csv";
    // FileReader filereader(dataset, ",");
    // vector<Point> insert_points = filereader.get_points();
    // int insertedTimes = 100;
    // int insertedNum = insert_points.size() / insertedTimes;
    // for (size_t i = 0; i < insertedTimes / 2; i++)
    // {
    //     vector<Point> tempPoints(insert_points.begin() + i * insertedNum, insert_points.begin() + (i + 1) * insertedNum);
    //     partition->insert(exp_recorder, tempPoints);
    //     file_writer.write_insert(exp_recorder);
    //     partition->point_query(exp_recorder, points);
    //     file_writer.write_insert_point_query(exp_recorder);
    // }
    // exp_recorder.clean();
}

void exp_LISA(FileWriter file_writer, ExpRecorder &exp_recorder, vector<Point> &points, map<string, vector<Mbr>> mbrs_map, vector<Point> query_points, string model_path)
{
    cout << "exp_LISA: " << endl;
    std::stringstream stream;
    stream << std::fixed << std::setprecision(1) << exp_recorder.model_reuse_threshold;
    string threshold = stream.str();
    if (exp_recorder.is_cost_model)
    {
        pre_train_zm::pre_train_1d_Z(Constants::RESOLUTION, threshold);
        Net::load_pre_trained_model_zm(threshold);
        // for get distribution
        Net::load_pre_trained_model_rsmi(threshold);
    }
    exp_recorder.clean();
    exp_recorder.set_structure_name("LISA");
    string model_root_path = Constants::TORCH_MODELS_ZM + distribution + "_" + to_string(cardinality) + "/";
    file_utils::check_dir(model_root_path);
    LISA *lisa = new LISA(model_root_path);
    lisa->build(exp_recorder, points);
    file_writer.write_build(exp_recorder);
    exp_recorder.clean();
    // lisa->point_query(exp_recorder, points);
    // file_writer.write_point_query(exp_recorder);
    // exp_recorder.clean();

    // if (exp_recorder.upper_level_lambda == 0.8)
    // {
    // TODO window
    for (size_t i = 0; i < 5; i++)
    {
        // zm->acc_window_query(exp_recorder, mbrs_map[to_string(areas[i]) + to_string(ratios[2])]);
        // file_writer.write_acc_window_query(exp_recorder);
        lisa->window_query(exp_recorder, mbrs_map[to_string(areas[i]) + to_string(ratios[2])]);
        exp_recorder.window_size = areas[i];
        exp_recorder.window_ratio = ratios[2];
        // exp_recorder.accuracy = ((double)exp_recorder.window_query_result_size) / exp_recorder.acc_window_query_result_size;
        file_writer.write_window_query(exp_recorder);
        exp_recorder.clean();
    }

    // // TODO knn
    // for (size_t i = 0; i < 5; i++)
    // {
    //     exp_recorder.k_num = ks[i];
    //     lisa->kNN_query(exp_recorder, query_points, exp_recorder.k_num);
    //     // exp_recorder.accuracy = knn_diff(exp_recorder.acc_knn_query_results, exp_recorder.knn_query_results);
    //     file_writer.write_kNN_query(exp_recorder);
    //     exp_recorder.clean();
    // }

    long inserted_all_size = exp_recorder.inserted_points.size();

    int inserted_time = sizeof(insertion_num) / sizeof(insertion_num[0]) - 1;
    if (is_update)
    {
        vector<Point> original_points = points;
        auto insert_bn = exp_recorder.inserted_points.begin();
        auto insert_en = exp_recorder.inserted_points.begin();
        int scale = exp_recorder.dataset_cardinality / 100;
        for (size_t i = 0; i < inserted_time; i++)
        {
            auto bn = exp_recorder.inserted_points.begin() + insertion_num[i] * scale;
            auto en = exp_recorder.inserted_points.begin() + insertion_num[i + 1] * scale;
            vector<Point> vec(bn, en);
            cout << "insert size: " << vec.size() << endl;
            bool is_rebuild = lisa->insert(exp_recorder, vec);
            file_writer.write_insert(exp_recorder);
            exp_recorder.clean();

            if (is_rebuildable && is_rebuild)
            {
                exp_recorder.is_rebuild = is_rebuild;
                insert_en = en;
                points.insert(points.end(), insert_bn, insert_en);
                insert_bn = insert_en;
                lisa->clear(exp_recorder);
                cout << "rebuild: " << points.size() << endl;
                lisa->build(exp_recorder, points);
                file_writer.write_build(exp_recorder);
                exp_recorder.clean();
            }

            lisa->point_query(exp_recorder, original_points);
            file_writer.write_insert_point_query(exp_recorder);
            exp_recorder.clean();

            // zm->acc_window_query(exp_recorder, mbrs_map[to_string(areas[2]) + to_string(ratios[2])]);
            // file_writer.write_insert_acc_window_query(exp_recorder);
            lisa->window_query(exp_recorder, mbrs_map[to_string(areas[2]) + to_string(ratios[2])]);

            exp_recorder.window_size = areas[2];
            exp_recorder.window_ratio = ratios[2];
            exp_recorder.accuracy = ((double)exp_recorder.window_query_result_size) / exp_recorder.acc_window_query_result_size;
            file_writer.write_insert_window_query(exp_recorder);
            exp_recorder.clean();

            exp_recorder.k_num = ks[2];
            // zm->acc_kNN_query(exp_recorder, query_points, exp_recorder.k_num);
            // file_writer.write_insert_acc_kNN_query(exp_recorder);
            lisa->kNN_query(exp_recorder, query_points, exp_recorder.k_num);
            // exp_recorder.accuracy = knn_diff(exp_recorder.acc_knn_query_results, exp_recorder.knn_query_results);
            file_writer.write_insert_kNN_query(exp_recorder);
            exp_recorder.clean();
        }
    }
}

void exp_ZM(FileWriter file_writer, ExpRecorder &exp_recorder, vector<Point> &points, map<string, vector<Mbr>> mbrs_map, vector<Point> query_points, string model_path)
{
    cout << "exp_ZM: " << endl;
    std::stringstream stream;
    stream << std::fixed << std::setprecision(1) << exp_recorder.model_reuse_threshold;
    string threshold = stream.str();
    if (exp_recorder.is_cost_model)
    {
        pre_train_zm::pre_train_1d_Z(Constants::RESOLUTION, threshold);
        Net::load_pre_trained_model_zm(threshold);
        // for get distribution
        Net::load_pre_trained_model_rsmi(threshold);
    }

    exp_recorder.clean();
    exp_recorder.set_structure_name("ZM");
    string model_root_path = Constants::TORCH_MODELS_ZM + distribution + "_" + to_string(cardinality) + "/";
    file_utils::check_dir(model_root_path);
    ZM *zm = new ZM(model_root_path);
    zm->sampling_rate = exp_recorder.sampling_rate;
    zm->threshold = exp_recorder.model_reuse_threshold;
    // zm->pre_train();
    zm->build(exp_recorder, points, Constants::RESOLUTION);
    file_writer.write_build(exp_recorder);
    exp_recorder.clean();
    zm->point_query(exp_recorder, points);
    file_writer.write_point_query(exp_recorder);
    exp_recorder.clean();

    if (exp_recorder.upper_level_lambda == 0.8)
    {
        // TODO window
        for (size_t i = 0; i < 5; i++)
        {
            zm->acc_window_query(exp_recorder, mbrs_map[to_string(areas[i]) + to_string(ratios[2])]);
            file_writer.write_acc_window_query(exp_recorder);
            zm->window_query(exp_recorder, mbrs_map[to_string(areas[i]) + to_string(ratios[2])]);
            exp_recorder.window_size = areas[i];
            exp_recorder.window_ratio = ratios[2];
            exp_recorder.accuracy = ((double)exp_recorder.window_query_result_size) / exp_recorder.acc_window_query_result_size;
            file_writer.write_window_query(exp_recorder);
            exp_recorder.clean();
        }

        // TODO knn
        for (size_t i = 0; i < 5; i++)
        {
            exp_recorder.k_num = ks[i];

            zm->acc_kNN_query(exp_recorder, query_points, exp_recorder.k_num);
            file_writer.write_acc_kNN_query(exp_recorder);

            zm->kNN_query(exp_recorder, query_points, exp_recorder.k_num);
            exp_recorder.accuracy = knn_diff(exp_recorder.acc_knn_query_results, exp_recorder.knn_query_results);
            file_writer.write_kNN_query(exp_recorder);
            exp_recorder.clean();
        }
    }
    else
    {
        zm->acc_window_query(exp_recorder, mbrs_map[to_string(areas[2]) + to_string(ratios[2])]);
        file_writer.write_acc_window_query(exp_recorder);
        zm->window_query(exp_recorder, mbrs_map[to_string(areas[2]) + to_string(ratios[2])]);
        exp_recorder.window_size = areas[2];
        exp_recorder.window_ratio = ratios[2];
        exp_recorder.accuracy = ((double)exp_recorder.window_query_result_size) / exp_recorder.acc_window_query_result_size;
        file_writer.write_window_query(exp_recorder);
        exp_recorder.clean();

        exp_recorder.k_num = ks[2];

        zm->acc_kNN_query(exp_recorder, query_points, exp_recorder.k_num);
        file_writer.write_acc_kNN_query(exp_recorder);

        zm->kNN_query(exp_recorder, query_points, exp_recorder.k_num);
        exp_recorder.accuracy = knn_diff(exp_recorder.acc_knn_query_results, exp_recorder.knn_query_results);
        file_writer.write_kNN_query(exp_recorder);
        exp_recorder.clean();
    }

    // long inserted_all_size = exp_recorder.inserted_points.size();
    // int inserted_time = 50;
    // int inserted_gap = inserted_all_size / inserted_time;
    // if (is_update)
    // {
    //     for (size_t i = 0; i < inserted_time; i++)
    //     {
    //         auto bn = exp_recorder.inserted_points.begin() + i * inserted_gap;
    //         auto en = exp_recorder.inserted_points.begin() + (i + 1) * inserted_gap;
    //         vector<Point> vec(bn, en);
    //         zm->insert(exp_recorder, vec);
    //         file_writer.write_insert(exp_recorder);
    //         exp_recorder.clean();

    //         zm->point_query(exp_recorder, points);
    //         file_writer.write_insert_point_query(exp_recorder);
    //         exp_recorder.clean();

    //         zm->acc_window_query(exp_recorder, mbrs_map[to_string(areas[2]) + to_string(ratios[2])]);
    //         file_writer.write_insert_acc_window_query(exp_recorder);
    //         zm->window_query(exp_recorder, mbrs_map[to_string(areas[2]) + to_string(ratios[2])]);

    //         exp_recorder.window_size = areas[2];
    //         exp_recorder.window_ratio = ratios[2];
    //         exp_recorder.accuracy = ((double)exp_recorder.window_query_result_size) / exp_recorder.acc_window_query_result_size;
    //         file_writer.write_insert_window_query(exp_recorder);
    //         exp_recorder.clean();

    //         exp_recorder.k_num = ks[2];
    //         zm->acc_kNN_query(exp_recorder, query_points, exp_recorder.k_num);
    //         file_writer.write_insert_acc_kNN_query(exp_recorder);
    //         zm->kNN_query(exp_recorder, query_points, exp_recorder.k_num);
    //         exp_recorder.accuracy = knn_diff(exp_recorder.acc_knn_query_results, exp_recorder.knn_query_results);
    //         file_writer.write_insert_kNN_query(exp_recorder);
    //         exp_recorder.clean();
    //     }
    // }
    long inserted_all_size = exp_recorder.inserted_points.size();

    int inserted_time = sizeof(insertion_num) / sizeof(insertion_num[0]) - 1;

    // int inserted_gap = inserted_all_size / inserted_time;
    if (is_update)
    {
        auto insert_bn = exp_recorder.inserted_points.begin();
        auto insert_en = exp_recorder.inserted_points.begin();
        int scale = exp_recorder.dataset_cardinality / 100;
        for (size_t i = 0; i < inserted_time; i++)
        {
            auto bn = exp_recorder.inserted_points.begin() + insertion_num[i] * scale;
            auto en = exp_recorder.inserted_points.begin() + insertion_num[i + 1] * scale;
            vector<Point> vec(bn, en);
            cout << "insert size: " << vec.size() << endl;
            bool is_rebuild = zm->insert(exp_recorder, vec);
            file_writer.write_insert(exp_recorder);
            exp_recorder.clean();

            if (is_rebuildable && is_rebuild)
            {
                exp_recorder.is_rebuild = is_rebuild;
                insert_en = en;
                points.insert(points.end(), insert_bn, insert_en);
                insert_bn = insert_en;
                zm->clear(exp_recorder);
                cout << "rebuild: " << points.size() << endl;
                zm->build(exp_recorder, points, Constants::RESOLUTION);
                file_writer.write_build(exp_recorder);
                exp_recorder.clean();
            }

            zm->point_query(exp_recorder, points);
            file_writer.write_insert_point_query(exp_recorder);
            exp_recorder.clean();

            zm->acc_window_query(exp_recorder, mbrs_map[to_string(areas[2]) + to_string(ratios[2])]);
            file_writer.write_insert_acc_window_query(exp_recorder);
            zm->window_query(exp_recorder, mbrs_map[to_string(areas[2]) + to_string(ratios[2])]);

            exp_recorder.window_size = areas[2];
            exp_recorder.window_ratio = ratios[2];
            exp_recorder.accuracy = ((double)exp_recorder.window_query_result_size) / exp_recorder.acc_window_query_result_size;
            file_writer.write_insert_window_query(exp_recorder);
            exp_recorder.clean();

            exp_recorder.k_num = ks[2];
            zm->acc_kNN_query(exp_recorder, query_points, exp_recorder.k_num);
            file_writer.write_insert_acc_kNN_query(exp_recorder);
            zm->kNN_query(exp_recorder, query_points, exp_recorder.k_num);
            exp_recorder.accuracy = knn_diff(exp_recorder.acc_knn_query_results, exp_recorder.knn_query_results);
            file_writer.write_insert_kNN_query(exp_recorder);
            exp_recorder.clean();

            // if (is_rebuild)
            // {
            //     exp_recorder.is_rebuild = is_rebuild;
            //     insert_en = en;
            //     points.insert(points.end(), insert_bn, insert_en);
            //     insert_bn = insert_en;
            //     zm->clear(exp_recorder);
            //     cout << "rebuild: " << points.size() << endl;
            //     zm->build(exp_recorder, points, Constants::RESOLUTION);
            //     file_writer.write_build(exp_recorder);
            //     exp_recorder.clean();
            // }
        }
    }
}

void exp_ML_index(FileWriter file_writer, ExpRecorder &exp_recorder, vector<Point> points, map<string, vector<Mbr>> mbrs_map, vector<Point> query_points, string model_path)
{
    cout << "exp_ML_index: " << endl;
    std::stringstream stream;
    stream << std::fixed << std::setprecision(1) << exp_recorder.model_reuse_threshold;
    string threshold = stream.str();
    pre_train_zm::pre_train_1d_Z(Constants::RESOLUTION, threshold);
    Net::load_pre_trained_model_zm(threshold);
    // for get distribution
    Net::load_pre_trained_model_rsmi(threshold);
    exp_recorder.clean();
    exp_recorder.set_structure_name("ML-index");
    string model_root_path = Constants::TORCH_MODELS_ZM + distribution + "_" + to_string(cardinality) + "/";
    file_utils::check_dir(model_root_path);
    MLIndex *mlindex = new MLIndex(exp_recorder.cluster_size);
    mlindex->build(exp_recorder, points);
    file_writer.write_build(exp_recorder);
    exp_recorder.clean();
    mlindex->point_query(exp_recorder, points);
    file_writer.write_point_query(exp_recorder);
    exp_recorder.clean();

    // // TODO window

    // TODO knn
    if (exp_recorder.upper_level_lambda == 0.8)
    {
        for (size_t i = 0; i < 5; i++)
        {
            mlindex->acc_window_query(exp_recorder, mbrs_map[to_string(areas[i]) + to_string(ratios[2])]);
            file_writer.write_acc_window_query(exp_recorder);
            mlindex->window_query(exp_recorder, mbrs_map[to_string(areas[i]) + to_string(ratios[2])]);
            exp_recorder.window_size = areas[i];
            exp_recorder.window_ratio = ratios[2];
            exp_recorder.accuracy = ((double)exp_recorder.window_query_result_size) / exp_recorder.acc_window_query_result_size;
            file_writer.write_window_query(exp_recorder);
            exp_recorder.clean();
        }

        for (size_t i = 0; i < 5; i++)
        {
            exp_recorder.k_num = ks[i];
            mlindex->acc_kNN_query(exp_recorder, query_points, exp_recorder.k_num);
            file_writer.write_acc_kNN_query(exp_recorder);
            mlindex->kNN_query(exp_recorder, query_points, exp_recorder.k_num);
            exp_recorder.accuracy = knn_diff(exp_recorder.acc_knn_query_results, exp_recorder.knn_query_results);
            file_writer.write_kNN_query(exp_recorder);
            exp_recorder.clean();
        }
    }
    else
    {
        // mlindex->acc_window_query(exp_recorder, mbrs_map[to_string(areas[2]) + to_string(ratios[2])]);
        // file_writer.write_acc_window_query(exp_recorder);
        // mlindex->window_query(exp_recorder, mbrs_map[to_string(areas[2]) + to_string(ratios[2])]);
        // exp_recorder.window_size = areas[2];
        // exp_recorder.window_ratio = ratios[2];
        // exp_recorder.accuracy = ((double)exp_recorder.window_query_result_size) / exp_recorder.acc_window_query_result_size;
        // file_writer.write_window_query(exp_recorder);
        // exp_recorder.clean();

        // exp_recorder.k_num = ks[2];
        // mlindex->acc_kNN_query(exp_recorder, query_points, exp_recorder.k_num);
        // file_writer.write_acc_kNN_query(exp_recorder);
        // mlindex->kNN_query(exp_recorder, query_points, exp_recorder.k_num);
        // exp_recorder.accuracy = knn_diff(exp_recorder.acc_knn_query_results, exp_recorder.knn_query_results);
        // file_writer.write_kNN_query(exp_recorder);
        // exp_recorder.clean();
    }

    long inserted_all_size = exp_recorder.inserted_points.size();

    int inserted_time = sizeof(insertion_num) / sizeof(insertion_num[0]) - 1;

    // int inserted_gap = inserted_all_size / inserted_time;
    if (is_update)
    {
        auto insert_bn = exp_recorder.inserted_points.begin();
        auto insert_en = exp_recorder.inserted_points.begin();
        int scale = exp_recorder.dataset_cardinality / 100;
        for (size_t i = 0; i < inserted_time; i++)
        {
            auto bn = exp_recorder.inserted_points.begin() + insertion_num[i] * scale;
            auto en = exp_recorder.inserted_points.begin() + insertion_num[i + 1] * scale;
            vector<Point> vec(bn, en);
            bool is_rebuild = mlindex->insert(exp_recorder, vec);
            file_writer.write_insert(exp_recorder);
            exp_recorder.clean();

            if (is_rebuildable && is_rebuild)
            {
                exp_recorder.is_rebuild = is_rebuild;
                insert_en = en;
                points.insert(points.end(), insert_bn, insert_en);
                insert_bn = insert_en;
                // TODO write points to
                FileWriter db_file_writer(Constants::DATASETS);
                db_file_writer.write_points(points, exp_recorder, "");
                mlindex->clear(exp_recorder);
                cout << "rebuild begin " << endl;
                mlindex->build(exp_recorder, points);
                cout << "rebuild finish " << endl;
                exp_recorder.is_rebuild = is_rebuild;
                file_writer.write_build(exp_recorder);
                exp_recorder.clean();
            }

            exp_recorder.is_rebuild = is_rebuild;
            mlindex->point_query(exp_recorder, points);
            file_writer.write_insert_point_query(exp_recorder);
            exp_recorder.clean();

            exp_recorder.is_rebuild = is_rebuild;
            mlindex->acc_window_query(exp_recorder, mbrs_map[to_string(areas[2]) + to_string(ratios[2])]);
            file_writer.write_insert_acc_window_query(exp_recorder);
            mlindex->window_query(exp_recorder, mbrs_map[to_string(areas[2]) + to_string(ratios[2])]);
            exp_recorder.window_size = areas[2];
            exp_recorder.window_ratio = ratios[2];
            exp_recorder.accuracy = ((double)exp_recorder.window_query_result_size) / exp_recorder.acc_window_query_result_size;
            file_writer.write_insert_window_query(exp_recorder);
            exp_recorder.clean();

            exp_recorder.is_rebuild = is_rebuild;
            exp_recorder.k_num = ks[2];
            mlindex->acc_kNN_query(exp_recorder, query_points, exp_recorder.k_num);
            file_writer.write_insert_acc_kNN_query(exp_recorder);
            mlindex->kNN_query(exp_recorder, query_points, exp_recorder.k_num);
            exp_recorder.accuracy = knn_diff(exp_recorder.acc_knn_query_results, exp_recorder.knn_query_results);
            file_writer.write_insert_kNN_query(exp_recorder);
            exp_recorder.clean();
        }
    }
}

void expHRR(FileWriter file_writer, ExpRecorder exp_recorder, vector<Point> points, map<string, vector<Mbr>> mbrs_map, vector<Point> query_points)
{
    exp_recorder.clean();
    exp_recorder.set_structure_name("HRR");
    cout << "expHRR" << endl;
    // vector<Mbr *> mbrs
    // vector<Point *> points
    HRR *hrr = new HRR(Constants::PAGESIZE);
    hrr->build(exp_recorder, points);
    cout << "build time: " << exp_recorder.time << endl;
    file_writer.write_build(exp_recorder);
    exp_recorder.clean();
    hrr->point_query(exp_recorder, points);
    cout << "point query time: " << exp_recorder.time << endl;
    file_writer.write_point_query(exp_recorder);
    exp_recorder.clean();

    for (size_t i = 0; i < 5; i++)
    {
        hrr->window_query(exp_recorder, mbrs_map[to_string(areas[i]) + to_string(ratios[2])]);
        exp_recorder.window_size = areas[i];
        exp_recorder.window_ratio = ratios[2];
        file_writer.write_window_query(exp_recorder);
        exp_recorder.clean();
    }

    exp_recorder.k_num = 25;
    hrr->kNN_query(exp_recorder, query_points, exp_recorder.k_num);
    file_writer.write_kNN_query(exp_recorder);
    exp_recorder.clean();

    long inserted_all_size = exp_recorder.inserted_points.size();
    int inserted_time = 50;
    int inserted_gap = inserted_all_size / inserted_time;
    if (is_update)
    {
        for (size_t i = 0; i < inserted_time; i++)
        {
            auto bn = exp_recorder.inserted_points.begin() + i * inserted_gap;
            auto en = exp_recorder.inserted_points.begin() + (i + 1) * inserted_gap;
            vector<Point> vec(bn, en);
            hrr->insert(exp_recorder, vec);
            file_writer.write_insert(exp_recorder);
            exp_recorder.clean();

            hrr->point_query(exp_recorder, points);
            file_writer.write_insert_point_query(exp_recorder);
            exp_recorder.clean();

            hrr->window_query(exp_recorder, mbrs_map[to_string(areas[2]) + to_string(ratios[2])]);
            exp_recorder.window_size = areas[2];
            exp_recorder.window_ratio = ratios[2];
            file_writer.write_insert_window_query(exp_recorder);
            exp_recorder.clean();

            exp_recorder.k_num = ks[2];
            hrr->kNN_query(exp_recorder, query_points, exp_recorder.k_num);
            file_writer.write_insert_kNN_query(exp_recorder);
            exp_recorder.clean();
        }
    }
}

void expGrid(FileWriter file_writer, ExpRecorder exp_recorder, vector<Point> points, map<string, vector<Mbr>> mbrs_map, vector<Point> query_points)
{
    exp_recorder.clean();
    exp_recorder.set_structure_name("Grid");
    cout << "expGrid" << endl;
    int side = (int)sqrt(points.size() / Constants::PAGESIZE);
    cout << "points.size(): " << points.size() << endl;
    cout << "side: " << side << endl;
    Grid *grid = new Grid(side, side);
    grid->build(exp_recorder, points);
    cout << "build time: " << exp_recorder.time << endl;
    file_writer.write_build(exp_recorder);
    exp_recorder.clean();
    grid->point_query(exp_recorder, points);
    cout << "point query time: " << exp_recorder.time << endl;
    file_writer.write_point_query(exp_recorder);
    exp_recorder.clean();

    for (size_t i = 0; i < 5; i++)
    {
        grid->window_query(exp_recorder, mbrs_map[to_string(areas[i]) + to_string(ratios[2])]);
        exp_recorder.window_size = areas[i];
        exp_recorder.window_ratio = ratios[2];
        file_writer.write_window_query(exp_recorder);
        exp_recorder.clean();
    }

    exp_recorder.k_num = 25;
    grid->kNN_query(exp_recorder, query_points, exp_recorder.k_num);
    file_writer.write_kNN_query(exp_recorder);
    exp_recorder.clean();

    long inserted_all_size = exp_recorder.inserted_points.size();
    int inserted_time = 50;
    int inserted_gap = inserted_all_size / inserted_time;
    if (is_update)
    {
        for (size_t i = 0; i < inserted_time; i++)
        {
            auto bn = exp_recorder.inserted_points.begin() + i * inserted_gap;
            auto en = exp_recorder.inserted_points.begin() + (i + 1) * inserted_gap;
            vector<Point> vec(bn, en);
            grid->insert(exp_recorder, vec);
            file_writer.write_insert(exp_recorder);
            exp_recorder.clean();

            grid->point_query(exp_recorder, points);
            file_writer.write_insert_point_query(exp_recorder);
            exp_recorder.clean();

            grid->window_query(exp_recorder, mbrs_map[to_string(areas[2]) + to_string(ratios[2])]);
            exp_recorder.window_size = areas[2];
            exp_recorder.window_ratio = ratios[2];
            file_writer.write_insert_window_query(exp_recorder);
            exp_recorder.clean();

            exp_recorder.k_num = ks[2];
            grid->kNN_query(exp_recorder, query_points, exp_recorder.k_num);
            file_writer.write_insert_kNN_query(exp_recorder);
            exp_recorder.clean();
        }
    }
}

void expKDB(FileWriter file_writer, ExpRecorder exp_recorder, vector<Point> points, map<string, vector<Mbr>> mbrs_map, vector<Point> query_points)
{
    exp_recorder.clean();
    exp_recorder.set_structure_name("KDB");
    cout << "expKDB" << endl;
    KDBTree *kdb = new KDBTree(points.size());
    kdb->build(exp_recorder, points);
    cout << "build time: " << exp_recorder.time << endl;
    file_writer.write_build(exp_recorder);
    exp_recorder.clean();
    kdb->point_query(exp_recorder, points);
    cout << "point query time: " << exp_recorder.time << endl;
    file_writer.write_point_query(exp_recorder);
    exp_recorder.clean();

    for (size_t i = 0; i < 5; i++)
    {
        kdb->window_query(exp_recorder, mbrs_map[to_string(areas[i]) + to_string(ratios[2])]);
        exp_recorder.window_size = areas[i];
        exp_recorder.window_ratio = ratios[2];
        file_writer.write_window_query(exp_recorder);
        exp_recorder.clean();
    }

    exp_recorder.k_num = 25;
    kdb->kNN_query(exp_recorder, query_points, exp_recorder.k_num);
    file_writer.write_kNN_query(exp_recorder);
    exp_recorder.clean();

    long inserted_all_size = exp_recorder.inserted_points.size();
    int inserted_time = 50;
    int inserted_gap = inserted_all_size / inserted_time;
    if (is_update)
    {
        for (size_t i = 0; i < inserted_time; i++)
        {
            auto bn = exp_recorder.inserted_points.begin() + i * inserted_gap;
            auto en = exp_recorder.inserted_points.begin() + (i + 1) * inserted_gap;
            vector<Point> vec(bn, en);
            kdb->insert(exp_recorder, vec);
            file_writer.write_insert(exp_recorder);
            exp_recorder.clean();

            kdb->point_query(exp_recorder, points);
            file_writer.write_insert_point_query(exp_recorder);
            exp_recorder.clean();

            kdb->window_query(exp_recorder, mbrs_map[to_string(areas[2]) + to_string(ratios[2])]);
            exp_recorder.window_size = areas[2];
            exp_recorder.window_ratio = ratios[2];
            file_writer.write_insert_window_query(exp_recorder);
            exp_recorder.clean();

            exp_recorder.k_num = ks[2];
            kdb->kNN_query(exp_recorder, query_points, exp_recorder.k_num);
            file_writer.write_insert_kNN_query(exp_recorder);
            exp_recorder.clean();
        }
    }
}

void exp1(FileWriter file_writer, ExpRecorder &exp_recorder, vector<Point> &points, map<string, vector<Mbr>> mbrs_map, vector<Point> query_points, string model_path)
{
    cout << "exp_ZM: " << endl;
    std::stringstream stream;
    stream << std::fixed << std::setprecision(1) << exp_recorder.model_reuse_threshold;
    string threshold = stream.str();

    pre_train_zm::pre_train_1d_Z(Constants::RESOLUTION, threshold);
    Net::load_pre_trained_model_zm(threshold);
    // for get distribution
    Net::load_pre_trained_model_rsmi(threshold);

    exp_recorder.set_structure_name("ZM");
    string model_root_path = Constants::TORCH_MODELS_ZM + distribution + "_" + to_string(cardinality) + "/";
    file_utils::check_dir(model_root_path);
    ZM *zm = new ZM(model_root_path);
    zm->sampling_rate = exp_recorder.sampling_rate;
    zm->threshold = exp_recorder.model_reuse_threshold;
    // zm->pre_train();
    zm->build(exp_recorder, points, Constants::RESOLUTION);
    file_writer.write_build(exp_recorder);
    exp_recorder.clean();
    zm->point_query(exp_recorder, points);
    file_writer.write_point_query(exp_recorder);
    file_writer.write_learned_cdf(exp_recorder, zm->predict_cdf());
    exp_recorder.clean();

    // zm->acc_window_query(exp_recorder, mbrs_map[to_string(areas[2]) + to_string(ratios[2])]);
    // file_writer.write_acc_window_query(exp_recorder);
    // zm->window_query(exp_recorder, mbrs_map[to_string(areas[2]) + to_string(ratios[2])]);
    // exp_recorder.window_size = areas[2];
    // exp_recorder.window_ratio = ratios[2];
    // exp_recorder.accuracy = ((double)exp_recorder.window_query_result_size) / exp_recorder.acc_window_query_result_size;
    // file_writer.write_window_query(exp_recorder);
    // exp_recorder.clean();

    // exp_recorder.k_num = ks[2];

    // zm->acc_kNN_query(exp_recorder, query_points, exp_recorder.k_num);
    // file_writer.write_acc_kNN_query(exp_recorder);

    // zm->kNN_query(exp_recorder, query_points, exp_recorder.k_num);
    // exp_recorder.accuracy = knn_diff(exp_recorder.acc_knn_query_results, exp_recorder.knn_query_results);
    // file_writer.write_kNN_query(exp_recorder);
    // exp_recorder.clean();
}

void parse(int argc, char **argv, ExpRecorder &exp_recorder)
{
    int c;
    static struct option long_options[] =
        {
            {"cardinality", required_argument, NULL, 'c'},
            {"distribution", required_argument, NULL, 'd'},
            {"skewness", required_argument, NULL, 's'},
            {"lambda", required_argument, NULL, 'l'},
            {"name", required_argument, NULL, 'n'},
            {"is_cost", no_argument, NULL, 't'},
            {"update", no_argument, NULL, 'u'},
            {"index", required_argument, NULL, 'i'},

        };

    while (1)
    {
        int opt_index = 0;
        c = getopt_long(argc, argv, "c:d:s:l:n:tu:ri:", long_options, &opt_index);

        if (-1 == c)
        {
            break;
        }
        switch (c)
        {
        case 'c':
            cardinality = atoll(optarg);
            break;
        case 'd':
            distribution = optarg;
            break;
        case 's':
            skewness = atoi(optarg);
            break;
        case 'l':
            lambda = atof(optarg);
            break;
        case 'n':
            name = optarg;
            break;
        case 't':
            is_cost = true;
            break;
        case 'u':
            is_update = true;
            insert_distribution = optarg;
            break;
        case 'r':
            is_rebuildable = true;
            break;
        case 'i':
            param_index = atoi(optarg);
            break;
        }
    }
    exp_recorder.dataset_cardinality = cardinality;
    exp_recorder.distribution = distribution;
    exp_recorder.skewness = skewness;
    // inserted_num = cardinality / 100;
    // inserted_num = 1000;
    // exp_recorder.insert_num = inserted_num;
    // exp_recorder.insert_times = 10;
    exp_recorder.upper_level_lambda = lambda;
    exp_recorder.lower_level_lambda = lambda;
    exp_recorder.insert_points_distribution = insert_distribution;
}

void get_query_points(vector<Point> &points, vector<Point> &query_points, ExpRecorder &exp_recorder)
{

    FileWriter query_file_writer(Constants::QUERYPROFILES);
    FileReader query_filereader;
    query_points = query_filereader.get_points((Constants::QUERYPROFILES + Constants::KNN + exp_recorder.distribution + "_" + to_string(exp_recorder.dataset_cardinality) + "_" + to_string(exp_recorder.skewness) + ".csv"), ",");
    if (query_points.size() == 0)
    {
        query_points = Point::get_points(points, query_k_num);
        query_file_writer.write_points(query_points, exp_recorder, Constants::KNN);
    }
}

map<string, vector<Mbr>> get_query_mbrs(vector<Point> &points, ExpRecorder &exp_recorder, map<string, vector<Mbr>> &mbrs_map)
{
    FileWriter query_file_writer(Constants::QUERYPROFILES);
    FileReader knn_reader((Constants::QUERYPROFILES + Constants::KNN + exp_recorder.distribution + "_" + to_string(exp_recorder.dataset_cardinality) + "_" + to_string(exp_recorder.k_num) + ".csv"), ",");
    FileReader query_filereader;

    for (size_t i = 0; i < window_length; i++)
    {
        for (size_t j = 0; j < ratio_length; j++)
        {
            exp_recorder.window_size = areas[i];
            exp_recorder.window_ratio = ratios[j];
            vector<Mbr> mbrs = query_filereader.get_mbrs((Constants::QUERYPROFILES + Constants::WINDOW + exp_recorder.distribution + "_" + to_string(exp_recorder.dataset_cardinality) + "_" + to_string(exp_recorder.skewness) + "_" + to_string(exp_recorder.window_size) + "_" + to_string(exp_recorder.window_ratio) + ".csv"), ",");
            mbrs_map.insert(pair<string, vector<Mbr>>(to_string(areas[i]) + to_string(ratios[j]), mbrs));
        }
    }

    for (size_t i = 0; i < window_length; i++)
    {
        for (size_t j = 0; j < ratio_length; j++)
        {
            exp_recorder.window_size = areas[i];
            exp_recorder.window_ratio = ratios[j];
            vector<Mbr> mbrs = Mbr::get_mbrs(points, exp_recorder.window_size, query_window_num, exp_recorder.window_ratio);
            query_file_writer.write_mbrs(mbrs, exp_recorder);
        }
    }

    return mbrs_map;
}

void build_rsmi_rebuild_model()
{
    ExpRecorder exp_recorder;
    exp_recorder.distribution = "skewed";
    exp_recorder.skewness = 4;
    exp_recorder.insert_num = 1280000;
    exp_recorder.insert_times = 1;
    exp_recorder.set_level(2)->set_cost_model(true);
    exp_recorder.sampling_rate = 0.0001;
    exp_recorder.rs_threshold_m = 10000;
    FileWriter file_writer(Constants::RECORDS);
    for (size_t j = 9; j <= 10; j++)
    {
        vector<Point> points;
        for (size_t i = 1; i <= j; i++)
        {
            string dataset = Constants::QUERYPROFILES + "update/skewed_128000000_4_12800000_" + to_string(i) + "_.csv";
            FileReader filereader(dataset, ",");
            vector<Point> temp_points = filereader.get_points();
            points.insert(points.end(), temp_points.begin(), temp_points.end());
        }
        exp_recorder.dataset_cardinality = points.size();
        string model_root_path = Constants::TORCH_MODELS + exp_recorder.distribution + "_" + to_string(exp_recorder.dataset_cardinality);
        file_utils::check_dir(model_root_path);
        string model_path = model_root_path + "/";
        vector<Point> query_points;
        map<string, vector<Mbr>> mbrs_map;
        exp_recorder.insert_rebuild_index = j + 1;
        if (j == 10)
        {
            exp_recorder.insert_times = 0;
        }
        exp_RSMI(file_writer, exp_recorder, points, mbrs_map, query_points, model_path);
    }
}

string RSMI::model_path_root = "";
int main(int argc, char **argv)
{
    // TODO traverse
    // pre_train_rsmi::pre_train_cost_model("or");
    // pre_train_rsmi::pre_train_cost_model("rs");
    // pre_train_rsmi::pre_train_cost_model("sp");
    // pre_train_rsmi::pre_train_cost_model("rl");
    // pre_train_rsmi::pre_train_cost_model("cl");
    // pre_train_rsmi::pre_train_cost_model("mr");
    // pre_train_rsmi::evaluate_cost_model(1.0);
    //----------------------------------------------------
    torch::manual_seed(0);
    ExpRecorder exp_recorder;
    parse(argc, argv, exp_recorder);
    //-------------------------------------------------
    FileWriter query_file_writer(Constants::QUERYPROFILES);

    cout << "exp_recorder.get_dataset_name():" << exp_recorder.get_dataset_name() << endl;
    FileReader filereader(exp_recorder.get_dataset_name(), ",");
    vector<Point> points = filereader.get_points();

    // int scale = exp_recorder.dataset_cardinality / 100;
    int inserted_num = exp_recorder.dataset_cardinality * 0.9;
    // TODO get inserted points
    if (is_update)
    {
        auto rng = std::default_random_engine{};
        std::shuffle(std::begin(points), std::end(points), rng);
        if (exp_recorder.insert_points_distribution == "normal")
        {
            vector<Point> inserted_points(points.begin(), points.begin() + inserted_num);
            exp_recorder.inserted_points = inserted_points;
        }
        points.erase(points.begin(), points.begin() + inserted_num);
        exp_recorder.dataset_cardinality = points.size();
        int scale = exp_recorder.dataset_cardinality / 100;
        exp_recorder.insert_num = scale * insertion_num[sizeof(insertion_num) / sizeof(insertion_num[0]) - 1];
        if (exp_recorder.insert_points_distribution == "skew" || exp_recorder.insert_points_distribution == "uniform")
        {
            exp_recorder.inserted_points = Point::get_inserted_points(exp_recorder.insert_num, exp_recorder.insert_points_distribution);
        }
        // // exp_recorder.insert_num = exp_recorder.dataset_cardinality / 2;
        // exp_recorder.inserted_points = Point::get_inserted_points(exp_recorder.insert_num, exp_recorder.insert_points_distribution);
        // // TODO write points!!!!
        // string name = "/home/research/datasets/" + str(exp_recorder.insert_num) + "_" + exp_recorder.insert_points_distribution + ".csv";
        // std::ifstream fin(name);
        // if (!fin)
        // {
        //     query_file_writer.write_inserted_points(exp_recorder, name);
        // }

        cout << "points.size(): " << points.size() << endl;
        cout << "exp_recorder.inserted_points.size(): " << exp_recorder.inserted_points.size() << endl;

        FileWriter db_file_writer(Constants::DATASETS);
        db_file_writer.write_points(points, exp_recorder, "");
    }
    //-------------------------------------------------
    if (is_cost)
    {
        rebuild_index::build_rebuild_model();
    }
    pre_train_zm::cost_model_build();
    pre_train_rsmi::cost_model_build();

    vector<Point> query_points;
    map<string, vector<Mbr>> mbrs_map;

    get_query_points(points, query_points, exp_recorder);
    get_query_mbrs(points, exp_recorder, mbrs_map);

    string model_root_path = Constants::TORCH_MODELS + distribution + "_" + to_string(cardinality);
    file_utils::check_dir(model_root_path);
    string model_path = model_root_path + "/";
    FileWriter file_writer(Constants::RECORDS);
    exp_recorder.set_level(2)->set_cost_model(is_cost);
    exp_recorder.sampling_rate = 0.0001;
    exp_recorder.rs_threshold_m = 10000;
    exp_recorder.cluster_size = 10;
    exp_recorder.bit_num = 6; // RL
    float rhos[] = {0.1, 0.01, 0.001, 0.0001};
    int cluster_num[] = {100, 1000, 10000};
    int rs[] = {128, 1000, 10000};
    int bits[] = {6, 8, 10};
    float model_reuse_thresholds[] = {0.1, 0.3, 0.5};

    //----------------------------------------------------
    // map<string, int> names = {{"hrr", 1}, {"kdb", 2}, {"grid", 3}, {"ml", 4}, {"zm", 5}, {"rsmi", 6}};
    map<string, int> names = {{"hrr", 1}, {"kdb", 2}, {"grid", 3}, {"ml", 4}, {"zm", 5}, {"rsmi", 6}, {"exp7", 7}, {"exp8", 8}, {"exp9", 9}, {"exp10", 10}, {"exp11", 11}, {"exp12", 12}, {"lisa", 13}};
    switch (names[name])
    {
    case 1:
        expHRR(file_writer, exp_recorder, points, mbrs_map, query_points);
        break;
    case 2:
        expKDB(file_writer, exp_recorder, points, mbrs_map, query_points);
        break;
    case 3:
        expGrid(file_writer, exp_recorder, points, mbrs_map, query_points);
        break;
    case 4:
        exp_ML_index(file_writer, exp_recorder, points, mbrs_map, query_points, model_path);
        break;
    case 5:
        exp_ZM(file_writer, exp_recorder, points, mbrs_map, query_points, model_path);
        break;
    case 6:
        exp_RSMI(file_writer, exp_recorder, points, mbrs_map, query_points, model_path);
        break;
    case 7:
        exp_recorder.set_level(1)->set_cost_model(false);
        exp1(file_writer, exp_recorder, points, mbrs_map, query_points, model_path);
        break;
    case 8:
        // for (size_t i = 1; i < 4; i++)
        // {
        exp_recorder.set_level(1)->test_sp();
        exp_recorder.sampling_rate = rhos[param_index];
        exp1(file_writer, exp_recorder, points, mbrs_map, query_points, model_path);
        // }
        break;
    case 9:
        // for (size_t i = 1; i < 3; i++)
        // {
        exp_recorder.set_level(1)->test_cluster();
        exp_recorder.cluster_size = cluster_num[param_index];
        exp1(file_writer, exp_recorder, points, mbrs_map, query_points, model_path);
        // }
        break;
    case 10:
        exp_recorder.set_level(1)->test_model_reuse();
        exp_recorder.model_reuse_threshold = model_reuse_thresholds[param_index];
        exp1(file_writer, exp_recorder, points, mbrs_map, query_points, model_path);
        break;
    case 11:
        // for (size_t i = 1; i < 3; i++)
        // {
        exp_recorder.set_level(1)->test_rs();
        exp_recorder.rs_threshold_m = rs[param_index];
        exp1(file_writer, exp_recorder, points, mbrs_map, query_points, model_path);
        // }
        break;
    case 12:
        // for (size_t i = 1; i < 3; i++)
        // {
        exp_recorder.set_level(1)->test_rl();
        exp_recorder.bit_num = bits[param_index];
        exp1(file_writer, exp_recorder, points, mbrs_map, query_points, model_path);
        // }
        break;
    case 13:
        exp_LISA(file_writer, exp_recorder, points, mbrs_map, query_points, model_path);
        break;
    default:
        break;
    }
    // exp_binary_search(file_writer, exp_recorder, points);

    // exp_ZM(file_writer, exp_recorder, points, mbrs_map, query_points, model_path);
    // exp_ML_index(file_writer, exp_recorder, points, mbrs_map, query_points, model_path);
    // exp_RSMI(file_writer, exp_recorder, points, mbrs_map, query_points, model_path);

    // for (size_t i = 10; i >= 0; i--)
    // {points = i * 0.1;
    //     exp_recorder.lower_level_lambda = i * 0.1;
    //     exp_ZM(file_writer, exp_recorder, points, mbrs_map, query_points, model_path);
    //     exp_ML_index(file_writer, exp_recorder, points, mbrs_map, query_points, model_path);
    //     exp_RSMI(file_writer, exp_recorder, points, mbrs_map, query_points, model_path);
    // }

    // cluster
    // exp_recorder.test_cluster()->set_level(1);
    // cout << "IS_CLUSTER" << endl;
    // exp_recorder.cluster_method = "kmeans";
    // exp_recorder.cluster_size = 10000;
    // exp_ZM(file_writer, exp_recorder, points, mbrs_map, query_points, insert_points, model_path, 1.0, 1);
}

#endif // use_gpu