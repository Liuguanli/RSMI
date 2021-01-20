#include <iostream>
#include <fstream>
#include <iterator>
#include <algorithm>
#include <vector>
#include <string>
#include <boost/algorithm/string.hpp>
#include "utils/FileReader.h"
#include "indices/ZM.h"
#include "indices/RSMI.h"
#include "utils/ExpRecorder.h"
#include "utils/Constants.h"
#include "utils/FileWriter.h"
#include "utils/util.h"
#include "utils/PreTrainZM.h"
#include "utils/PreTrainRSMI.h"
#include "utils/ModelTools.h"
#include "entities/SFC.h"
#include <torch/torch.h>

#include <xmmintrin.h>
#include <stdlib.h>
#include <stdio.h>
#include <getopt.h>

using namespace std;

#ifndef use_gpu
#define use_gpu

int ks[] = {1, 5, 25, 125, 625};
float areas[] = {0.000006, 0.000025, 0.0001, 0.0004, 0.0016};
float ratios[] = {0.25, 0.5, 1, 2, 4};
int Ns[] = {5000, 2500, 500};

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
            }
        }
    }
    return num * 1.0 / pred.size();
}

void exp_RSMI(FileWriter file_writer, ExpRecorder exp_recorder, vector<Point> points, map<string, vector<Mbr>> mbrs_map, vector<Point> query_poitns, vector<Point> insert_points, string model_path)
{
    exp_recorder.clean();
    exp_recorder.structure_name = "RSMI";
    RSMI::model_path_root = model_path;
    RSMI *partition = new RSMI(0, Constants::MAX_WIDTH);
    auto start = chrono::high_resolution_clock::now();
    partition->model_path = model_path;
    partition->build(exp_recorder, points);
    auto finish = chrono::high_resolution_clock::now();
    exp_recorder.time = chrono::duration_cast<chrono::nanoseconds>(finish - start).count();
    cout << "build time: " << exp_recorder.time << endl;
    exp_recorder.size = (2 * Constants::HIDDEN_LAYER_WIDTH + Constants::HIDDEN_LAYER_WIDTH * 1 + Constants::HIDDEN_LAYER_WIDTH * 1 + 1) * Constants::EACH_DIM_LENGTH * exp_recorder.non_leaf_node_num + (Constants::DIM * Constants::PAGESIZE + Constants::PAGESIZE + Constants::DIM * Constants::DIM) * Constants::EACH_DIM_LENGTH * exp_recorder.leaf_node_num;
    file_writer.write_build(exp_recorder);
    exp_recorder.clean();
    partition->point_query(exp_recorder, points);
    // cout << "finish point_query: pageaccess:" << exp_recorder.page_access << endl;
    cout << "finish point_query time: " << exp_recorder.time << endl;
    file_writer.write_point_query(exp_recorder);
    exp_recorder.clean();

    exp_recorder.window_size = areas[2];
    exp_recorder.window_ratio = ratios[2];
    partition->acc_window_query(exp_recorder, mbrs_map[to_string(areas[2]) + to_string(ratios[2])]);
    cout << "RSMI::acc_window_query time: " << exp_recorder.time << endl;
    cout << "RSMI::acc_window_query page_access: " << exp_recorder.page_access << endl;
    file_writer.write_acc_window_query(exp_recorder);
    partition->window_query(exp_recorder, mbrs_map[to_string(areas[2]) + to_string(ratios[2])]);
    exp_recorder.accuracy = ((double)exp_recorder.window_query_result_size) / exp_recorder.acc_window_query_qesult_size;
    cout << "window_query time: " << exp_recorder.time << endl;
    cout << "window_query page_access: " << exp_recorder.page_access << endl;
    cout << "exp_recorder.accuracy: " << exp_recorder.accuracy << endl;
    file_writer.write_window_query(exp_recorder);

    exp_recorder.clean();
    exp_recorder.k_num = ks[2];
    partition->acc_kNN_query(exp_recorder, query_poitns, ks[2]);
    cout << "exp_recorder.time: " << exp_recorder.time << endl;
    cout << "exp_recorder.page_access: " << exp_recorder.page_access << endl;
    file_writer.write_acc_kNN_query(exp_recorder);
    partition->kNN_query(exp_recorder, query_poitns, ks[2]);
    cout << "exp_recorder.time: " << exp_recorder.time << endl;
    cout << "exp_recorder.page_access: " << exp_recorder.page_access << endl;
    exp_recorder.accuracy = knn_diff(exp_recorder.acc_knn_query_results, exp_recorder.knn_query_results);
    cout << "exp_recorder.accuracy: " << exp_recorder.accuracy << endl;
    file_writer.write_kNN_query(exp_recorder);
    exp_recorder.clean();

    partition->insert(exp_recorder, insert_points);
    cout << "exp_recorder.insert_time: " << exp_recorder.insert_time << endl;
    exp_recorder.clean();
    partition->point_query(exp_recorder, points);
    // cout << "finish point_query: pageaccess:" << exp_recorder.page_access << endl;
    cout << "finish point_query time: " << exp_recorder.time << endl;
    exp_recorder.clean();
}

void exp_ZM(FileWriter file_writer, ExpRecorder exp_recorder, vector<Point> points, map<string, vector<Mbr>> mbrs_map, vector<Point> query_poitns, vector<Point> insert_points, string model_path)
{
    std::stringstream stream;
    stream << std::fixed << std::setprecision(1) << Constants::MODEL_REUSE_THRESHOLD;
    string threshold = stream.str();
    pre_train_zm::pre_train_1d_Z(Constants::RESOLUTION, threshold);
    Net::load_pre_trained_model_zm(threshold);
    exp_recorder.clean();
    exp_recorder.structure_name = "ZM";
    string model_root_path = Constants::TORCH_MODELS_ZM + distribution + "_" + to_string(cardinality) + "/";
    file_utils::check_dir(model_root_path);
    cout<< "points.size(): " << points.size() << endl;
    ZM *zm = new ZM(model_root_path);
    zm->threshold = Constants::MODEL_REUSE_THRESHOLD;
    // zm->pre_train();
    zm->build(exp_recorder, points, 1);
    file_writer.write_build(exp_recorder);
    exp_recorder.clean();
    zm->point_query(exp_recorder, points);
    file_writer.write_point_query(exp_recorder);
    exp_recorder.clean();
}

string RSMI::model_path_root = "";
int main(int argc, char **argv)
{

    int c;
    static struct option long_options[] =
        {
            {"cardinality", required_argument, NULL, 'c'},
            {"distribution", required_argument, NULL, 'd'},
            {"skewness", required_argument, NULL, 's'}};

    while (1)
    {
        int opt_index = 0;
        c = getopt_long(argc, argv, "c:d:s:", long_options, &opt_index);

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
        }
    }

    ExpRecorder exp_recorder;
    exp_recorder.dataset_cardinality = cardinality;
    exp_recorder.distribution = distribution;
    exp_recorder.skewness = skewness;
    inserted_num = cardinality / 2;

    // TODO change filename
    string dataset_filename = Constants::DATASETS + exp_recorder.distribution + "_" + to_string(exp_recorder.dataset_cardinality) + "_" + to_string(exp_recorder.skewness) + "_2_.csv";
    cout<< "dataset_filename: " << dataset_filename << endl;
    FileReader filereader(dataset_filename, ",");
    vector<Point> points = filereader.get_points();
    exp_recorder.insert_num = inserted_num;
    vector<Point> query_poitns;
    vector<Point> insert_points;
    //***********************write query data*********************
    // FileWriter query_file_writer(Constants::QUERYPROFILES);
    // query_poitns = Point::get_points(points, query_k_num);
    // query_file_writer.write_points(query_poitns, exp_recorder);
    // insert_points = Point::get_inserted_points(exp_recorder.insert_num);
    // query_file_writer.write_inserted_points(insert_points, exp_recorder);

    // for (size_t i = 0; i < window_length; i++)
    // {
    //     for (size_t j = 0; j < ratio_length; j++)
    //     {
    //         exp_recorder.window_size = areas[i];
    //         exp_recorder.window_ratio = ratios[j];
    //         vector<Mbr> mbrs = Mbr::get_mbrs(points, exp_recorder.window_size, query_window_num, exp_recorder.window_ratio);
    //         query_file_writer.write_mbrs(mbrs, exp_recorder);
    //     }
    // }
    //**************************prepare knn, window query, and insertion data******************
    FileReader knn_reader((Constants::QUERYPROFILES + Constants::KNN + exp_recorder.distribution + "_" + to_string(exp_recorder.dataset_cardinality) + "_" + to_string(exp_recorder.k_num) + ".csv"), ",");
    map<string, vector<Mbr>> mbrs_map;
    FileReader query_filereader;

    query_poitns = query_filereader.get_points((Constants::QUERYPROFILES + Constants::KNN + exp_recorder.distribution + "_" + to_string(exp_recorder.dataset_cardinality) + "_" + to_string(exp_recorder.skewness) + ".csv"), ",");
    insert_points = query_filereader.get_points((Constants::QUERYPROFILES + Constants::UPDATE + exp_recorder.distribution + "_" + to_string(exp_recorder.dataset_cardinality) + "_" + to_string(exp_recorder.skewness) + "_" + to_string(exp_recorder.insert_num) + ".csv"), ",");

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
    string model_root_path = Constants::TORCH_MODELS + distribution + "_" + to_string(cardinality);
    file_utils::check_dir(model_root_path);
    string model_path = model_root_path + "/";
    FileWriter file_writer(Constants::RECORDS);
    //************************* cannot comment above *************************
    // vector<int> sfc = pre_train_zm::test_Approximate_SFC(Constants::DATASETS, exp_recorder.distribution + "_" + to_string(exp_recorder.dataset_cardinality) + "_" + to_string(exp_recorder.skewness) + "_2_.csv");
    // vector<float> result_cdf;
    // vector<float> z_values;
    // pre_train_zm::cnn(128, sfc, 100000000, 100000, z_values, result_cdf);

    // pre_train_zm::write_approximate_SFC(Constants::DATASETS, exp_recorder.distribution + "_" + to_string(exp_recorder.dataset_cardinality) + "_" + to_string(exp_recorder.skewness) + "_2_.csv", 6);
    // 4004850 2 36208972 3763783, 0 0 0 0, 7909128 32457016 3824323 5417581, 0 1 0 1, 0 0 1 0, 0 0 0 0, 0 0 0 1,0 0 0 0, 
    // 965933 3765307 3627 1512124, 0 3 0 0, 0 138012 0 29323, 0 0 0 0, 0 0 0 0, 1 0 4 1, 0 0 0 1, 1 0 4 0
    // pre_train_zm::write_approximate_SFC(Constants::DATASETS, exp_recorder.distribution + "_" + to_string(exp_recorder.dataset_cardinality) + "_" + to_string(exp_recorder.skewness) + "_2_.csv", 22);
    // pre_train_zm::write_approximate_SFC(Constants::DATASETS, exp_recorder.distribution + "_" + to_string(exp_recorder.dataset_cardinality) + "_" + to_string(exp_recorder.skewness) + "_2_.csv", 38);
    // pre_train_zm::write_approximate_SFC(Constants::DATASETS, exp_recorder.distribution + "_" + to_string(exp_recorder.dataset_cardinality) + "_" + to_string(exp_recorder.skewness) + "_2_.csv", 54);
    // pre_train_zm::write_approximate_SFC(Constants::DATASETS, exp_recorder.distribution + "_" + to_string(exp_recorder.dataset_cardinality) + "_" + to_string(exp_recorder.skewness) + "_2_.csv", 8);
    // pre_train_zm::write_approximate_SFC(Constants::DATASETS, exp_recorder.distribution + "_" + to_string(exp_recorder.dataset_cardinality) + "_" + to_string(exp_recorder.skewness) + "_2_.csv", 10);

    // pre_train_zm::test_errors(dataset_filename, 1);
    // pre_train_zm::pre_train_1d_Z(1, 0.1);
    // exp_RSMI(file_writer, exp_recorder, points, mbrs_map, query_poitns, insert_points, model_path);
    exp_ZM(file_writer, exp_recorder, points, mbrs_map, query_poitns, insert_points, model_path);
    // int resolutions[] = {1};
    // int resolutions[] = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192,
    // 16384, 32768, 65536, 131072, 262144, 524288, 1048576, 2097152, 4194304, 8388608, 16777216};
    // for (size_t i = 0; i < sizeof(resolutions) / sizeof(resolutions[0]); i++)
    // {
    //     cout << "----------resolution = " + to_string(resolutions[i]) + "------------" << endl;
    //     pre_train_rsmi::test_errors("OSM_100000000_1_2_.csv", resolutions[i]);
    // }

    // int resolutions[] = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192,
    // 16384, 32768, 65536, 131072, 262144, 524288, 1048576, 2097152, 4194304, 8388608, 16777216};

    // for (size_t i = 0; i < sizeof(resolutions) / sizeof(resolutions[0]); i++)
    // {
    //     cout << "----------resolution = " + to_string(resolutions[i]) + "------------" << endl;
    //     pre_train_zm::test_errors("OSM_100000000_1_2_.csv", resolutions[i]);
    // }
    // Net::load_pre_trained_model();
}

#endif // use_gpu