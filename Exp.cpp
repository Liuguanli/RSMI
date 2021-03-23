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
#include "indices/ZM.h"
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

void exp_binary_search(FileWriter file_writer, ExpRecorder exp_recorder, vector<Point> points)
{
    cout << "exp_binary_search: " << endl;
    std::stringstream stream;
    stream << std::fixed << std::setprecision(1) << exp_recorder.model_reuse_threshold;
    string threshold = stream.str();
    if (exp_recorder.is_model_reuse)
    {
        pre_train_zm::pre_train_1d_Z(Constants::RESOLUTION, threshold);
        Net::load_pre_trained_model_zm(threshold);
    }
    exp_recorder.clean();
    exp_recorder.set_structure_name("BS");
    ZM *zm = new ZM();
    exp_recorder.clean();
    zm->binary_search(exp_recorder, points);
    file_writer.write_build(exp_recorder);
    file_writer.write_point_query(exp_recorder);
    exp_recorder.clean();
}

void exp_RSMI(FileWriter file_writer, ExpRecorder exp_recorder, vector<Point> points, map<string, vector<Mbr>> mbrs_map, vector<Point> query_poitns, vector<Point> insert_points, string model_path, float sampling_rate)
{
    // model reuse OSM
    // load time: 653787332
    // build time: 154s
    // finish point_query time: 425

    // model reuse SA
    // build time: 764s
    // finish point_query time: 395

    // train from scratch
    // build time: 14000s (build)
    // finish point_query time: 472
    cout << "exp_RSMI: " << endl;
    std::stringstream stream;
    stream << std::fixed << std::setprecision(1) << exp_recorder.model_reuse_threshold;
    string threshold = stream.str();
    if (exp_recorder.is_model_reuse)
    {
        auto start = chrono::high_resolution_clock::now();
        pre_train_rsmi::pre_train_2d_H(Constants::RESOLUTION, threshold);
        pre_train_rsmi::pre_train_2d_Z(Constants::RESOLUTION, threshold);
        Net::load_pre_trained_model_rsmi(threshold);
        auto finish = chrono::high_resolution_clock::now();
        cout << "load time: " << chrono::duration_cast<chrono::nanoseconds>(finish - start).count() << endl;
    }
    exp_recorder.clean();
    exp_recorder.set_structure_name("RSMI");
    RSMI::model_path_root = model_path;
    RSMI *partition = new RSMI(0, Constants::MAX_WIDTH);
    auto start = chrono::high_resolution_clock::now();
    partition->model_path = model_path;
    partition->sampling_rate = sampling_rate;
    partition->build(exp_recorder, points);
    auto finish = chrono::high_resolution_clock::now();
    exp_recorder.time = chrono::duration_cast<chrono::nanoseconds>(finish - start).count();
    cout << "build time: " << exp_recorder.time << endl;
    exp_recorder.sampling_rate = sampling_rate;
    exp_recorder.size = (2 * Constants::HIDDEN_LAYER_WIDTH + Constants::HIDDEN_LAYER_WIDTH * 1 + Constants::HIDDEN_LAYER_WIDTH * 1 + 1) * Constants::EACH_DIM_LENGTH * exp_recorder.non_leaf_node_num + (Constants::DIM * Constants::PAGESIZE + Constants::PAGESIZE + Constants::DIM * Constants::DIM) * Constants::EACH_DIM_LENGTH * exp_recorder.leaf_node_num;
    file_writer.write_build(exp_recorder);
    exp_recorder.clean();
    partition->point_query(exp_recorder, points);
    // cout << "finish point_query: pageaccess:" << exp_recorder.page_access << endl;
    cout << "finish point_query time: " << exp_recorder.time << endl;
    file_writer.write_point_query(exp_recorder);
    exp_recorder.clean();

    // exp_recorder.window_size = areas[2];
    // exp_recorder.window_ratio = ratios[2];
    // partition->acc_window_query(exp_recorder, mbrs_map[to_string(areas[2]) + to_string(ratios[2])]);
    // cout << "RSMI::acc_window_query time: " << exp_recorder.time << endl;
    // cout << "RSMI::acc_window_query page_access: " << exp_recorder.page_access << endl;
    // file_writer.write_acc_window_query(exp_recorder);
    // partition->window_query(exp_recorder, mbrs_map[to_string(areas[2]) + to_string(ratios[2])]);
    // exp_recorder.accuracy = ((double)exp_recorder.window_query_result_size) / exp_recorder.acc_window_query_qesult_size;
    // cout << "window_query time: " << exp_recorder.time << endl;
    // cout << "window_query page_access: " << exp_recorder.page_access << endl;
    // cout << "exp_recorder.accuracy: " << exp_recorder.accuracy << endl;
    // file_writer.write_window_query(exp_recorder);

    // exp_recorder.clean();
    // exp_recorder.k_num = ks[2];
    // partition->acc_kNN_query(exp_recorder, query_poitns, ks[2]);
    // cout << "exp_recorder.time: " << exp_recorder.time << endl;
    // cout << "exp_recorder.page_access: " << exp_recorder.page_access << endl;
    // file_writer.write_acc_kNN_query(exp_recorder);
    // partition->kNN_query(exp_recorder, query_poitns, ks[2]);
    // cout << "exp_recorder.time: " << exp_recorder.time << endl;
    // cout << "exp_recorder.page_access: " << exp_recorder.page_access << endl;
    // exp_recorder.accuracy = knn_diff(exp_recorder.acc_knn_query_results, exp_recorder.knn_query_results);
    // cout << "exp_recorder.accuracy: " << exp_recorder.accuracy << endl;
    // file_writer.write_kNN_query(exp_recorder);
    // exp_recorder.clean();

    // partition->insert(exp_recorder, insert_points);
    // cout << "exp_recorder.insert_time: " << exp_recorder.insert_time << endl;
    // exp_recorder.clean();
    // partition->point_query(exp_recorder, points);
    // // cout << "finish point_query: pageaccess:" << exp_recorder.page_access << endl;
    // cout << "finish point_query time: " << exp_recorder.time << endl;
    // exp_recorder.clean();
}

void exp_ZM(FileWriter file_writer, ExpRecorder exp_recorder, vector<Point> points, map<string, vector<Mbr>> mbrs_map, vector<Point> query_poitns, vector<Point> insert_points, string model_path, float sampling_rate, int m)
{
    cout << "exp_ZM: " << endl;
    std::stringstream stream;
    stream << std::fixed << std::setprecision(1) << exp_recorder.model_reuse_threshold;
    string threshold = stream.str();
    if (exp_recorder.is_model_reuse)
    {
        pre_train_zm::pre_train_1d_Z(Constants::RESOLUTION, threshold);
        Net::load_pre_trained_model_zm(threshold);
    }
    exp_recorder.clean();
    exp_recorder.set_structure_name("ZM");
    exp_recorder.representative_threshold_m = m;
    string model_root_path = Constants::TORCH_MODELS_ZM + distribution + "_" + to_string(cardinality) + "/";
    file_utils::check_dir(model_root_path);
    ZM *zm = new ZM(model_root_path);
    zm->sampling_rate = sampling_rate;
    zm->threshold = exp_recorder.model_reuse_threshold;
    // zm->pre_train();
    exp_recorder.sampling_rate = sampling_rate;
    zm->build(exp_recorder, points, Constants::RESOLUTION, m);
    file_writer.write_build(exp_recorder);
    exp_recorder.clean();
    zm->point_query(exp_recorder, points);
    file_writer.write_point_query(exp_recorder);
    exp_recorder.clean();
}

void expHRR(FileWriter file_writer, ExpRecorder exp_recorder, vector<Point> points, map<string, vector<Mbr>> mbrs_map, vector<Point> query_poitns, vector<Point> insert_points)
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
    hrr->pointQuery(exp_recorder, points);
    cout << "point query time: " << exp_recorder.time << endl;
    file_writer.write_point_query(exp_recorder);
    exp_recorder.clean();
}

void expGrid(FileWriter file_writer, ExpRecorder exp_recorder, vector<Point> points, map<string, vector<Mbr>> mbrs_map, vector<Point> query_poitns, vector<Point> insert_points)
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
    grid->pointQuery(exp_recorder, points);
    cout << "point query time: " << exp_recorder.time << endl;
    file_writer.write_point_query(exp_recorder);
    exp_recorder.clean();
}

void expKDB(FileWriter file_writer, ExpRecorder exp_recorder, vector<Point> points, map<string, vector<Mbr>> mbrs_map, vector<Point> query_poitns, vector<Point> insert_points)
{
    exp_recorder.clean();
    exp_recorder.set_structure_name("KDB");
    cout << "expKDB" << endl;
    KDBTree *kdb = new KDBTree(points.size());
    kdb->build(exp_recorder, points);
    cout << "build time: " << exp_recorder.time << endl;
    file_writer.write_build(exp_recorder);
    exp_recorder.clean();
    kdb->pointQuery(exp_recorder, points);
    cout << "point query time: " << exp_recorder.time << endl;
    file_writer.write_point_query(exp_recorder);
    exp_recorder.clean();
}

void parse(int argc, char **argv, ExpRecorder &exp_recorder)
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
    exp_recorder.dataset_cardinality = cardinality;
    exp_recorder.distribution = distribution;
    exp_recorder.skewness = skewness;
    inserted_num = cardinality / 2;
    exp_recorder.insert_num = inserted_num;
}

vector<Point> get_query_points(vector<Point> &points, vector<Point> &query_points, vector<Point> &insert_points, ExpRecorder &exp_recorder)
{
    FileWriter query_file_writer(Constants::QUERYPROFILES);
    FileReader query_filereader;
    query_points = query_filereader.get_points((Constants::QUERYPROFILES + Constants::KNN + exp_recorder.distribution + "_" + to_string(exp_recorder.dataset_cardinality) + "_" + to_string(exp_recorder.skewness) + ".csv"), ",");
    insert_points = query_filereader.get_points((Constants::QUERYPROFILES + Constants::UPDATE + exp_recorder.distribution + "_" + to_string(exp_recorder.dataset_cardinality) + "_" + to_string(exp_recorder.skewness) + "_" + to_string(exp_recorder.insert_num) + ".csv"), ",");
    if (query_points.size() == 0)
    {
        query_points = Point::get_points(points, query_k_num);
        query_file_writer.write_points(query_points, exp_recorder);
    }

    if (insert_points.size() == 0)
    {
        insert_points = Point::get_inserted_points(exp_recorder.insert_num);
        query_file_writer.write_inserted_points(insert_points, exp_recorder);
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
}

map<string, vector<Mbr>> get_query_mbrs(ExpRecorder &exp_recorder, map<string, vector<Mbr>> mbrs_map)
{
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
    return mbrs_map;
}

string RSMI::model_path_root = "";
int main(int argc, char **argv)
{
    ExpRecorder exp_recorder;
    parse(argc, argv, exp_recorder);
    cout << "exp_recorder.get_dataset_name():" << exp_recorder.get_dataset_name() << endl;

    FileReader filereader(exp_recorder.get_dataset_name(), ",");
    vector<Point> points = filereader.get_points();
    vector<Point> query_poitns;
    vector<Point> insert_points;
    map<string, vector<Mbr>> mbrs_map;
    // get_query_points(points, query_poitns, insert_points, exp_recorder);
    // get_query_mbrs(exp_recorder, mbrs_map);
    string model_root_path = Constants::TORCH_MODELS + distribution + "_" + to_string(cardinality);
    file_utils::check_dir(model_root_path);
    string model_path = model_root_path + "/";
    FileWriter file_writer(Constants::RECORDS);

    //************************* cannot comment above *************************
    // vector<int> sfc = pre_train_zm::test_Approximate_SFC(Constants::DATASETS, exp_recorder.distribution + "_" + to_string(exp_recorder.dataset_cardinality) + "_" + to_string(exp_recorder.skewness) + "_2_.csv");
    // vector<float> result_cdf;
    // vector<float> z_values;
    // pre_train_zm::cnn(128, sfc, 100000000, 100000, z_values, result_cdf);
    // auto start = chrono::high_resolution_clock::now();
    // pre_train_zm::train_top_level(points);
    // auto finish = chrono::high_resolution_clock::now();
    // cout << "load time: " << chrono::duration_cast<chrono::nanoseconds>(finish - start).count() << endl;

    // expHRR(file_writer, exp_recorder, points, mbrs_map, query_poitns, insert_points);
    // expGrid(file_writer, exp_recorder, points, mbrs_map, query_poitns, insert_points);
    // expKDB(file_writer, exp_recorder, points, mbrs_map, query_poitns, insert_points);

    // exp_RSMI(file_writer, exp_recorder, points, mbrs_map, query_poitns, insert_points, model_path, 0.1);

    // exp_ZM(file_writer, exp_recorder, points, mbrs_map, query_poitns, insert_points, model_path, 1.0, 1);

    // exp_binary_search(file_writer, exp_recorder, points);

    // // sampling
    exp_recorder.test_sp();
    cout << "IS_SAMPLING" << endl;
    float sampling_rates[] = {0.0001, 0.001};
    for (size_t i = 0; i < sizeof(sampling_rates) / sizeof(sampling_rates[0]); i++)
    {
        // exp_ZM(file_writer, exp_recorder, points, mbrs_map, query_poitns, insert_points, model_path, sampling_rates[i], 1024);
        exp_RSMI(file_writer, exp_recorder, points, mbrs_map, query_poitns, insert_points, model_path, sampling_rates[i]);
    }

    // // representative set
    // exp_recorder.test_rs();
    // cout << "IS_REPRESENTATIVE_SET" << endl;
    // int ms[] = {8192};
    // for (size_t i = 0; i < sizeof(ms) / sizeof(ms[0]); i++)
    // {
    //     exp_ZM(file_writer, exp_recorder, points, mbrs_map, query_poitns, insert_points, model_path, 1.0, ms[i]);
    // }

    // // reinforcement learning
    // exp_recorder.test_rl();
    // exp_ZM(file_writer, exp_recorder, points, mbrs_map, query_poitns, insert_points, model_path, 1.0, 1);

    // cluster
    // exp_recorder.test_cluster();
    // cout << "IS_CLUSTER" << endl;
    // exp_recorder.cluster_method = "kmeans";
    // exp_recorder.cluster_size = 10000;
    // exp_ZM(file_writer, exp_recorder, points, mbrs_map, query_poitns, insert_points, model_path, 1.0, 1);

    // // model reuse
    // exp_recorder.test_model_reuse();
    // exp_ZM(file_writer, exp_recorder, points, mbrs_map, query_poitns, insert_points, model_path, 1.0, 1);

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