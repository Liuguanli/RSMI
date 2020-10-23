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
#include "../utils/ExpRecorder.h"
#include "../entities/Mbr.h"
#include "../entities/Point.h"
#include "../entities/LeafNode.h"
#include "../entities/NonLeafNode.h"
#include "../curves/z.H"
// #include "../entities/Node.h"
#include "../utils/Constants.h"
#include "../utils/SortTools.h"
#include "../utils/ModelTools.h"
#include "../entities/NodeExtend.h"
// #include "../file_utils/SearchHelper.h"
// #include "../file_utils/CustomDataSet4ZM.h"

#include <torch/script.h>
#include <ATen/ATen.h>
#include <torch/torch.h>

#include <torch/nn/module.h>
// #include <torch/nn/modules/functional.h>
#include <torch/nn/modules/linear.h>
// #include <torch/nn/modules/sequential.h>
#include <torch/optim.h>
#include <torch/types.h>
#include <torch/utils.h>

using namespace std;
using namespace at;
using namespace torch::nn;
using namespace torch::optim;

class ZM
{
private:
    string file_name;
    int page_size;
    // long long side;
    int bit_num = 0;
    long long N = 0;
    long long gap;
    long long min_curve_val;
    long long max_curve_val;

public:
    ZM();
    ZM(int);

    vector<vector<std::shared_ptr<Net>>> index;

    vector<int> stages;

    vector<float> xs;
    vector<float> ys;

    int zm_max_error = 0;
    int zm_min_error = 0;
    // vector<long long> hs;

    vector<LeafNode *> leafnodes;

    // auto trainModel(vector<Point> points);
    void build(ExpRecorder &exp_recorder, vector<Point> points);

    void point_query(ExpRecorder &exp_recorder, Point query_point);
    void point_query_after_update(ExpRecorder &exp_recorder, Point query_point);
    long long get_point_index(ExpRecorder &exp_recorder, Point query_point);
    void point_query(ExpRecorder &exp_recorder, vector<Point> query_points);
    void point_query_after_update(ExpRecorder &exp_recorder, vector<Point> query_points);

    void window_query(ExpRecorder &exp_recorder, vector<Mbr> query_windows);
    vector<Point> window_query(ExpRecorder &exp_recorder, Mbr query_window);
    void acc_window_query(ExpRecorder &exp_recorder, vector<Mbr> query_windows);
    vector<Point> acc_window_query(ExpRecorder &exp_recorder, Mbr query_windows);

    void kNN_query(ExpRecorder &exp_recorder, vector<Point> query_points, int k);
    vector<Point> kNN_query(ExpRecorder &exp_recorder, Point query_point, int k);
    void acc_kNN_query(ExpRecorder &exp_recorder, vector<Point> query_points, int k);
    vector<Point> acc_kNN_query(ExpRecorder &exp_recorder, Point query_point, int k);

    void insert(ExpRecorder &exp_recorder, Point);
    void insert(ExpRecorder &exp_recorder, vector<Point>);

    void remove(ExpRecorder &exp_recorder, Point);
    void remove(ExpRecorder &exp_recorder, vector<Point>);
};

ZM::ZM()
{
    this->page_size = Constants::PAGESIZE;
}

ZM::ZM(int page_size)
{
    this->page_size = page_size;
}

void ZM::build(ExpRecorder &exp_recorder, vector<Point> points)
{
    auto start = chrono::high_resolution_clock::now();
    vector<vector<vector<Point>>> tmp_records;
    sort(points.begin(), points.end(), sortX());
    this->N = points.size();
    bit_num = ceil(log(N) / log(2));
    for (long i = 0; i < N; i++)
    {
        points[i].x_i = points[i].x * N;
        // xs.push_back(points[i]->x);
    }
    sort(points.begin(), points.end(), sortY());
    for (long long i = 0; i < N; i++)
    {
        points[i].y_i = points[i].y * N;
        // ys.push_back(points[i]->y);
        long long curve_val = compute_Z_value(points[i].x_i, points[i].y_i, bit_num);
        points[i].curve_val = curve_val;
    }
    // side = pow(2, bit_num);
    // side = pow(4, bit_num);
    // for (long long i = 0; i < N; i++)
    // {
    //     long long curve_val = compute_Z_value(points[i]->x_i, points[i]->y_i, bit_num);
    //     points[i]->curve_val = curve_val;
    // }
    sort(points.begin(), points.end(), sort_curve_val());
    min_curve_val = points[0].curve_val;
    max_curve_val = points[points.size() - 1].curve_val;
    this->gap = max_curve_val - min_curve_val;

    for (long long i = 0; i < N; i++)
    {
        // points[i]->index = i / Constants::PAGESIZE;
        points[i].index = i * 1.0 / N;
        points[i].normalized_curve_val = (points[i].curve_val - min_curve_val) * 1.0 / gap;
        // cout<< points[i]->normalized_curve_val <<endl;
    }

    int leaf_node_num = points.size() / page_size;
    // cout << "leaf_node_num:" << leaf_node_num << endl;
    for (int i = 0; i < leaf_node_num; i++)
    {
        LeafNode *leafnode = new LeafNode();
        auto bn = points.begin() + i * page_size;
        auto en = points.begin() + i * page_size + page_size;
        vector<Point> vec(bn, en);
        // cout << vec.size() << " " << vec[0]->x_i << " " << vec[99]->x_i << endl;
        leafnode->add_points(vec);
        LeafNode *temp = leafnode;
        leafnodes.push_back(temp);
    }
    exp_recorder.leaf_node_num += leaf_node_num;
    // for the last leafnode
    if (points.size() > page_size * leaf_node_num)
    {
        // TODO if do not delete will it last to the end of lifecycle?
        LeafNode *leafnode = new LeafNode();
        auto bn = points.begin() + page_size * leaf_node_num;
        auto en = points.end();
        vector<Point> vec(bn, en);
        // cout << vec.size() << " " << vec[0].x_i << " " << vec[99].x_i << endl;
        leafnode->add_points(vec);
        leafnodes.push_back(leafnode);
        exp_recorder.leaf_node_num++;
    }

    // long long N = (long long)leafnodes.size();
    stages.push_back(1);
    if (leafnodes.size() / Constants::PAGESIZE >= 4)
    {
        stages.push_back((int)(sqrt(leafnodes.size() / Constants::PAGESIZE)));
        stages.push_back(leafnodes.size() / Constants::PAGESIZE);
    }

    vector<vector<Point>> stage1;
    stage1.push_back(points);
    tmp_records.push_back(stage1);

    stage1.clear();
    stage1.shrink_to_fit();

    for (size_t i = 0; i < stages.size(); i++)
    {
        // initialize
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
        // build
        for (size_t j = 0; j < stages[i]; j++)
        {
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
                for (Point point : tmp_records[i][j])
                {
                    locations.push_back(point.normalized_curve_val);
                    labels.push_back(point.index);
                }
                net->train_model(locations, labels);
                net->get_parameters_ZM();
                // net->trainModel(tmp_records[i][j]);
                exp_recorder.non_leaf_node_num++;
                int max_error = 0;
                int min_error = 0;
                temp_index.push_back(net);
                for (Point point : tmp_records[i][j])
                {
                    torch::Tensor res = net->forward(torch::tensor({point.normalized_curve_val}));
                    int pos = 0;
                    if (i == stages.size() - 1)
                    {
                        pos = (int)(res.item().toFloat() * N);
                        // cout << "point->index: " << point->index << " predicted value: " << res.item().toFloat() << " pos: " << pos << endl;
                    }
                    else
                    {
                        // pos = res.item().toFloat() * stages[i + 1] / N;
                        pos = res.item().toFloat() * stages[i + 1];
                        // cout << "i: " << i << " pos: " << pos << endl;
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
                    }
                    else
                    {
                        int error = (int)(point.index * N) - pos;
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
                net->max_error = max_error;
                net->min_error = min_error;
                if ((max_error - min_error) > (zm_max_error - zm_min_error))
                {
                    zm_max_error = max_error;
                    zm_min_error = min_error;
                }
                // cout << net->parameters() << endl;
                // TODO initialize index and tmp_record
                // cout << "stage:" << i << " size:" << tmp_records[i][j].size() << endl;
                tmp_records[i][j].clear();
                tmp_records[i][j].shrink_to_fit();
            }
            catch (const std::exception &e)
            {
                std::cerr << e.what() << '\n';
            }
        }
        index.push_back(temp_index);
        // cout << "size of stages:" << stages[i] << endl;
    }

    // auto net = this->trainModel(tmp_records[0][0]);

    // torch::Tensor res = net->forward(torch::tensor({1.0}));
    // cout << res.item().toInt() << endl;
    // cout << typeid(net).name() << endl;
    // for (size_t i = 0; i < stages.size(); i++)
    // {
    //     for (size_t j = 0; j < stages[i]; j++)
    //     {
    //         cout << tmp_records[i][j].size() << endl;
    //     }
    //     cout << endl;
    // }
    auto finish = chrono::high_resolution_clock::now();
    exp_recorder.max_error = zm_max_error;
    exp_recorder.min_error = zm_min_error;
    exp_recorder.time = chrono::duration_cast<chrono::nanoseconds>(finish - start).count();
    exp_recorder.size = (1 * Constants::HIDDEN_LAYER_WIDTH + 1 * Constants::HIDDEN_LAYER_WIDTH + Constants::HIDDEN_LAYER_WIDTH * 1 + 1) * Constants::EACH_DIM_LENGTH * exp_recorder.non_leaf_node_num + (Constants::DIM * Constants::PAGESIZE * Constants::EACH_DIM_LENGTH + Constants::PAGESIZE * Constants::INFO_LENGTH + Constants::DIM * Constants::DIM * Constants::EACH_DIM_LENGTH) * exp_recorder.leaf_node_num;
}

void ZM::point_query(ExpRecorder &exp_recorder, Point query_point)
{
    long long curve_val = compute_Z_value(query_point.x * N, query_point.y * N, bit_num);
    float key = (curve_val - min_curve_val) * 1.0 / gap;
    int predicted_index = 0;
    int next_stage_length = 1;
    int min_error = 0;
    int max_error = 0;
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
    int front = predicted_index + min_error;
    front = front < 0 ? 0 : front;
    int back = predicted_index + max_error;
    back = back >= N ? N - 1 : back;
    // cout << "pos: " << pos << " front: " << front << " back: " << back << " width: " << width << endl;
    while (front <= back)
    {
        int mid = (front + back) / 2;
        int node_index = mid / Constants::PAGESIZE;

        LeafNode *leafnode = leafnodes[node_index];

        if(leafnode->mbr.contains(query_point))
        {
            vector<Point>::iterator iter = find(leafnode->children->begin(), leafnode->children->end(), query_point);
            exp_recorder.page_access += 1;
            if (iter != leafnode->children->end())
            {
                // cout << "find it" << endl;
                break;
            }
        }
        if ((*leafnode->children)[0].curve_val < curve_val)
        {
            front = mid + 1;
        }
        else
        {
            back = mid - 1;
        }
        if (front > back)
        {
            cout << "not found!" << endl;
        }
    }
}

void ZM::point_query_after_update(ExpRecorder &exp_recorder, Point query_point)
{
    long long curve_val = compute_Z_value(query_point.x * N, query_point.y * N, bit_num);
    float key = (curve_val - min_curve_val) * 1.0 / gap;
    int predicted_index = 0;
    int next_stage_length = 1;
    int min_error = 0;
    int max_error = 0;
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
    int front = predicted_index + min_error;
    front = front < 0 ? 0 : front;
    int back = predicted_index + max_error;
    back = back >= N ? N - 1 : back;
    // cout << "predicted_index: " << predicted_index << " front: " << front << " back: " << back << endl;
    front = front / Constants::PAGESIZE;
    back = back / Constants::PAGESIZE;
    for (size_t i = front; i <= back; i++)
    {
        vector<Point>::iterator iter = find(leafnodes[i]->children->begin(), leafnodes[i]->children->end(), query_point);
        exp_recorder.page_access += 1;
        if (iter != leafnodes[i]->children->end())
        {
            // cout<< "find it!" << endl;
            break;
        }
    }
}

void ZM::point_query(ExpRecorder &exp_recorder, vector<Point> query_points)
{
    cout << "point_query:" << query_points.size() << endl;
    auto start = chrono::high_resolution_clock::now();
    for (long i = 0; i < query_points.size(); i++)
    {
        point_query(exp_recorder, query_points[i]);
    }
    auto finish = chrono::high_resolution_clock::now();
    exp_recorder.time = chrono::duration_cast<chrono::nanoseconds>(finish - start).count() / query_points.size();
    cout << "finish point_query: pageaccess:" << exp_recorder.page_access << endl;
    exp_recorder.page_access = exp_recorder.page_access / query_points.size();
    cout << "finish point_query time: " << exp_recorder.time << endl;
}

void ZM::point_query_after_update(ExpRecorder &exp_recorder, vector<Point> query_points)
{
    auto start = chrono::high_resolution_clock::now();
    for (long i = 0; i < query_points.size(); i++)
    {
        point_query_after_update(exp_recorder, query_points[i]);
    }
    auto finish = chrono::high_resolution_clock::now();
    exp_recorder.time = chrono::duration_cast<chrono::nanoseconds>(finish - start).count() / query_points.size();
    exp_recorder.page_access = (double)exp_recorder.page_access / query_points.size();
    cout<< "point_query: " << exp_recorder.time << endl;
}

long long ZM::get_point_index(ExpRecorder &exp_recorder, Point query_point)
{
    long long curve_val = compute_Z_value(query_point.x * N, query_point.y * N, bit_num);
    query_point.curve_val = curve_val;
    float key = (curve_val - min_curve_val) * 1.0 / gap;
    query_point.normalized_curve_val = key;
    long long predicted_index = 0;
    long long next_stage_length = 1;
    int min_error = 0;
    int max_error = 0;
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
    exp_recorder.index_high = predicted_index + max_error;
    exp_recorder.index_low = predicted_index + min_error;
    return predicted_index;
}

vector<Point> ZM::window_query(ExpRecorder &exp_recorder, Mbr query_window)
{
    vector<Point> window_query_results;
    vector<Point> vertexes = query_window.get_corner_points();
    vector<long long> indices;
    for (Point point : vertexes)
    {
        get_point_index(exp_recorder, point);
        indices.push_back(exp_recorder.index_low);
        indices.push_back(exp_recorder.index_high);
    }
    sort(indices.begin(), indices.end());
    long front = indices.front() / page_size;
    long back = indices.back() / page_size;

    front = front < 0 ? 0 : front;
    back = back >= leafnodes.size() ? leafnodes.size() - 1 : back;
    // cout << "front: " << front << " back: " << back << endl;
    for (size_t i = front; i <= back; i++)
    {
        LeafNode *leafnode = leafnodes[i];
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
    // cout<< window_query_results.size() <<endl;
    return window_query_results;
}

void ZM::window_query(ExpRecorder &exp_recorder, vector<Mbr> query_windows)
{
    cout << "ZM::window_query" << endl;
    auto start = chrono::high_resolution_clock::now();
    for (int i = 0; i < query_windows.size(); i++)
    {
        vector<Point> window_query_results = window_query(exp_recorder, query_windows[i]);
        exp_recorder.window_query_result_size += window_query_results.size();
    }
    auto finish = chrono::high_resolution_clock::now();
    exp_recorder.time = chrono::duration_cast<chrono::nanoseconds>(finish - start).count() / query_windows.size();
    exp_recorder.page_access = (double)exp_recorder.page_access / query_windows.size();
}

vector<Point> ZM::acc_window_query(ExpRecorder &exp_recorder, Mbr query_window)
{
    vector<Point> window_query_results;
    for (LeafNode *leafnode : leafnodes)
    {
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
    // cout<< window_query_results.size() <<endl;
    return window_query_results;
}

void ZM::acc_window_query(ExpRecorder &exp_recorder, vector<Mbr> query_windows)
{
    cout << "ZM::acc_window_query" << endl;
    auto start = chrono::high_resolution_clock::now();
    for (int i = 0; i < query_windows.size(); i++)
    {
        exp_recorder.acc_window_query_qesult_size += acc_window_query(exp_recorder, query_windows[i]).size();
    }
    auto finish = chrono::high_resolution_clock::now();
    // cout << "end:" << end.tv_nsec << " begin" << begin.tv_nsec << endl;
    exp_recorder.time = chrono::duration_cast<chrono::nanoseconds>(finish - start).count() / query_windows.size();
    exp_recorder.page_access = (double)exp_recorder.page_access / query_windows.size();
}

void ZM::kNN_query(ExpRecorder &exp_recorder, vector<Point> query_points, int k)
{
    cout << "ZM::kNN_query" << endl;
    for (int i = 0; i < query_points.size(); i++)
    {
        auto start = chrono::high_resolution_clock::now();
        vector<Point> knn_result = kNN_query(exp_recorder, query_points[i], k);
        auto finish = chrono::high_resolution_clock::now();
        exp_recorder.time += chrono::duration_cast<chrono::nanoseconds>(finish - start).count();
        exp_recorder.knn_query_results.insert(exp_recorder.knn_query_results.end(), knn_result.begin(), knn_result.end());
        // cout << "knn_diff: " << knn_diff(acc_kNN_query(exp_recorder, query_points[i], k), kNN_query(exp_recorder, query_points[i], k)) << endl;
    }
    exp_recorder.time /= query_points.size();
    exp_recorder.page_access = (double)exp_recorder.page_access / query_points.size();
}

vector<Point> ZM::kNN_query(ExpRecorder &exp_recorder, Point query_point, int k)
{
    vector<Point> result;
    float knn_query_side = sqrt((float)k / N);
    while (true)
    {
        Mbr mbr = Mbr::get_mbr(query_point, knn_query_side);
        vector<Point> temp_result = window_query(exp_recorder, mbr);
        // cout << "mbr: " << mbr->get_self() << "size: " << temp_result.size() << endl;
        if (temp_result.size() >= k)
        {
            sort(temp_result.begin(), temp_result.end(), sortForKNN(query_point));
            Point last = temp_result[k - 1];
            // cout << " last dist : " << last->cal_dist(queryPoint) << " knnquerySide: " << knnquerySide << endl;
            if (last.cal_dist(query_point) <= knn_query_side)
            {
                // TODO get top K from the vector.
                auto bn = temp_result.begin();
                auto en = temp_result.begin() + k;
                vector<Point> vec(bn, en);
                result = vec;
                break;
            }
        }
        knn_query_side = knn_query_side * 2;
        // cout << " knnquerySide: " << knnquerySide << endl;
    }
    return result;
}

void ZM::acc_kNN_query(ExpRecorder &exp_recorder, vector<Point> query_points, int k)
{
    cout << "ZM::acc_kNN_query" << endl;
    for (int i = 0; i < query_points.size(); i++)
    {
        auto start = chrono::high_resolution_clock::now();
        vector<Point> knn_result = acc_kNN_query(exp_recorder, query_points[i], k);
        auto finish = chrono::high_resolution_clock::now();
        exp_recorder.time += chrono::duration_cast<chrono::nanoseconds>(finish - start).count();
        exp_recorder.acc_knn_query_results.insert(exp_recorder.acc_knn_query_results.end(), knn_result.begin(), knn_result.end());
    }
    exp_recorder.time /= query_points.size();
    exp_recorder.k_num = k;
    exp_recorder.page_access = (double)exp_recorder.page_access / query_points.size();
}

vector<Point> ZM::acc_kNN_query(ExpRecorder &exp_recorder, Point query_point, int k)
{
    vector<Point> result;
    float knn_query_side = sqrt((float)k / N);
    while (true)
    {
        Mbr mbr = Mbr::get_mbr(query_point, knn_query_side);
        vector<Point> temp_result = acc_window_query(exp_recorder, mbr);
        if (temp_result.size() >= k)
        {
            sort(temp_result.begin(), temp_result.end(), sortForKNN(query_point));
            Point last = temp_result[k - 1];
            if (last.cal_dist(query_point) <= knn_query_side)
            {
                // TODO get top K from the vector.
                auto bn = temp_result.begin();
                auto en = temp_result.begin() + k;
                vector<Point> vec(bn, en);
                result = vec;
                break;
            }
        }
        knn_query_side = knn_query_side * 2;
    }
    return result;
}

void ZM::insert(ExpRecorder &exp_recorder, Point point)
{
    // long long curve_val = compute_Z_value(point->x * width, point->y * width, bit_num);
    // point->curve_val = curve_val;
    // point->normalized_curve_val = (curve_val - min_curve_val) * 1.0 / gap;
    long long curve_val = compute_Z_value(point.x * N, point.y * N, bit_num);
    point.curve_val = curve_val;
    float key = (curve_val - min_curve_val) * 1.0 / gap;
    point.normalized_curve_val = key;
    long long predicted_index = 0;
    long long next_stage_length = 1;
    int min_error = 0;
    int max_error = 0;
    std::shared_ptr<Net> *net;
    int last_model_index = 0;
    for (int i = 0; i < stages.size(); i++)
    {
        if (i == stages.size() - 1)
        {
            next_stage_length = N;
            last_model_index = predicted_index;
            min_error = index[i][predicted_index]->min_error;
            max_error = index[i][predicted_index]->max_error;
        }
        else
        {
            next_stage_length = stages[i + 1];
        }
        predicted_index = index[i][predicted_index]->predict_ZM(key) * next_stage_length;

        net = &index[i][predicted_index];
        // predicted_index = net->forward(torch::tensor({key})).item().toFloat() * next_stage_length;
        if (predicted_index < 0)
        {
            predicted_index = 0;
        }
        if (predicted_index >= next_stage_length)
        {
            predicted_index = next_stage_length - 1;
        }
    }
    exp_recorder.index_high = predicted_index + max_error;
    exp_recorder.index_low = predicted_index + min_error;

    int inserted_index = predicted_index / Constants::PAGESIZE;

    LeafNode *leafnode = leafnodes[inserted_index];

    if (leafnode->is_full())
    {
        // int front = 0;
        // int back = leafnode->children->size() - 1;
        // int mid = 0;
        // while (front <= back)
        // {
        //     mid = (front + back) / 2;

        //     if ((*leafnode->children)[mid]->curve_val > point->curve_val)
        //     {
        //         back = mid - 1;
        //     }
        //     else if ((*leafnode->children)[mid]->curve_val < point->curve_val)
        //     {
        //         front = mid + 1;
        //     }
        //     else
        //     {
        //         break;
        //     }
        // }
        // leafnode->children->insert(leafnode->children->begin() + mid, point);
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

void ZM::insert(ExpRecorder &exp_recorder, vector<Point> points)
{
    cout << "ZM::insert" << endl;
    auto start = chrono::high_resolution_clock::now();
    for (int i = 0; i < points.size(); i++)
    {
        insert(exp_recorder, points[i]);
    }
    auto finish = chrono::high_resolution_clock::now();
    long long old_time_cost = exp_recorder.insert_time * exp_recorder.insert_num;
    exp_recorder.insert_num += points.size();
    exp_recorder.insert_time = (old_time_cost + chrono::duration_cast<chrono::nanoseconds>(finish - start).count()) / exp_recorder.insert_num;
    cout<< "exp_recorder.insert_time: " << exp_recorder.insert_time << endl;
}

void ZM::remove(ExpRecorder &exp_recorder, Point point)
{
    long long curve_val = compute_Z_value(point.x * N, point.y * N, bit_num);
    float key = (curve_val - min_curve_val) * 1.0 / gap;
    int predicted_index = 0;
    int next_stage_length = 1;
    int min_error = 0;
    int max_error = 0;
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
    int front = predicted_index + min_error;
    front = front < 0 ? 0 : front;
    int back = predicted_index + max_error;
    back = back >= N ? N - 1 : back;
    // cout << "pos: " << pos << " front: " << front << " back: " << back << " width: " << width << endl;
    back = back / Constants::PAGESIZE;
    front = front / Constants::PAGESIZE;
    for (size_t i = front; i <= back; i++)
    {
        LeafNode *leafnode = leafnodes[i];
        vector<Point>::iterator iter = find(leafnode->children->begin(), leafnode->children->end(), point);
        if (leafnode->mbr.contains(point) && leafnode->delete_point(point))
        {
            // cout << "remove it" << endl;
            break;
        }
    }

    // while (front <= back)
    // {
    //     int mid = (front + back) / 2;
    //     int node_index = mid / Constants::PAGESIZE;

    //     LeafNode *leafnode = leafnodes[node_index];

    //     vector<Point *>::iterator iter = find(leafnode->children->begin(), leafnode->children->end(), point);
    //     if (leafnode->mbr->contains(point) && leafnode->delete_point(point))
    //     {
    //         // cout << "remove it" << endl;
    //         break;
    //     }
    //     else
    //     {
    //         if ((*leafnode->children)[0]->curve_val < curve_val)
    //         {
    //             front = mid + 1;
    //         }
    //         else
    //         {
    //             back = mid - 1;
    //         }
    //     }
    //     // if (front > back)
    //     // {
    //     //     cout << "not found!" << endl;
    //     // }
    // }
}

void ZM::remove(ExpRecorder &exp_recorder, vector<Point> points)
{
    cout << "ZM::remove" << endl;
    auto start = chrono::high_resolution_clock::now();
    for (int i = 0; i < points.size(); i++)
    {
        remove(exp_recorder, points[i]);
    }
    auto finish = chrono::high_resolution_clock::now();
    // cout << "end:" << end.tv_nsec << " begin" << begin.tv_nsec << endl;
    long long old_time_cost = exp_recorder.delete_time * exp_recorder.delete_num;
    exp_recorder.delete_num += points.size();
    exp_recorder.delete_time = (old_time_cost + chrono::duration_cast<chrono::nanoseconds>(finish - start).count()) / exp_recorder.delete_num;
}