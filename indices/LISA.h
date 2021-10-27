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
#include "../entities/Shard.h"
#include "../curves/z.H"
// #include "../entities/Node.h"
#include "../utils/Constants.h"
#include "../utils/SortTools.h"
#include "../utils/ModelTools.h"
#include "../entities/NodeExtend.h"
// #include "../file_utils/SearchHelper.h"
// #include "../file_utils/CustomDataSet4ZM.h"
#include "../utils/PreTrainZM.h"
#include "../utils/Rebuild.h"

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

// #include "../utils/PreTrainZM.h"

using namespace std;
using namespace at;
using namespace torch::nn;
using namespace torch::optim;

class LISA
{
private:
    int page_size;
    int bit_num = 0;
    long long N = 0;
    string model_path_root;
    int n_parts = 240;
    float eta = 0.01;
    int n_models = 1000;
    float max_value_x = 1.0;
    int shard_id = 0;
    int page_id = 0;
    vector<Point> split_points;
    vector<long long> split_index_list;
    vector<std::shared_ptr<Net>> SP;
    vector<map<int, Shard>> shards;

    vector<double> mappings;
    vector<float> borders;
    vector<float> gaps;
    vector<float> x_split_points;
    vector<double> model_split_mapping;
    vector<int> shard_start_id_each_model;
    vector<int> partition_sizes;
    vector<double> min_key_list;

public:
    LISA();
    LISA(int);
    LISA(string);
    void build(ExpRecorder &exp_recorder, vector<Point> points);
    void point_query(ExpRecorder &exp_recorder, Point query_point);
    void point_query(ExpRecorder &exp_recorder, vector<Point> query_points);

    void window_query(ExpRecorder &exp_recorder, vector<Mbr> query_windows);
    vector<Point> window_query(ExpRecorder &exp_recorder, Mbr query_window, int partition_index);

    void kNN_query(ExpRecorder &exp_recorder, vector<Point> query_points, int k);
    vector<Point> kNN_query(ExpRecorder &exp_recorder, Point query_point, int k);

    void insert(ExpRecorder &exp_recorder, Point);
    bool insert(ExpRecorder &exp_recorder, vector<Point> &inserted_points);

    void remove(ExpRecorder &exp_recorder, Point);
    void remove(ExpRecorder &exp_recorder, vector<Point>);

    void clear(ExpRecorder &exp_recorder);

    bool check_distinctiveness(vector<double> vals);

    bool check_monotony(vector<int> positions);
    bool check_monotony1(vector<float> positions);
    bool check_monotony2(vector<double> positions);

    double get_mapped_key(Point point, int i);
    int get_partition_index(Point point);
    int get_model_index(double key);
};

LISA::LISA()
{
    this->page_size = Constants::PAGESIZE;
}

LISA::LISA(int page_size)
{
    this->page_size = page_size;
}

LISA::LISA(string path)
{
    this->model_path_root = path;
    this->page_size = Constants::PAGESIZE;
}

bool LISA::check_distinctiveness(vector<double> vals)
{
    set<double> s;
    for (size_t i = 0; i < vals.size(); i++)
    {
        s.insert(vals[i]);
    }
    cout << "s.size:" << s.size() << endl;

    double idx = vals[0];
    for (size_t i = 1; i < vals.size(); i++)
    {
        if (vals[i] < idx)
        {
            cout << "i: " << i << " positions[i]: " << vals[i] << " idx: " << idx << endl;
            return false;
        }
        idx = vals[i];
    }
    return s.size() == vals.size();
}

int LISA::get_model_index(double key)
{
    // TODO
    if (model_split_mapping[0] > key)
    {
        return 0;
    }
    if (model_split_mapping[n_models - 2] <= key)
    {
        return n_models - 1;
    }
    int begin = 1;
    int end = n_models - 1;
    while (begin < end)
    {
        int mid = (begin + end) / 2;
        if (model_split_mapping[mid - 1] <= key && key < model_split_mapping[mid])
        {
            return mid;
        }
        else if (model_split_mapping[mid - 1] > key)
        {
            end = mid;
        }
        else
        {
            begin = mid;
        }
    }
    return n_models - 1;
}

int LISA::get_partition_index(Point point)
{
    if (x_split_points[0] >= point.x)
    {
        return 0;
    }
    if (x_split_points[n_parts - 2] < point.x)
    {
        return n_parts - 1;
    }
    int begin = 1;
    int end = n_parts - 1;
    while (begin < end)
    {
        int mid = (begin + end) / 2;
        if (x_split_points[mid - 1] < point.x && point.x <= x_split_points[mid])
        {
            return mid;
        }
        else if (x_split_points[mid - 1] >= point.x)
        {
            end = mid;
        }
        else
        {
            begin = mid;
        }
    }
    return n_parts - 1;
}

double LISA::get_mapped_key(Point point, int i)
{
    // double mapped_val = point.x / max_value_x + (point.x - borders[i]) / gaps[i] + i * 2;
    // return mapped_val;

    // double measure = (point.x - borders[i]) / gaps[i];
    double mapped_val = point.y + i * 2;

    // double mapped_val = measure * eta * 0.000001 + point.y + i * 2;
    // double mapped_val = (point.y / 1 * (n_parts - 1)) + i * n_parts;
    // cout << "mapped_val: " << mapped_val << endl;
    return mapped_val;
}

void LISA::build(ExpRecorder &exp_recorder, vector<Point> points)
{

    cout << " build" << endl;
    auto build_start = chrono::high_resolution_clock::now();
    N = points.size();
    n_models = N / Constants::THRESHOLD;
    int partition_size = floor(N / n_parts);
    int remainder = N - n_parts * partition_size;
    cout << "remainder: " << remainder << " partition_size:" << partition_size << endl;
    vector<int> x_split_idxes;

    vector<int> model_split_idxes;

    sort(points.begin(), points.end(), sortX());
    // this->max_value_x = points[N - 1].x;
    // cout << "max_value_x: " << max_value_x << endl;

    borders.push_back(0.0);
    for (size_t i = 0; i < remainder; i++)
    {
        int idx = (i + 1) * (partition_size + 1);
        while (points[idx - 1].x == points[idx].x)
        {
            idx++;
        }
        x_split_idxes.push_back(idx);
        x_split_points.push_back(points[idx - 1].x);
        borders.push_back(points[idx - 1].x);
        gaps.push_back(x_split_points[i] - borders[i]);
    }

    for (size_t i = remainder; i < n_parts - 1; i++)
    {
        int idx = (i + 1) * partition_size + remainder;
        // x_split_points.push_back(points[idx - 1].x);
        // borders.push_back(points[idx - 1].x);
        while (points[idx - 1].x == points[idx].x)
        {
            idx++;
        }
        x_split_idxes.push_back(idx);
        x_split_points.push_back(points[idx - 1].x);
        borders.push_back(points[idx - 1].x);
        gaps.push_back(x_split_points[i] - borders[i]);
    }

    x_split_points.push_back(max_value_x);
    x_split_idxes.push_back(N);
    gaps.push_back(x_split_points[n_parts - 1] - borders[n_parts - 1]);
    // for (size_t i = 0; i < n_parts; i++)
    // {
    //     cout << x_split_points[i] << " " << x_split_idxes[i] << " " << borders[i] << " " << gaps[i] << endl;
    // }
    cout << "x_split_idxes.size(): " << x_split_idxes.size() << endl;
    int start = 0;
    int index = 0;

    vector<float> float_mappings;

    for (size_t i = 0; i < x_split_idxes.size(); i++)
    {
        // cout << "-----------" << i << "-----------" << endl;
        vector<Point> part_data(points.begin() + start, points.begin() + x_split_idxes[i]);
        start = x_split_idxes[i];

        sort(part_data.begin(), part_data.end(), sortY());

        for (size_t j = 0; j < part_data.size(); j++)
        {
            // TODO chagen mapping
            // double mapped_val = ((part_data[j].x - borders[i]) / gaps[i]) * eta + (part_data[j].x / max_value_x * (n_parts - 1)) + (i * n_parts) * 1.0;
            // double mapped_val = part_data[j].x / max_value_x + (part_data[j].x - borders[i]) / gaps[i] + i * 2;
            int partition_index = get_partition_index(part_data[j]);
            // if (239 == i)
            // {
            //     cout << "index: " << index << " j: " << j << endl;
            // }
            double mapped_val = get_mapped_key(part_data[j], partition_index);
            mappings.push_back(mapped_val);
            float_mappings.push_back(mapped_val);
            part_data[j].key = mapped_val;
            points[index++] = part_data[j];
        }
    }
    // if (!check_monotony2(mappings))
    // {
    //     cout << "mappings: xxxxxxxxxxxxxxxxxxxxxxxxxx" << endl;
    // }
    // cout << "-----------finish mapping-----------" << endl;
    // cout << check_distinctiveness(mappings) << endl;
    // cout << "-----------finish check_distinctiveness-----------" << endl;

    Histogram mappings_histogram(pow(2, Constants::UNIFIED_Z_BIT_NUM), float_mappings);
    exp_recorder.ogiginal_histogram = mappings_histogram;
    exp_recorder.changing_histogram = mappings_histogram;

    int offset = N / n_models;
    for (size_t i = 1; i < n_models; i++)
    {
        int idx = i * offset;
        while (mappings[idx] == mappings[idx + 1])
        {
            idx++;
        }
        model_split_mapping.push_back(mappings[idx]);
        model_split_idxes.push_back(idx);
    }
    model_split_mapping.push_back(mappings[N - 1]);
    model_split_idxes.push_back(N);

    start = 0;

    for (size_t i = 0; i < n_models; i++)
    {
        // cout << "-----------" << i << "-----------" << endl;
        vector<Point> part_data(points.begin() + start, points.begin() + model_split_idxes[i]);
        vector<float> keys;
        vector<float> labels;
        // float min_key = part_data[0].key;
        double min_key = mappings[start];
        min_key_list.push_back(min_key);
        int part_data_size = part_data.size();
        partition_sizes.push_back(part_data_size);
        keys.push_back(0.0);
        labels.push_back(0.0);

        for (size_t j = 1; j < part_data_size; j++)
        {
            keys.push_back(mappings[start + j] - min_key);
            labels.push_back(j * 1.0 / part_data_size);
        }

        // if (!check_monotony1(keys))
        // {
        //     cout << "keys: -------------i------------- " << i << endl;
        // }

        auto net = std::make_shared<Net>(1);
#ifdef use_gpu
        net->to(torch::kCUDA);
#endif
        // TODO begin BASE framework
        Histogram histogram(pow(2, Constants::UNIFIED_Z_BIT_NUM), keys);
        exp_recorder.ogiginal_histogram = histogram;
        exp_recorder.changing_histogram = histogram;

        if (exp_recorder.is_cost_model)
        {
            pre_train_zm::cost_model_predict(exp_recorder, exp_recorder.upper_level_lambda, keys.size() * 1.0 / 10000, pre_train_zm::get_distribution(histogram));
        }

        if (exp_recorder.is_model_reuse || exp_recorder.is_rs)
        {
            auto start_mr = chrono::high_resolution_clock::now();
            exp_recorder.model_reuse_num++;
            std::stringstream stream;
            stream << std::fixed << std::setprecision(1) << exp_recorder.model_reuse_threshold;
            string threshold = stream.str();
            string model_path = "";
            if (net->is_reusable_zm(histogram, threshold, model_path))
            {
                torch::load(net, model_path);
                auto finish_mr = chrono::high_resolution_clock::now();
                exp_recorder.extra_time += chrono::duration_cast<chrono::nanoseconds>(finish_mr - start_mr).count();
            }
            else
            {
                auto start_train = chrono::high_resolution_clock::now();
                net->train_model(keys, labels);
                auto end_train = chrono::high_resolution_clock::now();
                exp_recorder.training_cost += chrono::duration_cast<chrono::nanoseconds>(end_train - start_train).count();
            }
        }
        else if (exp_recorder.is_sp)
        {
            exp_recorder.sp_num++;
            auto start_sp = chrono::high_resolution_clock::now();
            vector<float> sub_keys;
            vector<float> sub_labels;
            int sample_gap = 1 / sqrt(exp_recorder.sampling_rate);
            for (int j = 0; j < part_data_size; j++)
            {
                if (j % sample_gap == 0)
                {
                    // features.push_back(point.curve_val);
                    sub_keys.push_back(part_data[j].key);
                    sub_labels.push_back(part_data[j].index);
                }
            }
            auto finish_sp = chrono::high_resolution_clock::now();
            exp_recorder.extra_time += chrono::duration_cast<chrono::nanoseconds>(finish_sp - start_sp).count();
            auto start_train = chrono::high_resolution_clock::now();
            net->train_model(sub_keys, sub_labels);
            auto end_train = chrono::high_resolution_clock::now();
            exp_recorder.training_cost += chrono::duration_cast<chrono::nanoseconds>(end_train - start_train).count();
        }
        // else if (exp_recorder.is_rs)
        // {
        //     auto start_rs = chrono::high_resolution_clock::now();
        //     exp_recorder.rs_num++;
        //     vector<Point> temp_part_data = pre_train_zm::get_rep_set_space(sqrt(exp_recorder.rs_threshold_m), 0, 0, 0.5, 0.5, part_data);
        //     int temp_N = temp_part_data.size();
        //     vector<float> sub_keys;
        //     vector<float> sub_labels;
        //     for (int j = 0; j < temp_N; j++)
        //     {
        //         sub_keys.push_back(temp_part_data[i].key);
        //         sub_labels.push_back(temp_part_data[i].index);
        //     }
        //     auto finish_rs = chrono::high_resolution_clock::now();
        //     exp_recorder.extra_time += chrono::duration_cast<chrono::nanoseconds>(finish_rs - start_rs).count();
        //     auto start_train = chrono::high_resolution_clock::now();
        //     net->train_model(sub_keys, sub_labels);
        //     auto end_train = chrono::high_resolution_clock::now();
        //     exp_recorder.training_cost += chrono::duration_cast<chrono::nanoseconds>(end_train - start_train).count();
        // }
        // else if (exp_recorder.is_cluster)
        // {
        //     exp_recorder.cluster_num++;
        // }
        // else if (exp_recorder.is_rl)
        // {
        // }
        else
        {
            exp_recorder.original_num++;
            auto start_train = chrono::high_resolution_clock::now();
            net->train_model(keys, labels);
            auto end_train = chrono::high_resolution_clock::now();
            exp_recorder.training_cost += chrono::duration_cast<chrono::nanoseconds>(end_train - start_train).count();
        }
        // net->train_model(keys, labels);
        net->get_parameters_ZM();

        map<int, int> entries_count;

        vector<int> positions;
        vector<float> predicts;
        for (size_t j = 0; j < part_data_size; j++)
        {
            // cout << "keys[j]: " << keys[j] << endl;
            // cout << "net->predict_ZM(keys[j]): " << net->predict_ZM(keys[j]) << endl;
            predicts.push_back(net->predict_ZM(keys[j]));
            int idx = net->predict_ZM(keys[j]) * part_data_size / page_size;
            positions.push_back(idx);
            entries_count[idx]++;
        }
        // TODO comment it when record time
        // if (!check_monotony(positions))
        // {
        //     if (!check_monotony1(predicts))
        //     {
        //         cout << "check_monotony1: i " << i << endl;
        //     }
        // }
        map<int, int>::iterator iter = entries_count.begin();
        int start_idx = 0;
        int end_idx = 0;
        shard_start_id_each_model.push_back(shard_id);
        map<int, Shard> model_shards;
        while (iter != entries_count.end())
        {
            int idx = iter->first;
            int shard_size = iter->second;
            end_idx += shard_size;

            vector<Point> shard_points(part_data.begin() + start_idx, part_data.begin() + end_idx);
            Shard shard(shard_id++, page_size);
            shard.gen_local_model(shard_points, page_id);
            start_idx = end_idx;
            model_shards.insert(pair<int, Shard>(idx, shard));
            iter++;
            // cout << "------------idx-----------: " << idx << endl;
        }
        // cout << "shard_id: " << shard_id << endl;
        // cout << "page_id: " << page_id << endl;
        start = model_split_idxes[i];
        SP.push_back(net);
        shards.push_back(model_shards);

        // if (i >= 93)
        //     break;
    }
    auto build_end = chrono::high_resolution_clock::now();
    exp_recorder.time = chrono::duration_cast<chrono::nanoseconds>(build_end - build_start).count();
    cout << "build time: " << exp_recorder.time << endl;
}

void LISA::point_query(ExpRecorder &exp_recorder, Point query_point)
{
    double key = get_mapped_key(query_point, get_partition_index(query_point));
    query_point.key = key;
    // cout << "key: " << key << " partition: " << get_partition_index(query_point) << endl;
    int model_index = get_model_index(key);
    // TODO get
    int partition_size = partition_sizes[model_index];
    auto start = chrono::high_resolution_clock::now();
    int shard_index = SP[model_index]->predict_ZM(key - min_key_list[model_index]) * partition_size / page_size;
    auto end = chrono::high_resolution_clock::now();
    exp_recorder.prediction_time += chrono::duration_cast<chrono::nanoseconds>(end - start).count();

    // cout << "model_index: " << model_index << " shard_index: " << shard_index << endl;

    auto it = shards[model_index].find(shard_index);
    start = chrono::high_resolution_clock::now();
    if (!it->second.search_point(exp_recorder, query_point))
    {
        exp_recorder.point_not_found++;
        // query_point.print();
        // (x=0.195941,y=0.352417) index=0 curve_val=0
    }
    end = chrono::high_resolution_clock::now();
    exp_recorder.search_time += chrono::duration_cast<chrono::nanoseconds>(end - start).count();
}

void LISA::point_query(ExpRecorder &exp_recorder, vector<Point> query_points)
{
    cout << "point_query:" << query_points.size() << endl;
    auto start = chrono::high_resolution_clock::now();
    long res = 0;
    for (long i = 0; i < query_points.size(); i++)
    {
        // cout << "--------------" << i << "---------------" << endl;

        // if (abs(query_points[i].x - 0.195941) < 0.000001 && abs(query_points[i].y - 0.352417) < 0.000001)
        // {
        auto start1 = chrono::high_resolution_clock::now();
        point_query(exp_recorder, query_points[i]);
        auto finish1 = chrono::high_resolution_clock::now();
        res += chrono::duration_cast<chrono::nanoseconds>(finish1 - start1).count();
        // }
    }
    auto finish = chrono::high_resolution_clock::now();
    exp_recorder.time = res / query_points.size();
    exp_recorder.page_access /= query_points.size();
    exp_recorder.search_time /= query_points.size();
    exp_recorder.prediction_time /= query_points.size();
    cout << "finish point_query time: " << exp_recorder.time << endl;
    cout << "point_not_found: " << exp_recorder.point_not_found << endl;
}

void LISA::window_query(ExpRecorder &exp_recorder, vector<Mbr> query_windows)
{
    exp_recorder.is_window = true;
    long long time_cost = 0;
    int length = query_windows.size();
    for (int i = 0; i < length; i++)
    {
        auto start = chrono::high_resolution_clock::now();
        window_query(exp_recorder, query_windows[i], -1);
        auto finish = chrono::high_resolution_clock::now();
        exp_recorder.window_query_result_size += exp_recorder.window_query_results.size();
        exp_recorder.window_query_results.clear();
        exp_recorder.window_query_results.shrink_to_fit();
        exp_recorder.time += chrono::duration_cast<chrono::nanoseconds>(finish - start).count();
    }
    // cout << "exp_recorder.window_query_result_size: " << exp_recorder.window_query_result_size << endl;
    exp_recorder.time /= length;
    exp_recorder.page_access = (double)exp_recorder.page_access / length;
    exp_recorder.prediction_time /= length;
    cout << "finish window_query time: " << exp_recorder.time << endl;
    cout << "finish page_access time: " << exp_recorder.page_access << endl;
    cout << "finish exp_recorder.window_query_result_size: " << exp_recorder.window_query_result_size << endl;
}

vector<Point> LISA::window_query(ExpRecorder &exp_recorder, Mbr query_window, int partition_index)
{
    vector<Point> result;
    Point pl(query_window.x1, query_window.y1);
    Point pu(query_window.x2, query_window.y2);
    int partition_l = partition_index == -1 ? get_partition_index(pl) : partition_index;
    int partition_u = partition_index == -1 ? get_partition_index(pu) : partition_index;
    if (partition_l == partition_u)
    {
        double key_l = get_mapped_key(pl, partition_l);
        int start_model_index = get_model_index(key_l);
        int start_partition_size = partition_sizes[start_model_index];
        int start_shard_index = SP[start_model_index]->predict_ZM(key_l - min_key_list[start_model_index]) * start_partition_size / page_size + shard_start_id_each_model[start_model_index];
        // auto it = shards[start_model_index].find(start_shard_index);
        double key_u = get_mapped_key(pu, partition_u);
        int end_model_index = get_model_index(key_u);
        int end_partition_size = partition_sizes[end_model_index];
        int end_shard_index = SP[end_model_index]->predict_ZM(key_u - min_key_list[end_model_index]) * end_partition_size / page_size + shard_start_id_each_model[end_model_index];

        auto it_start = shards[start_model_index].find(start_shard_index);
        auto it_end = shards[start_model_index].end();
        while (it_start != shards[start_model_index].end())
        {
            it_start->second.window_query(exp_recorder, query_window);
            it_start++;
        }

        for (size_t i = start_model_index + 1; i < end_model_index; i++)
        {
            it_start = shards[i].begin();
            it_end = shards[i].end();
            while (it_start != it_end)
            {
                it_start->second.window_query(exp_recorder, query_window);
                it_start++;
            }
        }

        it_start = shards[end_model_index].begin();
        it_end = shards[end_model_index].find(start_shard_index);
        it_end++;
        while (it_start != it_end)
        {
            it_start->second.window_query(exp_recorder, query_window);
            it_start++;
        }
        // it_end->second.window_query(exp_recorder, query_window);
    }
    else
    {
        Mbr first_query_window = query_window;
        first_query_window.x2 = x_split_points[partition_l];
        vector<Point> first_result = window_query(exp_recorder, first_query_window, partition_l);
        result.insert(result.end(), first_result.begin(), first_result.end());
        for (size_t i = partition_l + 1; i < partition_u; i++)
        {
            Mbr new_query_window = query_window;
            new_query_window.x1 = x_split_points[i - 1];
            new_query_window.x2 = x_split_points[i];
            window_query(exp_recorder, new_query_window, i);
            // vector<Point> temp_result = window_query(exp_recorder, new_query_window);
            // result.insert(result.end(), temp_result.begin(), temp_result.end());
        }
        Mbr last_query_window = query_window;
        last_query_window.x1 = x_split_points[partition_u - 1];
        window_query(exp_recorder, last_query_window, partition_u);
        // vector<Point> last_result = window_query(exp_recorder, last_query_window);
        // result.insert(result.end(), last_result.begin(), last_result.end());
    }
    return result;
}

void LISA::kNN_query(ExpRecorder &exp_recorder, vector<Point> query_points, int k)
{
    exp_recorder.is_knn = true;
    for (int i = 0; i < query_points.size(); i++)
    {
        auto start = chrono::high_resolution_clock::now();
        vector<Point> knn_result = kNN_query(exp_recorder, query_points[i], k);
        auto finish = chrono::high_resolution_clock::now();
        exp_recorder.window_query_results.clear();
        exp_recorder.window_query_results.shrink_to_fit();
        exp_recorder.time += chrono::duration_cast<chrono::nanoseconds>(finish - start).count();
        exp_recorder.knn_query_results.insert(exp_recorder.knn_query_results.end(), knn_result.begin(), knn_result.end());
        // cout << "knnDiff: " << knnDiff(accKNNQuery(exp_recorder, query_points[i], k), kNN_query(exp_recorder, query_points[i], k)) << endl;
    }
    exp_recorder.time /= query_points.size();
    exp_recorder.search_time /= query_points.size();
    exp_recorder.search_length /= query_points.size();
    exp_recorder.prediction_time /= query_points.size();
    exp_recorder.page_access = (double)exp_recorder.page_access / query_points.size();
}

vector<Point> LISA::kNN_query(ExpRecorder &exp_recorder, Point query_point, int k)
{
    float knn_query_side = sqrt((float)k / N) * 2;
    vector<Point> result;
    while (true)
    {
        vector<Point> temp_result;
        Mbr mbr = Mbr::get_mbr(query_point, knn_query_side);
        window_query(exp_recorder, mbr, -1);
        vector<Point> window_result = exp_recorder.window_query_results;

        for (size_t i = 0; i < window_result.size(); i++)
        {
            if (window_result[i].cal_dist(query_point) <= knn_query_side)
            {
                temp_result.push_back(window_result[i]);
            }
        }
        if (temp_result.size() < k)
        {
            if (temp_result.size() == 0)
            {
                knn_query_side *= 2;
            }
            else
            {
                knn_query_side *= sqrt((float)k / temp_result.size());
            }
            exp_recorder.window_query_results.clear();
            exp_recorder.window_query_results.shrink_to_fit();
            continue;
        }
        else
        {
            sort(temp_result.begin(), temp_result.end(), sort_for_kNN(query_point));
            vector<Point> vec(temp_result.begin(), temp_result.begin() + k);
            result = vec;
            break;
        }
    }
    return result;
}

bool LISA::insert(ExpRecorder &exp_recorder, vector<Point> &inserted_points)
{
    auto start = chrono::high_resolution_clock::now();
    for (int i = 0; i < inserted_points.size(); i++)
    {
        insert(exp_recorder, inserted_points[i]);
    }
    exp_recorder.previous_insert_num += inserted_points.size();
    // cout << "is_rebuild: " << rebuild_index::is_rebuild(exp_recorder, "Z") << endl;
    auto finish = chrono::high_resolution_clock::now();
    bool is_rebuild = rebuild_index::is_rebuild(exp_recorder, "Z");
    long long previous_time = exp_recorder.insert_time * exp_recorder.previous_insert_num;
    exp_recorder.insert_time = (previous_time + chrono::duration_cast<chrono::nanoseconds>(finish - start).count()) / exp_recorder.previous_insert_num;
    return is_rebuild;
}

void LISA::insert(ExpRecorder &exp_recorder, Point point)
{
    exp_recorder.update_num++;
    double key = get_mapped_key(point, get_partition_index(point));
    point.key = key;
    // cout << "key: " << key << " partition: " << get_partition_index(query_point) << endl;
    int model_index = get_model_index(key);
    exp_recorder.changing_histogram.update(key);
    int partition_size = partition_sizes[model_index];
    auto start = chrono::high_resolution_clock::now();
    int shard_index = SP[model_index]->predict_ZM(key - min_key_list[model_index]) * partition_size / page_size;
    auto end = chrono::high_resolution_clock::now();
    exp_recorder.prediction_time += chrono::duration_cast<chrono::nanoseconds>(end - start).count();

    // cout << "model_index: " << model_index << " shard_index: " << shard_index << endl;

    auto it = shards[model_index].find(shard_index);
    if (it != shards[model_index].end())
    {
        it->second.insert(exp_recorder, point);
    }
    else
    {
        vector<Point> shard_points;
        shard_points.push_back(point);
        Shard shard(0, page_size);
        shard.gen_local_model(shard_points, shard_id);
        shards[model_index].insert(pair<int, Shard>(shard_index, shard));
    }
}

bool LISA::check_monotony(vector<int> positions)
{
    // for (vector<int>::const_iterator i = positions.begin(); i != positions.end(); ++i)
    // {
    //     cout << *i << ' ';
    // }
    if (positions.size() == 0)
    {
        return true;
    }
    int idx = positions[0];
    for (size_t i = 1; i < positions.size(); i++)
    {
        if (positions[i] < idx)
        {
            cout << "i: " << i << " positions[i]: " << positions[i] << " idx: " << idx << endl;
            return false;
        }
        idx = positions[i];
    }
    return true;
}

bool LISA::check_monotony1(vector<float> predicts)
{
    // for (vector<int>::const_iterator i = positions.begin(); i != positions.end(); ++i)
    // {
    //     cout << *i << ' ';
    // }
    if (predicts.size() == 0)
    {
        return true;
    }
    float idx = predicts[0];
    for (size_t i = 1; i < predicts.size(); i++)
    {
        if (predicts[i] < idx)
        {
            cout << "i: " << i << " idx: " << idx << " positions[i]: " << predicts[i] << endl;
            return false;
        }
        idx = predicts[i];
    }
    return true;
}

bool LISA::check_monotony2(vector<double> predicts)
{
    // for (vector<int>::const_iterator i = positions.begin(); i != positions.end(); ++i)
    // {
    //     cout << *i << ' ';
    // }
    if (predicts.size() == 0)
    {
        return true;
    }
    double idx = predicts[0];
    for (size_t i = 1; i < predicts.size(); i++)
    {
        if (predicts[i] < idx)
        {
            cout << "i: " << i << " idx: " << idx << " positions[i]: " << predicts[i] << endl;
            return false;
        }
        idx = predicts[i];
    }
    return true;
}

// TODO clean all objects
void LISA::clear(ExpRecorder &exp_recorder)
{

    shard_id = 0;
    page_id = 0;

    split_points.clear();
    split_points.shrink_to_fit();

    split_index_list.clear();
    split_index_list.shrink_to_fit();

    SP.clear();

    shards.clear();

    mappings.clear();
    mappings.shrink_to_fit();

    borders.clear();
    borders.shrink_to_fit();

    gaps.clear();
    gaps.shrink_to_fit();

    x_split_points.clear();
    x_split_points.shrink_to_fit();

    model_split_mapping.clear();
    model_split_mapping.shrink_to_fit();

    shard_start_id_each_model.clear();
    shard_start_id_each_model.shrink_to_fit();

    partition_sizes.clear();
    partition_sizes.shrink_to_fit();

    min_key_list.clear();
    min_key_list.shrink_to_fit();
}