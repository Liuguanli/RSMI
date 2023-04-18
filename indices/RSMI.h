#include <iostream>
#include <vector>
#include "../entities/Node.h"
#include "../entities/Point.h"
#include "../entities/Mbr.h"
#include "../entities/NonLeafNode.h"
#include "../entities/LeafNode.h"
#include <typeinfo>
#include "../utils/ExpRecorder.h"
#include "../utils/SortTools.h"
#include "../utils/ModelTools.h"
#include "../curves/hilbert.H"
#include "../curves/hilbert4.H"
#include "../curves/z.H"
#include <map>
#include <boost/smart_ptr/make_shared_object.hpp>
#include <torch/script.h>
#include <ATen/ATen.h>
#include <torch/torch.h>

#include <torch/optim.h>
#include <torch/types.h>
#include <torch/utils.h>

using namespace at;
using namespace torch::nn;
using namespace torch::optim;
using namespace std;

class RSMI
{

private:
    int level;
    int index;
    int max_partition_num;
    long long N = 0;
    int max_error = 0;
    int min_error = 0;
    int width = 0;
    int leaf_node_num;
    
    bool is_last;
    Mbr mbr;
    std::shared_ptr<Net> net;

public:
    string model_path;
    static string model_path_root;
    map<int, RSMI> children;
    vector<LeafNode> leafnodes;

    RSMI();
    RSMI(int index, int max_partition_num);
    RSMI(int index, int level, int max_partition_num);
    void build(ExpRecorder &exp_recorder, vector<Point> points);
    void print_index_info(ExpRecorder &exp_recorder);

    bool point_query(ExpRecorder &exp_recorder, Point query_point);
    void point_query(ExpRecorder &exp_recorder, vector<Point> query_points);

    void window_query(ExpRecorder &exp_recorder, vector<Mbr> query_windows);
    // vector<Point> window_query(ExpRecorder &exp_recorder, Mbr query_window);
    void window_query(ExpRecorder &exp_recorder, vector<Point> vertexes, Mbr query_window, float boundary, int k, Point query_point, float &);
    void window_query(ExpRecorder &exp_recorder, vector<Point> vertexes, Mbr query_window);
    void acc_window_query(ExpRecorder &exp_recorder, vector<Mbr> query_windows);
    vector<Point> acc_window_query(ExpRecorder &exp_recorder, Mbr query_windows);

    void kNN_query(ExpRecorder &exp_recorder, vector<Point> query_points, int k);
    vector<Point> kNN_query(ExpRecorder &exp_recorder, Point query_point, int k);
    void acc_kNN_query(ExpRecorder &exp_recorder, vector<Point> query_points, int k);
    vector<Point> acc_kNN_query(ExpRecorder &exp_recorder, Point query_point, int k);
    double cal_rho(Point point);
    double knn_diff(vector<Point> acc, vector<Point> pred);

    void insert(ExpRecorder &exp_recorder, Point);
    void insert(ExpRecorder &exp_recorder, vector<Point>);

    void remove(ExpRecorder &exp_recorder, Point);
    void remove(ExpRecorder &exp_recorder, vector<Point>);
};

RSMI::RSMI()
{
    // leafnodes = vector<LeafNode>(10);
}

RSMI::RSMI(int index, int max_partition_num)
{
    this->index = index;
    this->max_partition_num = max_partition_num;
    this->level = 0;
}

RSMI::RSMI(int index, int level, int max_partition_num)
{
    this->index = index;
    this->level = level;
    this->max_partition_num = max_partition_num;
}

void RSMI::build(ExpRecorder &exp_recorder, vector<Point> points)
{

    int page_size = Constants::PAGESIZE;
    auto start = chrono::high_resolution_clock::now();
    if (points.size() <= exp_recorder.N)
    {
        this->model_path += "_" + to_string(level) + "_" + to_string(index);
        if (exp_recorder.depth < level)
        {
            exp_recorder.depth = level;
        }
        exp_recorder.last_level_model_num++;
        is_last = true;
        N = points.size();
        long long side = pow(2, ceil(log(points.size()) / log(2)));
        sort(points.begin(), points.end(), sortX());
        for (int i = 0; i < N; i++)
        {
            points[i].x_i = i;
            mbr.update(points[i].x, points[i].y);
        }
        sort(points.begin(), points.end(), sortY());
        for (int i = 0; i < N; i++)
        {
            points[i].y_i = i;
            long long curve_val = compute_Hilbert_value(points[i].x_i, points[i].y_i, side);
            points[i].curve_val = curve_val;
        }
        sort(points.begin(), points.end(), sort_curve_val());
        width = N - 1;
        if (N == 1)
        {
            points[0].index = 0;
        }
        else
        {
            for (long i = 0; i < N; i++)
            {
                points[i].index = i * 1.0 / (N - 1);
            }
        }
        leaf_node_num = points.size() / page_size;
        for (int i = 0; i < leaf_node_num; i++)
        {
            LeafNode leafNode;
            auto bn = points.begin() + i * page_size;
            auto en = points.begin() + i * page_size + page_size;
            vector<Point> vec(bn, en);
            // cout << vec.size() << " " << vec[0]->x_i << " " << vec[99]->x_i << endl;
            leafNode.add_points(vec);
            leafnodes.push_back(leafNode);
        }

        // for the last leafNode
        if (points.size() > page_size * leaf_node_num)
        {
            // TODO if do not delete will it last to the end of lifecycle?
            LeafNode leafNode;
            auto bn = points.begin() + page_size * leaf_node_num;
            auto en = points.end();
            vector<Point> vec(bn, en);
            leafNode.add_points(vec);
            leafnodes.push_back(leafNode);
            leaf_node_num++;
        }
        exp_recorder.leaf_node_num += leaf_node_num;
        net = std::make_shared<Net>(2, leaf_node_num / 2 + 2);
        #ifdef use_gpu
            net->to(torch::kCUDA);
        #endif
        vector<float> locations;
        vector<float> labels;
        for (Point point : points)
        {
            locations.push_back(point.x);
            locations.push_back(point.y);
            labels.push_back(point.index);
        }
        
        std::ifstream fin(this->model_path);
        if (!fin)
        {
            net->train_model(locations, labels);
            // torch::save(net, this->model_path);
        }
        else
        {
            torch::load(net, this->model_path);
        }
        net->get_parameters();

        exp_recorder.non_leaf_node_num++;
        for (int i = 0; i < N; i++)
        {
            Point point = points[i];
            int predicted_index = (int)(net->predict(point) * leaf_node_num);
            predicted_index = predicted_index < 0 ? 0 : predicted_index;
            predicted_index = predicted_index >= leaf_node_num ? leaf_node_num - 1 : predicted_index;

            int error = i / page_size - predicted_index;

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
        exp_recorder.average_max_error += max_error;
        exp_recorder.average_min_error += min_error;
        if ((max_error - min_error) > (exp_recorder.max_error - exp_recorder.min_error))
        {
            exp_recorder.max_error = max_error;
            exp_recorder.min_error = min_error;
        }

    }
    else
    {
        is_last = false;
        N = (long long)points.size();
        int bit_num = max_partition_num;
        int partition_size = ceil(points.size() * 1.0 / pow(bit_num, 2));
        sort(points.begin(), points.end(), sortX());
        long long side = pow(bit_num, 2);
        width = side - 1;
        map<int, vector<Point>> points_map;
        int each_item_size = partition_size * bit_num;
        long long point_index = 0;

        vector<float> locations(N * 2);
        vector<float> labels(N);

        for (size_t i = 0; i < bit_num; i++)
        {
            long long bn_index = i * each_item_size;
            long long end_index = bn_index + each_item_size;
            if (bn_index >= N)
            {
                break;
            }
            else
            {
                if (end_index > N)
                {
                    end_index = N;
                }
            }
            auto bn = points.begin() + bn_index;
            auto en = points.begin() + end_index;
            vector<Point> vec(bn, en);
            sort(vec.begin(), vec.end(), sortY());
            for (size_t j = 0; j < bit_num; j++)
            {
                long long sub_bn_index = j * partition_size;
                long long sub_end_index = sub_bn_index + partition_size;
                if (sub_bn_index >= vec.size())
                {
                    break;
                }
                else
                {
                    if (sub_end_index > vec.size())
                    {
                        sub_end_index = vec.size();
                    }
                }
                auto sub_bn = vec.begin() + sub_bn_index;
                auto sub_en = vec.begin() + sub_end_index;
                vector<Point> sub_vec(sub_bn, sub_en);
                int Z_value = compute_Z_value(i, j, side);
                int sub_point_index = 1;
                long sub_size = sub_vec.size();
                for (Point point : sub_vec)
                {
                    point.index = Z_value * 1.0 / width;
                    locations[point_index * 2] = point.x;
                    locations[point_index * 2 + 1] = point.y;
                    labels[point_index] = point.index;
                    point_index++;
                    mbr.update(point.x, point.y);
                    sub_point_index++;
                }
                vector<Point> temp;
                points_map.insert(pair<int, vector<Point>>(Z_value, temp));
            }
        }

        int epoch = Constants::START_EPOCH;
        bool is_retrain = false;
        do
        {
            net = std::make_shared<Net>(2);
            #ifdef use_gpu
                net->to(torch::kCUDA);
            #endif

            this->model_path += "_" + to_string(level) + "_" + to_string(index);
            std::ifstream fin(this->model_path);
                net->train_model(locations, labels);

            // if (!fin)
            // {
            //     net->train_model(locations, labels);
            //     // torch::save(net, this->model_path);
            // }
            // else
            // {
            //     torch::load(net, this->model_path);
            // }
            net->get_parameters();

            for (Point point : points)
            {
                int predicted_index = (int)(net->predict(point) * width);

                predicted_index = predicted_index < 0 ? 0 : predicted_index;
                predicted_index = predicted_index >= width ? width - 1 : predicted_index;
                points_map[predicted_index].push_back(point);
            }

            map<int, vector<Point>>::iterator iter1;
            iter1 = points_map.begin();
            int map_size = 0;
            while (iter1 != points_map.end())
            {
                if (iter1->second.size() > 0)
                {
                    map_size++;
                }
                iter1++;
            }
            if (map_size < 2)
            {
                int predicted_index = (int)(net->predict(points[0]) * width);
                predicted_index = predicted_index < 0 ? 0 : predicted_index;
                predicted_index = predicted_index >= width ? width - 1 : predicted_index;

                points_map[predicted_index].clear();
                points_map[predicted_index].shrink_to_fit();
                is_retrain = true;
                epoch = Constants::EPOCH_ADDED;
            }
            else
            {
                is_retrain = false;
            }

        } while (is_retrain);
        auto finish = chrono::high_resolution_clock::now();
        
        exp_recorder.non_leaf_node_num++;

        points.clear();
        points.shrink_to_fit();

        map<int, vector<Point>>::iterator iter;
        iter = points_map.begin();

        while (iter != points_map.end())
        {
            if (iter->second.size() > 0)
            {
                RSMI partition(iter->first, level + 1, max_partition_num);
                partition.model_path = model_path;
                partition.build(exp_recorder, iter->second);
                iter->second.clear();
                iter->second.shrink_to_fit();
                children.insert(pair<int, RSMI>(iter->first, partition));
            }
            iter++;
        }
    }
}

void RSMI::print_index_info(ExpRecorder &exp_recorder)
{
    cout << "finish point_query max_error: " << exp_recorder.max_error << endl;
    cout << "finish point_query min_error: " << exp_recorder.min_error << endl;
    cout << "finish point_query average_max_error: " << exp_recorder.average_max_error << endl;
    cout << "finish point_query average_min_error: " << exp_recorder.average_min_error << endl;
    cout << "last_level_model_num: " << exp_recorder.last_level_model_num << endl;
    cout << "leaf_node_num: " << exp_recorder.leaf_node_num << endl;
    cout << "non_leaf_node_num: " << exp_recorder.non_leaf_node_num << endl;
    cout << "depth: " << exp_recorder.depth << endl;
}

bool RSMI::point_query(ExpRecorder &exp_recorder, Point query_point)
{
    if (is_last)
    {
        int predicted_index = 0;
        predicted_index = net->predict(query_point) * leaf_node_num;
        predicted_index = predicted_index < 0 ? 0 : predicted_index;
        predicted_index = predicted_index >= leaf_node_num ? leaf_node_num - 1 : predicted_index;
        LeafNode leafnode = leafnodes[predicted_index];
        if (leafnode.mbr.contains(query_point))
        {
            exp_recorder.page_access += 1;
            vector<Point>::iterator iter = find(leafnode.children->begin(), leafnode.children->end(), query_point);
            if (iter != leafnode.children->end())
            {
                return true;
            }
        }
        // predicted result is not correct
        int front = predicted_index + min_error;
        front = front < 0 ? 0 : front;
        int back = predicted_index + max_error;
        back = back >= leaf_node_num ? leaf_node_num - 1 : back;

        int gap = 1;
        int predicted_index_left = predicted_index - gap;
        int predicted_index_right = predicted_index + gap;
        while (predicted_index_left >= front && predicted_index_right <= back)
        {
            // search left
            LeafNode leafnode = leafnodes[predicted_index_left];
            if (leafnode.mbr.contains(query_point))
            {
                exp_recorder.page_access += 1;
                for (Point point : (*leafnode.children))
                {
                    if (query_point.x == point.x && query_point.y == point.y)
                    {
                        return true;
                    }
                }
            }

            // search right
            leafnode = leafnodes[predicted_index_right];
            if (leafnode.mbr.contains(query_point))
            {
                exp_recorder.page_access += 1;
                for (Point point : (*leafnode.children))
                {
                    if (query_point.x == point.x && query_point.y == point.y)
                    {
                        return true;
                    }
                }
            }
            gap++;
            predicted_index_left = predicted_index - gap;
            predicted_index_right = predicted_index + gap;
        }

        while (predicted_index_left >= front)
        {
            LeafNode leafnode = leafnodes[predicted_index_left];

            if (leafnode.mbr.contains(query_point))
            {
                exp_recorder.page_access += 1;
                for (Point point : (*leafnode.children))
                {
                    if (query_point.x == point.x && query_point.y == point.y)
                    {
                        return true;
                    }
                }
            }
            gap++;
            predicted_index_left = predicted_index - gap;
        }

        while (predicted_index_right <= back)
        {
            LeafNode leafnode = leafnodes[predicted_index_right];

            if (leafnode.mbr.contains(query_point))
            {
                exp_recorder.page_access += 1;
                for (Point point : (*leafnode.children))
                {
                    if (query_point.x == point.x && query_point.y == point.y)
                    {
                        return true;
                    }
                }
            }
            gap++;
            predicted_index_right = predicted_index + gap;
        }
        // cout<< "not find" << endl;
        // query_point.print();
        return false;
    }
    else
    {
        int predicted_index = net->predict(query_point) * width;
        predicted_index = predicted_index < 0 ? 0 : predicted_index;
        predicted_index = predicted_index >= width ? width - 1 : predicted_index;
        if (children.count(predicted_index) == 0)
        {
            return false;
        }
        return children[predicted_index].point_query(exp_recorder, query_point);
    }
}

void RSMI::point_query(ExpRecorder &exp_recorder, vector<Point> query_points)
{
    long size = query_points.size();
    for (long i = 0; i < size; i++)
    {
        auto start = chrono::high_resolution_clock::now();
        point_query(exp_recorder, query_points[i]);
        auto finish = chrono::high_resolution_clock::now();
        exp_recorder.time += chrono::duration_cast<chrono::nanoseconds>(finish - start).count();
    }
    exp_recorder.time /= size;
    exp_recorder.page_access = exp_recorder.page_access / size;
}

void RSMI::window_query(ExpRecorder &exp_recorder, vector<Mbr> query_windows)
{
    long long time_cost = 0;
    int length = query_windows.size();
    // length = 1;
    for (int i = 0; i < length; i++)
    {
        vector<Point> vertexes = query_windows[i].get_corner_points();
        auto start = chrono::high_resolution_clock::now();
        window_query(exp_recorder, vertexes, query_windows[i]);
        auto finish = chrono::high_resolution_clock::now();
        exp_recorder.window_query_result_size += exp_recorder.window_query_results.size();
        exp_recorder.window_query_results.clear();
        exp_recorder.window_query_results.shrink_to_fit();
        exp_recorder.time += chrono::duration_cast<chrono::nanoseconds>(finish - start).count();
    }
    exp_recorder.time /= length;
    exp_recorder.page_access = (double)exp_recorder.page_access / length;
}

void RSMI::window_query(ExpRecorder &exp_recorder, vector<Point> vertexes, Mbr query_window)
{
    if (is_last)
    {
        int leafnodes_size = leafnodes.size();
        int front = leafnodes_size - 1;
        int back = 0;
        if (leaf_node_num == 0)
        {
            return;
        }
        else if (leaf_node_num < 2)
        {
            front = 0;
            back = 0;
        }
        else
        {
            int max = 0;
            int min = width;
            for (size_t i = 0; i < vertexes.size(); i++)
            {
                int predicted_index = net->predict(vertexes[i]) * leaf_node_num;
                predicted_index = predicted_index < 0 ? 0 : predicted_index;
                predicted_index = predicted_index > width ? width : predicted_index;
                int predicted_index_max = predicted_index + max_error;
                int predicted_index_min = predicted_index + min_error;
                if (predicted_index_min < min)
                {
                    min = predicted_index_min;
                }
                if (predicted_index_max > max)
                {
                    max = predicted_index_max;
                }
            }

            front = min < 0 ? 0 : min;
            back = max >= leafnodes_size ? leafnodes_size - 1 : max;
        }
        for (size_t i = front; i <= back; i++)
        {
            LeafNode leafnode = leafnodes[i];
            if (leafnode.mbr.interact(query_window))
            {
                exp_recorder.page_access += 1;
                for (Point point : (*leafnode.children))
                {
                    if (query_window.contains(point))
                    {
                        exp_recorder.window_query_results.push_back(point);
                        // exp_recorder.window_query_result_size++;
                    }
                }
            }
        }
        return;
    }
    else
    {
        int children_size = children.size();
        int front = children_size - 1;
        int back = 0;
        for (size_t i = 0; i < vertexes.size(); i++)
        {
            int predicted_index = net->predict(vertexes[i]) * children.size();
            predicted_index = predicted_index < 0 ? 0 : predicted_index;
            predicted_index = predicted_index >= children_size ? children_size - 1 : predicted_index;
            if (predicted_index < front)
            {
                front = predicted_index;
            }
            if (predicted_index > back)
            {
                back = predicted_index;
            }
        }

        for (size_t i = front; i <= back; i++)
        {
            if (children.count(i) == 0)
            {
                continue;
            }
            if (children[i].mbr.interact(query_window))
            {
                children[i].window_query(exp_recorder, vertexes, query_window);
            }
        }
    }
}

// this method is for knn query
void RSMI::window_query(ExpRecorder &exp_recorder, vector<Point> vertexes, Mbr query_window, float boundary, int k, Point query_point, float &kth)
{
    if (is_last)
    {
        int leafnodesSize = leafnodes.size();
        int front = leafnodesSize - 1;
        int back = 0;
        if (leaf_node_num == 0)
        {
            return;
        }
        else if (leaf_node_num < 2)
        {
            front = 0;
            back = 0;
        }
        else
        {
            int max = 0;
            int min = width;
            for (size_t i = 0; i < vertexes.size(); i++)
            {
                auto start = chrono::high_resolution_clock::now();
                int predicted_index = net->predict(vertexes[i]) * leaf_node_num;
                auto finish = chrono::high_resolution_clock::now();
                predicted_index = predicted_index < 0 ? 0 : predicted_index;
                predicted_index = predicted_index > width ? width : predicted_index;
                int predictedIndexMax = predicted_index + max_error;
                int predictedIndexMin = predicted_index + min_error;
                if (predictedIndexMin < min)
                {
                    min = predictedIndexMin;
                }
                if (predictedIndexMax > max)
                {
                    max = predictedIndexMax;
                }
            }

            front = min < 0 ? 0 : min;
            back = max >= leafnodesSize ? leafnodesSize - 1 : max;
        }
        for (size_t i = front; i <= back; i++)
        {
            LeafNode leafnode = leafnodes[i];
            float dis = leafnode.mbr.cal_dist(query_point);
            if (dis > boundary)
            {
                continue;
            }
            if (exp_recorder.pq.size() >= k && dis > kth)
            {
                continue;
            }
            if (leafnode.mbr.interact(query_window))
            {
                exp_recorder.page_access += 1;
                for (Point point : (*leafnode.children))
                {
                    if (query_window.contains(point))
                    {
                        if (point.cal_dist(query_point) <= boundary)
                        {
                            exp_recorder.pq.push(point);
                        }
                    }
                }
            }
        }
        return;
    }
    else
    {
        int front = width;
        int back = 0;
        for (size_t i = 0; i < vertexes.size(); i++)
        {
            auto start = chrono::high_resolution_clock::now();
            int predicted_index = net->predict(vertexes[i]) * width;
            auto finish = chrono::high_resolution_clock::now();
            predicted_index = predicted_index < 0 ? 0 : predicted_index;
            predicted_index = predicted_index >= width ? width - 1 : predicted_index;
            if (predicted_index < front)
            {
                front = predicted_index;
            }
            if (predicted_index > back)
            {
                back = predicted_index;
            }
        }
        for (size_t i = front; i <= back; i++)
        {
            if (children.count(i) == 0)
            {
                continue;
            }
            if (exp_recorder.pq.size() >= k && children[i].mbr.cal_dist(query_point) > kth)
            {
                continue;
            }
            if (children[i].mbr.interact(query_window))
            {
                children[i].window_query(exp_recorder, vertexes, query_window, boundary, k, query_point, kth);
            }
        }
    }
}

vector<Point> RSMI::acc_window_query(ExpRecorder &exp_recorder, Mbr query_window)
{
    vector<Point> window_query_results;
    if (is_last)
    {
        for (LeafNode leafnode : leafnodes)
        {
            if (leafnode.mbr.interact(query_window))
            {
                exp_recorder.page_access += 1;
                for (Point point : (*leafnode.children))
                {
                    if (query_window.contains(point))
                    {
                        window_query_results.push_back(point);
                    }
                }
            }
        }
    }
    else
    {
        map<int, RSMI>::iterator iter = children.begin();
        while (iter != children.end())
        {
            if (iter->second.mbr.interact(query_window))
            {
                vector<Point> tempResult = iter->second.acc_window_query(exp_recorder, query_window);
                window_query_results.insert(window_query_results.end(), tempResult.begin(), tempResult.end());
            }
            iter++;
        }
    }
    return window_query_results;
}

void RSMI::acc_window_query(ExpRecorder &exp_recorder, vector<Mbr> query_windows)
{
    int length = query_windows.size();
    for (int i = 0; i < length; i++)
    {
        auto start = chrono::high_resolution_clock::now();
        exp_recorder.acc_window_query_qesult_size += acc_window_query(exp_recorder, query_windows[i]).size();
        auto finish = chrono::high_resolution_clock::now();
        exp_recorder.time += chrono::duration_cast<chrono::nanoseconds>(finish - start).count();
    }
    exp_recorder.time = exp_recorder.time / length;
    exp_recorder.page_access = (double)exp_recorder.page_access / length;
}

void RSMI::kNN_query(ExpRecorder &exp_recorder, vector<Point> query_points, int k)
{
    int length = query_points.size();
    exp_recorder.time = 0;
    exp_recorder.page_access = 0;
    // length = 2;
    for (int i = 0; i < length; i++)
    {
        priority_queue<Point , vector<Point>, sortForKNN2> temp_pq;
        exp_recorder.pq = temp_pq;
        auto start = chrono::high_resolution_clock::now();
        vector<Point> knnresult = kNN_query(exp_recorder, query_points[i], k);
        auto finish = chrono::high_resolution_clock::now();
        long long temp_time = chrono::duration_cast<chrono::nanoseconds>(finish - start).count();
        exp_recorder.time += temp_time;
        exp_recorder.knn_query_results.insert(exp_recorder.knn_query_results.end(), knnresult.begin(), knnresult.end());
    }
    exp_recorder.time /= length;
    exp_recorder.page_access = (double)exp_recorder.page_access / length;
    exp_recorder.k_num = k;
}

double RSMI::cal_rho(Point point)
{
    // return 1;
    int predicted_index = net->predict(point) * width;
    predicted_index = predicted_index < 0 ? 0 : predicted_index;
    predicted_index = predicted_index >= width ? width - 1 : predicted_index;
    long long bk = 0;
    for (int i = 0; i < predicted_index; i++)
    {
        if (children[i].N == 0)
        {
            return 1;
        }
        bk += children[i].N;
    }
    if (children[predicted_index].N == 0)
    {
        return 1;
    }
    long long bk1 = bk + children[predicted_index].N;
    double result = bk1 * 1.0 / bk;
    // TODO use 4 to avoid a large number
    result = result > 1 ? result : 1;
    result = result > 4 ? 4 : result;
    return result;
}

vector<Point> RSMI::kNN_query(ExpRecorder &exp_recorder, Point query_point, int k)
{
    // double rh0 = cal_rho(query_point);
    // float knnquery_side = sqrt((float)k / N) * rh0;
    vector<Point> result;
    float knnquery_side = sqrt((float)k / N) * 4;
    while (true)
    {
        Mbr mbr = Mbr::get_mbr(query_point, knnquery_side);
        vector<Point> vertexes = mbr.get_corner_points();

        int size = 0;
        float kth = 0.0;
        window_query(exp_recorder, vertexes, mbr, knnquery_side, k, query_point, kth);
        size = exp_recorder.pq.size();
        if (size >= k)
        {
            for (size_t i = 0; i < k; i++)
            {
                result.push_back(exp_recorder.pq.top());
                exp_recorder.pq.pop();
            }
            break;
        }
        knnquery_side *= 2;
    }
    return result;
}

void RSMI::acc_kNN_query(ExpRecorder &exp_recorder, vector<Point> query_points, int k)
{
    int length = query_points.size();
    // length = 1;
    for (int i = 0; i < length; i++)
    {
        auto start = chrono::high_resolution_clock::now();
        vector<Point> knnresult = acc_kNN_query(exp_recorder, query_points[i], k);
        auto finish = chrono::high_resolution_clock::now();
        exp_recorder.time += chrono::duration_cast<chrono::nanoseconds>(finish - start).count();
        exp_recorder.acc_knn_query_results.insert(exp_recorder.acc_knn_query_results.end(), knnresult.begin(), knnresult.end());
    }
    exp_recorder.time /= length;
    exp_recorder.k_num = k;
    exp_recorder.page_access = (double)exp_recorder.page_access / length;
}

vector<Point> RSMI::acc_kNN_query(ExpRecorder &exp_recorder, Point query_point, int k)
{
    vector<Point> result;
    float knnquery_side = sqrt((float)k / N);
    while (true)
    {
        Mbr mbr = Mbr::get_mbr(query_point, knnquery_side);
        vector<Point> tempResult = acc_window_query(exp_recorder, mbr);
        if (tempResult.size() >= k)
        {
            sort(tempResult.begin(), tempResult.end(), sortForKNN(query_point));
            Point last = tempResult[k - 1];
            if (last.cal_dist(query_point) <= knnquery_side)
            {
                // TODO get top K from the vector.
                auto bn = tempResult.begin();
                auto en = tempResult.begin() + k;
                vector<Point> vec(bn, en);
                result = vec;
                break;
            }
        }
        knnquery_side = knnquery_side * 2;
    }
    return result;
}

// TODO when rebuild!!!
void RSMI::insert(ExpRecorder &exp_recorder, Point point)
{
    int predicted_index = net->predict(point) * width;
    predicted_index = predicted_index < 0 ? 0 : predicted_index;
    predicted_index = predicted_index >= width ? width - 1 : predicted_index;
    if (is_last)
    {
        if (N == Constants::THRESHOLD)
        {
            // cout << "rebuild: " << endl;
            is_last = false;
            vector<Point> points;
            for (LeafNode leafNode : leafnodes)
            {
                points.insert(points.end(), leafNode.children->begin(), leafNode.children->end());
            }
            points.push_back(point);
            auto start = chrono::high_resolution_clock::now();
            build(exp_recorder, points);
            auto finish = chrono::high_resolution_clock::now();
            exp_recorder.rebuild_time += chrono::duration_cast<chrono::nanoseconds>(finish - start).count();
            exp_recorder.rebuild_num++;
        }
        else
        {
            int insertedIndex = predicted_index / Constants::PAGESIZE;
            LeafNode leafnode = leafnodes[insertedIndex];
            if (leafnode.is_full())
            {
                LeafNode right = leafnode.split1();
                leafnodes.insert(leafnodes.begin() + insertedIndex + 1, right);
                leaf_node_num++;
            }
            leafnode.add_point(point);
            N++;
            width++;
        }
    }
    else
    {
        if (children.count(predicted_index) == 0)
        {
            // TODO
            return;
        }
        children[predicted_index].insert(exp_recorder, point);
    }
}

void RSMI::insert(ExpRecorder &exp_recorder, vector<Point> points)
{
    auto start = chrono::high_resolution_clock::now();
    for (Point point : points)
    {
        insert(exp_recorder, point);
    }
    auto finish = chrono::high_resolution_clock::now();
    exp_recorder.insert_time = (chrono::duration_cast<chrono::nanoseconds>(finish - start).count()) / exp_recorder.insert_num;
}

void RSMI::remove(ExpRecorder &exp_recorder, Point point)
{
    int predicted_index = net->predict(point) * width;
    predicted_index = predicted_index < 0 ? 0 : predicted_index;
    predicted_index = predicted_index >= N ? N - 1 : predicted_index;
    if (is_last)
    {
        // cout << "predicted_index: " << predicted_index << endl;
        int front = predicted_index + min_error;
        front = front < 0 ? 0 : front;
        int back = predicted_index + max_error;
        back = back >= N ? N - 1 : back;
        front = front / Constants::PAGESIZE;
        back = back / Constants::PAGESIZE;
        for (size_t i = front; i <= back; i++)
        {
            LeafNode leafnode = leafnodes[i];
            vector<Point>::iterator iter = find(leafnode.children->begin(), leafnode.children->end(), point);
            if (leafnode.mbr.contains(point) && leafnode.delete_point(point))
            {
                N--;
                break;
            }
        }
    }
    else
    {
        if (children.count(predicted_index) == 0)
        {
            return;
        }
        children[predicted_index].remove(exp_recorder, point);
    }
}

void RSMI::remove(ExpRecorder &exp_recorder, vector<Point> points)
{
    auto start = chrono::high_resolution_clock::now();
    for (Point point : points)
    {
        remove(exp_recorder, point);
    }
    auto finish = chrono::high_resolution_clock::now();
    long long oldTimeCost = exp_recorder.delete_time * exp_recorder.delete_num;
    exp_recorder.delete_num += points.size();
    exp_recorder.delete_time = (oldTimeCost + oldTimeCost + chrono::duration_cast<chrono::nanoseconds>(finish - start).count()) / exp_recorder.delete_num;
}