#ifndef MODELTOOLS_H
#define MODELTOOLS_H

#include <iostream>
#include <fstream>
#include <vector>
#include <iterator>
#include <string>
#include <algorithm>
#include <random>
#include <math.h>
#include <cmath>

#include <torch/script.h>
#include <ATen/ATen.h>
#include <torch/torch.h>

#include <torch/nn/module.h>
#include <torch/nn/modules/linear.h>
#include <torch/optim.h>
#include <torch/types.h>
#include <torch/utils.h>


#include <xmmintrin.h> //SSE指令集需包含词头文件
// #include <immintrin.h>

#include <dirent.h>

#include "../entities/Feature.h"
#include "../entities/SFC.h"
#include "SimMeasurement.h"
#include "../utils/FileReader.h"
#include "../curves/hilbert.H"
#include "../curves/hilbert4.H"

using namespace at;
using namespace torch::nn;
using namespace torch::optim;
using namespace std;

struct Net : torch::nn::Module
{

public:
    int input_width;
    int max_error = 0;
    int min_error = 0;
    int width = 0;

    float learning_rate = Constants::LEARNING_RATE;

    // float w1[Constants::HIDDEN_LAYER_WIDTH * 2];
    // float w1_[Constants::HIDDEN_LAYER_WIDTH];
    // float w2[Constants::HIDDEN_LAYER_WIDTH];
    // float b1[Constants::HIDDEN_LAYER_WIDTH];

    float *w1_0 = (float *)_mm_malloc(Constants::HIDDEN_LAYER_WIDTH * sizeof(float), 32);
    float *w1_1 = (float *)_mm_malloc(Constants::HIDDEN_LAYER_WIDTH * sizeof(float), 32);
    float *w2_ = (float *)_mm_malloc(Constants::HIDDEN_LAYER_WIDTH * sizeof(float), 32);
    float *b1_ = (float *)_mm_malloc(Constants::HIDDEN_LAYER_WIDTH * sizeof(float), 32);

    float *w1__ = (float *)_mm_malloc(Constants::HIDDEN_LAYER_WIDTH * sizeof(float), 32);

    float b2 = 0.0;

    Net(int input_width)
    {
        this->width = Constants::HIDDEN_LAYER_WIDTH;
        this->input_width = input_width;
        fc1 = register_module("fc1", torch::nn::Linear(input_width, width));
        fc2 = register_module("fc2", torch::nn::Linear(width, 1));
        torch::nn::init::uniform_(fc1->weight, 0, 1);
        torch::nn::init::uniform_(fc2->weight, 0, 1);
        // torch::nn::init::normal_(fc1->weight, 0, 1);
        // torch::nn::init::normal_(fc2->weight, 0, 1);
    }

    // RSMI
    Net(int input_width, int width)
    {
        this->width = width;
        this->width = this->width >= Constants::HIDDEN_LAYER_WIDTH ? Constants::HIDDEN_LAYER_WIDTH : this->width;
        // this->width = Constants::HIDDEN_LAYER_WIDTH;
        // this->width = Constants::HIDDEN_LAYER_WIDTH;
        this->input_width = input_width;
        fc1 = register_module("fc1", torch::nn::Linear(input_width, this->width));
        fc2 = register_module("fc2", torch::nn::Linear(this->width, 1));
        // torch::nn::init::uniform_(fc1->weight, 0, 0.1);
        // torch::nn::init::uniform_(fc2->weight, 0, 0.1);
        torch::nn::init::uniform_(fc1->weight, 0, 1.0 / width);
        torch::nn::init::uniform_(fc1->bias, 0, 1.0 / width);
        torch::nn::init::uniform_(fc2->weight, 0, 1.0 / width);
        // torch::nn::init::normal_(fc1->weight, 0, 1);
        // torch::nn::init::normal_(fc2->weight, 0, 1);
    }

    void get_parameters_ZM()
    {
        torch::Tensor p1 = this->parameters()[0];
        torch::Tensor p2 = this->parameters()[1];
        torch::Tensor p3 = this->parameters()[2];
        torch::Tensor p4 = this->parameters()[3];
        p1 = p1.reshape({width, 1});
        for (size_t i = 0; i < width; i++)
        {
            w1__[i] = p1.select(0, i).item().toFloat();
        }

        p2 = p2.reshape({width, 1});
        for (size_t i = 0; i < width; i++)
        {
            b1_[i] = p2.select(0, i).item().toFloat();
        }

        p3 = p3.reshape({width, 1});
        for (size_t i = 0; i < width; i++)
        {
            w2_[i] = p3.select(0, i).item().toFloat();
        }
        b2 = p4.item().toFloat();
    }

    void get_parameters()
    {
        torch::Tensor p1 = this->parameters()[0];
        torch::Tensor p2 = this->parameters()[1];
        torch::Tensor p3 = this->parameters()[2];
        torch::Tensor p4 = this->parameters()[3];
        p1 = p1.reshape({2 * width, 1});
        for (size_t i = 0; i < width; i++)
        {
            // w1[i * 2] = p1.select(0, 2 * i).item().toFloat();
            // w1[i * 2 + 1] = p1.select(0, 2 * i + 1).item().toFloat();

            w1_0[i] = p1.select(0, 2 * i).item().toFloat();
            w1_1[i] = p1.select(0, 2 * i + 1).item().toFloat();
        }

        p2 = p2.reshape({width, 1});
        for (size_t i = 0; i < width; i++)
        {
            // b1[i] = p2.select(0, i).item().toFloat();

            b1_[i] = p2.select(0, i).item().toFloat();
        }

        p3 = p3.reshape({width, 1});
        for (size_t i = 0; i < width; i++)
        {
            // w2[i] = p3.select(0, i).item().toFloat();

            w2_[i] = p3.select(0, i).item().toFloat();
        }
        b2 = p4.item().toFloat();
    }

    void print_parameters()
    {
        cout<< "W1" << endl;
        for (size_t i = 0; i < width; i++)
        {
            cout<< w1__[i] << " ";
        }
        cout<< endl;
        cout<< "b1" << endl;
        for (size_t i = 0; i < width; i++)
        {
            cout<< b1_[i] << " ";
        }
        cout<< endl;
        cout<< "W2" << endl;
        for (size_t i = 0; i < width; i++)
        {
            cout<< w2_[i] << " ";
        }
        cout<< endl;
        cout<< "b2" << endl;
        cout<< b2 << endl;
    }

    torch::Tensor forward(torch::Tensor x)
    {
        // Use one of many tensor manipulation functions.
        // x = torch::sigmoid(fc1->forward(x));
        x = torch::relu(fc1->forward(x));
        // x = fc1->forward(x);
        x = fc2->forward(x);
        // x = torch::dropout(x, /*p=*/0.5, /*train=*/is_training());
        // x = torch::relu(fc2->forward(x));
        // x = fc2->forward(x);
        return x;
        // return fc2->forward(fc1->forward(x));
    }

    torch::Tensor predict(torch::Tensor x)
    {
        x = torch::relu(fc1->forward(x));
        x = fc2->forward(x);
        return x;
    }

    // float predictZM(float key)
    // {
    //     float result;
    //     for (size_t i = 0; i < Constants::HIDDEN_LAYER_WIDTH; i++)
    //     {
    //         result += activation(key * w1_[i] + b1[i]) * w2[i];
    //     }
    //     result += b2;
    //     return result;
    // }

    float predict_ZM(float key)
    {
        int blocks = width / 4;
        int rem = width % 4;
        int move_back = blocks * 4;
        __m128 fLoad_w1, fLoad_b1, fLoad_w2;
        __m128 temp1, temp2, temp3;
        __m128 fSum0 = _mm_setzero_ps();
        __m128 fLoad0_x, fLoad0_zeros;
        // _mm_load1_ps
        fLoad0_x = _mm_set_ps(key, key, key, key);
        fLoad0_zeros = _mm_set_ps(0, 0, 0, 0);
        float result;
        for (int i = 0; i < blocks; i++)
        {
            fLoad_w1 = _mm_load_ps(w1__);
            fLoad_b1 = _mm_load_ps(b1_);
            fLoad_w2 = _mm_load_ps(w2_);
            temp1 = _mm_mul_ps(fLoad0_x, fLoad_w1);
            temp2 = _mm_add_ps(temp1, fLoad_b1);

            temp1 = _mm_max_ps(temp2, fLoad0_zeros);

            temp2 = _mm_mul_ps(temp1, fLoad_w2);
            fSum0 = _mm_add_ps(fSum0, temp2);

            w1__ += 4;
            b1_ += 4;
            w2_ += 4;
        }
        result = 0;
        if(blocks > 0)
        {
            result += fSum0[0] + fSum0[1] + fSum0[2] + fSum0[3];
        }
        for (size_t i = 0; i < rem; i++)
        {
            result += activation(key * w1__[i] + b1_[i]) * w2_[i];
        }
        result += b2;
        w1__ -= move_back;
        b1_ -= move_back;
        w2_ -= move_back;
        return result;
    }

    float predict(Point point)
    {
        float x1 = point.x;
        float x2 = point.y;
        int blocks = width / 4;
        int rem = width % 4;
        int move_back = blocks * 4;
        __m128 fLoad_w1_1, fLoad_w1_2, fLoad_b1, fLoad_w2;
        __m128 temp1, temp2, temp3;
        __m128 fSum0 = _mm_setzero_ps();
        __m128 fLoad0_x1, fLoad0_x2, fLoad0_zeros;
        // _mm_load1_ps
        fLoad0_x1 = _mm_set_ps(x1, x1, x1, x1);
        fLoad0_x2 = _mm_set_ps(x2, x2, x2, x2);
        fLoad0_zeros = _mm_set_ps(0, 0, 0, 0);
        float result;
        for (int i = 0; i < blocks; i++)
        {
            fLoad_w1_1 = _mm_load_ps(w1_0);
            fLoad_w1_2 = _mm_load_ps(w1_1);
            fLoad_b1 = _mm_load_ps(b1_);
            fLoad_w2 = _mm_load_ps(w2_);
            temp1 = _mm_mul_ps(fLoad0_x1, fLoad_w1_1);
            temp2 = _mm_mul_ps(fLoad0_x2, fLoad_w1_2);
            temp2 = _mm_add_ps(temp1, temp2);
            temp2 = _mm_add_ps(temp2, fLoad_b1);

            temp1 = _mm_max_ps(temp2, fLoad0_zeros);

            temp2 = _mm_mul_ps(temp1, fLoad_w2);
            fSum0 = _mm_add_ps(fSum0, temp2);

            w1_0 += 4;
            w1_1 += 4;
            b1_ += 4;
            w2_ += 4;
        }
        result = 0;
        if(blocks > 0)
        {
            result += fSum0[0] + fSum0[1] + fSum0[2] + fSum0[3];
        }
        for (size_t i = 0; i < rem; i++)
        {
            result += activation(x1 * w1_0[i] + x2 * w1_1[i] + b1_[i]) * w2_[i];
        }
        result += b2;
        w1_0 -= move_back;
        w1_1 -= move_back;
        b1_ -= move_back;
        w2_ -= move_back;
        return result;
    }

    // float predict(Point point)
    // {
    //     float x1 = point.x;
    //     float x2 = point.y;
    //     float result;
    //     for (int i = 0; i < width; ++i)
    //     {
    //         result += activation(x1 * w1[i * 2] + x2 * w1[i * 2 + 1] + b1[i]) * w2[i];
    //     }
    //     result += b2;
    //     return result;
    // }

    float activation(float val)
    {
        if (val > 0.0)
        {
            return val;
        }
        return 0.0;
    }

    void train_model(vector<float> locations, vector<float> labels)
    {
        long long N = labels.size();
        
        #ifdef use_gpu
            torch::Tensor x = torch::tensor(locations, at::kCUDA).reshape({N, this->input_width});
            torch::Tensor y = torch::tensor(labels, at::kCUDA).reshape({N, 1});
        #else
            torch::Tensor x = torch::tensor(locations).reshape({N, this->input_width});
            torch::Tensor y = torch::tensor(labels).reshape({N, 1});
        #endif
        // torch::Tensor x = torch::tensor(locations).reshape({N, this->input_width});
        // torch::Tensor y = torch::tensor(labels).reshape({N, 1});
        // auto net = isRetrain ? this->net : std::make_shared<Net>(2, width);
        // auto net = std::make_shared<Net>(this->input_width, this->width);
        torch::optim::Adam optimizer(this->parameters(), torch::optim::AdamOptions(this->learning_rate));
        if (N > 64000000)
        {
            int batch_num = 4;

            auto x_chunks = x.chunk(batch_num, 0);
            auto y_chunks = y.chunk(batch_num, 0);
            for (size_t epoch = 0; epoch < Constants::EPOCH; epoch++)
            {
                for (size_t i = 0; i < batch_num; i++)
                {
                    optimizer.zero_grad();
                    torch::Tensor loss = torch::l1_loss(this->forward(x_chunks[i]), y_chunks[i]);
                    #ifdef use_gpu
                        loss.to(torch::kCUDA);
                    #endif
                    loss.backward();
                    optimizer.step();
                }
            }
        }
        else
        {
            for (size_t epoch = 0; epoch < Constants::EPOCH; epoch++)
            {
                optimizer.zero_grad();
                torch::Tensor loss = torch::l1_loss(this->forward(x), y);
                #ifdef use_gpu
                    loss.to(torch::kCUDA);
                #endif
                loss.backward();
                optimizer.step();
            }
        }
    }

    // cal the total errors can compare the error
    // long get_error()
    // {
        
    // }

    // paras SFC target, SFC source , threshold
    bool is_reusable(SFC target, Histogram histogram, string threshold, string &model_path)
    {
        double min_dist = 1.0;
        std::map<string, Histogram>::iterator iter;
        iter = pre_trained_histograms.begin();

        while (iter != pre_trained_histograms.end())
        {
            double temp_dist = iter->second.cal_similarity(histogram);
            // if (iter->first == "uniform_1000_scale_1")
            // {
            //     cout<< iter->second.hist << endl;
            // }
            if (temp_dist < min_dist)
            {
                min_dist = temp_dist;
                model_path = Constants::PRE_TRAIN_MODEL_PATH_ZM + to_string(Constants::RESOLUTION) + "/" + threshold + "/" + iter->first + ".pt";
            }
            iter++;
        }
        // cout<< "min_dist: " << min_dist << endl;
        // if (min_dist > 0.9)
        // {
        //     cout<< "hist: " << histogram.hist << endl;
        //     cout<< "data: " << histogram.data << endl;
        // }
        return true;
    }

    bool is_reusable(Histogram histogram, string threshold, string &model_path)
    {
        double min_dist = 1.0;
        std::map<string, SFC>::iterator iter;
        iter = pre_trained_features.begin();

        while (iter != pre_trained_features.end())
        {
            double temp_dist = iter->second.cal_similarity(histogram);
            if (temp_dist < min_dist)
            {
                min_dist = temp_dist;
                model_path = Constants::PRE_TRAIN_MODEL_PATH_ZM + to_string(Constants::RESOLUTION) + "/" + threshold + "/" + iter->first + ".pt";
            }
            iter++;
        }
        return true;
    }

    bool is_reusable(SFC target, string threshold, string &model_path)
    {
        double min_dist = 1.0;
        std::map<string, SFC>::iterator iter;
        iter = pre_trained_features.begin();

        while (iter != pre_trained_features.end())
        {
            double temp_dist = target.cal_similarity(iter->second);
            if (temp_dist < min_dist)
            {
                min_dist = temp_dist;
                model_path = Constants::PRE_TRAIN_MODEL_PATH_ZM + to_string(Constants::RESOLUTION) + "/" + threshold + "/" + iter->first + ".pt";
            }
            iter++;
        }
        return true;
    }

    // inline static map<string, std::shared_ptr<Net>> pre_trained_models;
    inline static map<string, SFC> pre_trained_features;
    inline static map<string, Histogram> pre_trained_histograms;

    inline static void load_pre_trained_model_zm(string threshold)
    {
        if (!Constants::IS_MODEL_REUSE)
        {
            return;
        }
        if (pre_trained_features.size() > 0)
        {
            return;
        }
        string ppath = Constants::PRE_TRAIN_MODEL_PATH_ZM + to_string(Constants::RESOLUTION) + "/" + threshold + "/";
        cout<< "load_pre_trained_model_zm: ppath:" << ppath << endl;
        struct dirent *ptr;
        DIR *dir;
        dir = opendir(ppath.c_str());
        while ((ptr = readdir(dir)) != NULL)
        {
            if (ptr->d_name[0] == '.')
                continue;
            string file_name_s = ptr->d_name;
            int find_result = file_name_s.find(".pt");
            if (find_result > 0 && find_result <= file_name_s.length())
            {
                string prefix = file_name_s.substr(0, file_name_s.find(".pt"));
                // if (prefix == "OSM_100000000_1_2_")
                //     continue;
                // auto net = std::make_shared<Net>(1);
                // string model_path = Constants::PRE_TRAIN_MODEL_PATH_ZM + "1/" + to_string(threshold) + "/" + file_name_s;
                // torch::load(net, model_path);
                // net->get_parameters_ZM();
                string feature_path = Constants::FEATURES_PATH_ZM + to_string(Constants::RESOLUTION) + "/" + threshold + "/" + prefix + ".csv";
                FileReader reader;
                // TODO use Histogram!!!
                string path = Constants::PRE_TRAIN_1D_DATA + threshold + "/";
                vector<float> features = get_features_z(path, prefix + ".csv");
                Histogram histogram(pow(2, Constants::UNIFIED_Z_BIT_NUM), features);
                pre_trained_histograms.insert(pair<string, Histogram>(prefix, histogram));
            }
            // break;
        }
        cout<< "load finish..." << pre_trained_histograms.size() << endl;
    }

    inline static vector<long long> get_points_bitnum_h(string folder, string file_name, int bit_num)
    {
        FileReader filereader(folder + file_name, ",");
        vector<Point> points = filereader.get_points();
        long long N = points.size();
        long long width = pow(2, bit_num / 2);
        vector<long long> result;
        sort(points.begin(), points.end(), sortX());
        for (int i = 0; i < N; i++)
        {
            points[i].x_i = i;
        }
        sort(points.begin(), points.end(), sortY());
        for (int i = 0; i < N; i++)
        {
            points[i].y_i = i;
            long long curve_val = compute_Hilbert_value(points[i].x_i, points[i].y_i, width);
            points[i].curve_val = curve_val;
            result.push_back(curve_val);
        }
        sort(points.begin(), points.end(), sort_curve_val());
        return result;
    }

    inline static vector<float> get_features_z(string folder, string file_name)
    {
        FileReader filereader(folder + file_name, ",");
        vector<float> features = filereader.read_features();
        return features;
    }

    inline static vector<long long> get_points_bitnum_z(string folder, string file_name, int bit_num)
    {
        FileReader filereader(folder + file_name, ",");
        vector<Point> points = filereader.get_points();
        long long N = points.size();
        int width = pow(2, bit_num / 2);
        vector<long long> result;
        for (int i = 0; i < N; i++)
        {
            points[i].x_i = points[i].x * width;
            points[i].y_i = points[i].y * width;
            long long curve_val = compute_Z_value(points[i].x_i, points[i].y_i, bit_num);
            result.push_back(curve_val);
        }
        sort(result.begin(), result.end());
        return result;
    }

    torch::nn::Linear fc1{nullptr}, fc2{nullptr};
};

#endif