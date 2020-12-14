#ifndef PRETRAINRSMI_H
#define PRETRAINRSMI_H

#include <dirent.h>
#include <iostream>
#include <string.h>

#include "FileReader.h"
#include "FileWriter.h"

#include "util.h"

namespace pre_train_rsmi
{
    vector<Point> get_points(string folder, string file_name, int resolution)
    {
        FileReader filereader(folder + file_name, ",");
        vector<Point> points = filereader.get_points();
        long long N = points.size();
        long long side = pow(2, ceil(log(N / resolution) / log(2)));
        cout<< "side: " << side << endl;
        sort(points.begin(), points.end(), sortX());
        for (int i = 0; i < N; i++)
        {
            points[i].x_i = i / resolution;
        }
        sort(points.begin(), points.end(), sortY());
        for (int i = 0; i < N; i++)
        {
            points[i].y_i = i / resolution;
            points[i].curve_val = compute_Hilbert_value(points[i].x_i, points[i].y_i, side);
        }
        sort(points.begin(), points.end(), sort_curve_val());

        long min_curve_val = points[0].curve_val;
        long max_curve_val = points[points.size() - 1].curve_val;
        long gap = max_curve_val - min_curve_val;
        for (long long i = 0; i < N; i++)
        {
            points[i].index = i * 1.0 / N;
            points[i].normalized_curve_val = (points[i].curve_val - min_curve_val) * 1.0 / gap;
        }
        cout<< "file_name: " << file_name << endl;
        cout<< "N: " << N << " min_curve_val: " << min_curve_val << " max_curve_val: " << max_curve_val << endl;
        return points;
    }

    void train_model(vector<Point> points, string folder, string file_name, int resolution)
    {
        // TODO generate hist according to normalized_curve_val
        auto net = std::make_shared<Net>(1);
        #ifdef use_gpu
            net->to(torch::kCUDA);
        #endif
        vector<float> locations;
        vector<float> labels;
        for (Point point : points)
        {
            locations.push_back(point.normalized_curve_val);
            labels.push_back(point.index);
        }
        file_name = file_name.substr(0, file_name.find(".csv"));
        cout<< "file_name: " << file_name << endl;
        string features_path = Constants::FEATURES_PATH_RSMI + to_string(resolution) + "/";
        FileWriter SFC_writer(features_path);
        SFC_writer.write_SFC(locations, file_name + ".csv");
        
        string model_path = Constants::PRE_TRAIN_MODEL_PATH_RSMI + to_string(resolution) + "/" + file_name + ".pt";
        std::ifstream fin(model_path);
        if (!fin)
        {
            net->train_model(locations, labels);
            torch::save(net, model_path);
        }
        else
        {
            torch::load(net, model_path);
            net->get_parameters_ZM();
        }
        vector<float> locations1;
        int N = points.size();
        for (size_t i = 0; i < N; i++)
        {
            float pred = net->predict_ZM(points[i].normalized_curve_val);
            pred = pred < 0 ? 0 : pred;
            pred = pred > 1 ? 1 : pred;
            locations1.push_back(pred);
        }
        SFC_writer.write_SFC(locations1, file_name + "learned.csv");
    }

    void test_errors(string file_name_t, int resolution)
    {
        // string path = "OSM_100000000_1_2_.csv";
        // train target to compare.
        string file_name = file_name_t.substr(0, file_name_t.find(".csv"));
        string model_path = Constants::PRE_TRAIN_MODEL_PATH_RSMI + to_string(resolution) + "/" + file_name + ".pt";
        std::ifstream fin(model_path);
        string floder = "/home/liuguanli/Documents/datasets/RLRtree/raw/";
        vector<Point> points = get_points(floder, file_name_t, resolution);
        if (!fin)
        {
            pre_train_rsmi::train_model(points, floder, file_name_t, resolution);
        }
        else
        {
            vector<float> locations;
            for (Point point : points)
            {
                locations.push_back(point.normalized_curve_val);
            }
            string features_path = Constants::FEATURES_PATH_RSMI + to_string(resolution) + "/";
            FileWriter SFC_writer(features_path);
            SFC_writer.write_SFC(locations, file_name + ".csv");
        }
        
        string ppath = Constants::PRE_TRAIN_MODEL_PATH_RSMI + to_string(1) + "/";
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
                auto net = std::make_shared<Net>(1);
                model_path = Constants::PRE_TRAIN_MODEL_PATH_RSMI + "1/" + file_name_s;
                torch::load(net, model_path);
                net->get_parameters_ZM();
                // net->print_parameters();
                double errs = 0;
                int N = points.size();
                for (size_t i = 0; i < N; i++)
                {
                    float pred = net->predict_ZM(points[i].normalized_curve_val);
                    // cout<< "pred: " << pred << endl;
                    if (pred < 0)
                    {
                        pred = 0;
                    }
                    if (pred > 1)
                    {
                        pred = 1;
                    }
                    // cout<< "error:" << abs(pred - points[i].index) << endl;
                    errs += abs(pred - points[i].index);
                    // if (i % 1000000 == 0)
                    // {
                    //     cout<< "pred: " << pred << " index: " << points[i].index << " abs: " << abs(pred - points[i].index) << " errs: " << errs << endl;
                    // }
                    // break;
                }
                // 1.67772e+07 10,0000000
                cout << file_name_s << " errs: " << errs << endl;
            }
            // break;
        }
    }

    void pre_train_1d_Z(int resolution)
    {
        string ppath = Constants::PRE_TRAIN_DATA;
        struct dirent *ptr;
        DIR *dir;
        dir = opendir(ppath.c_str());
        while ((ptr = readdir(dir)) != NULL)
        {
            if (ptr->d_name[0] == '.')
                continue;
            string path = ptr->d_name;
            int find_result = path.find(".csv");
            if (find_result > 0 && find_result <= path.length())
            {
                train_model(get_points(ppath, path, resolution), ppath, path, resolution);
            }
        }
    }
}; // namespace pre_train

#endif