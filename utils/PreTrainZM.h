#ifndef PRETRAINZM_H
#define PRETRAINZM_H

#include <dirent.h>
#include <iostream>
#include <string.h>

#include "FileReader.h"
#include "FileWriter.h"
#include "Constants.h"
#include "../entities/SFC.h"

#include "util.h"

namespace pre_train_zm
{

    void each_cnn(vector<int> sfc, int threshold, int z_value_begin, vector<int> & result, vector<float> & z_values) 
    {
        int length = sfc.size();
        int gap = 1;
        // vector<int> result = sfc;
        // vector<float> z_values;
        result = sfc;
        for (size_t i = 0; i < length; i++)
        {
            z_values.push_back(z_value_begin + 1.0 * i / length);
        }
        while (gap <= length)
        {
            vector<int> temp_result;
            vector<float> temp_z_values;
            for (size_t i = 0; i < length; i += gap)
            {
                int temp_sum = 0;
                for (size_t j = 0; j < gap; j++)
                {
                    temp_sum += sfc[j + i];
                }
                if (temp_sum > threshold)
                {
                    return;
                }
                else
                {
                    temp_result.push_back(temp_sum);
                    temp_z_values.push_back(z_value_begin + i * 1.0 / length);
                }
            }
            result = temp_result;
            z_values = temp_z_values;
            gap = gap * 4;
        }
    }

    void cnn(int conv_width, vector<int> sfc, long cardinality, int threshold, vector<float> & z_values, vector<float> & result_cdf)
    {
        // have a large concolution and constantly shrink it until each cell contains n points && n <= threshold
        vector<float> cdf;
        vector<float> keys;
        long N = sfc.size();
        int gap = pow(conv_width, 2);

        vector<int> result_sfc;
        // vector<float> result_cdf;
        // vector<float> z_values;
        long sum = 0;
        int z_value_begin = 0;
        for (size_t i = 0; i < N; i = i + gap)
        {
            vector<int> sub_sfc(sfc.begin() + i, sfc.begin() + i + gap);
            vector<int> approx_sub_sfc;
            vector<float> approx_sub_z_values;
            each_cnn(sub_sfc, threshold, z_value_begin, approx_sub_sfc, approx_sub_z_values);
            result_sfc.insert(result_sfc.end(), approx_sub_sfc.begin(), approx_sub_sfc.end());
            // z_values.insert(z_values.end(), approx_sub_z_values.begin(), approx_sub_z_values.end());
            for (size_t j = 0; j < approx_sub_sfc.size(); j++)
            {
                sum += approx_sub_sfc[j];
                if (sum < cardinality)
                {
                    result_cdf.push_back(sum * 1.0 / cardinality);
                    z_values.push_back(approx_sub_z_values[j]);
                }
            }
            z_value_begin++;
        }
        float z_max = z_values[z_values.size() - 1];
        float z_min = z_values[0];
        float z_gap = z_max - z_min;
        for (size_t i = 0; i < z_values.size(); i++)
        {
            z_values[i] = (z_values[i] - z_min ) / z_gap;
        }
    }

    vector<int> test_Approximate_SFC(string folder, string file_name)
    {
        FileReader filereader(folder + file_name, ",");
        vector<Point> points = filereader.get_points();
        long long N = points.size();
        int bit_num = ceil((log(N)) / log(2));
        long z_value_num = pow(2, bit_num);
        int side = pow(2, bit_num / 2);
        for (int i = 0; i < N; i++)
        {
            points[i].x_i = points[i].x * side;
            points[i].y_i = points[i].y * side;
            long long curve_val = compute_Z_value(points[i].x_i, points[i].y_i, bit_num / 2);
            points[i].curve_val = curve_val;
        }
        sort(points.begin(), points.end(), sort_curve_val());
        long index = 0;
        vector<int> sfc;
        long sum = 0;
        for (int i = 0; i < z_value_num; i++)
        {
            int num = 0;
            while (points[index].curve_val == i && index < N)
            {
                index++;
                num++;
            }
            sum += num;
            sfc.push_back(num);
        }
        return sfc;
    }

    void test_SFC(string folder, string file_name, int bit_num)
    {
        FileReader filereader(folder + file_name, ",");
        vector<Point> points = filereader.get_points();
        long long N = points.size();
        int side = pow(2, bit_num / 2);
        vector<long long> sfcs;
        vector<float> features;
        vector<int> counter;
        for (int i = 0; i < 64; i++)
        {
            counter.push_back(0);
        }
        for (int i = 0; i < N; i++)
        {
            points[i].x_i = points[i].x * side;
            points[i].y_i = points[i].y * side;
            long long curve_val = compute_Z_value(points[i].x_i, points[i].y_i, bit_num / 2);
            points[i].curve_val = curve_val;
            counter[curve_val] += 1;
        }
        sort(points.begin(), points.end(), sort_curve_val());
        long long max_curve_val = points[N - 1].curve_val;
        long long min_curve_val = points[0].curve_val;
        long long gap = max_curve_val - min_curve_val;
        for (int i = 0; i < N; i++)
        {
            // sfcs.push_back(points[i].curve_val);
            features.push_back((points[i].curve_val - min_curve_val) * 1.0 / gap);
        }

        // A method to record how many curve values are the same
        long num_of_same = 0;
        for (int i = 1; i < N; i++)
        {
            // sfcs.push_back(points[i].curve_val);
            if (points[i-1].curve_val == points[i].curve_val)
            {
                num_of_same++;
            }
            
        }

        // num_of_same: 99999975  25
        // num_of_same: 99776164  223835
        // num_of_same: 14316443  85683557
        // num_of_same: 424660
        cout<< "num_of_same: " << num_of_same << endl;

        cout<< "N: " << N << " min_curve_val: " << min_curve_val << " max_curve_val: " << max_curve_val << endl;

        cout<< "counter: " << counter << endl;
        cout<< "counter.size(): " << counter.size() << endl;
        // SFC sfc(bit_num, sfcs);
        // // cout<< "sfcs.size(): " << sfcs.size() << endl;
        // vector<float> weighted_SFC = sfc.get_weighted_curve();
        // cout<< "weighted_SFC: " << weighted_SFC << endl;

        // vector<int> counted_SFC = sfc.get_counted_courve();
        // cout<< "counted_SFC: " << counted_SFC << endl;

        // string sfc_weight_path = Constants::SFC_Z_WEIGHT + "bit_num_" + to_string(bit_num) + "/";
        // FileWriter SFC_writer(sfc_weight_path);
        // SFC_writer.write_weighted_SFC(weighted_SFC, file_name);

        // string sfc_count_path = Constants::SFC_Z_COUNT + "bit_num_" + to_string(bit_num) + "/";
        // FileWriter SFC_writer_count(sfc_count_path);
        // SFC_writer_count.write_counted_SFC(counted_SFC, file_name);

        string features_path = Constants::FEATURES_PATH_ZM;
        file_utils::check_dir(features_path);
        FileWriter SFC_writer_feature(features_path);

        SFC_writer_feature.write_SFC(features, to_string(bit_num) + "_" + file_name);

        // return sfc;
    }

    vector<Point> get_points(string folder, string file_name, int resolution)
    {
        FileReader filereader(folder + file_name, ",");
        vector<Point> points = filereader.get_points();
        long long N = points.size();
        cout<< "get_points: " << N << endl;
        // cout<< log((N / resolution) * (N / resolution)) << endl;
        int bit_num = 2 * ceil(log((N / resolution)) / log(2));
        cout<< "bit_num: " << bit_num << endl;
        for (int i = 0; i < N; i++)
        {
            points[i].x_i = points[i].x * (N / resolution);
            points[i].y_i = points[i].y * (N / resolution);
            long long curve_val = compute_Z_value(points[i].x_i, points[i].y_i, bit_num);
            points[i].curve_val = curve_val;
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

    void train_model_1d(vector<float> features, string folder, string file_name, int resolution, double threshold)
    {
        // TODO generate hist according to normalized_curve_val
        auto net = std::make_shared<Net>(1);
        #ifdef use_gpu
            net->to(torch::kCUDA);
        #endif
        vector<float> labels;
        long N = features.size();
        sort(features.begin(), features.end());
        for (size_t i = 1; i <= N; i++)
        {
            labels.push_back(i * 1.0 / N);
        }

        file_name = file_name.substr(0, file_name.find(".csv"));
        // cout<< "file_name: " << file_name << endl;
        string features_path = Constants::FEATURES_PATH_ZM + to_string(resolution) + "/" + to_string(threshold) + "/";
        file_utils::check_dir(features_path);
        FileWriter SFC_writer(features_path);
        SFC_writer.write_SFC(features, file_name + ".csv");
        string model_path = Constants::PRE_TRAIN_MODEL_PATH_ZM + to_string(resolution) + "/" + to_string(threshold) + "/";
        file_utils::check_dir(model_path);
        model_path = model_path + file_name + ".pt";
        std::ifstream fin(model_path);
        if (!fin)
        {
            net->train_model(features, labels);
            torch::save(net, model_path);
        }
        else
        {
            torch::load(net, model_path);
            net->get_parameters_ZM();
        }
        vector<float> locations1;
        for (size_t i = 0; i < N; i++)
        {
            float pred = net->predict_ZM(features[i]);
            pred = pred < 0 ? 0 : pred;
            pred = pred > 1 ? 1 : pred;
            locations1.push_back(pred);
        }
        SFC_writer.write_SFC(locations1, file_name + "learned.csv");
    }

    void train_model(vector<Point> points, string folder, string file_name, int resolution, double threshold)
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
        // cout<< "file_name: " << file_name << endl;
        string features_path = Constants::FEATURES_PATH_ZM + to_string(resolution) + "/" + to_string(threshold) + "/";
        file_utils::check_dir(features_path);
        FileWriter SFC_writer(features_path);
        SFC_writer.write_SFC(locations, file_name + ".csv");
        // cout<< "features_path: " << features_path << endl;
        string model_path = Constants::PRE_TRAIN_MODEL_PATH_ZM + to_string(resolution) + "/" + to_string(threshold) + "/";
        file_utils::check_dir(model_path);
        // cout<< "model_path: " << model_path << endl;
        model_path = model_path + file_name + ".pt";
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
        file_name_t = "uniform_1000000_1_2_.csv";
        // train target to compare.
        cout<< "file name: " << file_name_t << endl;
        string file_name = file_name_t.substr(0, file_name_t.find(".csv"));
        string model_path = Constants::PRE_TRAIN_MODEL_PATH_ZM + to_string(resolution) + "/" + file_name + ".pt";
        std::ifstream fin(model_path);
        string floder = "/home/liuguanli/Documents/datasets/RLRtree/raw/";
        vector<Point> points = get_points(floder, file_name_t, resolution);
        if (!fin)
        {
            pre_train_zm::train_model(points, floder, file_name_t, resolution, 0.1);
        }
        else
        {
            vector<float> locations;
            for (Point point : points)
            {
                locations.push_back(point.normalized_curve_val);
            }
            string features_path = Constants::FEATURES_PATH_ZM + to_string(resolution) + "/";
            FileWriter SFC_writer(features_path);
            SFC_writer.write_SFC(locations, file_name + ".csv");
        }
        
        // string ppath = "pre_train/models_zm/" + to_string(1) + "/";
        // struct dirent *ptr;
        // DIR *dir;
        // dir = opendir(ppath.c_str());
        // while ((ptr = readdir(dir)) != NULL)
        // {
        //     if (ptr->d_name[0] == '.')
        //         continue;
        //     string file_name_s = ptr->d_name;
        //     int find_result = file_name_s.find(".pt");
        //     if (find_result > 0 && find_result <= file_name_s.length())
        //     {
        //         auto net = std::make_shared<Net>(1);
        //         model_path = "pre_train/models_zm/1/" + file_name_s;
        //         torch::load(net, model_path);
        //         net->get_parameters_ZM();
        //         // net->print_parameters();
        //         double errs = 0;
        //         int N = points.size();
        //         for (size_t i = 0; i < N; i++)
        //         {
        //             float pred = net->predict_ZM(points[i].normalized_curve_val);
        //             // cout<< "pred: " << pred << endl;
        //             if (pred < 0)
        //             {
        //                 pred = 0;
        //             }
        //             if (pred > 1)
        //             {
        //                 pred = 1;
        //             }
        //             // cout<< "error:" << abs(pred - points[i].index) << endl;
        //             errs += abs(pred - points[i].index);
        //             // if (i % 1000000 == 0)
        //             // {
        //             //     cout<< "pred: " << pred << " index: " << points[i].index << " abs: " << abs(pred - points[i].index) << " errs: " << errs << endl;
        //             // }
        //             // break;
        //         }
        //         // 1.67772e+07 10,0000000
        //         cout << file_name_s << " errs: " << errs << endl;
        //     }
        //     // break;
        // }
    }

    void pre_train_1d_Z(int resolution, double threshold)
    {
        string ppath = Constants::PRE_TRAIN_1D_DATA + to_string(threshold) + "/";
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
                FileReader filereader(ppath + path, ",");
                vector<float> features = filereader.read_features();
                train_model_1d(features, ppath, path, resolution, threshold);
            }
        }
    }

    void pre_train_multid_Z(int resolution, double threshold)
    {
        string ppath = Constants::PRE_TRAIN_DATA + to_string(threshold) + "/";
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
                train_model(get_points(ppath, path, resolution), ppath, path, resolution, threshold);
            }
        }
    }

}; // namespace pre_train

#endif