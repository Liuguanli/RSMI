#ifndef REBUILD_H
#define REBUILD_H

#include <dirent.h>
#include <iostream>
#include <string.h>

#include <iomanip>
#include <sstream>

#include "FileReader.h"
#include "FileWriter.h"
#include "Constants.h"
#include "PreTrainRSMI.h"

#include "util.h"

namespace rebuild_index
{
    auto rebuild_model = std::make_shared<Net>(7, 32);
    void build_rebuild_model()
    {
        cout << "build_rebuild_model: " << endl;

        FileReader filereader(",");
        string path = "/home/liuguanli/Dropbox/shared/VLDB20/codes/rsmi/rebuild_model/train_set_formatted.csv";
        string rebuild_model_path = "/home/liuguanli/Dropbox/shared/VLDB20/codes/rsmi/rebuild_model/rebuild_model.pt";
        std::ifstream fin_rebuild(rebuild_model_path);
        if (fin_rebuild)
        {
            torch::load(rebuild_model, rebuild_model_path);
        }
        else
        {
            vector<float> parameters;
            vector<float> labels;
            filereader.get_rebuild_data(path, parameters, labels);
#ifdef use_gpu
            rebuild_model->to(torch::kCUDA);
#endif
            rebuild_model->train_model(parameters, labels);
            torch::save(rebuild_model, rebuild_model_path);
        }
    }

    bool is_rebuild(ExpRecorder &exp_recorder, string type)
    {

        float cardinality = (exp_recorder.previous_insert_num + exp_recorder.dataset_cardinality) * 1.0 / 100000000;
        float cdf_change = exp_recorder.ogiginal_histogram.cal_cdf_dist(exp_recorder.changing_histogram.hist);
        // float relative_depth = exp_recorder.new_depth * 1.0 / exp_recorder.depth;
        float relative_depth = 1.0;
        float update_ratio = exp_recorder.previous_insert_num * 1.0 / exp_recorder.dataset_cardinality;
        string distribution = pre_train_rsmi::get_distribution(exp_recorder.changing_histogram, type);
        vector<float> parameters;
        map<string, vector<float>> distributions = {
            {"normal", {1, 0, 0}}, {"skewed", {0, 1, 0}}, {"uniform", {0, 0, 1}}};
        vector<float> distribution_list = distributions[distribution];
        parameters.push_back(cardinality);
        parameters.push_back(cdf_change);
        parameters.push_back(relative_depth);
        parameters.push_back(update_ratio);
        parameters.insert(parameters.end(), distribution_list.begin(), distribution_list.end());

#ifdef use_gpu
        torch::Tensor x = torch::tensor(parameters, at::kCUDA).reshape({1, 7});
#else
        torch::Tensor x = torch::tensor(parameters).reshape({1, 7});
#endif
        float res = rebuild_model->predict(x).item().toFloat();
        // cout << "rebuild x: " << x << endl;
        // cout << "rebuild res: " << res << endl;
        return res >= 0.5;
    }

    string get_distribution(Histogram hist)
    {
        Histogram uniform = Net::pre_trained_histograms["index_446"];
        Histogram normal = Net::pre_trained_histograms["index_446"];

        double distance = 0;
        distance = hist.cal_similarity(uniform.hist);
        if (distance < 0.1)
        {
            return "uniform";
        }
        distance = hist.cal_similarity(normal.hist);
        if (distance < 0.1)
        {
            return "normal";
        }
        return "skewed";
    }
}

#endif